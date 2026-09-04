import os
import queue
import socket
import struct
import threading
from dotenv import load_dotenv

load_dotenv()

from audio_capture import LinuxAudioCapture
from config import BACKEND, CTRL_PORT
from ctrl_server import SpeakMute, start_in_background
from vad import VoiceActivityDetector

if BACKEND == "groq":
    from groq_transcriber import GroqTranscriber as Transcriber
elif BACKEND == "local":
    from transcriber import Transcriber
else:
    raise ValueError(f"Unknown BACKEND: {BACKEND!r} (expected 'local' or 'groq')")

Target_IP = "192.168.68.60"

# Timeout de conexion al orquestador: sin esto, socket.connect() usa el
# retry de TCP del SO (puede tardar minutos) si el host no responde ni
# rechaza ni acepta (IP vieja, firewall, WSL que cambio de IP al reiniciar
# el orquestador). Eso colgaba el hilo que lo llama indefinidamente.
ORCHESTRATOR_CONNECT_TIMEOUT_S = 3.0


# --- NUEVA FUNCIÓN DE RED ---
def send_to_orchestrator(text: str, ip: str, port: int) -> None:
    """
    Envía el string al orquestador respetando el protocolo:
    [4 bytes de tamaño en Big Endian] + [N bytes del string]
    """
    try:
        encoded_text = text.encode('utf-8')

        # '!I' empaqueta un entero sin signo (I) de 32 bits en Big Endian (!)
        length_prefix = struct.pack('!I', len(encoded_text))

        # Abrimos el socket, enviamos y cerramos en cada mensaje (matchea
        # tcp_receiver.accept()), con timeout de conexión y de envío para no
        # quedarnos colgados si el orquestador no está realmente accesible
        # (antes usaba socket.connect() a secas, sin timeout: si el host no
        # respondía ni con un rechazo, se podía colgar minutos).
        with socket.create_connection(
            (ip, port), timeout=ORCHESTRATOR_CONNECT_TIMEOUT_S
        ) as s:
            s.settimeout(ORCHESTRATOR_CONNECT_TIMEOUT_S)
            s.sendall(length_prefix + encoded_text)

    except ConnectionRefusedError:
        print(f"[ERROR DE RED] No se pudo conectar al orquestador en {ip}:{port} (¿Está encendido?)")
    except OSError as e:
        print(f"[ERROR DE RED] No se pudo hablar con el orquestador en {ip}:{port}: {e}")
# -----------------------------


def transcribe_worker(
    audio_queue: "queue.Queue",
    transcriber,
    orchestrator_ip: str,
    orchestrator_port: int,
) -> None:
    """Consume utterances cerradas por el VAD y hace el trabajo lento (STT +
    red) fuera del hilo de captura de audio.

    Antes, transcribe() + send_to_orchestrator() corrían inline en el mismo
    loop que lee del stream de PortAudio: mientras esperaban a Whisper/Groq o
    a la conexión TCP, no se drenaban frames del mic y el buffer de captura
    se atrasaba/perdía contenido, sumando desfase a las respuestas del robot.
    """
    while True:
        audio = audio_queue.get()
        if audio is None:  # señal de shutdown
            return
        text = transcriber.transcribe(audio)
        if text:
            print(f">>> {text}")
            send_to_orchestrator(text, orchestrator_ip, orchestrator_port)


def main() -> None:
    # --- CONFIGURACIÓN DE RED ---
    # Lee la IP desde tu archivo .env, o usa una IP fija de respaldo
    ORCHESTRATOR_IP = os.getenv("ORCHESTRATOR_IP", Target_IP)  # <-- ¡Cambia esto por la IP de tu PC!
    ORCHESTRATOR_PORT = 9000

    print("Available audio devices:")
    LinuxAudioCapture.list_devices()
    print()

    # Mic opcional por .env (AUDIO_INPUT_DEVICE = índice de sounddevice).
    # Vacío = dispositivo default del sistema, igual que siempre en la Pi.
    device_env = os.getenv("AUDIO_INPUT_DEVICE", "").strip()
    audio_device = int(device_env) if device_env else None

    print(f"Loading transcriber (backend={BACKEND})...")
    transcriber = Transcriber()
    vad = VoiceActivityDetector()
    capture = LinuxAudioCapture(device_id=audio_device)

    # Mute remoto: el orquestador avisa SPEAK_START/SPEAK_END mientras habla
    # y acá se descartan los frames, así el robot no se transcribe a sí mismo.
    mute = SpeakMute()
    start_in_background(CTRL_PORT, mute)
    print(f"Speak-mute control server on 0.0.0.0:{CTRL_PORT}")

    # STT + envío al orquestador corren en un hilo aparte (ver
    # transcribe_worker): son las dos operaciones lentas/bloqueantes del
    # pipeline y no deben frenar la lectura del stream de audio.
    audio_queue: "queue.Queue" = queue.Queue()
    worker = threading.Thread(
        target=transcribe_worker,
        args=(audio_queue, transcriber, ORCHESTRATOR_IP, ORCHESTRATOR_PORT),
        daemon=True,
    )
    worker.start()

    print(f"Listening. Sending outputs to {ORCHESTRATOR_IP}:{ORCHESTRATOR_PORT}")
    print("Press Ctrl+C to stop.")

    try:
        for frame in capture.frames():
            # El robot está hablando: ignorar audio (evita el autoescucha).
            if mute.is_muted():
                continue
            closed, audio = vad.process_frame(frame)
            if closed and audio is not None:
                audio_queue.put(audio)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
