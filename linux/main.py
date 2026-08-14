import os
import socket
import struct
from dotenv import load_dotenv

load_dotenv()

from audio_capture import LinuxAudioCapture
from config import BACKEND, CTRL_PORT
from ctrl_server import SpeakMute, start_in_background
from vad import VoiceActivityDetector

Target_IP = "192.168.68.60"

if BACKEND == "groq":
    from groq_transcriber import GroqTranscriber as Transcriber
elif BACKEND == "local":
    from transcriber import Transcriber
else:
    raise ValueError(f"Unknown BACKEND: {BACKEND!r} (expected 'local' or 'groq')")

# --- NUEVA FUNCIÓN DE RED ---
def send_to_orchestrator(text: str, ip: str, port: int) -> None:
    """
    Envía el string al orquestador C++ respetando el protocolo:
    [4 bytes de tamaño en Big Endian] + [N bytes del string]
    """
    try:
        encoded_text = text.encode('utf-8')
        
        # '!I' empaqueta un entero sin signo (I) de 32 bits en Big Endian (!)
        length_prefix = struct.pack('!I', len(encoded_text))
        
        # Abrimos el socket, enviamos y cerramos en cada mensaje
        # (Esto hace match con cómo funciona tcp_receiver_accept)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ip, port))
            # sendall asegura que se manden todos los bytes, mandamos prefijo + texto
            s.sendall(length_prefix + encoded_text)
            
    except ConnectionRefusedError:
        print(f"[ERROR DE RED] No se pudo conectar al orquestador en {ip}:{port} (¿Está encendido?)")
    except Exception as e:
        print(f"[ERROR DE RED] Excepción enviando datos: {e}")
# -----------------------------

def main() -> None:
    # --- CONFIGURACIÓN DE RED ---
    # Lee la IP desde tu archivo .env, o usa una IP fija de respaldo
    ORCHESTRATOR_IP = os.getenv("ORCHESTRATOR_IP", Target_IP) # <-- ¡Cambia esto por la IP de tu PC!
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

    print(f"Listening. Sending outputs to {ORCHESTRATOR_IP}:{ORCHESTRATOR_PORT}")
    print("Press Ctrl+C to stop.")

    try:
        for frame in capture.frames():
            # El robot está hablando: ignorar audio (evita el autoescucha).
            if mute.is_muted():
                continue
            closed, audio = vad.process_frame(frame)
            if closed and audio is not None:
                text = transcriber.transcribe(audio)
                if text:
                    print(f">>> {text}")
                    # Enviar el texto si no está vacío
                    send_to_orchestrator(text, ORCHESTRATOR_IP, ORCHESTRATOR_PORT)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()