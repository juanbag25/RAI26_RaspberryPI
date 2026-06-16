import os
import socket
import struct
from dotenv import load_dotenv

load_dotenv()

from audio_capture import LinuxAudioCapture
from config import BACKEND, RESPONSE_PORT
from response_server import start_in_background
from tts_player import TtsPlayer
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

    print(f"Loading transcriber (backend={BACKEND})...")
    transcriber = Transcriber()
    vad = VoiceActivityDetector()
    capture = LinuxAudioCapture()

    # TTS: arranca el servidor que recibe la respuesta del orquestador y la
    # reproduce. `tts.speaking` queda en True mientras suena el audio, para
    # silenciar el micrófono y no transcribir la propia voz del robot.
    tts = TtsPlayer()
    start_in_background(RESPONSE_PORT, tts.speak)
    print(f"TTS response server on 0.0.0.0:{RESPONSE_PORT}")

    print(f"Listening. Sending outputs to {ORCHESTRATOR_IP}:{ORCHESTRATOR_PORT}")
    print("Press Ctrl+C to stop.")

    try:
        for frame in capture.frames():
            # Ignorar audio mientras el robot habla (evita el bucle de feedback).
            if tts.speaking.is_set():
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