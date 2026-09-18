import os
import queue
import socket
import struct
import sys
import threading
import time
from dotenv import load_dotenv

load_dotenv()

# Si stdout no es una terminal (systemd, nohup, `> log.txt`), Python bufferea
# por bloques y los prints aparecen minutos después o nunca. Forzar line
# buffering acá evita tener que acordarse de `python -u`.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

from audio_capture import LinuxAudioCapture
from battery import POWER_LOG_S, report_power
from config import (
    BACKEND,
    CTRL_PORT,
    FRAME_MS,
    NEAR_RMS_THRESHOLD,
    STT_PROMPT,
    WAKE_WINDOW_S,
    WAKE_WORD_ENABLED,
)
from ctrl_server import SpeakMute, start_in_background
from log import DEBUG, log
from vad import VoiceActivityDetector
from wake_word import WakeWord, normalize

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

# Cada cuánto imprimir el resumen de "qué está viendo el mic" (0 = nunca).
# Es la línea a mirar cuando "se queda escuchando y no pasa nada".
HEARTBEAT_S = float(os.getenv("LOG_HEARTBEAT_S", "10") or 0)

_NORMALIZED_PROMPT = normalize(STT_PROMPT)


def is_prompt_echo(text: str) -> bool:
    """Whisper devolvió (parte de) STT_PROMPT en vez de transcribir.

    Pasa con audio flojo o ruido: el modelo se agarra del prompt y lo repite.
    Sin este chequeo, ese eco despertaría al robot solo. Se piden >=4 palabras
    para no descartar frases cortas legítimas ("rai", "hola") que casualmente
    aparecen en el prompt.
    """
    normalized = normalize(text)
    return len(normalized.split()) >= 4 and normalized in _NORMALIZED_PROMPT


# --- NUEVA FUNCIÓN DE RED ---
def send_to_orchestrator(text: str, ip: str, port: int) -> bool:
    """
    Envía el string al orquestador respetando el protocolo:
    [4 bytes de tamaño en Big Endian] + [N bytes del string]
    Devuelve True si se envió.
    """
    t0 = time.monotonic()
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
        log(f"[NET] enviado al orquestador {ip}:{port} "
            f"({len(encoded_text)} B en {time.monotonic() - t0:.2f}s): «{text}»")
        return True

    except ConnectionRefusedError:
        log(f"[NET ERROR] {ip}:{port} rechazó la conexión (¿orquestador encendido?)", err=True)
    except socket.timeout:
        log(f"[NET ERROR] timeout ({ORCHESTRATOR_CONNECT_TIMEOUT_S}s) conectando a "
            f"{ip}:{port} (¿IP correcta? ¿misma red? ¿firewall?)", err=True)
    except OSError as e:
        log(f"[NET ERROR] no se pudo hablar con el orquestador en {ip}:{port}: {e}", err=True)
    return False
# -----------------------------


class Counters:
    """Totales del proceso, para el heartbeat."""

    def __init__(self) -> None:
        self.transcribed = 0    # llamadas a STT
        self.empty = 0          # STT devolvió vacío
        self.sent = 0           # mensajes que llegaron al orquestador
        self.send_failed = 0
        self.muted_frames = 0   # frames descartados por SPEAK_START
        self.unfocused = 0      # utterances más flojas que quien nos llamó


def transcribe_worker(
    audio_queue: "queue.Queue",
    transcriber,
    wake: WakeWord,
    orchestrator_ip: str,
    orchestrator_port: int,
    counters: Counters,
) -> None:
    """Consume utterances cerradas por el VAD y hace el trabajo lento (STT +
    red) fuera del hilo de captura de audio.

    Antes, transcribe() + send_to_orchestrator() corrían inline en el mismo
    loop que lee del stream de PortAudio: mientras esperaban a Whisper/Groq o
    a la conexión TCP, no se drenaban frames del mic y el buffer de captura
    se atrasaba/perdía contenido, sumando desfase a las respuestas del robot.
    """
    while True:
        item = audio_queue.get()
        if item is None:  # señal de shutdown
            return
        audio, level = item
        seconds = len(audio) / 16000.0
        pending = audio_queue.qsize()
        log(f"[STT] transcribiendo {seconds:.1f}s de audio"
            + (f" ({pending} más en cola)" if pending else "") + "...")
        t0 = time.monotonic()
        text = transcriber.transcribe(audio)
        elapsed = time.monotonic() - t0
        counters.transcribed += 1
        if not text:
            counters.empty += 1
            log(f"[STT] vacío en {elapsed:.2f}s (error de Groq arriba, o Whisper no "
                f"entendió nada: ¿audio muy flojo?)")
            continue
        log(f"[STT] {elapsed:.2f}s >>> {text}")
        if is_prompt_echo(text):
            log("[STT] eco del prompt (ruido), descartado")
            continue
        # Wake word: hasta que lo llamen por su nombre, no sale nada de acá.
        payload = wake.filter(text, level)
        if payload:
            if send_to_orchestrator(payload, orchestrator_ip, orchestrator_port):
                counters.sent += 1
            else:
                counters.send_failed += 1


def heartbeat_worker(
    vad: VoiceActivityDetector,
    mute: SpeakMute,
    wake: WakeWord,
    audio_queue: "queue.Queue",
    counters: Counters,
) -> None:
    """Cada HEARTBEAT_S imprime un resumen del estado del pipeline.

    Con esto se ve de un vistazo dónde se traba: si `frames` no crece, el mic
    no entrega audio; si `voz` es 0 mientras hablás, webrtcvad no la detecta
    (¿mic equivocado?); si `voz` sube pero `voz>umbral` no, el nivel no llega
    (subí la ganancia o bajá RMS_THRESHOLD); si se abren utterances pero no se
    aceptan, mirá los `[VAD] descartada` de arriba.
    """
    last_muted = 0
    last_power = time.monotonic()
    while True:
        time.sleep(HEARTBEAT_S)
        st = vad.pop_stats()
        muted_now = counters.muted_frames
        muted_delta, last_muted = muted_now - last_muted, muted_now
        expected = int(HEARTBEAT_S * 1000 / FRAME_MS)
        audio_note = ""
        if st.frames + muted_delta == 0:
            audio_note = "  <-- SIN AUDIO DEL MIC"
        elif st.frames + muted_delta < expected * 0.8:
            audio_note = f"  <-- llegan pocos frames (esperados ~{expected})"
        log(
            f"[HB] frames={st.frames} muteados={muted_delta} "
            f"voz={st.speech_frames} voz>umbral={st.loud_speech_frames} | "
            f"rms max={st.max_rms:.4f} media={st.mean_rms:.4f} "
            f"ruido={vad.noise_floor:.4f} umbral_abrir={vad.open_threshold():.4f} "
            f"cerca={NEAR_RMS_THRESHOLD} | "
            f"utt abiertas={st.opened} ok={st.accepted} desc={st.rejected} "
            f"en_utt={'sí' if vad.in_speech else 'no'} | "
            f"cola_stt={audio_queue.qsize()} stt={counters.transcribed} "
            f"vacías={counters.empty} enviadas={counters.sent} "
            f"fallidas={counters.send_failed} | "
            f"mute={'SÍ' if mute.is_muted() else 'no'} "
            f"wake={'despierto' if wake.is_awake() else 'dormido'}"
            + (f" foco={wake.focus_level():.4f} mínimo={wake.min_level():.4f} "
               f"ignoradas_por_foco={counters.unfocused}" if wake.is_awake() else "")
            + f"{audio_note}"
        )
        if POWER_LOG_S > 0 and time.monotonic() - last_power >= POWER_LOG_S:
            last_power = time.monotonic()
            report_power()


def main() -> None:
    # --- CONFIGURACIÓN DE RED ---
    # Lee la IP desde tu archivo .env, o usa una IP fija de respaldo
    ORCHESTRATOR_IP = os.getenv("ORCHESTRATOR_IP", Target_IP)  # <-- ¡Cambia esto por la IP de tu PC!
    ORCHESTRATOR_PORT = 9000

    log("=== STT Pi arrancando ===")

    # Alimentación: tensión de entrada y flags de undervoltage de la Pi.
    report_power()

    log("Available audio devices:")
    LinuxAudioCapture.list_devices()
    print()

    # Mic opcional por .env (AUDIO_INPUT_DEVICE = índice de sounddevice).
    # Vacío = dispositivo default del sistema, igual que siempre en la Pi.
    device_env = os.getenv("AUDIO_INPUT_DEVICE", "").strip()
    audio_device = int(device_env) if device_env else None
    log(f"Mic: {'default del sistema' if audio_device is None else f'device {audio_device}'}"
        f" (AUDIO_INPUT_DEVICE={device_env!r})")

    log(f"Loading transcriber (backend={BACKEND})...")
    transcriber = Transcriber()
    vad = VoiceActivityDetector()
    capture = LinuxAudioCapture(device_id=audio_device)
    wake = WakeWord()
    counters = Counters()

    # Mute remoto: el orquestador avisa SPEAK_START/SPEAK_END mientras habla
    # y acá se descartan los frames, así el robot no se transcribe a sí mismo.
    # El SPEAK_END además renueva la ventana del wake word: el robot acaba de
    # contestar, lo natural es que le sigan hablando sin repetir el nombre.
    mute = SpeakMute(on_speak_end=wake.refresh)
    start_in_background(CTRL_PORT, mute)
    log(f"Speak-mute control server on 0.0.0.0:{CTRL_PORT}")
    if WAKE_WORD_ENABLED:
        log(f"Wake word activo: decile «rai» para que escuche "
            f"(ventana de {WAKE_WINDOW_S:.0f}s por turno)")
    else:
        log("Wake word DESACTIVADO (WAKE_WORD_ENABLED=False en config.py)")
    log(f"Foco del mic: umbral de cercanía NEAR_RMS_THRESHOLD="
        f"{NEAR_RMS_THRESHOLD} (calibrar con mic_level.py)")
    log(f"Logs: heartbeat cada {HEARTBEAT_S:.0f}s (LOG_HEARTBEAT_S), "
        f"debug por frame={'ON' if DEBUG else 'off'} (LOG_DEBUG=1), "
        f"alimentación cada {POWER_LOG_S:.0f}s (POWER_LOG_S)")

    # STT + envío al orquestador corren en un hilo aparte (ver
    # transcribe_worker): son las dos operaciones lentas/bloqueantes del
    # pipeline y no deben frenar la lectura del stream de audio.
    audio_queue: "queue.Queue" = queue.Queue()
    worker = threading.Thread(
        target=transcribe_worker,
        args=(audio_queue, transcriber, wake, ORCHESTRATOR_IP, ORCHESTRATOR_PORT, counters),
        daemon=True,
    )
    worker.start()

    if HEARTBEAT_S > 0:
        threading.Thread(
            target=heartbeat_worker,
            args=(vad, mute, wake, audio_queue, counters),
            daemon=True,
        ).start()

    log(f"Listening. Sending outputs to {ORCHESTRATOR_IP}:{ORCHESTRATOR_PORT}")
    log("Press Ctrl+C to stop.")

    try:
        first_frame = True
        was_muted = False
        for frame in capture.frames():
            if first_frame:
                first_frame = False
                log("[AUDIO] primer frame recibido del mic: el stream funciona")
            # El robot está hablando: ignorar audio (evita el autoescucha).
            if mute.is_muted():
                if not was_muted:
                    was_muted = True
                    # Lo que quedó a medio decir cuando el robot arrancó a
                    # hablar es viejo: no dejar que se cierre (y se envíe)
                    # recién al desmutear.
                    vad.discard_open_utterance()
                counters.muted_frames += 1
                continue
            was_muted = False
            closed, audio = vad.process_frame(frame)
            if closed and audio is not None:
                # Atención: despierto, sólo transcribimos lo que suena tan
                # fuerte como quien dijo "rai" (el fondo no gasta Whisper).
                level = vad.last_level
                if not wake.accepts_level(level):
                    counters.unfocused += 1
                    continue
                audio_queue.put((audio, level))
                log(f"[MAIN] utterance encolada para STT (nivel={level:.4f}, "
                    f"cola={audio_queue.qsize()})")

    except KeyboardInterrupt:
        log("Stopped.")


if __name__ == "__main__":
    main()
