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
    ORCH_EVENT_PREFIX,
    STT_PROMPT,
    WAKE_EVENTS_ENABLED,
    WAKE_MODE,
    WAKE_PHRASES,
    WAKE_WINDOW_S,
    WAKE_WORD_ENABLED,
)
from ctrl_server import SpeakMute, start_in_background
from log import DEBUG, dbg, dim, drop, err, fmt, info, ok, warn
from vad import VoiceActivityDetector
from wake_sound import WakeSound
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
def send_to_orchestrator(text: str, ip: str, port: int, *, quiet: bool = False) -> bool:
    """
    Envía el string al orquestador respetando el protocolo:
    [4 bytes de tamaño en Big Endian] + [N bytes del string]
    Devuelve True si se envió. `quiet` no loguea el envío exitoso (eventos
    de wake: la línea WAKE ya lo cuenta).
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
        elapsed = time.monotonic() - t0
        if quiet:
            dbg(f"enviado {text!r} a {ip}:{port} ({len(encoded_text)} B, {elapsed:.2f}s)", "NET")
        else:
            ok("NET", f"enviado «{text}» ({elapsed:.2f}s)")
        return True

    except ConnectionRefusedError:
        err("NET", f"{ip}:{port} rechazó la conexión (¿orquestador encendido?)")
    except socket.timeout:
        err("NET", f"timeout ({ORCHESTRATOR_CONNECT_TIMEOUT_S}s) conectando a "
            f"{ip}:{port} (¿IP correcta? ¿misma red? ¿firewall?)")
    except OSError as e:
        err("NET", f"no se pudo hablar con el orquestador en {ip}:{port}: {e}")
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
        self.wakes = 0          # disparos del spotter de audio
        self.asleep = 0         # utterances descartadas por estar dormido (modo audio)
        self.beep_frames = 0    # frames descartados mientras sonaba el beep


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
        if pending:
            warn("STT", f"{pending} utterances esperando en cola (Groq lento?)")
        dbg(f"transcribiendo {seconds:.1f}s de audio...", "STT")
        t0 = time.monotonic()
        text = transcriber.transcribe(audio)
        elapsed = time.monotonic() - t0
        counters.transcribed += 1
        if not text:
            counters.empty += 1
            drop("STT", "Groq no devolvió texto (error arriba, o audio inaudible)",
                 audio_s=f"{seconds:.1f}", nivel=level)
            continue
        info("STT", f"«{text}» ({elapsed:.2f}s)")
        if is_prompt_echo(text):
            drop("STT", "eco del prompt de Whisper: es ruido, no habló nadie")
            continue
        # Wake word: hasta que lo llamen por su nombre, no sale nada de acá.
        payload = wake.filter(text, level)
        if payload:
            if send_to_orchestrator(payload, orchestrator_ip, orchestrator_port):
                counters.sent += 1
            else:
                counters.send_failed += 1


# Cada cuánto el watcher mira si el wake word cambió de estado. Marca el
# retraso máximo del chime de "despierto" respecto del "oye rai" (más el TCP).
WAKE_EVENT_POLL_S = 0.1


def wake_event_worker(
    wake: WakeWord,
    orchestrator_ip: str,
    orchestrator_port: int,
) -> None:
    """Avisa al orquestador cada transición dormido<->despierto.

    Se hace por polling y no con un callback en WakeWord porque dormirse no es
    un evento: la ventana simplemente vence (`is_awake()` pasa a False solo).
    Cubre las dos formas de despertar (spotter de audio y "rai" en el texto) y
    las dos de dormirse (vencimiento y `sleep()`). El envío bloquea hasta 3 s
    si el orquestador no responde, por eso corre en su propio hilo.
    """
    was_awake = wake.is_awake()
    while True:
        time.sleep(WAKE_EVENT_POLL_S)
        awake = wake.is_awake()
        if awake == was_awake:
            continue
        was_awake = awake
        name = "awake" if awake else "asleep"
        if awake:
            dim("WAKE", "aviso al orquestador: despierto")
        else:
            info("WAKE", f"DORMIDO (pasaron {WAKE_WINDOW_S:.0f}s sin hablarme), aviso al orquestador")
        send_to_orchestrator(ORCH_EVENT_PREFIX + name, orchestrator_ip, orchestrator_port,
                             quiet=True)


def heartbeat_worker(
    vad: VoiceActivityDetector,
    mute: SpeakMute,
    wake: WakeWord,
    spotter,
    audio_queue: "queue.Queue",
    counters: Counters,
) -> None:
    """Cada HEARTBEAT_S imprime un resumen del estado del pipeline.

    Con esto se ve de un vistazo dónde se traba: `SIN AUDIO DEL MIC`, el mic
    no entrega audio; `sin voz` mientras hablás, webrtcvad no la detecta (¿mic
    equivocado?); `voz` sube pero `fuerte` queda en 0, el nivel no llega al
    umbral de apertura (subí la ganancia o bajá RMS_THRESHOLD); `utt desc` sin
    `ok`, mirá los `VAD ✗ DESCARTADO` de arriba. Sale en gris si todo está
    bien y en amarillo si detecta un problema.
    """
    last_muted = 0
    last_beep = 0
    last_power = time.monotonic()
    while True:
        time.sleep(HEARTBEAT_S)
        st = vad.pop_stats()
        muted_delta, last_muted = counters.muted_frames - last_muted, counters.muted_frames
        beep_delta, last_beep = counters.beep_frames - last_beep, counters.beep_frames
        got_frames = st.frames + muted_delta + beep_delta
        expected = int(HEARTBEAT_S * 1000 / FRAME_MS)

        problem = False
        parts: list[str] = []

        # 1. ¿Llega audio?
        if got_frames == 0:
            parts.append("SIN AUDIO DEL MIC")
            problem = True
        elif got_frames < expected * 0.8:
            parts.append(f"pocos frames: {got_frames} de ~{expected}")
            problem = True
        if muted_delta:
            parts.append(f"mute {muted_delta * FRAME_MS / 1000:.1f}s")

        # 2. ¿webrtcvad ve voz y llega al umbral de apertura?
        if st.speech_frames == 0:
            parts.append(f"sin voz (rms max {st.max_rms:.4f})")
        else:
            parts.append(f"voz {st.speech_frames * FRAME_MS / 1000:.1f}s, "
                         f"fuerte {st.loud_speech_frames * FRAME_MS / 1000:.1f}s")
        parts.append(fmt(ruido=vad.noise_floor, abre=vad.open_threshold(),
                         cerca=vad.near_threshold()))

        # 3. Utterances de la ventana.
        if st.opened or st.accepted or st.rejected:
            parts.append(f"utt ok={st.accepted} desc={st.rejected}"
                         + (" (una abierta)" if vad.in_speech else ""))

        # 4. Totales del proceso.
        totals = f"total stt={counters.transcribed} env={counters.sent}"
        if counters.empty:
            totals += f" vacías={counters.empty}"
        if counters.send_failed:
            totals += f" FALLIDAS={counters.send_failed}"
            problem = True
        if audio_queue.qsize():
            totals += f" cola={audio_queue.qsize()}"
        parts.append(totals)

        # 5. Estado.
        if mute.is_muted():
            parts.append("MUTE")
        if wake.is_awake():
            parts.append("despierto " + fmt(foco=wake.focus_level(), minimo=wake.min_level()))
        else:
            parts.append("dormido")
        if spotter is not None:
            sp = spotter.pop_stats()
            if sp.last_text:
                parts.append(f"spotter oyó «{sp.last_text}»")
            if sp.dropped:
                parts.append(f"spotter atrasado: {sp.dropped} frames perdidos (¿CPU?)")
                problem = True

        line = " · ".join(parts)
        if problem:
            warn("HB", line)
        else:
            dim("HB", line)
        if POWER_LOG_S > 0 and time.monotonic() - last_power >= POWER_LOG_S:
            last_power = time.monotonic()
            report_power()


def main() -> None:
    # --- CONFIGURACIÓN DE RED ---
    # Lee la IP desde tu archivo .env, o usa una IP fija de respaldo
    ORCHESTRATOR_IP = os.getenv("ORCHESTRATOR_IP", Target_IP)  # <-- ¡Cambia esto por la IP de tu PC!
    ORCHESTRATOR_PORT = 9000

    info("MAIN", "=== STT Pi arrancando ===")

    # Alimentación: tensión de entrada y flags de undervoltage de la Pi.
    report_power()

    dim("AUDIO", "dispositivos de audio:")
    LinuxAudioCapture.list_devices()

    # Mic opcional por .env (AUDIO_INPUT_DEVICE = índice de sounddevice).
    # Vacío = dispositivo default del sistema, igual que siempre en la Pi.
    device_env = os.getenv("AUDIO_INPUT_DEVICE", "").strip()
    audio_device = int(device_env) if device_env else None

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
    dim("CTRL", f"espero SPEAK_START/SPEAK_END del orquestador en :{CTRL_PORT}")

    # Wake por audio: spotter local (Vosk) + beep. Si no puede arrancar, el
    # robot sigue funcionando con el wake por texto de siempre.
    spotter = None
    sound = WakeSound()
    if WAKE_WORD_ENABLED and WAKE_MODE == "audio":
        try:
            from wake_spotter import WakeSpotter
            spotter = WakeSpotter()
        except Exception as exc:  # noqa: BLE001 - ImportError, modelo ausente...
            warn("SPOT", f"no pude arrancar el wake por audio ({type(exc).__name__}: {exc}); "
                 f"caigo a WAKE_MODE=text")
    elif WAKE_WORD_ENABLED and WAKE_MODE != "text":
        warn("SPOT", f"WAKE_MODE={WAKE_MODE!r} desconocido, uso 'text'")

    if not WAKE_WORD_ENABLED:
        warn("WAKE", "wake word DESACTIVADO (WAKE_WORD_ENABLED=False): se envía todo")
    elif spotter is not None:
        info("WAKE", f"modo AUDIO: decí «{WAKE_PHRASES[0]}» y esperá el beep; "
             f"dormido no transcribo nada; ventana {WAKE_WINDOW_S:.0f}s")
    else:
        info("WAKE", f"modo TEXTO: decí «rai» al principio de la frase; "
             f"ventana {WAKE_WINDOW_S:.0f}s")
    dim("MAIN", f"heartbeat cada {HEARTBEAT_S:.0f}s (LOG_HEARTBEAT_S), "
        f"debug {'ON' if DEBUG else 'off'} (LOG_DEBUG=1)")

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

    # Chime remoto: el orquestador suena al despertar / dormirse (ver
    # wake_event_worker). Independiente del beep local (WAKE_SOUND).
    if WAKE_WORD_ENABLED and WAKE_EVENTS_ENABLED:
        threading.Thread(
            target=wake_event_worker,
            args=(wake, ORCHESTRATOR_IP, ORCHESTRATOR_PORT),
            daemon=True,
        ).start()

    if HEARTBEAT_S > 0:
        threading.Thread(
            target=heartbeat_worker,
            args=(vad, mute, wake, spotter, audio_queue, counters),
            daemon=True,
        ).start()

    info("NET", f"orquestador en {ORCHESTRATOR_IP}:{ORCHESTRATOR_PORT}")

    try:
        first_frame = True
        was_muted = False
        for frame in capture.frames():
            if first_frame:
                first_frame = False
                ok("AUDIO", "primer frame del mic: escuchando")
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
            # Suena el beep de wake: no escucharse a sí mismo.
            if sound.is_playing():
                counters.beep_frames += 1
                continue
            if spotter is not None:
                spotter.feed(frame)
                heard = spotter.take_detection()
                if heard:
                    counters.wakes += 1
                    # El nivel de la utterance en curso es el de quien dijo
                    # "oye rai": a eso le prestamos atención.
                    wake.wake_from_audio(vad.current_level())
                    # El "oye rai" ya cumplió: no gastar Groq en transcribirlo.
                    # Si la persona sigue hablando, el VAD abre otra utterance
                    # enseguida (pre-buffer de 200 ms) y esa sí va a Groq.
                    vad.discard_open_utterance(reason="era el «oye rai», no hace falta transcribirlo")
                    sound.play()
                    continue
            closed, audio = vad.process_frame(frame)
            if closed and audio is not None:
                level = vad.last_level
                # Modo audio: dormido no se transcribe nada. Sólo el spotter
                # escucha, y hasta que no oiga "oye rai" Groq no se entera.
                if spotter is not None and not wake.is_awake():
                    counters.asleep += 1
                    drop("WAKE", f"dormido: no transcribo hasta oír «{WAKE_PHRASES[0]}»",
                         nivel=level)
                    continue
                # Atención: despierto, sólo transcribimos lo que suena tan
                # fuerte como quien dijo "rai" (el fondo no gasta Whisper).
                if not wake.accepts_level(level):
                    counters.unfocused += 1
                    continue
                audio_queue.put((audio, level))

    except KeyboardInterrupt:
        info("MAIN", "Stopped.")


if __name__ == "__main__":
    main()
