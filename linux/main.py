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

from collections import deque

from audio_capture import AudioFrame, LinuxAudioCapture
from battery import POWER_LOG_S, report_power
from config import (
    ATTENTION_LEVEL_RATIO,
    ATTENTION_LEVEL_RATIO_ARRAY,
    BACKEND,
    CTRL_PORT,
    DOA_MIN_IN_FOCUS,
    DOA_POLL_HZ,
    DOA_SPEAKER_SECTOR,
    DOA_TOLERANCE_DEG,
    FRAME_MS,
    MIC_MODE,
    ORCH_EVENT_PREFIX,
    RESPEAKER_ENABLED,
    RESPEAKER_LED_BRIGHTNESS,
    RESPEAKER_PARAMS,
    SPEAK_BARGE_RATIO,
    SPEAK_LISTEN_MODE,
    SPEAK_STOP_PHRASES,
    STT_PROMPT,
    WAKE_ACK_DECIDE_MS,
    WAKE_DOA_WINDOW_S,
    WAKE_FOLLOW_SPEECH_MS,
    WAKE_MODE,
    WAKE_PHRASES,
    WAKE_WINDOW_S,
    WAKE_WORD_ENABLED,
)
from ctrl_server import SpeakMute, start_in_background
from doa import DoaFocus, DoaReading, in_any_sector, parse_sectors, to_robot_frame
from log import DEBUG, dbg, dim, drop, err, fmt, info, ok, warn
from respeaker import ReSpeaker
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

# Si una utterance lleva más de esto esperando en la cola de STT, se descarta
# en vez de transcribirla: sin este tope, si STT+red tardan más que el ritmo
# al que habla la gente, se arma un backlog y el robot termina contestando
# cosas viejas mucho después de que se dijeron.
MAX_QUEUE_AGE_S = float(os.getenv("MAX_QUEUE_AGE_S", "4") or 4)

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
        self.off_focus = 0      # utterances descartadas por dirección (DoA)
        self.barge_ins = 0      # órdenes de corte enviadas mientras el robot hablaba
        self.dropped_stale = 0  # utterances descartadas por vieja/atrasada en cola


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
        # Si mientras esperábamos se acumularon más utterances, sólo nos
        # importa la más nueva: si no, se responde tarde a algo que la
        # persona ya dio por perdido (y probablemente repitió).
        while True:
            try:
                newer = audio_queue.get_nowait()
            except queue.Empty:
                break
            counters.dropped_stale += 1
            if newer is None:  # señal de shutdown
                return
            drop("STT", "superada por una más nueva en cola")
            item = newer
        # wake_prefix: la frase empieza con el «oye rai» que oyó el spotter
        # (la persona siguió hablando sin esperar el «Sí, dime»).
        audio, level, queued_at, wake_prefix = item
        age = time.monotonic() - queued_at
        if age > MAX_QUEUE_AGE_S:
            counters.dropped_stale += 1
            drop("STT", "vieja en cola", edad_s=f"{age:.1f}", nivel=level)
            continue
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
        if wake_prefix:
            text = wake.strip_wake(text)
            if not text:
                # Al final era sólo el nombre: ahora sí, «Sí, dime».
                dim("WAKE", "la frase era sólo el «oye rai»: pido el «Sí, dime»")
                wake.announce()
                continue
        # Wake word: hasta que lo llamen por su nombre, no sale nada de acá.
        payload = wake.filter(text, level)
        if payload:
            if send_to_orchestrator(payload, orchestrator_ip, orchestrator_port):
                counters.sent += 1
                # No dormirse mientras el robot piensa la respuesta.
                wake.await_reply()
            else:
                counters.send_failed += 1


# Cada cuánto el emisor de eventos mira si hubo un wake o si se durmió. Marca
# el retraso máximo del «Sí, dime» respecto del «oye rai» (más el TCP).
WAKE_EVENT_POLL_S = 0.1


class OrchestratorEvents:
    """Eventos para el orquestador (`@@event:<nombre>`, mismo socket que el
    texto). La Pi no tiene parlante: así se entera el orquestador de que tiene
    que decir «Sí, dime» (awake), sonar el chime de dormirse (asleep), cortar
    lo que está diciendo (stop) o registrar la dirección (doa:<grados>).

    `awake` sale por CADA wake (`WakeWord.wake_seq`), también estando ya
    despierto: otra persona dijo «oye rai» y toma el foco. `asleep` sale al
    vencerse la ventana: no es un evento, es tiempo que pasa, por eso se mira
    por polling. Los envíos bloquean hasta 3 s si el orquestador no responde:
    corren en su propio hilo, en orden.
    """

    def __init__(self, wake: WakeWord, ip: str, port: int, on_asleep=None) -> None:
        self._wake = wake
        self._ip = ip
        self._port = port
        self._on_asleep = on_asleep
        self._queue: "queue.Queue[str]" = queue.Queue()

    def send(self, name: str) -> None:
        self._queue.put(name)

    def start(self) -> None:
        threading.Thread(target=self._run, name="orch-events", daemon=True).start()

    def _run(self) -> None:
        seen_seq = self._wake.wake_seq()
        was_awake = self._wake.is_awake()
        while True:
            try:
                name = self._queue.get(timeout=WAKE_EVENT_POLL_S)
            except queue.Empty:
                name = None
            seq = self._wake.wake_seq()
            if seq != seen_seq:
                seen_seq = seq
                dim("WAKE", "aviso al orquestador: despierto («Sí, dime»)")
                self._post("awake")
            awake = self._wake.is_awake()
            if was_awake and not awake:
                info("WAKE", f"DORMIDO (pasaron {WAKE_WINDOW_S:.0f}s sin hablarme), aviso al orquestador")
                if self._on_asleep is not None:
                    self._on_asleep()
                self._post("asleep")
            was_awake = awake
            if name is not None:
                self._post(name)

    def _post(self, name: str) -> None:
        send_to_orchestrator(ORCH_EVENT_PREFIX + name, self._ip, self._port, quiet=True)


class Spatial:
    """Dirección de la voz con el ReSpeaker (None-safe: sin array, todo pasa).

    Junta respeaker.py (lecturas de DoA) y doa.py (foco) para lo que main
    necesita: dirección del «oye rai», juicio de cada frase y LEDs.
    """

    def __init__(self, array: ReSpeaker | None) -> None:
        self.array = array
        self.focus = DoaFocus()
        self.speaker_sector = parse_sectors(DOA_SPEAKER_SECTOR)

    @property
    def enabled(self) -> bool:
        return self.array is not None and self.array.polling

    def angles(self, t0: float, t1: float) -> list[float]:
        """Ángulos (marco del robot) de las lecturas CON VOZ en [t0, t1]."""
        if not self.enabled:
            return []
        return [to_robot_frame(s.angle) for s in self.array.samples_between(t0, t1) if s.voice]

    def reading(self, t0: float, t1: float) -> DoaReading:
        return self.focus.summarize(self.angles(t0, t1))

    def from_speaker(self, t0: float, t1: float) -> bool:
        """¿La voz de [t0, t1] viene mayormente del parlante del robot?"""
        angles = self.angles(t0, t1)
        if not self.speaker_sector or len(angles) < 2:
            return False
        return sum(in_any_sector(a, self.speaker_sector) for a in angles) / len(angles) >= 0.5

    def leds_awake(self) -> None:
        if self.array is not None:
            self.array.leds_listen()

    def leds_asleep(self) -> None:
        if self.array is not None:
            self.array.leds_off()

    def on_asleep(self) -> None:
        self.focus.clear()
        self.leds_asleep()


def _deg(angle: float | None) -> str:
    return "?" if angle is None else f"{angle:.0f}°"


class BargeIn:
    """SPEAK_LISTEN_MODE=keyword: mientras el robot habla, el spotter sigue
    escuchando sólo las órdenes de corte. Para no cortarse solo con su propia
    voz, una detección vale si (1) no viene del sector del parlante y (2) en el
    último ~0.8 s hubo un pico SPEAK_BARGE_RATIO veces más fuerte que el nivel
    típico del parlante durante este mute (mediana de los frames anteriores)."""

    _RECENT_S = 0.8

    def __init__(self) -> None:
        self._levels: deque[float] = deque(maxlen=int(10_000 / FRAME_MS))  # ~10 s

    def reset(self) -> None:
        self._levels.clear()

    def feed(self, rms: float) -> None:
        self._levels.append(rms)

    def loud_enough(self) -> tuple[bool, float, float]:
        recent_n = max(1, int(self._RECENT_S * 1000 / FRAME_MS))
        levels = list(self._levels)
        if len(levels) <= recent_n + 5:
            return False, 0.0, 0.0
        recent, before = levels[-recent_n:], sorted(levels[:-recent_n])
        speaker = before[len(before) // 2]
        peak = max(recent)
        return peak >= speaker * SPEAK_BARGE_RATIO, peak, speaker


def heartbeat_worker(
    vad: VoiceActivityDetector,
    mute: SpeakMute,
    wake: WakeWord,
    spotter,
    spatial: Spatial,
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
    last_power = time.monotonic()
    while True:
        time.sleep(HEARTBEAT_S)
        st = vad.pop_stats()
        muted_delta, last_muted = counters.muted_frames - last_muted, counters.muted_frames
        got_frames = st.frames + muted_delta
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
        if counters.dropped_stale:
            totals += f" descartadas_viejas={counters.dropped_stale}"
        if counters.off_focus:
            totals += f" fuera_de_foco={counters.off_focus}"
        if counters.barge_ins:
            totals += f" cortes={counters.barge_ins}"
        parts.append(totals)

        # 5. Estado.
        if mute.is_muted():
            parts.append("MUTE")
        if wake.is_awake():
            awake = "despierto"
            if spatial.enabled:
                awake += f" foco={_deg(spatial.focus.focus)}±{DOA_TOLERANCE_DEG:.0f}"
            if wake.min_level() > 0:
                awake += " " + fmt(nivel_ref=wake.focus_level(), minimo=wake.min_level())
            parts.append(awake)
        else:
            parts.append("dormido")
        if spatial.array is not None:
            if not spatial.enabled:
                parts.append("DoA CAÍDO")
                problem = True
            else:
                last = spatial.array.latest()
                if last is not None:
                    parts.append(f"doa {_deg(to_robot_frame(last.angle))}"
                                 f"{' voz' if last.voice else ''} ({spatial.array.poll_hz:.0f}/s)")
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

    # Mic array: parámetros DSP + lecturas de dirección (DoA) por USB.
    info("AUDIO", f"MIC_MODE={MIC_MODE} "
         + ("(ReSpeaker: foco por dirección)" if MIC_MODE == "array" else "(mic común, mono)"))
    if MIC_MODE == "array" and not capture.is_array:
        warn("AUDIO", "MIC_MODE=array pero no encontré el ReSpeaker: uso el mic default "
             "como mono, con los umbrales *_ARRAY (¿era MIC_MODE=normal?)")
    array = ReSpeaker.open() if RESPEAKER_ENABLED else None
    if array is not None and not capture.is_array:
        warn("ARRAY", "el ReSpeaker está conectado pero el audio sale de otro mic "
             "(AUDIO_INPUT_DEVICE): su DoA no sirve para ese audio, lo ignoro")
        array = None
    if array is not None:
        array.apply_params(RESPEAKER_PARAMS)
        array.leds_brightness(RESPEAKER_LED_BRIGHTNESS)
        array.leds_off()
        array.start_polling(DOA_POLL_HZ)
        ok("ARRAY", f"ReSpeaker: foco por dirección activo (DoA a {DOA_POLL_HZ:.0f} Hz)")
    elif capture.is_array:
        warn("ARRAY", "audio del ReSpeaker pero sin control USB (pyusb/permisos, ver "
             "README): sin DoA ni parámetros DSP, foco sólo por nivel")
    spatial = Spatial(array)

    wake = WakeWord(level_ratio=(ATTENTION_LEVEL_RATIO_ARRAY if spatial.enabled
                                 else ATTENTION_LEVEL_RATIO))
    counters = Counters()

    # Mute remoto: el orquestador avisa SPEAK_START/SPEAK_END mientras habla
    # y acá se descartan los frames, así el robot no se transcribe a sí mismo.
    # El SPEAK_END además renueva la ventana del wake word: el robot acaba de
    # contestar, lo natural es que le sigan hablando sin repetir el nombre.
    # El SPEAK_START congela la ventana mientras el robot habla.
    mute = SpeakMute(on_speak_end=wake.refresh, on_speak_start=wake.hold)
    start_in_background(CTRL_PORT, mute)
    dim("CTRL", f"espero SPEAK_START/SPEAK_END del orquestador en :{CTRL_PORT}")

    # Wake por audio: spotter local (Vosk). Si no puede arrancar, el robot
    # sigue funcionando con el wake por texto de siempre.
    barge_mode = SPEAK_LISTEN_MODE == "keyword"
    if SPEAK_LISTEN_MODE not in ("mute", "keyword"):
        warn("CTRL", f"SPEAK_LISTEN_MODE={SPEAK_LISTEN_MODE!r} desconocido, uso 'mute'")
        barge_mode = False
    spotter = None
    if WAKE_WORD_ENABLED and WAKE_MODE == "audio":
        try:
            from wake_spotter import WakeSpotter
            spotter = WakeSpotter(stop_phrases=SPEAK_STOP_PHRASES if barge_mode else ())
        except Exception as exc:  # noqa: BLE001 - ImportError, modelo ausente...
            warn("SPOT", f"no pude arrancar el wake por audio ({type(exc).__name__}: {exc}); "
                 f"caigo a WAKE_MODE=text")
    elif WAKE_WORD_ENABLED and WAKE_MODE != "text":
        warn("SPOT", f"WAKE_MODE={WAKE_MODE!r} desconocido, uso 'text'")

    if not WAKE_WORD_ENABLED:
        warn("WAKE", "wake word DESACTIVADO (WAKE_WORD_ENABLED=False): se envía todo")
    elif spotter is not None:
        info("WAKE", f"modo AUDIO: decí «{WAKE_PHRASES[0]}» y esperá el «Sí, dime» del robot; "
             f"dormido no transcribo nada; ventana {WAKE_WINDOW_S:.0f}s")
    else:
        info("WAKE", f"modo TEXTO: decí «rai» al principio de la frase; "
             f"ventana {WAKE_WINDOW_S:.0f}s")
    if barge_mode and spotter is None:
        warn("CTRL", "SPEAK_LISTEN_MODE=keyword necesita el spotter de audio: uso 'mute'")
        barge_mode = False
    elif barge_mode:
        info("CTRL", "mientras el robot habla escucho órdenes de corte: "
             + ", ".join(f"«{p}»" for p in SPEAK_STOP_PHRASES))
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

    # Eventos al orquestador: él habla por el parlante de la Jetson («Sí,
    # dime» en cada wake, chime al dormirse). La Pi no emite sonido.
    events = OrchestratorEvents(wake, ORCHESTRATOR_IP, ORCHESTRATOR_PORT,
                                on_asleep=spatial.on_asleep)
    if WAKE_WORD_ENABLED:
        events.start()

    if HEARTBEAT_S > 0:
        threading.Thread(
            target=heartbeat_worker,
            args=(vad, mute, wake, spotter, spatial, audio_queue, counters),
            daemon=True,
        ).start()

    def handle_wake(heard: str, now: float, announce: bool = True) -> bool:
        """El spotter oyó la frase de wake. Despierta (o pasa el foco a quien
        la dijo) y fija la dirección. False si no corresponde atenderla.
        `announce=False`: el «Sí, dime» lo decide el loop (WAKE_ACK_DECIDE_MS)."""
        reading = spatial.reading(now - WAKE_DOA_WINDOW_S, now)
        if spatial.enabled and reading.n >= 2 and reading.blocked >= 0.6:
            drop("WAKE", f"«{heard}» desde un sector de ruido del robot",
                 dir=_deg(reading.direction))
            return False
        if wake.is_awake():
            # Traspaso: otra persona (o la misma) vuelve a llamarlo. Tiene que
            # sonar cerca (filtro general) o venir del foco actual: un «rai»
            # de fondo no se roba el robot.
            level, near = vad.current_level(), vad.near_threshold()
            same_place = (spatial.focus.focus is not None
                          and reading.in_focus >= DOA_MIN_IN_FOCUS)
            if level < near and not same_place:
                drop("WAKE", f"otro «{heard}» pero lejano: no cambio el foco",
                     nivel=level, umbral=near, dir=_deg(reading.direction))
                return False
        old_focus = spatial.focus.focus
        wake.wake_from_audio(announce=announce)
        if spatial.enabled:
            new_focus = reading.direction if reading.n >= 2 else None
            spatial.focus.lock(new_focus)
            if old_focus is not None and new_focus is not None:
                ok("DOA", f"NUEVO FOCO dir {_deg(old_focus)}→{_deg(new_focus)} "
                   f"({reading.n} lecturas)")
            else:
                ok("DOA", f"foco en {_deg(new_focus)} ({reading.n} lecturas con voz)")
            if new_focus is not None:
                events.send(f"doa:{new_focus:.0f}")
            spatial.leds_awake()
        return True

    barge = BargeIn()

    info("NET", f"orquestador en {ORCHESTRATOR_IP}:{ORCHESTRATOR_PORT}")

    # «Oye rai» y después ¿siguió hablando? (ver WAKE_ACK_DECIDE_MS)
    # ack_pending_t: cuándo disparó el spotter, mientras se decide.
    # wake_utt_open: la utterance abierta empieza con el «oye rai» y trae la
    # orden: al cerrarse va entera a Whisper (y se le saca el nombre).
    ack_pending_t: float | None = None
    wake_utt_open = False

    try:
        first_frame = True
        was_muted = False
        for af in capture.frames():
            frame = af.pcm
            if first_frame:
                first_frame = False
                ok("AUDIO", "primer frame del mic: escuchando")
            # El robot está hablando: ignorar audio (evita el autoescucha).
            if mute.is_muted():
                if not was_muted:
                    was_muted = True
                    barge.reset()
                    # Lo que quedó a medio decir cuando el robot arrancó a
                    # hablar es viejo: no dejar que se cierre (y se envíe)
                    # recién al desmutear.
                    vad.discard_open_utterance()
                    ack_pending_t = None
                    wake_utt_open = False
                    # Utterances que ya se habían cerrado y encolado antes de
                    # este mute también son viejas: sin esto, se transcriben
                    # y mandan igual, recién cuando el robot ya terminó de
                    # hablar, como si fueran del turno actual.
                    drained = 0
                    while True:
                        try:
                            audio_queue.get_nowait()
                        except queue.Empty:
                            break
                        drained += 1
                    if drained:
                        counters.dropped_stale += drained
                        info("MAIN", f"{drained} utterance(s) en cola descartada(s) "
                             "por mute (robot empezó a hablar)")
                counters.muted_frames += 1
                if barge_mode:
                    # Orden de corte: el spotter sigue oyendo mientras habla.
                    barge.feed(af.rms if af.rms is not None else vad.frame_rms(frame))
                    spotter.feed(frame)
                    heard = spotter.take_detection()
                    if heard:
                        loud, peak, speaker = barge.loud_enough()
                        if spatial.from_speaker(af.t - WAKE_DOA_WINDOW_S, af.t):
                            drop("CTRL", f"«{heard}» vino del parlante del robot")
                        elif not loud:
                            drop("CTRL", f"«{heard}» no se destaca sobre la voz del robot",
                                 pico=peak, parlante=speaker, factor=SPEAK_BARGE_RATIO)
                        else:
                            counters.barge_ins += 1
                            ok("CTRL", f"ORDEN DE CORTE «{heard}» " + fmt(pico=peak, parlante=speaker))
                            events.send("stop")
                            # «oye rai» en vez de «para rai»: además re-despierta
                            # (el orquestador contesta «Sí, dime»).
                            if not spotter.is_stop(heard) and handle_wake(heard, af.t):
                                counters.wakes += 1
                continue
            was_muted = False
            if spotter is not None:
                spotter.feed(frame)
                heard = spotter.take_detection()
                # Las órdenes de corte sólo valen mientras el robot habla.
                if heard and not spotter.is_stop(heard):
                    if handle_wake(heard, af.t, announce=False):
                        counters.wakes += 1
                        # No se tira la utterance: si la persona sigue de
                        # largo ("oye rai, sentate"), la orden ya empezó acá y
                        # cortarla se comía la primera palabra. Se marca y en
                        # WAKE_ACK_DECIDE_MS se decide (abajo).
                        wake_utt_open = False
                        if vad.mark():
                            ack_pending_t = af.t
                        else:
                            ack_pending_t = None
                            wake.announce()
            closed, audio = vad.process_frame(frame, af.rms, af.t)
            if ack_pending_t is not None:
                if closed or not vad.in_speech:
                    # Se cerró antes de decidir: era sólo el nombre.
                    ack_pending_t = None
                    wake.announce()
                    if closed:
                        dim("WAKE", "era sólo el «oye rai», no lo transcribo")
                        continue
                elif af.t - ack_pending_t >= WAKE_ACK_DECIDE_MS / 1000.0:
                    ack_pending_t = None
                    follow_ms = vad.speech_ms_since_mark()
                    if follow_ms >= WAKE_FOLLOW_SPEECH_MS:
                        wake_utt_open = True
                        info("WAKE", f"siguió hablando tras el «oye rai» ({follow_ms} ms de voz): "
                             "mando la frase entera, sin «Sí, dime»")
                    else:
                        wake.announce()
                        vad.discard_open_utterance(
                            reason="era el «oye rai», no hace falta transcribirlo")
            # Mientras le están hablando no se duerme: si la ventana vence a
            # mitad de frase, al cerrarla se descartaría por "dormido".
            if vad.in_speech:
                wake.refresh()
            wake_prefix = False
            if wake_utt_open and not vad.in_speech:
                # La frase del «oye rai» terminó: si el VAD la rechazó, igual
                # está despierto, así que «Sí, dime» y que repita la orden.
                wake_prefix, wake_utt_open = True, False
                if not closed:
                    wake.announce()
            if closed and audio is not None:
                level = vad.last_level
                # Modo audio: dormido no se transcribe nada. Sólo el spotter
                # escucha, y hasta que no oiga "oye rai" Groq no se entera.
                if spotter is not None and not wake.is_awake():
                    counters.asleep += 1
                    drop("WAKE", f"dormido: no transcribo hasta oír «{WAKE_PHRASES[0]}»",
                         nivel=level)
                    continue
                # Dirección: ¿viene de quien lo llamó? (y nunca de un sector
                # de ruido propio del robot). Sin array, no decide nada.
                if spatial.enabled and vad.last_span is not None:
                    reading = spatial.reading(*vad.last_span)
                    accepted, why = spatial.focus.judge(reading)
                    if not accepted:
                        counters.off_focus += 1
                        drop("DOA", why, dir=_deg(reading.direction),
                             foco=f"{_deg(spatial.focus.focus)}±{DOA_TOLERANCE_DEG:.0f}",
                             en_foco=f"{reading.in_focus:.0%}", lecturas=reading.n)
                        continue
                    if wake.is_awake() and reading.n >= 2:
                        spatial.focus.follow(reading.direction)
                    dbg(f"dirección {_deg(reading.direction)} ({why}, "
                        f"{reading.in_focus:.0%} en foco, {reading.n} lecturas)", "DOA")
                # Atención por nivel (sin array): despierto, sólo lo que llega
                # al nivel de esta conversación (el fondo no gasta Whisper).
                if not wake.accepts_level(level):
                    counters.unfocused += 1
                    continue
                audio_queue.put((audio, level, time.monotonic(), wake_prefix))

    except KeyboardInterrupt:
        info("MAIN", "Stopped.")
    finally:
        if array is not None:
            array.stop_polling()
            array.leds_off()


if __name__ == "__main__":
    main()
