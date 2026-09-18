"""Wake word sobre AUDIO: detector liviano de "oye rai" que corre siempre.

Es la etapa barata de la cascada (spotter -> VAD -> Groq), como el chip
always-on de un celular: mientras el robot duerme NO se manda nada a Groq;
sólo este módulo escucha, y cuando reconoce la frase de wake despierta al
robot en ~0.3 s en vez de los ~2 s que tarda "esperar silencio + Groq +
buscar 'rai' en el texto" del modo texto (wake_word.py).

Motor: Vosk (Kaldi) con el modelo chico en español y una GRAMÁTICA cerrada:
el reconocedor sólo puede devolver las frases de WAKE_PHRASES o "[unk]"
(cualquier otra cosa). Con eso deja de ser un STT y pasa a ser un keyword
spotter: ~40x tiempo real en una PC, sobra en la Pi 5, y todo lo que no
suena a la frase cae en [unk]. No es infalible (es un modelo de 40 MB y "rai"
no es una palabra del español: suele salir "ray"/"rey", por eso esas variantes
están en la gramática), pero para "despertá cuando suene a 'oye rai'" alcanza.

El match se hace sobre los resultados PARCIALES (WAKE_ON_PARTIAL): apenas el
decoder ve la frase completa dispara, sin esperar a que cierre la utterance.
Después resetea el reconocedor y espera WAKE_COOLDOWN_S para no disparar dos
veces por la misma frase (parcial + final).

Uso desde main.py: `spotter.feed(frame)` por cada frame de 30 ms (mismo
formato que el VAD: int16 mono 16 kHz) y `spotter.take_detection()` en el loop
de captura para enterarse. El decodificado corre en su propio hilo con una
cola acotada: si la Pi se atrasa se DESCARTAN frames (se pierde algún wake,
el mic no se traba nunca).

Prueba en vivo con el mic (imprime lo que reconoce y los disparos):

    python wake_spotter.py
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass

from config import (
    FRAME_MS,
    SAMPLE_RATE,
    WAKE_COOLDOWN_S,
    WAKE_MODEL_PATH,
    WAKE_ON_PARTIAL,
    WAKE_PHRASES,
    WAKE_QUEUE_S,
)
from log import DEBUG, dbg, log
from wake_word import normalize


@dataclass
class SpotterStats:
    """Contadores desde el último `pop_stats()` (para el heartbeat)."""
    frames: int = 0
    dropped: int = 0      # cola llena: el decoder no dio abasto
    detections: int = 0
    last_text: str = ""   # último parcial/final no vacío que devolvió Vosk


class WakeSpotter:
    def __init__(self) -> None:
        # Import acá y no arriba: main.py tiene que poder arrancar en modo
        # texto aunque vosk no esté instalado.
        from vosk import KaldiRecognizer, Model, SetLogLevel

        if not os.path.isdir(WAKE_MODEL_PATH):
            raise FileNotFoundError(
                f"modelo Vosk no encontrado en {WAKE_MODEL_PATH} "
                f"(ver README: descargar vosk-model-small-es-0.42 en models/)")
        # Kaldi es muy charlatán por stderr; sólo con LOG_DEBUG=1.
        SetLogLevel(0 if DEBUG else -1)

        self._phrases = tuple(dict.fromkeys(normalize(p) for p in WAKE_PHRASES if normalize(p)))
        if not self._phrases:
            raise ValueError("WAKE_PHRASES vacío")
        grammar = json.dumps(list(self._phrases) + ["[unk]"])

        t0 = time.monotonic()
        self._model = Model(WAKE_MODEL_PATH)
        self._rec = KaldiRecognizer(self._model, SAMPLE_RATE, grammar)
        log(f"[SPOT] Vosk cargado en {time.monotonic() - t0:.1f}s: "
            f"{os.path.basename(WAKE_MODEL_PATH)}, frases={list(self._phrases)}, "
            f"parcial={'sí' if WAKE_ON_PARTIAL else 'no'}, cooldown={WAKE_COOLDOWN_S}s")

        frames_per_s = 1000 // FRAME_MS
        self._queue: "queue.Queue[bytes | None]" = queue.Queue(
            maxsize=max(10, int(WAKE_QUEUE_S * frames_per_s)))
        self._lock = threading.Lock()
        self._stats = SpotterStats()
        self._detected = threading.Event()
        self._detected_text = ""
        # Cooldown en TIEMPO DE AUDIO (frames procesados), no de reloj: así
        # es determinista y no depende de si el decoder va atrasado.
        self._frames_done = 0
        self._cooldown_frames = int(WAKE_COOLDOWN_S * 1000 / FRAME_MS)
        self._cooldown_until_frame = 0
        self._thread = threading.Thread(target=self._run, name="wake-spotter", daemon=True)
        self._thread.start()

    # -- API para el loop de captura ------------------------------------------

    def feed(self, frame: bytes) -> None:
        """Encola un frame (no bloquea nunca)."""
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            with self._lock:
                self._stats.dropped += 1

    def take_detection(self) -> str | None:
        """Texto con el que disparó el spotter desde la última consulta, o None."""
        if not self._detected.is_set():
            return None
        self._detected.clear()
        return self._detected_text

    def pop_stats(self) -> SpotterStats:
        with self._lock:
            stats, self._stats = self._stats, SpotterStats(last_text=self._stats.last_text)
        return stats

    def queue_size(self) -> int:
        return self._queue.qsize()

    def stop(self) -> None:
        self._queue.put(None)

    # -- hilo de decodificado -------------------------------------------------

    def _run(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:
                return
            with self._lock:
                self._stats.frames += 1
            self._frames_done += 1
            try:
                if self._rec.AcceptWaveform(frame):
                    text = json.loads(self._rec.Result()).get("text", "")
                    self._check(text, final=True)
                elif WAKE_ON_PARTIAL:
                    text = json.loads(self._rec.PartialResult()).get("partial", "")
                    self._check(text, final=False)
            except Exception as exc:  # noqa: BLE001 - nunca matar el spotter
                log(f"[SPOT ERROR] {type(exc).__name__}: {exc}", err=True)

    def _matches(self, text: str) -> str | None:
        n = normalize(text)
        for phrase in self._phrases:
            if n == phrase or n.endswith(" " + phrase):
                return phrase
        return None

    def _check(self, text: str, *, final: bool) -> None:
        if not text or text == "[unk]":
            return
        with self._lock:
            if text != self._stats.last_text:
                self._stats.last_text = text
                dbg(f"[SPOT] {'final' if final else 'parcial'}: {text!r}")
        phrase = self._matches(text)
        if phrase is None:
            return
        # Vaciar el decoder: lo que sigue es la instrucción, no otra vez el nombre.
        self._rec.Reset()
        if self._frames_done < self._cooldown_until_frame:
            dbg(f"[SPOT] {phrase!r} en cooldown, ignorado")
            return
        self._cooldown_until_frame = self._frames_done + self._cooldown_frames
        with self._lock:
            self._stats.detections += 1
        self._detected_text = text
        self._detected.set()
        log(f"[SPOT] wake detectado ({'final' if final else 'parcial'}): «{text}»")


def _live() -> None:
    """Prueba en vivo: mic -> spotter, sin VAD ni Groq."""
    from audio_capture import LinuxAudioCapture
    from wake_sound import WakeSound

    device_env = os.getenv("AUDIO_INPUT_DEVICE", "").strip()
    spotter = WakeSpotter()
    sound = WakeSound()
    log("Hablale al mic. Decí una de las frases de wake; Ctrl+C para salir.")
    last_hb = time.monotonic()
    for frame in LinuxAudioCapture(device_id=int(device_env) if device_env else None).frames():
        spotter.feed(frame)
        text = spotter.take_detection()
        if text:
            sound.play()
            log(f">>> DESPIERTO por «{text}»")
        if time.monotonic() - last_hb >= 5:
            last_hb = time.monotonic()
            st = spotter.pop_stats()
            log(f"[HB] frames={st.frames} descartados={st.dropped} cola={spotter.queue_size()} "
                f"último_texto={st.last_text!r}")


if __name__ == "__main__":
    try:
        _live()
    except KeyboardInterrupt:
        pass
