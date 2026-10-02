"""VAD + filtro de cercanía (foco del micrófono).

webrtcvad sólo dice "esto es voz humana", no "esto me lo están diciendo a mí":
con un mic omnidireccional, una charla del otro lado de la sala abre utterances
y Whisper las transcribe. Encima de webrtcvad va entonces un filtro de energía
en dos etapas, porque la voz de quien le habla al robot de cerca llega mucho
más fuerte que la de fondo:

1. Apertura (frame a frame): hacen falta ONSET_SPEECH_FRAMES frames seguidos de
   voz por encima de max(RMS_THRESHOLD, piso_de_ruido * NEAR_SNR_RATIO).
2. Cierre (sobre la utterance entera): el percentil 90 de sus frames de voz
   tiene que llegar a NEAR_RMS_THRESHOLD y seguir NEAR_SNR_RATIO por encima del
   piso de ruido. Una frase que arrancó fuerte (un portazo, una sílaba) pero es
   de lejos se cae acá y nunca llega a Whisper.

El piso de ruido se mide en vivo con los frames descartados (incluye el murmullo
lejano), así que en una sala ruidosa el filtro se endurece solo; está topeado en
NOISE_FLOOR_MAX para que un ruido fuerte y sostenido no deje sordo al robot.

Con el ReSpeaker el nivel (RMS) no se mide sobre el audio que se transcribe
(ch0, con AGC) sino sobre los mics crudos: lo pasa audio_capture.py en
`AudioFrame.rms`. Cada utterance guarda además su intervalo de tiempo
(`last_span`) para cruzarlo con las lecturas de dirección (doa.py).

Los umbrales se calibran con `python mic_level.py`.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import webrtcvad

from log import dbg, dim, drop, fmt, ok

from config import (
    CLOSE_RMS_RATIO,
    CONTINUE_LEVEL_RATIO,
    FRAME_MS,
    MAX_UTTERANCE_MS,
    MIN_UTTERANCE_MS,
    NEAR_RMS_THRESHOLD,
    NEAR_SNR_RATIO,
    NOISE_FLOOR_ALPHA,
    NOISE_FLOOR_INIT,
    NOISE_FLOOR_MAX,
    NOISE_WINDOW_MS,
    ONSET_SPEECH_FRAMES,
    PRE_SPEECH_PADDING_MS,
    RMS_THRESHOLD,
    SAMPLE_RATE,
    SILENCE_MS,
    VAD_AGGRESSIVENESS,
)


@dataclass
class VadStats:
    """Contadores desde el último `pop_stats()` (los imprime el heartbeat de
    main.py). Sirven para responder, sin adivinar, por qué "no escucha":
    ¿llega audio? ¿webrtcvad ve voz? ¿el nivel llega al umbral de apertura?"""
    frames: int = 0
    speech_frames: int = 0        # webrtcvad dijo "voz"
    loud_speech_frames: int = 0   # voz Y por encima del umbral de apertura
    max_rms: float = 0.0
    sum_rms: float = 0.0
    opened: int = 0               # utterances abiertas
    accepted: int = 0             # cerradas y aceptadas (van a Whisper)
    rejected: int = 0             # cerradas y descartadas (cortas/lejanas)

    @property
    def mean_rms(self) -> float:
        return self.sum_rms / self.frames if self.frames else 0.0


class VoiceActivityDetector:
    def __init__(self) -> None:
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        padding_frames = max(1, PRE_SPEECH_PADDING_MS // FRAME_MS)
        self._pre_buffer: deque[bytes] = deque(maxlen=padding_frames)
        self._silence_frames_to_close = max(1, SILENCE_MS // FRAME_MS)
        self._min_speech_frames = max(1, MIN_UTTERANCE_MS // FRAME_MS)
        self._max_utterance_frames = max(1, MAX_UTTERANCE_MS // FRAME_MS)
        self._noise_floor = NOISE_FLOOR_INIT
        self._recent_rms: deque[float] = deque(maxlen=max(1, NOISE_WINDOW_MS // FRAME_MS))
        # Nivel (p90) de la última utterance aceptada: main.py lo lee justo
        # después de process_frame() para la atención del wake word.
        self.last_level = 0.0
        # (inicio, fin) en time.monotonic() de la última utterance cerrada y
        # aceptada: main.py lo cruza con las muestras de DoA.
        self.last_span: tuple[float, float] | None = None
        self._utt_start_t = 0.0
        self._stats = VadStats()
        self._reset_utterance()
        dim("VAD", f"filtro: abre>={RMS_THRESHOLD} cerca>={NEAR_RMS_THRESHOLD} "
            f"snr x{NEAR_SNR_RATIO} min_voz={MIN_UTTERANCE_MS}ms silencio={SILENCE_MS}ms")

    def pop_stats(self) -> VadStats:
        """Devuelve y reinicia los contadores de la ventana."""
        stats, self._stats = self._stats, VadStats()
        return stats

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def discard_open_utterance(self, reason: str = "el robot empezó a hablar") -> bool:
        """Tira la utterance en curso (si hay) sin transcribirla.

        Lo llama main.py al entrar en mute: si el robot empezó a hablar con
        una utterance abierta, los frames se saltean mientras suena y el VAD
        queda congelado a mitad de frase. Al desmutear vería silencio, la
        cerraría y mandaría a transcribir audio de ANTES de que el robot
        hablara — una frase vieja que el orquestador contestaría como si fuera
        nueva. También lo llama cuando el spotter de audio dispara: el "oye
        rai" ya cumplió, no hace falta transcribirlo. Devuelve True si había
        algo abierto.
        """
        if not self._in_speech:
            return False
        drop("VAD", reason, voz_ms=self._speech_frame_count * FRAME_MS)
        self._stats.rejected += 1
        self._reset_utterance()
        return True

    def _reset_utterance(self) -> None:
        self._in_speech = False
        self._utterance: list[bytes] = []
        self._silence_count = 0
        self._speech_frame_count = 0
        self._speech_levels: list[float] = []
        self._onset_count = 0

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    def open_threshold(self) -> float:
        """Umbral vivo para abrir: el absoluto o el relativo al ruido, el mayor."""
        return max(RMS_THRESHOLD, self._noise_floor * NEAR_SNR_RATIO)

    def near_threshold(self) -> float:
        """Umbral vivo de cercanía (segunda etapa, sobre la utterance entera)."""
        return max(NEAR_RMS_THRESHOLD, self._noise_floor * NEAR_SNR_RATIO)

    def continue_threshold(self) -> float:
        """Umbral para que un frame siga la utterance abierta: el de apertura
        (× CLOSE_RMS_RATIO) o una fracción de la voz de esta misma frase, el
        mayor. Lo segundo es lo que corta el ruido del robot caminando: está
        por encima del umbral absoluto pero muy por debajo de quien habla."""
        absolute = self.open_threshold() * CLOSE_RMS_RATIO
        return max(absolute, self._utterance_level() * CONTINUE_LEVEL_RATIO)

    def process_frame(self, frame_bytes: bytes, rms: float | None = None,
                      t: float | None = None) -> tuple[bool, np.ndarray | None]:
        """`rms`: nivel a usar para los umbrales (con el ReSpeaker, el de los
        mics crudos); None = calcularlo de `frame_bytes`. `t`: time.monotonic()
        del frame (para `last_span`); None = ahora."""
        is_speech = self._vad.is_speech(frame_bytes, SAMPLE_RATE)
        if rms is None:
            rms = self._frame_rms(frame_bytes)
        if t is None:
            t = time.monotonic()

        st = self._stats
        st.frames += 1
        st.sum_rms += rms
        if rms > st.max_rms:
            st.max_rms = rms
        if is_speech:
            st.speech_frames += 1
            if rms >= self.open_threshold():
                st.loud_speech_frames += 1

        self._track_window_floor(rms)

        if not self._in_speech:
            self._pre_buffer.append(frame_bytes)
            if is_speech and rms >= self.open_threshold():
                self._onset_count += 1
                if self._onset_count >= ONSET_SPEECH_FRAMES:
                    # El pre-buffer (200 ms) ya trae los frames del onset, así
                    # que exigir varios no come el arranque de la palabra.
                    self._in_speech = True
                    self._utterance = list(self._pre_buffer)
                    self._utt_start_t = t - len(self._utterance) * FRAME_MS / 1000.0
                    self._silence_count = 0
                    self._speech_frame_count = self._onset_count
                    self._speech_levels = [rms]
                    st.opened += 1
                    dim("VAD", "▶ voz " + fmt(rms=rms, abre=self.open_threshold()))
            else:
                if is_speech:
                    # Voz según webrtcvad pero floja: no abre. Si esto aparece
                    # todo el tiempo mientras le hablás, el umbral está alto.
                    dbg("voz floja, no abre " + fmt(rms=rms, abre=self.open_threshold()), "VAD")
                self._onset_count = 0
                # Todo lo que no abrió (silencio, ventilador, voces lejanas)
                # es, por definición, el fondo contra el que hay que destacarse.
                self._update_noise_floor(rms)
            return False, None

        self._utterance.append(frame_bytes)
        # Sólo la voz fuerte mantiene abierta la utterance: con ruido de fondo
        # webrtcvad dice "voz" casi siempre y la frase no cerraba nunca.
        if is_speech and rms >= self.continue_threshold():
            self._silence_count = 0
            self._speech_frame_count += 1
            self._speech_levels.append(rms)
        else:
            self._silence_count += 1

        too_long = len(self._utterance) >= self._max_utterance_frames
        if too_long:
            dim("VAD", f"utterance de {MAX_UTTERANCE_MS} ms sin silencio: la cierro igual "
                + fmt(ruido=self._noise_floor, abre=self.open_threshold()))
        if too_long or self._silence_count >= self._silence_frames_to_close:
            accepted = self._utterance_accepted()
            audio = self._finalize_utterance() if accepted else None
            if accepted:
                st.accepted += 1
                self.last_level = self._utterance_level()
                self.last_span = (self._utt_start_t, t)
                ok("VAD", f"voz {self._speech_frame_count * FRAME_MS} ms "
                   + fmt(nivel=self.last_level, ruido=self._noise_floor))
            else:
                st.rejected += 1
            self._reset_utterance()
            return (accepted, audio)
        return False, None

    def _utterance_accepted(self) -> bool:
        """Segunda etapa del filtro: ¿fue voz real y de cerca?"""
        if self._speech_frame_count < self._min_speech_frames:
            drop("VAD", "muy corta", voz_ms=self._speech_frame_count * FRAME_MS,
                 minimo_ms=MIN_UTTERANCE_MS)
            return False

        level = self._utterance_level()
        near_threshold = self.near_threshold()
        if level < near_threshold:
            drop("VAD", "lejana/floja", nivel=level, umbral=near_threshold,
                 ruido=self._noise_floor)
            return False
        return True

    def open_span(self, now: float) -> tuple[float, float] | None:
        """Intervalo de la utterance ABIERTA hasta `now` (None si no hay)."""
        return (self._utt_start_t, now) if self._in_speech else None

    def current_level(self) -> float:
        """Nivel (p90) de la utterance ABIERTA hasta ahora (0 si no hay).

        Lo usa main.py cuando el spotter de audio dispara a mitad de frase:
        el "oye rai" todavía no cerró, pero ya sabemos cuán fuerte suena
        quien lo dijo para fijar el foco de atención.
        """
        return self._utterance_level() if self._in_speech else 0.0

    def _utterance_level(self) -> float:
        """Percentil 90 del RMS de los frames de voz.

        Percentil y no promedio: una frase normal trae pausas y consonantes
        flojas que hunden la media aunque la persona esté al lado del robot.
        """
        if not self._speech_levels:
            return 0.0
        return float(np.percentile(self._speech_levels, 90))

    def _track_window_floor(self, rms: float) -> None:
        """Sube el piso al percentil 10 de los últimos NOISE_WINDOW_MS, con o
        sin utterance abierta. Sólo sube: bajar lo sigue haciendo la EMA de
        los frames descartados. Se recalcula cada 10 frames (300 ms)."""
        self._recent_rms.append(rms)
        window = self._recent_rms
        if len(window) < window.maxlen or self._stats.frames % 10:
            return
        floor = min(float(np.percentile(window, 10)), NOISE_FLOOR_MAX)
        if floor > self._noise_floor:
            self._noise_floor = floor

    def _update_noise_floor(self, rms: float) -> None:
        # Sube rápido y baja lento (así el murmullo de fondo eleva el umbral
        # enseguida), con tope para no quedarse sordo tras un ruido fuerte.
        alpha = NOISE_FLOOR_ALPHA if rms > self._noise_floor else NOISE_FLOOR_ALPHA * 0.2
        floor = (1.0 - alpha) * self._noise_floor + alpha * rms
        self._noise_floor = min(floor, NOISE_FLOOR_MAX)

    def _finalize_utterance(self) -> np.ndarray:
        raw = b"".join(self._utterance)
        samples = np.frombuffer(raw, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0

    @staticmethod
    def frame_rms(frame_bytes: bytes) -> float:
        return VoiceActivityDetector._frame_rms(frame_bytes)

    @staticmethod
    def _frame_rms(frame_bytes: bytes) -> float:
        samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(samples ** 2)))
