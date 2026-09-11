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

Los umbrales se calibran con `python mic_level.py`.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import webrtcvad

from config import (
    FRAME_MS,
    MIN_UTTERANCE_MS,
    NEAR_RMS_THRESHOLD,
    NEAR_SNR_RATIO,
    NOISE_FLOOR_ALPHA,
    NOISE_FLOOR_INIT,
    NOISE_FLOOR_MAX,
    ONSET_SPEECH_FRAMES,
    PRE_SPEECH_PADDING_MS,
    RMS_THRESHOLD,
    SAMPLE_RATE,
    SILENCE_MS,
    VAD_AGGRESSIVENESS,
)


class VoiceActivityDetector:
    def __init__(self) -> None:
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        padding_frames = max(1, PRE_SPEECH_PADDING_MS // FRAME_MS)
        self._pre_buffer: deque[bytes] = deque(maxlen=padding_frames)
        self._silence_frames_to_close = max(1, SILENCE_MS // FRAME_MS)
        self._min_speech_frames = max(1, MIN_UTTERANCE_MS // FRAME_MS)
        self._noise_floor = NOISE_FLOOR_INIT
        self._reset_utterance()

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

    def process_frame(self, frame_bytes: bytes) -> tuple[bool, np.ndarray | None]:
        is_speech = self._vad.is_speech(frame_bytes, SAMPLE_RATE)
        rms = self._frame_rms(frame_bytes)

        if not self._in_speech:
            self._pre_buffer.append(frame_bytes)
            if is_speech and rms >= self.open_threshold():
                self._onset_count += 1
                if self._onset_count >= ONSET_SPEECH_FRAMES:
                    # El pre-buffer (200 ms) ya trae los frames del onset, así
                    # que exigir varios no come el arranque de la palabra.
                    self._in_speech = True
                    self._utterance = list(self._pre_buffer)
                    self._silence_count = 0
                    self._speech_frame_count = self._onset_count
                    self._speech_levels = [rms]
            else:
                self._onset_count = 0
                # Todo lo que no abrió (silencio, ventilador, voces lejanas)
                # es, por definición, el fondo contra el que hay que destacarse.
                self._update_noise_floor(rms)
            return False, None

        self._utterance.append(frame_bytes)
        if is_speech:
            self._silence_count = 0
            self._speech_frame_count += 1
            self._speech_levels.append(rms)
            return False, None

        self._silence_count += 1
        if self._silence_count >= self._silence_frames_to_close:
            accepted = self._utterance_accepted()
            audio = self._finalize_utterance() if accepted else None
            self._reset_utterance()
            return (accepted, audio)
        return False, None

    def _utterance_accepted(self) -> bool:
        """Segunda etapa del filtro: ¿fue voz real y de cerca?"""
        if self._speech_frame_count < self._min_speech_frames:
            print(f"[VAD] descartada: muy corta "
                  f"({self._speech_frame_count * FRAME_MS} ms)", flush=True)
            return False

        level = self._utterance_level()
        near_threshold = max(NEAR_RMS_THRESHOLD, self._noise_floor * NEAR_SNR_RATIO)
        if level < near_threshold:
            print(f"[VAD] descartada: lejana/floja (nivel={level:.4f} < "
                  f"{near_threshold:.4f}, ruido={self._noise_floor:.4f})", flush=True)
            return False
        return True

    def _utterance_level(self) -> float:
        """Percentil 90 del RMS de los frames de voz.

        Percentil y no promedio: una frase normal trae pausas y consonantes
        flojas que hunden la media aunque la persona esté al lado del robot.
        """
        if not self._speech_levels:
            return 0.0
        return float(np.percentile(self._speech_levels, 90))

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
    def _frame_rms(frame_bytes: bytes) -> float:
        samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(samples ** 2)))
