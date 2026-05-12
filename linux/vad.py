from __future__ import annotations

from collections import deque

import numpy as np
import webrtcvad

from config import (
    FRAME_MS,
    PRE_SPEECH_PADDING_MS,
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
        self._reset_utterance()

    def _reset_utterance(self) -> None:
        self._in_speech = False
        self._utterance: list[bytes] = []
        self._silence_count = 0

    def process_frame(self, frame_bytes: bytes) -> tuple[bool, np.ndarray | None]:
        is_speech = self._vad.is_speech(frame_bytes, SAMPLE_RATE)

        if not self._in_speech:
            self._pre_buffer.append(frame_bytes)
            if is_speech:
                self._in_speech = True
                self._utterance = list(self._pre_buffer)
                self._silence_count = 0
            return False, None

        self._utterance.append(frame_bytes)
        if is_speech:
            self._silence_count = 0
            return False, None

        self._silence_count += 1
        if self._silence_count >= self._silence_frames_to_close:
            audio = self._finalize_utterance()
            self._reset_utterance()
            return True, audio
        return False, None

    def _finalize_utterance(self) -> np.ndarray:
        raw = b"".join(self._utterance)
        samples = np.frombuffer(raw, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0
