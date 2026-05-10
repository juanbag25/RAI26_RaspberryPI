import sys

import numpy as np
from faster_whisper import WhisperModel

from config import COMPUTE_TYPE, LANGUAGE, MODEL_SIZE


class Transcriber:
    def __init__(self) -> None:
        self._model = WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)

    def transcribe(self, audio_np: np.ndarray) -> str:
        try:
            segments, _ = self._model.transcribe(
                audio_np,
                language=LANGUAGE,
                beam_size=1,
            )
            return "".join(segment.text for segment in segments).strip()
        except Exception as exc:
            print(f"Transcription error: {exc}", file=sys.stderr)
            return ""
