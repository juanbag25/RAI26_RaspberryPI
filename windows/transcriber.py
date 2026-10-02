import math
import sys

import numpy as np
from faster_whisper import WhisperModel

from config import COMPUTE_TYPE, LANGUAGE, MODEL_PATH


class Transcriber:
    def __init__(self) -> None:
        self._model = WhisperModel(MODEL_PATH, device="cpu", compute_type=COMPUTE_TYPE)

    def transcribe(self, audio_np: np.ndarray) -> tuple[str, float]:
        try:
            segments, _ = self._model.transcribe(
                audio_np,
                language=LANGUAGE,
                beam_size=1,
            )
            # El generador se consume una sola vez: materializarlo para poder
            # usarlo tanto para el texto como para la confianza.
            segments = list(segments)
            text = "".join(segment.text for segment in segments).strip()
            # Peor segmento del utterance: una sola parte con voces
            # superpuestas/ilegible basta para marcar todo poco confiable.
            stt_confidence = (
                min(math.exp(segment.avg_logprob) for segment in segments)
                if segments else 0.0
            )
            return text, stt_confidence
        except Exception as exc:
            print(f"Transcription error: {exc}", file=sys.stderr)
            return "", 0.0
