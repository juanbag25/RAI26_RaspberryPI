import io
import math
import sys
import wave

import numpy as np
from groq import Groq

from config import GROQ_MODEL, LANGUAGE, SAMPLE_RATE


class GroqTranscriber:
    def __init__(self) -> None:
        self._client = Groq()

    def transcribe(self, audio_np: np.ndarray) -> tuple[str, float]:
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                samples = np.clip(audio_np * 32768.0, -32768, 32767).astype(np.int16)
                wav.writeframes(samples.tobytes())
            buf.seek(0)

            result = self._client.audio.transcriptions.create(
                file=("audio.wav", buf.read()),
                model=GROQ_MODEL,
                language=LANGUAGE,
                response_format="verbose_json",
            )
            text = result.text.strip()
            # segments[] viene como lista de dicts (no objetos): confirmado
            # contra la respuesta real del SDK groq 1.7.0.
            segments = result.segments or []
            stt_confidence = (
                min(math.exp(seg["avg_logprob"]) for seg in segments)
                if segments else 0.0
            )
            return text, stt_confidence
        except Exception as exc:
            print(f"Groq transcription error: {exc}", file=sys.stderr)
            return "", 0.0
