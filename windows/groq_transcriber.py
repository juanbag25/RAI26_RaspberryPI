import io
import sys
import wave

import numpy as np
from groq import Groq

from config import GROQ_MODEL, LANGUAGE, SAMPLE_RATE


class GroqTranscriber:
    def __init__(self) -> None:
        self._client = Groq()

    def transcribe(self, audio_np: np.ndarray) -> str:
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
            )
            return result.text.strip()
        except Exception as exc:
            print(f"Groq transcription error: {exc}", file=sys.stderr)
            return ""
