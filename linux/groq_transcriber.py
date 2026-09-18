import io
import wave

import numpy as np
from groq import Groq

from config import GROQ_MODEL, LANGUAGE, SAMPLE_RATE, STT_PROMPT
from log import dim, err, warn


class GroqTranscriber:
    def __init__(self) -> None:
        # Groq() lee GROQ_API_KEY del entorno; si falta, explota recién en la
        # primera transcripción con un error poco claro. Avisar ya.
        import os
        if not os.getenv("GROQ_API_KEY"):
            warn("STT", "GROQ_API_KEY no está definida (¿falta linux/.env?): "
                 "toda transcripción va a fallar")
        self._client = Groq()
        dim("STT", f"groq {GROQ_MODEL} ({LANGUAGE})")

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
                # Sesga a Whisper hacia el dominio: escribe "RAI" (clave para
                # el wake word) y alucina menos con audio flojo.
                prompt=STT_PROMPT,
                temperature=0.0,
            )
            return result.text.strip()
        except Exception as exc:
            err("STT", f"Groq falló ({type(exc).__name__}): {exc}")
            return ""
