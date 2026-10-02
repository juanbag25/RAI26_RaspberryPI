import io
import math
import wave

import numpy as np
from groq import Groq

from config import (
    GROQ_MODEL,
    LANGUAGE,
    SAMPLE_RATE,
    STT_HALLUCINATIONS,
    STT_MAX_COMPRESSION,
    STT_MAX_NO_SPEECH_PROB,
    STT_MIN_LOGPROB,
    STT_NO_SPEECH_LOGPROB,
    STT_PROMPT,
)
from log import dim, drop, err, warn
from wake_word import normalize

# STT_HALLUCINATIONS: "*frase" = descartar si el texto la CONTIENE; sin "*",
# sólo si el texto entero es eso.
_HALLUCINATION_EXACT = {normalize(h) for h in STT_HALLUCINATIONS if not h.startswith("*")}
_HALLUCINATION_SUBSTR = tuple(normalize(h[1:]) for h in STT_HALLUCINATIONS if h.startswith("*"))


def _field(segment, name: str, default: float) -> float:
    value = segment.get(name) if isinstance(segment, dict) else getattr(segment, name, None)
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _segment_rejection(segment) -> str | None:
    """Por qué descartar este segmento de Whisper (None = se queda)."""
    no_speech = _field(segment, "no_speech_prob", 0.0)
    logprob = _field(segment, "avg_logprob", 0.0)
    compression = _field(segment, "compression_ratio", 0.0)
    if no_speech > STT_MAX_NO_SPEECH_PROB and logprob < STT_NO_SPEECH_LOGPROB:
        return f"sin voz (no_speech={no_speech:.2f}, logprob={logprob:.2f})"
    if logprob < STT_MIN_LOGPROB:
        return f"poco probable (logprob={logprob:.2f})"
    if compression > STT_MAX_COMPRESSION:
        return f"repetitivo (compresión={compression:.1f})"
    return None


def is_known_hallucination(text: str) -> bool:
    """Frases que Whisper inventa con silencio/ruido ("gracias por ver")."""
    n = normalize(text)
    return n in _HALLUCINATION_EXACT or any(h in n for h in _HALLUCINATION_SUBSTR)


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
                # Sesga a Whisper hacia el dominio: escribe "RAI" (clave para
                # el wake word) y alucina menos con audio flojo.
                prompt=STT_PROMPT,
                temperature=0.0,
                # verbose_json trae la confianza de cada segmento
                # (no_speech_prob, avg_logprob, compression_ratio): se usa acá
                # para filtrar segmentos poco confiables del texto, y además
                # para armar un stt_confidence agregado que viaja hasta el
                # orchestrator (ver _confidence).
                response_format="verbose_json",
            )
        except Exception as exc:
            err("STT", f"Groq falló ({type(exc).__name__}): {exc}")
            return "", 0.0
        text = self._filter(result)
        stt_confidence = self._confidence(result)
        return text, stt_confidence

    @staticmethod
    def _segments_of(result) -> list:
        segments = getattr(result, "segments", None)
        if segments is None:
            segments = (getattr(result, "model_extra", None) or {}).get("segments")
        return segments or []

    @staticmethod
    def _filter(result) -> str:
        """Texto de los segmentos confiables. Sin segmentos (respuesta sin
        verbose_json) se usa el texto tal cual."""
        segments = GroqTranscriber._segments_of(result)
        if not segments:
            text = (getattr(result, "text", "") or "").strip()
        else:
            kept = []
            for segment in segments:
                seg_text = (segment.get("text") if isinstance(segment, dict)
                            else getattr(segment, "text", "")) or ""
                why = _segment_rejection(segment)
                if why:
                    drop("STT", f"segmento descartado, {why}", texto=f"«{seg_text.strip()}»")
                    continue
                kept.append(seg_text)
            text = "".join(kept).strip()
        if text and is_known_hallucination(text):
            drop("STT", "alucinación típica de Whisper", texto=f"«{text}»")
            return ""
        return text

    @staticmethod
    def _confidence(result) -> float:
        """Confianza del utterance completo (antes de filtrar segmentos): el
        peor segmento de Whisper, convertido de log-prob a un score 0-1. Se
        calcula sobre TODOS los segmentos (no sólo los que `_filter` termina
        conservando) a propósito: un segmento con voces superpuestas/ruido ya
        viene con avg_logprob bajo, así que lo "ve" acá aunque `_filter` lo
        descarte del texto final — es justo la señal que se quiere mandar."""
        segments = GroqTranscriber._segments_of(result)
        if not segments:
            return 0.0
        return min(math.exp(_field(segment, "avg_logprob", 0.0)) for segment in segments)
