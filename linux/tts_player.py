"""Texto -> voz -> reproducción, con motor de TTS intercambiable.

Motores (config.py `TTS_ENGINE`):
  - "edge"          -> edge-tts: voces neuronales de Microsoft Edge. Gratis, SIN
                       API key ni billing. Necesita internet. (default)
  - "google_cloud"  -> Google Cloud TTS (requiere credenciales + billing).

El audio (MP3 de edge o WAV de Google) se decodifica con soundfile y se
reproduce con sounddevice, así el resto del código no depende del motor.
"""

from __future__ import annotations

import asyncio
import io
import threading

import sounddevice as sd
import soundfile as sf

from config import (
    TTS_ENGINE,
    TTS_OUTPUT_DEVICE,
    EDGE_VOICE,
    EDGE_RATE,
    EDGE_PITCH,
    TTS_LANGUAGE,
    TTS_VOICE,
    TTS_SAMPLE_RATE,
    TTS_SPEAKING_RATE,
)


class TtsPlayer:
    def __init__(self) -> None:
        self._engine = TTS_ENGINE
        self._play_lock = threading.Lock()

        # Set mientras suena el audio: la captura usa esto para silenciar el
        # micrófono y no transcribir la propia voz del robot (feedback loop).
        self.speaking = threading.Event()

        if self._engine == "google_cloud":
            self._init_google_cloud()
        elif self._engine == "edge":
            self._init_edge()
        else:
            raise ValueError(f"TTS_ENGINE desconocido: {self._engine!r} "
                             f"(esperaba 'edge' o 'google_cloud')")

    # ---- edge-tts ----------------------------------------------------------
    def _init_edge(self) -> None:
        import edge_tts  # lazy: solo si se usa este motor
        self._edge_tts = edge_tts

    def _synthesize_edge(self, text: str) -> bytes:
        async def _run() -> bytes:
            communicate = self._edge_tts.Communicate(
                text, voice=EDGE_VOICE, rate=EDGE_RATE, pitch=EDGE_PITCH
            )
            buf = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    buf.extend(chunk["data"])
            return bytes(buf)

        return asyncio.run(_run())  # MP3 bytes

    # ---- Google Cloud ------------------------------------------------------
    def _init_google_cloud(self) -> None:
        from google.cloud import texttospeech  # lazy
        self._gc = texttospeech
        self._gc_client = texttospeech.TextToSpeechClient()
        self._gc_voice = texttospeech.VoiceSelectionParams(
            language_code=TTS_LANGUAGE, name=TTS_VOICE or None
        )
        self._gc_audio = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=TTS_SAMPLE_RATE,
            speaking_rate=TTS_SPEAKING_RATE,
        )

    def _synthesize_google_cloud(self, text: str) -> bytes:
        resp = self._gc_client.synthesize_speech(
            input=self._gc.SynthesisInput(text=text),
            voice=self._gc_voice,
            audio_config=self._gc_audio,
        )
        return resp.audio_content  # WAV bytes

    # ---- común -------------------------------------------------------------
    def synthesize(self, text: str) -> bytes:
        """Devuelve bytes de audio (MP3 o WAV según el motor)."""
        if self._engine == "edge":
            return self._synthesize_edge(text)
        return self._synthesize_google_cloud(text)

    def play_audio(self, audio_bytes: bytes) -> None:
        """Decodifica (MP3/WAV) y reproduce en el dispositivo de salida."""
        data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="int16")
        with self._play_lock:
            sd.play(data, samplerate=sample_rate, device=TTS_OUTPUT_DEVICE)
            sd.wait()

    def speak(self, text: str) -> None:
        """Sintetiza y reproduce `text`. Los errores se loguean, no se propagan."""
        text = text.strip()
        if not text:
            return

        self.speaking.set()
        try:
            self.play_audio(self.synthesize(text))
        except Exception as exc:  # noqa: BLE001 - mantener vivo el loop
            print(f"[TTS ERROR] {exc}")
        finally:
            self.speaking.clear()
