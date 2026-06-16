"""Google Cloud Text-to-Speech -> WAV -> playback.

Receives plain text (the LLM reply forwarded by the orchestrator), synthesizes
it with Google Cloud TTS as LINEAR16 WAV, and plays it through the default
output device using sounddevice (already a project dependency).

Credentials: the Google Cloud client picks up the service-account JSON pointed
to by the GOOGLE_APPLICATION_CREDENTIALS environment variable. See linux/README.
"""

from __future__ import annotations

import io
import threading
import wave

import numpy as np
import sounddevice as sd
from google.cloud import texttospeech

from config import (
    TTS_LANGUAGE,
    TTS_VOICE,
    TTS_SAMPLE_RATE,
    TTS_SPEAKING_RATE,
    TTS_OUTPUT_DEVICE,
)


class TtsPlayer:
    def __init__(self) -> None:
        # One client + one config, reused across calls.
        self._client = texttospeech.TextToSpeechClient()

        self._voice = texttospeech.VoiceSelectionParams(
            language_code=TTS_LANGUAGE,
            name=TTS_VOICE or None,
        )

        self._audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=TTS_SAMPLE_RATE,
            speaking_rate=TTS_SPEAKING_RATE,
        )

        # Serialize playback so overlapping replies don't fight for the device.
        self._play_lock = threading.Lock()

        # Set while audio is playing so the capture side can mute the mic and
        # avoid transcribing the robot's own voice (feedback loop).
        self.speaking = threading.Event()

    def synthesize(self, text: str) -> bytes:
        """Return WAV (LINEAR16) bytes for `text`."""
        response = self._client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=self._voice,
            audio_config=self._audio_config,
        )
        # With LINEAR16, audio_content is a complete WAV container.
        return response.audio_content

    def play_wav(self, wav_bytes: bytes) -> None:
        """Decode a WAV blob and play it on the default output device."""
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels)

        with self._play_lock:
            sd.play(audio, samplerate=sample_rate, device=TTS_OUTPUT_DEVICE)
            sd.wait()

    def speak(self, text: str) -> None:
        """Synthesize and play `text`. Errors are logged, never raised."""
        text = text.strip()
        if not text:
            return

        self.speaking.set()
        try:
            wav_bytes = self.synthesize(text)
            self.play_wav(wav_bytes)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive
            print(f"[TTS ERROR] {exc}")
        finally:
            self.speaking.clear()
