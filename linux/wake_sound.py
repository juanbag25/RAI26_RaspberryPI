"""Sonido de confirmación al despertar (el "ding" de Siri).

Se reproduce por el parlante de la Pi con sounddevice, sin bloquear el loop
de captura. Mientras suena, main.py descarta los frames del mic para no
transcribir el propio beep (el mic omnidireccional lo escucha perfecto).

WAKE_SOUND: "beep" (dos tonos generados, no necesita archivo), "none", o la
ruta a un .wav (mono o estéreo, 16-bit). AUDIO_OUTPUT_DEVICE elige el
parlante (índice de sounddevice, mismo listado que el mic). Si no hay salida
de audio, se avisa una vez y se sigue sin sonido: el wake funciona igual.

Prueba: `python wake_sound.py` reproduce el sonido una vez.
"""

from __future__ import annotations

import os
import time
import wave

import numpy as np

from config import WAKE_SOUND, WAKE_SOUND_DEVICE, WAKE_SOUND_VOLUME
from log import dim, warn


def _tone(rate: int, freq: float, ms: int, volume: float) -> np.ndarray:
    n = int(rate * ms / 1000)
    t = np.arange(n) / rate
    signal = np.sin(2 * np.pi * freq * t)
    # Rampas de 5 ms para que no haga "clic".
    ramp = max(1, int(rate * 0.005))
    env = np.ones(n)
    env[:ramp] = np.linspace(0, 1, ramp)
    env[-ramp:] = np.linspace(1, 0, ramp)
    return (signal * env * volume).astype(np.float32)


def _beep(rate: int, volume: float) -> np.ndarray:
    # Dos notas cortas ascendentes (~150 ms en total): se reconoce como
    # "te escuché" y es lo bastante corto como para no comerse la frase.
    return np.concatenate([_tone(rate, 880, 60, volume), _tone(rate, 1175, 90, volume)])


def _load_wav(path: str, volume: float) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        width = w.getsampwidth()
        channels = w.getnchannels()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: se espera wav de 16-bit, tiene {width * 8}-bit")
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return (samples * volume).astype(np.float32), rate


class WakeSound:
    def __init__(self) -> None:
        self._enabled = WAKE_SOUND.strip().lower() not in ("", "none", "off", "0", "false")
        self._playing_until = 0.0
        self._data: np.ndarray | None = None
        self._rate = 0
        self._duration = 0.0
        self._device = WAKE_SOUND_DEVICE
        if not self._enabled:
            dim("SOUND", "beep desactivado (WAKE_SOUND=none)")
            return
        try:
            import sounddevice as sd
            self._sd = sd
            if WAKE_SOUND.strip().lower() == "beep":
                info = sd.query_devices(self._device, "output")
                self._rate = int(info["default_samplerate"]) or 44100
                self._data = _beep(self._rate, WAKE_SOUND_VOLUME)
            else:
                path = os.path.expanduser(WAKE_SOUND)
                self._data, self._rate = _load_wav(path, WAKE_SOUND_VOLUME)
            self._duration = len(self._data) / self._rate
            dim("SOUND", f"beep={WAKE_SOUND} ({self._duration * 1000:.0f} ms) por device="
                f"{'default' if self._device is None else self._device}")
        except Exception as exc:  # noqa: BLE001
            self._enabled = False
            warn("SOUND", f"sin salida de audio, sigo sin beep ({type(exc).__name__}: {exc}). "
                 f"Revisá AUDIO_OUTPUT_DEVICE o poné WAKE_SOUND=none")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def play(self) -> None:
        if not self._enabled or self._data is None:
            return
        try:
            self._sd.play(self._data, self._rate, device=self._device)
            # +80 ms de margen por la latencia de salida de PortAudio/ALSA.
            self._playing_until = time.monotonic() + self._duration + 0.08
        except Exception as exc:  # noqa: BLE001
            self._enabled = False
            warn("SOUND", f"falló la reproducción, desactivo el beep "
                 f"({type(exc).__name__}: {exc})")

    def is_playing(self) -> bool:
        return time.monotonic() < self._playing_until


if __name__ == "__main__":
    s = WakeSound()
    s.play()
    while s.is_playing():
        time.sleep(0.05)
    time.sleep(0.2)
