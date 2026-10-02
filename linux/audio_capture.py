from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterator

# USB mics rarely support 16 kHz natively; route ALSA through "plug" so
# PortAudio gets transparent sample-rate conversion.
os.environ.setdefault("PA_ALSA_PLUGHW", "1")

import numpy as np
import sounddevice as sd

from config import FRAME_MS, RESPEAKER_ENABLED, SAMPLE_RATE
from log import dbg, err, info, warn

# Cómo aparece el ReSpeaker USB Mic Array v2.0 en PortAudio/ALSA.
_ARRAY_NAMES = ("respeaker", "arrayuac10")
# Firmware de 6 canales: 0 = procesado por el chip, 1-4 = mics crudos,
# 5 = referencia de playback (siempre 0: no reproducimos por el array).
_ARRAY_CHANNELS = 6
_ARRAY_PROCESSED = 0
_ARRAY_RAW = slice(1, 5)


@dataclass(frozen=True)
class AudioFrame:
    """Un bloque de FRAME_MS ms.

    `pcm`: int16 mono 16 kHz (lo que va al spotter, al VAD y a Whisper; con el
    array es el canal procesado). `rms`: nivel normalizado [0, 1] para el
    filtro de cercanía — con el array, de los mics CRUDOS (sin AGC); sin
    array, None y el VAD lo calcula de `pcm`. `t`: time.monotonic() de la
    lectura, para cruzarlo con las muestras de DoA.
    """
    pcm: bytes
    rms: float | None
    t: float


def find_array_device() -> int | None:
    """Índice de PortAudio del ReSpeaker con >= 6 canales de entrada."""
    for i, d in enumerate(sd.query_devices()):
        name = d["name"].lower()
        if d["max_input_channels"] >= _ARRAY_CHANNELS and any(n in name for n in _ARRAY_NAMES):
            return i
    return None


class LinuxAudioCapture:
    def __init__(self, device_id: int | None = None) -> None:
        self._device_id = device_id
        self._blocksize = SAMPLE_RATE * FRAME_MS // 1000
        self.is_array = False
        if RESPEAKER_ENABLED:
            if device_id is None:
                self._device_id = find_array_device()
                self.is_array = self._device_id is not None
            else:
                d = sd.query_devices(device_id)
                self.is_array = (d["max_input_channels"] >= _ARRAY_CHANNELS
                                 and any(n in d["name"].lower() for n in _ARRAY_NAMES))

    @staticmethod
    def list_devices() -> None:
        print(sd.query_devices())

    def frames(self) -> Iterator[AudioFrame]:
        channels = _ARRAY_CHANNELS if self.is_array else 1
        dbg(f"abriendo stream: device={self._device_id} {SAMPLE_RATE} Hz {channels} canal(es) int16, "
            f"bloque={self._blocksize} samples ({FRAME_MS} ms)", "AUDIO")
        try:
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=self._blocksize,
                channels=channels,
                dtype="int16",
                device=self._device_id,
            )
        except sd.PortAudioError as exc:
            # "Error querying device -1" = PortAudio no ve NINGUNA entrada:
            # mic USB desconectado o no enumerado por ALSA.
            err("AUDIO", f"no pude abrir el mic (device={self._device_id}): {exc}")
            err("AUDIO", "¿mic USB conectado? Revisá `arecord -l` y el listado de "
                "arriba; si aparece pero no es el default, poné su índice en "
                "AUDIO_INPUT_DEVICE (.env)")
            raise
        with stream:
            kind = ("ReSpeaker 6 canales: ch0 procesado -> STT, ch1-4 crudos -> nivel"
                    if self.is_array else "mono")
            info("AUDIO", f"mic abierto: device="
                 f"{'default' if self._device_id is None else self._device_id} ({kind}), "
                 f"{SAMPLE_RATE} Hz, latencia {stream.latency * 1000:.0f} ms")
            try:
                while True:
                    data, overflowed = stream.read(self._blocksize)
                    now = time.monotonic()
                    if overflowed:
                        # El ring buffer de PortAudio no se drenó a tiempo
                        # (algo bloqueó este loop demasiado); frames de audio
                        # se perdieron/pisaron, lo que puede desincronizar al
                        # VAD del tiempo real.
                        warn("AUDIO", "input overflow: se perdieron frames del mic")
                    if not self.is_array:
                        yield AudioFrame(bytes(data), None, now)
                        continue
                    block = np.frombuffer(data, dtype=np.int16).reshape(-1, _ARRAY_CHANNELS)
                    raw = block[:, _ARRAY_RAW].astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(raw ** 2)))
                    pcm = np.ascontiguousarray(block[:, _ARRAY_PROCESSED]).tobytes()
                    yield AudioFrame(pcm, rms, now)
            except KeyboardInterrupt:
                return
