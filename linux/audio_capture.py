from __future__ import annotations

import os
from typing import Iterator

# USB mics rarely support 16 kHz natively; route ALSA through "plug" so
# PortAudio gets transparent sample-rate conversion.
os.environ.setdefault("PA_ALSA_PLUGHW", "1")

import sounddevice as sd

from config import FRAME_MS, SAMPLE_RATE
from log import log


class LinuxAudioCapture:
    def __init__(self, device_id: int | None = None) -> None:
        self._device_id = device_id
        self._blocksize = SAMPLE_RATE * FRAME_MS // 1000

    @staticmethod
    def list_devices() -> None:
        print(sd.query_devices())

    def frames(self) -> Iterator[bytes]:
        log(f"[AUDIO] abriendo stream: device={self._device_id} "
            f"{SAMPLE_RATE} Hz mono int16, bloque={self._blocksize} samples ({FRAME_MS} ms)")
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=self._blocksize,
            channels=1,
            dtype="int16",
            device=self._device_id,
        ) as stream:
            log(f"[AUDIO] stream abierto (latencia={stream.latency * 1000:.0f} ms)")
            try:
                while True:
                    data, overflowed = stream.read(self._blocksize)
                    if overflowed:
                        # El ring buffer de PortAudio no se drenó a tiempo
                        # (algo bloqueó este loop demasiado); frames de audio
                        # se perdieron/pisaron, lo que puede desincronizar al
                        # VAD del tiempo real.
                        log("[AUDIO WARN] input overflow: se perdieron frames de mic")
                    yield bytes(data)
            except KeyboardInterrupt:
                return
