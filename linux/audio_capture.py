from __future__ import annotations

from typing import Iterator

import sounddevice as sd

from config import FRAME_MS, SAMPLE_RATE


class LinuxAudioCapture:
    def __init__(self, device_id: int | None = None) -> None:
        self._device_id = device_id
        self._blocksize = SAMPLE_RATE * FRAME_MS // 1000

    @staticmethod
    def list_devices() -> None:
        print(sd.query_devices())

    def frames(self) -> Iterator[bytes]:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=self._blocksize,
            channels=1,
            dtype="int16",
            device=self._device_id,
        ) as stream:
            try:
                while True:
                    data, _ = stream.read(self._blocksize)
                    yield bytes(data)
            except KeyboardInterrupt:
                return
