from __future__ import annotations

import os
from typing import Iterator

# USB mics rarely support 16 kHz natively; route ALSA through "plug" so
# PortAudio gets transparent sample-rate conversion.
os.environ.setdefault("PA_ALSA_PLUGHW", "1")

import sounddevice as sd

from config import FRAME_MS, SAMPLE_RATE
from log import dbg, err, info, warn


class LinuxAudioCapture:
    def __init__(self, device_id: int | None = None) -> None:
        self._device_id = device_id
        self._blocksize = SAMPLE_RATE * FRAME_MS // 1000

    @staticmethod
    def list_devices() -> None:
        print(sd.query_devices())

    def frames(self) -> Iterator[bytes]:
        dbg(f"abriendo stream: device={self._device_id} {SAMPLE_RATE} Hz mono int16, "
            f"bloque={self._blocksize} samples ({FRAME_MS} ms)", "AUDIO")
        try:
            stream = sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=self._blocksize,
                channels=1,
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
            info("AUDIO", f"mic abierto: device="
                 f"{'default' if self._device_id is None else self._device_id}, "
                 f"{SAMPLE_RATE} Hz, latencia {stream.latency * 1000:.0f} ms")
            try:
                while True:
                    data, overflowed = stream.read(self._blocksize)
                    if overflowed:
                        # El ring buffer de PortAudio no se drenó a tiempo
                        # (algo bloqueó este loop demasiado); frames de audio
                        # se perdieron/pisaron, lo que puede desincronizar al
                        # VAD del tiempo real.
                        warn("AUDIO", "input overflow: se perdieron frames del mic")
                    yield bytes(data)
            except KeyboardInterrupt:
                return
