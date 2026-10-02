"""Silero VAD (red neuronal) como detector de "¿esto es voz humana?".

Reemplaza a webrtcvad en vad.py (VAD_ENGINE=silero). webrtcvad es un modelo
estadístico que reacciona a energía/espectro: pasos, motores y ventiladores le
parecen voz, y con el robot caminando las frases se abrían solas o no
cerraban nunca. Silero está entrenado justamente para separar voz de ruido.

Corre local en la Pi (onnxruntime en CPU, <1 ms por bloque, ~2 % de un núcleo),
sin PyTorch: el modelo es `models/silero_vad.onnx` (~2 MB, licencia MIT,
snakers4/silero-vad). Protocolo del modelo v5 a 16 kHz, igual que su
OnnxWrapper oficial: bloques de 512 muestras precedidos de las últimas 64 del
bloque anterior (contexto) y un estado recurrente (2, 1, 128) que se arrastra.

Nuestros frames son de 30 ms (480 muestras, lo que exige webrtcvad y usa todo
el pipeline): se acumulan y se evalúa cada vez que se juntan 512; entre medio
vale la última probabilidad. Histéresis como recomienda Silero: para EMPEZAR a
ser voz hace falta prob >= umbral; para SEGUIR siéndolo alcanza con
umbral - 0.15 (las sílabas flojas no cortan la frase).
"""

from __future__ import annotations

import numpy as np

_CHUNK = 512
_CONTEXT = 64
_SAMPLE_RATE = 16000
_HYSTERESIS = 0.15


class SileroVad:
    def __init__(self, model_path: str, threshold: float = 0.5) -> None:
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.threshold = threshold
        self._sr = np.array(_SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self.prob = 0.0
        self._speaking = False

    def _run(self, chunk: np.ndarray) -> float:
        x = np.concatenate([self._context, chunk])[np.newaxis, :]
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr})
        self._context = x[0, -_CONTEXT:]
        return float(out[0][0])

    def is_speech(self, frame_bytes: bytes, sample_rate: int = _SAMPLE_RATE) -> bool:
        """Misma firma que webrtcvad.Vad.is_speech."""
        if sample_rate != _SAMPLE_RATE:
            raise ValueError(f"Silero VAD acá sólo a {_SAMPLE_RATE} Hz")
        samples = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._pending = np.concatenate([self._pending, samples])
        while len(self._pending) >= _CHUNK:
            self.prob = self._run(self._pending[:_CHUNK])
            self._pending = self._pending[_CHUNK:]
        limit = self.threshold - _HYSTERESIS if self._speaking else self.threshold
        self._speaking = self.prob >= limit
        return self._speaking
