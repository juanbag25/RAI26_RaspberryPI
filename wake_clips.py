"""Clips de audio para medir y ajustar el wake word con datos.

Dormido no se transcribe nada, así que un «oye rai» que el spotter (Vosk) no
reconoce no deja rastro: sólo una frase `dormido` en el log. Este módulo
guarda en disco el audio de esas frases y de las que sí sabemos que traían el
nombre, para escucharlas después y armar un set de prueba del spotter:

- `dormido`: frase cercana cerrada estando dormido (candidata a wake perdido).
- `wake`:    lo que el spotter reconoció como frase de wake (positivo).
- `nombre`:  despierto, Whisper transcribió el nombre («Oye, RAI») en una
             frase que no venía marcada por el spotter (si `spot` no lo trae,
             el spotter no lo oyó).

Cada clip es un .wav (16 kHz mono int16) en WAKE_CLIPS_DIR y una línea en
`clips.jsonl` con tag, lo que oyó Vosk (`spot`), texto y nivel. Se guardan los
últimos WAKE_CLIPS_KEEP (0 = desactivado). La escritura va en un hilo propio:
el loop de captura nunca espera al disco.

Es audio de quien hable cerca del robot y queda SÓLO en la Pi (data/ no se
versiona). Para apagarlo: WAKE_CLIPS_KEEP=0 en .env.
"""

from __future__ import annotations

import glob
import json
import os
import queue
import threading
import time
import wave

import numpy as np

from config import WAKE_CLIPS_DIR, WAKE_CLIPS_KEEP, WAKE_CLIPS_MAX_S
from log import err, info


class WakeClips:
    def __init__(self, directory: str = WAKE_CLIPS_DIR, keep: int = WAKE_CLIPS_KEEP,
                 max_s: float = WAKE_CLIPS_MAX_S) -> None:
        self.enabled = keep > 0
        self._dir = directory
        self._keep = keep
        self._max_s = max_s
        self._seq = 0
        self._queue: "queue.Queue[tuple | None]" = queue.Queue(maxsize=32)
        if not self.enabled:
            return
        os.makedirs(directory, exist_ok=True)
        threading.Thread(target=self._run, name="wake-clips", daemon=True).start()
        info("CLIP", f"guardo audio de wakes y frases dormido en {directory} "
             f"(últimos {keep}; WAKE_CLIPS_KEEP=0 para apagar)")

    def save(self, audio: "np.ndarray | None", tag: str, **meta) -> str | None:
        """Encola un clip (float32 -1..1). Devuelve el nombre del archivo, o
        None si no se guarda (apagado, vacío, muy largo o cola llena)."""
        if not self.enabled or audio is None or len(audio) == 0:
            return None
        if len(audio) / 16000.0 > self._max_s:
            return None
        self._seq += 1
        name = time.strftime("%Y%m%d-%H%M%S") + f"-{self._seq % 1000:03d}_{tag}.wav"
        try:
            self._queue.put_nowait((name, audio.copy(), tag, meta))
        except queue.Full:
            return None
        return name

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            name, audio, tag, meta = item
            try:
                pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
                with wave.open(os.path.join(self._dir, name), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(16000)
                    w.writeframes(pcm.tobytes())
                line = dict(t=round(time.time(), 3), file=name, tag=tag,
                            audio_s=round(len(audio) / 16000.0, 2), **meta)
                with open(os.path.join(self._dir, "clips.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                self._prune()
            except Exception as exc:  # noqa: BLE001 - nunca romper el STT por esto
                err("CLIP", f"{type(exc).__name__}: {exc}")

    def _prune(self) -> None:
        files = sorted(glob.glob(os.path.join(self._dir, "*.wav")))
        for path in files[:-self._keep]:
            try:
                os.remove(path)
            except OSError:
                pass
