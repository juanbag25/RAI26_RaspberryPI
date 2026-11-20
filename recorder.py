"""Grabación continua del mic, para armar benchmarks con audio real del robot.

Graba TODO lo que entra por el mic (también mientras el robot habla y el STT
está muteado) en `RECORD_DIR` (default `data/recordings/`, no se versiona),
en WAV de 16 kHz cortados cada `RECORD_SEGMENT_S` segundos:

    rec-AAAAMMDD-HHMMSS-<run>.wav     audio (int16; 1 canal = ch0 procesado
                                      del ReSpeaker, o los 6 con RECORD_CHANNELS=all)
    index.jsonl                       una línea al abrir cada archivo y cada
                                      minuto: {"file", "sample", "t"} para
                                      ubicar cualquier evento del log
                                      (stt-*.jsonl, `t` = la misma hora de pared)
                                      en el audio, aunque se hayan perdido frames
    budget.json                       bytes grabados EN TOTAL, de todas las corridas

Tope duro acumulado: `RECORD_MAX_GB` (2) entre todas las sesiones. Al llegar,
corta y no vuelve a grabar nunca más, aunque se borren los WAV: el contador
vive en `budget.json` (borrarlo a mano es la única forma de habilitar otra
tanda). También corta si al disco le quedan menos de `RECORD_MIN_FREE_GB`.
`RECORD_AUDIO=0` lo apaga.

Escribir nunca frena la captura: los frames van a una cola y un hilo escribe;
si la tarjeta SD se traba y la cola se llena, se pierden frames de la
grabación (`dropped`), no del STT.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import threading
import time
import wave

from log import event, info, warn

BUDGET_FILE = "budget.json"
INDEX_FILE = "index.jsonl"
WAV_HEADER = 44
SYNC_EVERY_S = 60.0
BATCH_S = 1.0


class Recorder:
    def __init__(self, directory: str, *, max_bytes: int, channels: int = 1, sample_rate: int = 16000,
                 segment_s: float = 300.0, run: str | None = None, min_free_bytes: int = 0,
                 queue_s: float = 20.0, frame_ms: int = 30) -> None:
        self.dir = os.path.abspath(directory)
        self.max_bytes = int(max_bytes)
        self.channels = channels
        self.sr = sample_rate
        self.segment_bytes = int(segment_s * sample_rate) * 2 * channels
        self.run = run or (time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}")
        self.min_free = int(min_free_bytes)
        self.dropped = 0
        self.written = 0  # bytes de esta corrida
        self.stopped_reason: str | None = None
        self._q: queue.Queue = queue.Queue(maxsize=max(10, int(queue_s * 1000 / frame_ms)))
        self._wav: wave.Wave_write | None = None
        self._file: str | None = None
        self._file_bytes = 0
        self._file_samples = 0
        self._last_sync = 0.0
        self._thread: threading.Thread | None = None
        os.makedirs(self.dir, exist_ok=True)
        self.total = self._load_budget()

    # ------------------------------------------------------------ presupuesto
    def _load_budget(self) -> int:
        """Bytes grabados en total. Si el JSON se perdió o quedó atrás (corte
        de luz entre escribir audio y actualizarlo), manda lo que hay en disco."""
        total = 0
        try:
            with open(os.path.join(self.dir, BUDGET_FILE), encoding="utf-8") as f:
                total = int(json.load(f).get("bytes", 0))
        except (OSError, ValueError, TypeError):
            pass
        on_disk = sum(os.path.getsize(os.path.join(self.dir, n)) for n in os.listdir(self.dir)
                      if n.startswith("rec-") and n.endswith(".wav"))
        return max(total, on_disk)

    def _save_budget(self) -> None:
        path = os.path.join(self.dir, BUDGET_FILE)
        data = {"bytes": self.total, "max_bytes": self.max_bytes, "full": self.full,
                "updated": round(time.time(), 3), "run": self.run}
        try:
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(path + ".tmp", path)
        except OSError:
            pass

    @property
    def full(self) -> bool:
        return self.total >= self.max_bytes

    @property
    def remaining(self) -> int:
        return max(0, self.max_bytes - self.total)

    # ------------------------------------------------------------------ API
    def start(self) -> bool:
        """Arranca el hilo. False (y no graba) si el tope ya se alcanzó."""
        if self.full:
            self.stopped_reason = "tope"
            info("REC", f"grabación apagada: ya se grabó el tope ({self.total / 1e9:.2f} GB de "
                 f"{self.max_bytes / 1e9:.2f} GB, {os.path.join(self.dir, BUDGET_FILE)})")
            event("rec", state="tope", total_bytes=self.total, max_bytes=self.max_bytes)
            return False
        self._thread = threading.Thread(target=self._loop, name="recorder", daemon=True)
        self._thread.start()
        info("REC", f"grabando el mic en {self.dir} ({self.channels} canal(es)); quedan "
             f"{self.remaining / 1e9:.2f} GB de {self.max_bytes / 1e9:.2f} GB "
             f"(~{self.remaining / (self.sr * 2 * self.channels) / 3600:.1f} h)")
        event("rec", state="start", channels=self.channels, total_bytes=self.total,
              max_bytes=self.max_bytes, dir=self.dir)
        return True

    @property
    def active(self) -> bool:
        return self._thread is not None and self.stopped_reason is None

    def push(self, pcm: bytes, t: float | None = None) -> None:
        """Un bloque de audio (int16 intercalado). Nunca bloquea."""
        if not self.active:
            return
        try:
            self._q.put_nowait((pcm, time.time() if t is None else t))
        except queue.Full:
            self.dropped += 1

    def close(self, timeout: float = 5.0) -> None:
        if self._thread is None:
            return
        try:
            self._q.put(None, timeout=1.0)
        except queue.Full:
            pass
        self._thread.join(timeout)
        self._thread = None

    # --------------------------------------------------------------- escritor
    def _loop(self) -> None:
        try:
            done = False
            while not done:
                batch, t_first = [], None
                deadline = time.monotonic() + BATCH_S
                while time.monotonic() < deadline:
                    try:
                        item = self._q.get(timeout=max(0.01, deadline - time.monotonic()))
                    except queue.Empty:
                        break
                    if item is None:
                        done = True
                        break
                    if t_first is None:
                        t_first = item[1]
                    batch.append(item[0])
                if batch and not self._write(b"".join(batch), t_first):
                    break
        except Exception as exc:  # noqa: BLE001 - grabar nunca tumba el STT
            self.stopped_reason = f"error: {type(exc).__name__}: {exc}"
            warn("REC", f"grabación cortada por error ({type(exc).__name__}: {exc})")
        finally:
            self._close_file()
            self._save_budget()
            event("rec", state="stop", reason=self.stopped_reason or "fin", run_bytes=self.written,
                  total_bytes=self.total, dropped=self.dropped)

    def _write(self, data: bytes, t: float) -> bool:
        """Escribe respetando el tope y el disco. False = no se graba más."""
        frame = 2 * self.channels
        if self.min_free and shutil.disk_usage(self.dir).free < self.min_free:
            return self._stop("disco", f"al disco le quedan menos de {self.min_free / 1e9:.1f} GB")
        while data:
            if self._wav is None or self._file_bytes >= self.segment_bytes:
                if self.remaining <= WAV_HEADER + frame:
                    break
                self._open_file(t)
            allowed = (self.remaining // frame) * frame
            n = min(len(data), allowed, self.segment_bytes - self._file_bytes)
            if n <= 0:
                break
            if time.time() - self._last_sync >= SYNC_EVERY_S:
                self._index(t)  # muestra donde empieza este tramo = hora de su primer frame
            self._wav.writeframes(data[:n])  # wave reescribe el header: el WAV queda válido si se corta la luz
            self._file_bytes += n
            self._file_samples += n // frame
            self.written += n
            self.total += n
            t += n / (frame * self.sr)
            data = data[n:]
        self._save_budget()
        if data or self.remaining <= WAV_HEADER + frame:
            return self._stop("tope", "se llegó al tope")
        return True

    def _stop(self, reason: str, why: str) -> bool:
        self.stopped_reason = reason
        warn("REC", f"grabación terminada: {why} ({self.total / 1e9:.2f} GB grabados en total)"
             + (": no se graba más (borrar budget.json para otra tanda)" if reason == "tope" else ""))
        return False

    def _open_file(self, t: float) -> None:
        self._close_file()
        name = f"rec-{time.strftime('%Y%m%d-%H%M%S', time.localtime(t))}-{self.run}.wav"
        path = os.path.join(self.dir, name)
        wav = wave.open(path, "wb")
        wav.setnchannels(self.channels)
        wav.setsampwidth(2)
        wav.setframerate(self.sr)
        wav.writeframes(b"")  # escribe el header ya
        self._wav, self._file = wav, name
        self._file_bytes = self._file_samples = 0
        self.total += WAV_HEADER
        self.written += WAV_HEADER
        self._index(t, opened=True)

    def _index(self, t: float, opened: bool = False) -> None:
        self._last_sync = time.time()
        rec = {"file": self._file, "sample": self._file_samples, "t": round(t, 3), "run": self.run,
               "channels": self.channels, "dropped": self.dropped}
        if opened:
            rec["open"] = True
        try:
            with open(os.path.join(self.dir, INDEX_FILE), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def _close_file(self) -> None:
        if self._wav is not None:
            try:
                self._wav.close()
            except OSError:
                pass
            self._wav = None
