"""rai_perf — métricas de rendimiento por turno, mismo formato en los 3 repos.

COPIA IDÉNTICA en los tres repos; la canónica es la de
`R-AI-026/observability/rai_perf.py` y el contrato está en
`R-AI-026/observability/PERF.md`. No editar una copia sola: cambiar la
canónica y copiarla (el test de R-AI-026 compara las tres si están al lado).

Cada registro es una línea JSON en `perf-AAAA-MM-DD.jsonl`:

    {"v": 1, "ts": "2026-11-13T18:04:05.123Z", "host": "rai-pi", "svc": "stt",
     "run": "...", "ver": "a1b2c3d", "turn": "3f9c2a71d0", "stage": "pi.utterance",
     "ms": 812.4, ...campos de la etapa}

- `ms`: duración de la etapa, medida con el reloj monotónico de ESA máquina
  (nunca se restan horas de equipos distintos: la Pi no tiene RTC).
- `turn`: id del turno, lo crea la Pi y viaja Pi -> orquestador -> LLM
  server. Es lo que une los tres archivos.
- `ver`: commit del código (+ "-dirty"): con eso se compara antes/después.
- Campos extra: sólo escalares (None, bool, int, float, str corto). Nunca
  texto de la conversación: largos, conteos y resultados.

Sólo stdlib, Python 3.8+. Escribir nunca tira excepciones: si el disco falla,
se pierde la métrica y el robot sigue igual.
"""
import datetime
import json
import math
import os
import socket
import subprocess
import threading
import time
import uuid

SCHEMA = 1
RESERVED = ("v", "ts", "host", "svc", "run", "ver", "turn", "stage", "ms")
MAX_STR = 200
PREFIX = "perf-"


def now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_turn_id():
    """Id de turno: 10 hex al azar (único sin depender del reloj)."""
    return uuid.uuid4().hex[:10]


def ms_since(t0):
    """Milisegundos desde `t0` (time.monotonic()), con 1 decimal."""
    return round((time.monotonic() - t0) * 1000.0, 1)


def to_ms(seconds):
    return None if seconds is None else round(float(seconds) * 1000.0, 1)


def git_version(path):
    """Commit corto del repo que contiene `path` (+ "-dirty" si hay cambios
    sin commitear en archivos versionados). "?" si no se puede saber."""
    here = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    try:
        rev = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=3).stdout.strip()
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=3).stdout.strip()
        return (rev or "?") + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "?"


def scalar(value):
    """Valor apto para un campo: escalar JSON, str recortado, NaN -> None."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item"):  # numpy
        try:
            return scalar(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)[:MAX_STR]


def make_record(stage, *, svc, run, ver, turn=None, ms=None, ts=None, host=None, **fields):
    """Registro con el esquema común. Campos no escalares se convierten a str
    (recortado); nombres reservados en `fields` se ignoran."""
    record = {"v": SCHEMA, "ts": ts or now_iso(), "host": host or socket.gethostname(),
              "svc": svc, "run": run, "ver": ver, "turn": turn, "stage": stage,
              "ms": None if ms is None else scalar(float(ms))}
    for key, value in fields.items():
        if key not in RESERVED:
            record[key] = scalar(value)
    return record


class PerfLog:
    """`perf-AAAA-MM-DD.jsonl` en `directory` (fecha UTC). Thread-safe.

    Retención: al abrir cada día se borran los archivos `perf-*` de más de
    `keep_days` y, si el total pasa `max_mb`, los más viejos. Sólo toca
    archivos `perf-*.jsonl` de su carpeta.
    """

    def __init__(self, directory, svc, ver=None, run=None, keep_days=90.0, max_mb=200.0):
        self.dir = os.path.abspath(os.path.expanduser(directory))
        self.svc = svc
        self.ver = ver if ver is not None else "?"
        self.run = run or (time.strftime("%Y%m%d-%H%M%S") + "-" + str(os.getpid()))
        self.keep_days = keep_days
        self.max_bytes = max_mb * 1024 * 1024
        self.host = socket.gethostname()
        self._lock = threading.Lock()
        self._day = ""
        self._file = None
        os.makedirs(self.dir, exist_ok=True)

    def write(self, stage, ms=None, turn=None, **fields):
        """Una etapa. `ms` en milisegundos (o None si es un evento sin duración)."""
        self.write_record(make_record(stage, svc=self.svc, run=self.run, ver=self.ver,
                                      turn=turn, ms=ms, host=self.host, **fields))

    def write_record(self, record):
        """Un registro ya armado (lo usa el collector de la Jetson)."""
        try:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            return
        with self._lock:
            try:
                self._open_for_today()
                self._file.write(line + "\n")
                self._file.flush()
            except OSError:
                pass

    def close(self):
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
                self._file = None

    def _open_for_today(self):
        day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        if day == self._day and self._file is not None:
            return
        if self._file is not None:
            self._file.close()
            self._file = None
        self._day = day
        self._file = open(os.path.join(self.dir, PREFIX + day + ".jsonl"), "a", encoding="utf-8")
        self._cleanup()

    def _cleanup(self):
        try:
            files = sorted(f for f in os.listdir(self.dir) if f.startswith(PREFIX) and f.endswith(".jsonl"))
            current = PREFIX + self._day + ".jsonl"
            limit = time.time() - self.keep_days * 86400
            for name in list(files):
                path = os.path.join(self.dir, name)
                if name != current and self.keep_days > 0 and os.path.getmtime(path) < limit:
                    os.remove(path)
                    files.remove(name)
            sizes = {f: os.path.getsize(os.path.join(self.dir, f)) for f in files}
            for name in files:  # más viejo primero
                if sum(sizes.values()) <= self.max_bytes or name == current:
                    break
                os.remove(os.path.join(self.dir, name))
                del sizes[name]
        except OSError:
            pass


def from_env(svc, default_dir, ver=None, run=None):
    """PerfLog configurado por entorno: PERF_DIR (carpeta), PERF_KEEP_DAYS (90),
    PERF_MAX_MB (200), PERF_ENABLED=0 lo apaga (devuelve None). Si la carpeta
    no se puede crear, también None."""
    if os.getenv("PERF_ENABLED", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    try:
        keep = float(os.getenv("PERF_KEEP_DAYS", "") or 90)
        max_mb = float(os.getenv("PERF_MAX_MB", "") or 200)
    except ValueError:
        keep, max_mb = 90.0, 200.0
    try:
        return PerfLog(os.getenv("PERF_DIR", "").strip() or default_dir, svc, ver=ver, run=run,
                       keep_days=keep, max_mb=max_mb)
    except OSError:
        return None
