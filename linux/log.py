"""Log mínimo con timestamp y flush.

Todo lo que imprime el pipeline pasa por acá para que (a) cada línea tenga la
hora (sin eso no se puede medir cuánto tardó el VAD en cerrar, Groq en
contestar, etc.) y (b) salga al instante aunque stdout no sea una terminal
(systemd / nohup / `python main.py > log.txt`): sin flush, Python bufferea y
parece que "no pasa nada" durante minutos.
"""

from __future__ import annotations

import os
import sys
import time

# LOG_DEBUG=1 en .env habilita los logs por frame/verbosos ([DBG] ...).
DEBUG = os.getenv("LOG_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def _stamp() -> str:
    now = time.time()
    return time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}"


def log(message: str, *, err: bool = False) -> None:
    stream = sys.stderr if err else sys.stdout
    print(f"[{_stamp()}] {message}", file=stream, flush=True)


def dbg(message: str) -> None:
    if DEBUG:
        log(f"[DBG] {message}")
