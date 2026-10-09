"""Log del pipeline: una línea por evento, con hora, color por etapa y flush.

Formato de cada línea:

    HH:MM:SS.mmm ETAPA  símbolo mensaje

- La hora sirve para medir cuánto tardó cada paso (VAD en cerrar, Groq en
  contestar, etc.).
- ETAPA es quién habla (AUDIO, VAD, SPOT, WAKE, STT, NET, CTRL, HB...) y tiene
  siempre el mismo color, así se sigue el recorrido de una frase de un vistazo.
- El símbolo dice qué pasó: ✓ aceptado/hecho, ✗ DESCARTADO (con la razón y sólo
  los valores que la explican), ✂ se conservó sólo una parte, ! aviso,
  ✖ error, · info sin importancia.
- Todo sale con flush: si stdout no es una terminal (systemd, nohup,
  `> log.txt`) Python bufferea y parece que "no pasa nada" durante minutos.

Colores: se activan si stdout es una terminal. LOG_COLOR=1 los fuerza (útil con
`| tee`), LOG_COLOR=0 o NO_COLOR los apaga. LOG_DEBUG=1 habilita las líneas
por frame / verbosas (`dbg`), que por defecto no salen.

Uso:

    from log import ok, drop, info, warn, err, dim, dbg, fmt, event
    ok("VAD", f"voz {ms} ms {fmt(nivel=level)}")
    drop("VAD", "muy corta", voz_ms=120, minimo_ms=400)
    event("utt", accepted=True, level=0.08)   # sólo al archivo, no a consola

Log persistente (`persist()`, lo llama main.py al arrancar): además de la
consola, TODO se escribe en `LOG_DIR/stt-AAAA-MM-DD.jsonl` (por defecto
`<repo>/logs/`), un objeto JSON por línea, para analizar después (ver
log_report.py). Sobrevive reinicios del servicio y de la Pi; nunca guarda
audio. Dos clases de registro:

- Líneas de consola: {"t", "run", "k": ok|drop|info|warn|err|dim|dbg, "tag",
  "msg"} (+ los valores de drop() como campos).
- Eventos estructurados (`event()`): {"t", "run", "k": "ev", "ev": nombre,
  ...campos}. Son los que se analizan: config al arrancar, cada utterance y
  su destino, heartbeat, wake, voz.

Rotación: un archivo por día; al arrancar y al cambiar de día se borran los
de más de LOG_KEEP_DAYS días (60) y, si el total pasa LOG_MAX_MB (500), los
más viejos. LOG_PERSIST=0 lo apaga.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

DEBUG = os.getenv("LOG_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def _color_enabled() -> bool:
    forced = os.getenv("LOG_COLOR", "").strip().lower()
    if forced in ("0", "false", "no", "off"):
        return False
    if forced in ("1", "true", "yes", "on"):
        return True
    if os.getenv("NO_COLOR"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


COLOR = _color_enabled()

if COLOR and os.name == "nt":
    # Habilita el procesamiento de secuencias ANSI en la consola de Windows.
    os.system("")

# Que un símbolo unicode nunca tire el log (consolas con codepage viejo).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"
_WHITE = "\033[37m"
_BRIGHT_RED = "\033[91m"
_BRIGHT_GREEN = "\033[92m"
_BRIGHT_YELLOW = "\033[93m"
_BRIGHT_BLUE = "\033[94m"
_BRIGHT_MAGENTA = "\033[95m"
_BRIGHT_CYAN = "\033[96m"

# Color fijo por etapa: cada módulo del pipeline habla con su propio color.
_TAG_COLORS = {
    "AUDIO": _BLUE,           # captura del mic
    "VAD": _CYAN,             # detección de voz + filtro de cercanía
    "SPOT": _BRIGHT_YELLOW,   # spotter de "oye rai" (Vosk)
    "WAKE": _BRIGHT_MAGENTA,  # estado despierto/dormido y foco de atención
    "STT": _BRIGHT_BLUE,      # transcripción (Groq / Whisper)
    "NET": _BRIGHT_GREEN,     # envío al orquestador
    "CTRL": _WHITE,           # SPEAK_START / SPEAK_END del orquestador
    "ARRAY": _BLUE,           # ReSpeaker: parámetros DSP, LEDs, lecturas USB
    "DOA": _BRIGHT_CYAN,      # dirección de la voz y foco espacial
    "SPK": _MAGENTA,          # verificación de hablante (huella de voz)
    "MIX": _BRIGHT_CYAN,      # recorte de frases con dos voces (mix_trim.py)
    "LOG": _DIM,
    "HB": _DIM,               # heartbeat
    "POWER": _DIM,
    "MAIN": _WHITE,
    "CONFIG": _DIM,
    "DBG": _DIM,
}
_TAG_WIDTH = 5


def _stamp() -> str:
    now = time.time()
    return time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}"


def _paint(text: str, *codes: str) -> str:
    if not COLOR or not codes:
        return text
    return "".join(codes) + text + _RESET


def fmt(**values: object) -> str:
    """`fmt(nivel=0.0312, umbral=0.055)` -> `nivel=0.0312 umbral=0.0550`.

    Floats con 4 decimales (escala del RMS normalizado), el resto tal cual.
    """
    parts = []
    for key, value in values.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


class _Sink:
    """Archivo JSONL del día. Thread-safe; abre/rota solo."""

    def __init__(self, directory: str, keep_days: float, max_mb: float, run: str) -> None:
        self.dir = directory
        self.keep_days = keep_days
        self.max_bytes = max_mb * 1024 * 1024
        self.run = run
        self._lock = threading.Lock()
        self._day = ""
        self._file = None
        os.makedirs(directory, exist_ok=True)

    def _open_for_today(self) -> None:
        day = time.strftime("%Y-%m-%d")
        if day == self._day and self._file is not None:
            return
        if self._file is not None:
            self._file.close()
        self._day = day
        self._file = open(os.path.join(self.dir, f"stt-{day}.jsonl"), "a", encoding="utf-8")
        self._cleanup()

    def _cleanup(self) -> None:
        files = sorted(f for f in os.listdir(self.dir)
                       if f.startswith("stt-") and f.endswith(".jsonl"))
        current = f"stt-{self._day}.jsonl"
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

    def write(self, record: dict) -> None:
        record = {"t": round(time.time(), 3), "run": self.run, **record}
        line = json.dumps(record, ensure_ascii=False, default=_jsonable)
        with self._lock:
            try:
                self._open_for_today()
                self._file.write(line + "\n")
                self._file.flush()
            except OSError:
                pass  # disco lleno / SD de sólo lectura: la consola sigue


def _jsonable(value: object) -> object:
    """numpy y demás -> tipos JSON (float32, int64, arrays chicos)."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


_sink: _Sink | None = None


def persist(directory: str | None = None) -> str | None:
    """Activa el log persistente (ver docstring del módulo). Devuelve la
    carpeta, o None si quedó apagado (LOG_PERSIST=0 o no se pudo crear)."""
    global _sink
    if os.getenv("LOG_PERSIST", "1").strip().lower() in ("0", "false", "no", "off"):
        return None
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
    directory = directory or os.getenv("LOG_DIR", "").strip() or root
    try:
        keep = float(os.getenv("LOG_KEEP_DAYS", "") or 60)
        max_mb = float(os.getenv("LOG_MAX_MB", "") or 500)
    except ValueError:
        keep, max_mb = 60.0, 500.0
    run = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    try:
        _sink = _Sink(os.path.abspath(directory), keep, max_mb, run)
    except OSError as exc:
        warn("LOG", f"no pude crear {directory} ({exc}): sin log persistente")
        return None
    return _sink.dir


def event(name: str, **fields: object) -> None:
    """Evento estructurado: sólo al archivo (no ensucia la consola)."""
    if _sink is not None:
        _sink.write({"k": "ev", "ev": name, **fields})


def _emit(tag: str, symbol: str, message: str, *, symbol_color: str = "",
          message_color: str = "", err: bool = False, kind: str = "",
          plain: str | None = None, fields: dict | None = None,
          lines: list[str] | None = None) -> None:
    if _sink is not None:
        record = {"k": kind, "tag": tag, "msg": message if plain is None else plain}
        if fields:
            record["f"] = fields
        if lines:
            record["lines"] = lines
        _sink.write(record)
    stream = sys.stderr if err else sys.stdout
    tag_color = _TAG_COLORS.get(tag, "")
    line = (
        _paint(_stamp(), _DIM)
        + " "
        + _paint(f"{tag:<{_TAG_WIDTH}}", tag_color, _BOLD)
        + " "
        + (_paint(symbol, symbol_color) + " " if symbol else "")
        + _paint(message, message_color)
    )
    if lines:
        # Debajo del mensaje, alineadas con él. Un solo print: otro hilo no
        # puede meter una línea suya en el medio del bloque.
        indent = " " * (len(_stamp()) + 1 + _TAG_WIDTH + 1 + 2)
        line += "".join("\n" + indent + _paint(extra, _DIM) for extra in lines)
    print(line, file=stream, flush=True)


# --- API ---------------------------------------------------------------------

def info(tag: str, message: str) -> None:
    """Algo pasó y vale la pena verlo (sin juicio de valor)."""
    _emit(tag, "·", message, kind="info")


def ok(tag: str, message: str) -> None:
    """Paso superado: la frase avanza a la siguiente etapa."""
    _emit(tag, "✓", message, symbol_color=_BRIGHT_GREEN, kind="ok")


def drop(tag: str, reason: str, **values: object) -> None:
    """La frase NO sigue. `reason` dice por qué; `values` sólo los números
    que lo explican (nivel medido vs umbral, ms de voz vs mínimo, etc.).

    Todos los descartes del pipeline pasan por acá para que se lean igual:

        VAD   ✗ DESCARTADO lejana/floja  nivel=0.0263 umbral=0.0550 ruido=0.0036
    """
    detail = fmt(**values)
    message = _paint("DESCARTADO ", _BRIGHT_YELLOW, _BOLD) + _paint(reason, _BRIGHT_YELLOW)
    if detail:
        message += "  " + _paint(detail, _DIM)
    _emit(tag, "✗", message, symbol_color=_BRIGHT_YELLOW, kind="drop",
          plain=f"DESCARTADO {reason}", fields=values or None)


def drop_block(tag: str, reason: str, lines: list[str], **values: object) -> None:
    """drop() con un bloque de detalle debajo (ej. el mapa de tramos de
    mix_trim.py), impreso de una sola vez."""
    detail = fmt(**values)
    message = _paint("DESCARTADO ", _BRIGHT_YELLOW, _BOLD) + _paint(reason, _BRIGHT_YELLOW)
    if detail:
        message += "  " + _paint(detail, _DIM)
    _emit(tag, "✗", message, symbol_color=_BRIGHT_YELLOW, kind="drop",
          plain=f"DESCARTADO {reason}", fields=values or None, lines=lines)


def cut(tag: str, message: str, lines: list[str]) -> None:
    """Se conservó sólo una parte de la frase (✂), con el detalle debajo."""
    _emit(tag, "✂", message, symbol_color=_BRIGHT_CYAN, kind="cut", lines=lines)


def warn(tag: str, message: str) -> None:
    _emit(tag, "!", message, symbol_color=_YELLOW, message_color=_YELLOW, kind="warn")


def err(tag: str, message: str) -> None:
    _emit(tag, "✖", message, symbol_color=_BRIGHT_RED, message_color=_BRIGHT_RED, err=True,
          kind="err")


def dim(tag: str, message: str) -> None:
    """Contexto de fondo (heartbeat, estado): visible pero apagado."""
    _emit(tag, "", message, message_color=_DIM, kind="dim")


def dbg(message: str, tag: str = "DBG") -> None:
    """Sólo con LOG_DEBUG=1: por frame, parciales de Vosk, detalle de red."""
    if DEBUG:
        _emit(tag, "", message, message_color=_DIM, kind="dbg")

