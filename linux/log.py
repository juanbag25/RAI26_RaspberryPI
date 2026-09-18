"""Log del pipeline: una línea por evento, con hora, color por etapa y flush.

Formato de cada línea:

    HH:MM:SS.mmm ETAPA  símbolo mensaje

- La hora sirve para medir cuánto tardó cada paso (VAD en cerrar, Groq en
  contestar, etc.).
- ETAPA es quién habla (AUDIO, VAD, SPOT, WAKE, STT, NET, CTRL, HB...) y tiene
  siempre el mismo color, así se sigue el recorrido de una frase de un vistazo.
- El símbolo dice qué pasó: ✓ aceptado/hecho, ✗ DESCARTADO (con la razón y sólo
  los valores que la explican), ! aviso, ✖ error, · info sin importancia.
- Todo sale con flush: si stdout no es una terminal (systemd, nohup,
  `> log.txt`) Python bufferea y parece que "no pasa nada" durante minutos.

Colores: se activan si stdout es una terminal. LOG_COLOR=1 los fuerza (útil con
`| tee`), LOG_COLOR=0 o NO_COLOR los apaga. LOG_DEBUG=1 habilita las líneas
por frame / verbosas (`dbg`), que por defecto no salen.

Uso:

    from log import ok, drop, info, warn, err, dim, dbg, fmt
    ok("VAD", f"voz {ms} ms {fmt(nivel=level)}")
    drop("VAD", "muy corta", voz_ms=120, minimo_ms=400)
"""

from __future__ import annotations

import os
import sys
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
    "SOUND": _DIM,            # beep local
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


def _emit(tag: str, symbol: str, message: str, *, symbol_color: str = "",
          message_color: str = "", err: bool = False) -> None:
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
    print(line, file=stream, flush=True)


# --- API ---------------------------------------------------------------------

def info(tag: str, message: str) -> None:
    """Algo pasó y vale la pena verlo (sin juicio de valor)."""
    _emit(tag, "·", message)


def ok(tag: str, message: str) -> None:
    """Paso superado: la frase avanza a la siguiente etapa."""
    _emit(tag, "✓", message, symbol_color=_BRIGHT_GREEN)


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
    _emit(tag, "✗", message, symbol_color=_BRIGHT_YELLOW)


def warn(tag: str, message: str) -> None:
    _emit(tag, "!", message, symbol_color=_YELLOW, message_color=_YELLOW)


def err(tag: str, message: str) -> None:
    _emit(tag, "✖", message, symbol_color=_BRIGHT_RED, message_color=_BRIGHT_RED, err=True)


def dim(tag: str, message: str) -> None:
    """Contexto de fondo (heartbeat, estado): visible pero apagado."""
    _emit(tag, "", message, message_color=_DIM)


def dbg(message: str, tag: str = "DBG") -> None:
    """Sólo con LOG_DEBUG=1: por frame, parciales de Vosk, detalle de red."""
    if DEBUG:
        _emit(tag, "", message, message_color=_DIM)

