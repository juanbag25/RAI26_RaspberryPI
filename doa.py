"""Foco espacial: ¿de qué dirección vino cada frase y es la de quien me llamó?

Usa las lecturas de DoA del ReSpeaker (respeaker.py) entre la apertura y el
cierre de cada utterance. Medido con el mic real (README, "Mic array"):

- En silencio DOAANGLE queda pegado en el último valor o sigue al ruido: sólo
  sirven las lecturas con VOICEACTIVITY=1.
- Con voz, ~2/3 de las lecturas caen a pocos grados de la persona y ~1/3 son
  reflexiones (en una sala con RT60 ~0.9 s). Por eso NO se promedia la frase
  (el promedio se va hacia las paredes): se mira la dirección dominante
  (histograma) y la FRACCIÓN de lecturas que cae dentro del foco.

Ángulos en el marco del robot: 0° = frente (DOA_FORWARD_OFFSET_DEG corrige
cómo esté montado el array), sentido igual al del chip.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

from config import (
    DOA_BIN_DEG,
    DOA_BLOCKED_SECTORS,
    DOA_FOLLOW_ALPHA,
    DOA_FORWARD_OFFSET_DEG,
    DOA_MIN_IN_FOCUS,
    DOA_MIN_SAMPLES,
    DOA_TOLERANCE_DEG,
)

Sector = tuple[float, float]


def ang_diff(a: float, b: float) -> float:
    """Distancia angular en grados, 0..180."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def circular_mean(angles: list[float]) -> float:
    s = sum(math.sin(math.radians(a)) for a in angles)
    c = sum(math.cos(math.radians(a)) for a in angles)
    return math.degrees(math.atan2(s, c)) % 360.0


def to_robot_frame(chip_angle: float) -> float:
    return (chip_angle - DOA_FORWARD_OFFSET_DEG) % 360.0


def parse_sectors(spec: str) -> list[Sector]:
    """"170-200, 350-10" -> [(170, 200), (350, 10)] (cruzar 0° vale)."""
    sectors: list[Sector] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        if not sep:
            raise ValueError(f"sector inválido {part!r}: usar desde-hasta en grados (ej. 170-200)")
        sectors.append((float(lo) % 360.0, float(hi) % 360.0))
    return sectors


def in_sector(angle: float, sector: Sector) -> bool:
    lo, hi = sector
    angle %= 360.0
    return lo <= angle <= hi if lo <= hi else angle >= lo or angle <= hi


def in_any_sector(angle: float, sectors: list[Sector]) -> bool:
    return any(in_sector(angle, s) for s in sectors)


def dominant_direction(angles: list[float], bin_deg: float = DOA_BIN_DEG) -> float | None:
    """Dirección con más lecturas: el bin más poblado (sumando sus dos vecinos,
    así una persona justo en el borde de dos bins no se parte en dos) y la
    media circular de las lecturas cercanas a él. Robusto a reflexiones."""
    if not angles:
        return None
    nbins = max(1, int(round(360.0 / bin_deg)))
    width = 360.0 / nbins
    counts = [0] * nbins
    for a in angles:
        counts[int((a % 360.0) // width) % nbins] += 1
    best = max(range(nbins),
               key=lambda i: counts[i - 1] + counts[i] + counts[(i + 1) % nbins])
    center = (best + 0.5) * width
    near = [a for a in angles if ang_diff(a, center) <= 1.5 * width]
    return circular_mean(near)


@dataclass(frozen=True)
class DoaReading:
    """Resumen espacial de una frase (ángulos en el marco del robot)."""
    n: int                    # lecturas con voz en la frase
    direction: float | None   # dirección dominante
    in_focus: float           # fracción de lecturas dentro del foco (0..1)
    blocked: float            # fracción de lecturas en sectores bloqueados


def summarize(angles: list[float], focus: float | None,
              tolerance: float = DOA_TOLERANCE_DEG,
              blocked: list[Sector] | None = None) -> DoaReading:
    blocked = blocked or []
    n = len(angles)
    if n == 0:
        return DoaReading(0, None, 0.0, 0.0)
    in_focus = (sum(1 for a in angles if ang_diff(a, focus) <= tolerance) / n
                if focus is not None else 0.0)
    in_blocked = sum(1 for a in angles if in_any_sector(a, blocked)) / n
    return DoaReading(n, dominant_direction(angles), in_focus, in_blocked)


class DoaFocus:
    """Dirección de la persona a la que el robot le presta atención.

    `lock()` al despertar (la dirección del «oye rai»), `judge()` por cada
    frase mientras está despierto, `follow()` con las aceptadas (la persona se
    mueve un poco), `clear()` al dormirse.
    """

    def __init__(self, blocked: list[Sector] | None = None) -> None:
        self._lock = threading.Lock()
        self._focus: float | None = None
        self.blocked = parse_sectors(DOA_BLOCKED_SECTORS) if blocked is None else blocked

    @property
    def focus(self) -> float | None:
        with self._lock:
            return self._focus

    def lock(self, direction: float | None) -> None:
        with self._lock:
            self._focus = direction

    def clear(self) -> None:
        self.lock(None)

    def follow(self, direction: float | None) -> None:
        """EMA circular hacia `direction`."""
        if direction is None:
            return
        with self._lock:
            if self._focus is None:
                self._focus = direction
                return
            delta = (direction - self._focus + 180.0) % 360.0 - 180.0
            self._focus = (self._focus + DOA_FOLLOW_ALPHA * delta) % 360.0

    def summarize(self, angles: list[float]) -> DoaReading:
        return summarize(angles, self.focus, blocked=self.blocked)

    def judge(self, reading: DoaReading) -> tuple[bool, str]:
        """(aceptada, motivo). Con pocas lecturas se abstiene (acepta): una
        frase corta sin datos de dirección no se pierde por eso."""
        if reading.n < DOA_MIN_SAMPLES:
            return True, "sin datos de dirección"
        if reading.blocked >= 0.6:
            return False, "sector bloqueado (ruido propio del robot)"
        focus = self.focus
        if focus is None:
            return True, "sin foco"
        if reading.in_focus >= DOA_MIN_IN_FOCUS:
            return True, "en foco"
        return False, "fuera de foco"
