"""Recorte de frases mezcladas: dos personas hablando a la vez.

Problema: el VAD no separa voces. Si alguien se mete mientras le hablan al
robot, todo cae en la misma utterance y los filtros la juzgan ENTERA:

- dirección (doa.py): las lecturas se reparten entre las dos personas y el
  porcentaje en foco no llega a DOA_MIN_IN_FOCUS -> «fuera de foco».
- huella de voz (speaker_id.py): un solo embedding de las dos voces mezcladas
  da una similitud intermedia -> «otra voz».

Y se tira todo, aunque la mitad la haya dicho quien llamó. Con
MIX_TRIM_ENABLED=1, en vez de tirarla se juzga por tramos:

1. La frase se parte en tramos de MIX_TRIM_SEG_S (0.5 s). Los tramos flojos
   (pausas) son «silencio» y no deciden.
2. Dirección por tramo: las lecturas de DoA de ese medio segundo, mismo foco
   y mismo DOA_MIN_IN_FOCUS que la frase entera.
3. Voz por tramo: huellas de ventanas de MIX_TRIM_WIN_S (1 s, salto de un
   tramo) contra la referencia de quien llamó; cada tramo se queda con el
   promedio de las ventanas que lo cubren. Menos de 1 s da huellas ruidosas.
4. Se conserva un tramo si ni la voz ni la dirección dicen que es de otro.
   Las pausas se conservan sólo entre dos tramos conservados.
5. Lo conservado (con MIX_TRIM_PAD_S de margen para no comer palabras) se
   junta con un silencio corto entre pedazos, se verifica de nuevo entero
   contra quien llamó, y eso es lo que va a Whisper.

Si no se puede separar (no hay tramos de otro, o queda menos de
MIX_TRIM_MIN_KEEP_S, o lo que queda tampoco se parece a quien llamó) se
descarta como antes.

Límite: lo que las dos personas dicen EXACTAMENTE a la vez no se separa (eso
pediría separación de fuentes); se rescatan los pedazos donde habla sólo una.

Log: cada frase recortada sale como UN bloque (log.cut / log.drop_block) con
el mapa de tramos, qué se sacó y por qué, y qué decía la frase entera, así no
hay que reconstruir la charla desde líneas sueltas:

    MIX   ✂ frase mezclada (otra voz sim=0.22): conservo 1.5 de 3.5 s  [cómputo 0.31 s]
            voz  ██▒▒▒▒████··   █ quien llamó  ▒ otra  ? sin dato  · pausa   (0.5 s/casilla)
            dir  ████▒▒████··   █ en foco 32°  ▒ fuera
            uso  ██────████──
            saqué 1.0–2.0 s: otra voz (sim 0.05–0.12) · 1.5–2.0 s ...
            lo que quedó vs quien llamó: sim 0.48
            la frase entera decía: «prendé la luz che viste el partido»
    STT   · «prendé la luz» (0.61s, confianza=0.92)  [recortada 1.5/3.5 s]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from config import (
    DOA_MIN_IN_FOCUS,
    MIX_TRIM_MIN_KEEP_S,
    MIX_TRIM_MIN_SIM,
    MIX_TRIM_PAD_S,
    MIX_TRIM_SEG_S,
    MIX_TRIM_WIN_S,
    SAMPLE_RATE,
    SPEAKER_MIN_AUDIO_S,
    SPEAKER_MIN_SIMILARITY,
)
from doa import DoaFocus, DoaReading
from log import cut, drop_block

# Silencio entre pedazos conservados: que Whisper no pegue la última palabra
# de uno con la primera del siguiente.
_GAP_S = 0.25
# Un tramo es «pausa» si su nivel queda por debajo de esta fracción del p90
# de los tramos de la frase.
_QUIET_RATIO = 0.2
# Con menos lecturas con voz que esto, la dirección del tramo es «sin dato».
_MIN_DOA_SAMPLES = 2

_SAME, _OTHER, _UNKNOWN, _QUIET = "█", "▒", "?", "·"


def _seg_samples() -> int:
    return max(1, int(MIX_TRIM_SEG_S * SAMPLE_RATE))


@dataclass
class MixContext:
    """Lo que el hilo de captura sabe de la frase y el de STT no: la
    dirección por tramo (hay que leerla del ReSpeaker mientras las lecturas
    siguen en el buffer) y si la dirección ya la rechazó entera."""
    doa: list[DoaReading] | None      # None = sin array
    focus: float | None               # foco al cerrar la frase
    doa_rejected: str | None = None   # motivo, si la dirección la rechazó entera
    doa_values: dict = field(default_factory=dict)  # para el drop("DOA") de siempre


def doa_by_segment(angles: Callable[[float, float], list[float]], focus: DoaFocus,
                   t0: float, n_samples: int) -> list[DoaReading]:
    """Resumen de dirección de cada tramo. `t0` = time.monotonic() del primer
    sample de la frase (vad.last_span[0]); `angles(t0, t1)` = Spatial.angles."""
    seg = _seg_samples()
    out = []
    for start in range(0, n_samples, seg):
        a = t0 + start / SAMPLE_RATE
        b = t0 + min(start + seg, n_samples) / SAMPLE_RATE
        out.append(focus.summarize(angles(a, b)))
    return out


@dataclass
class _Segment:
    start: int
    end: int
    quiet: bool = False
    voice: str = _UNKNOWN
    sim: float | None = None
    doa: str = _UNKNOWN
    angle: float | None = None
    keep: bool = False


@dataclass
class TrimResult:
    audio: np.ndarray | None          # None = no se pudo separar: se descarta
    final_sim: float | None = None    # lo conservado vs quien llamó
    summary: dict = field(default_factory=dict)   # para el log persistente
    # False = no había mezcla (toda la frase es de otro, o nada es claramente
    # de otro): trim() no logueó nada y el que llama loguea el descarte de
    # siempre, en una línea.
    mixed: bool = True

    @property
    def kept_s(self) -> float:
        return self.summary.get("kept_s", 0.0)

    @property
    def total_s(self) -> float:
        return self.summary.get("total_s", 0.0)


def _fmt_t(samples: int) -> str:
    return f"{samples / SAMPLE_RATE:.1f}"


def _deg(angle: float | None) -> str:
    return "?" if angle is None else f"{angle:.0f}°"


class MixTrimmer:
    """`trim()` corre en el hilo de STT (las huellas tardan); el contexto de
    dirección lo arma el de captura con `doa_by_segment()`."""

    def __init__(self, speaker) -> None:
        self._speaker = speaker   # speaker_id.SpeakerLock

    def trim(self, audio: np.ndarray, why: str, ctx: MixContext | None,
             full_text: str | None = None) -> TrimResult:
        """Intenta quedarse con lo de quien llamó. Si hubo mezcla (tramos de
        quien llamó Y de otro) loguea un bloque con el resultado, recortada o
        descartada. Si no (`mixed=False`), no loguea nada."""
        t_start = time.monotonic()
        segs = self._segments(audio)
        doa = ctx.doa if ctx is not None else None
        if doa is not None:
            self._judge_direction(segs, doa, ctx.focus)
        has_voice = self._judge_voice(segs, audio)
        self._decide(segs)

        removed = [s for s in segs if not s.quiet and not s.keep]
        kept = [s for s in segs if not s.quiet and s.keep]
        runs = self._runs(segs, len(audio))
        kept_samples = sum(b - a for a, b in runs)
        summary = {
            "why": why, "total_s": round(len(audio) / SAMPLE_RATE, 2),
            "kept_s": round(kept_samples / SAMPLE_RATE, 2),
            "voz": "".join(s.voice if not s.quiet else _QUIET for s in segs),
            "dir": "".join(s.doa if not s.quiet else _QUIET for s in segs) if doa else None,
            "uso": "".join("█" if s.keep else "─" for s in segs),
            "sims": [None if s.sim is None else round(s.sim, 2) for s in segs],
        }

        if (not has_voice and doa is None) or not removed or not kept:
            summary["result"] = ("sin datos por tramo" if not has_voice and doa is None
                                 else "ningún tramo es claramente de otro" if not removed
                                 else "ningún tramo es de quien llamó")
            summary["compute_s"] = round(time.monotonic() - t_start, 3)
            return TrimResult(None, None, summary, mixed=False)

        reason = None
        final_sim = None
        trimmed = None
        if kept_samples < MIX_TRIM_MIN_KEEP_S * SAMPLE_RATE:
            reason = (f"quedó muy poco de quien llamó ({kept_samples / SAMPLE_RATE:.1f} s, "
                      f"mínimo {MIX_TRIM_MIN_KEEP_S:g} s)")
        else:
            pieces = [audio[a:b] for a, b in runs]
            joined = np.concatenate(pieces)
            if has_voice and len(joined) >= SPEAKER_MIN_AUDIO_S * SAMPLE_RATE:
                final_sim = self._speaker.similarity(joined)
                if final_sim is not None and final_sim < SPEAKER_MIN_SIMILARITY:
                    reason = (f"lo que quedó tampoco se parece a quien llamó "
                              f"(sim {final_sim:.2f} < {SPEAKER_MIN_SIMILARITY:g})")
            if reason is None:
                gap = np.zeros(int(_GAP_S * SAMPLE_RATE), dtype=audio.dtype)
                parts: list[np.ndarray] = []
                for piece in pieces:
                    if parts:
                        parts.append(gap)
                    parts.append(piece)
                trimmed = np.concatenate(parts)
        elapsed = time.monotonic() - t_start
        summary.update(final_sim=None if final_sim is None else round(float(final_sim), 3),
                       result="recortada" if trimmed is not None else reason,
                       compute_s=round(elapsed, 3))

        lines = self._render(segs, doa is not None, has_voice, ctx, removed,
                             final_sim, full_text)
        if trimmed is not None:
            cut("MIX", f"frase mezclada ({why}): conservo {kept_samples / SAMPLE_RATE:.1f} de "
                f"{len(audio) / SAMPLE_RATE:.1f} s  [cómputo {elapsed:.2f} s]", lines)
        else:
            drop_block("MIX", f"frase mezclada ({why}): {reason}", lines,
                       computo_s=f"{elapsed:.2f}")
        return TrimResult(trimmed, final_sim, summary)

    # --- tramos ----------------------------------------------------------------

    @staticmethod
    def _segments(audio: np.ndarray) -> list[_Segment]:
        seg = _seg_samples()
        segs = [_Segment(a, min(a + seg, len(audio))) for a in range(0, len(audio), seg)]
        rms = np.array([float(np.sqrt(np.mean(np.square(audio[s.start:s.end]))))
                        for s in segs])
        floor = float(np.percentile(rms, 90)) * _QUIET_RATIO if len(rms) else 0.0
        for s, level in zip(segs, rms):
            s.quiet = level < floor
        return segs

    @staticmethod
    def _judge_direction(segs: list[_Segment], doa: list[DoaReading],
                         focus: float | None) -> None:
        for s, reading in zip(segs, doa):
            s.angle = reading.direction
            if reading.n < _MIN_DOA_SAMPLES:
                s.doa = _UNKNOWN
            elif reading.blocked >= 0.6:
                s.doa = _OTHER
            elif focus is None:
                s.doa = _UNKNOWN
            else:
                s.doa = _SAME if reading.in_focus >= DOA_MIN_IN_FOCUS else _OTHER

    def _judge_voice(self, segs: list[_Segment], audio: np.ndarray) -> bool:
        """Huella por ventanas. False = no hay referencia (no decide la voz)."""
        speaker = self._speaker
        if speaker is None or not speaker.enabled or speaker.ref_seconds() <= 0:
            return False
        n = len(audio)
        win = int(MIX_TRIM_WIN_S * SAMPLE_RATE)
        hop = _seg_samples()
        if n <= win:
            starts = [0]
            win = n
        else:
            starts = list(range(0, n - win + 1, hop))
            if starts[-1] + win < n:
                starts.append(n - win)
        if win < SPEAKER_MIN_AUDIO_S * SAMPLE_RATE:
            return False
        windows: list[tuple[int, int, float]] = []
        for a in starts:
            b = a + win
            # Ventana sólo de pausas: no vale la pena la huella.
            if all(s.quiet for s in segs if s.start < b and s.end > a):
                continue
            sim = speaker.similarity(audio[a:b])
            if sim is None:
                return False
            windows.append((a, b, sim))
        for s in segs:
            half = (s.end - s.start) / 2
            covering = [sim for a, b, sim in windows
                        if min(b, s.end) - max(a, s.start) >= half]
            if covering:
                s.sim = float(np.mean(covering))
                s.voice = _SAME if s.sim >= MIX_TRIM_MIN_SIM else _OTHER
        return True

    @staticmethod
    def _decide(segs: list[_Segment]) -> None:
        for s in segs:
            if not s.quiet:
                s.keep = s.voice != _OTHER and s.doa != _OTHER
        # Pausas: sólo entre dos tramos conservados (no cortar la frase de
        # quien llamó en pedazos por respirar).
        loud = [i for i, s in enumerate(segs) if not s.quiet]
        for i, s in enumerate(segs):
            if not s.quiet:
                continue
            before = [j for j in loud if j < i]
            after = [j for j in loud if j > i]
            s.keep = bool(before and after and segs[before[-1]].keep and segs[after[0]].keep)

    @staticmethod
    def _runs(segs: list[_Segment], n: int) -> list[tuple[int, int]]:
        """Rangos [a, b) conservados, con margen, fusionados si se tocan."""
        pad = int(MIX_TRIM_PAD_S * SAMPLE_RATE)
        runs: list[list[int]] = []
        for s in segs:
            if not s.keep:
                continue
            if runs and runs[-1][1] >= s.start:
                runs[-1][1] = s.end
            else:
                runs.append([s.start, s.end])
        padded: list[list[int]] = []
        for a, b in runs:
            a, b = max(0, a - pad), min(n, b + pad)
            if padded and padded[-1][1] >= a:
                padded[-1][1] = b
            else:
                padded.append([a, b])
        return [(a, b) for a, b in padded]

    # --- log -------------------------------------------------------------------

    @staticmethod
    def _render(segs: list[_Segment], has_doa: bool, has_voice: bool,
                ctx: MixContext | None, removed: list[_Segment],
                final_sim: float | None, full_text: str | None) -> list[str]:
        lines = []
        if has_voice:
            row = "".join(_QUIET if s.quiet else s.voice for s in segs)
            lines.append(f"voz  {row}   █ quien llamó  ▒ otra  ? sin dato  · pausa  "
                         f"({MIX_TRIM_SEG_S:g} s/casilla)")
        else:
            lines.append("voz  (sin huella de referencia todavía: decide sólo la dirección)")
        if has_doa:
            row = "".join(_QUIET if s.quiet else s.doa for s in segs)
            focus = _deg(ctx.focus) if ctx is not None else "?"
            lines.append(f"dir  {row}   █ en foco {focus}  ▒ fuera")
        lines.append("uso  " + "".join("█" if s.keep else "─" for s in segs))

        # Qué se sacó, agrupado en pedazos contiguos (sin contar pausas).
        groups: list[list[_Segment]] = []
        for s in removed:
            if groups and groups[-1][-1].end == s.start:
                groups[-1].append(s)
            else:
                groups.append([s])
        described = []
        for g in groups:
            why = []
            if any(s.voice == _OTHER for s in g):
                sims = [s.sim for s in g if s.sim is not None]
                lo, hi = min(sims), max(sims)
                why.append(f"otra voz (sim {lo:.2f}–{hi:.2f})" if hi - lo >= 0.005
                           else f"otra voz (sim {lo:.2f})")
            if any(s.doa == _OTHER for s in g):
                angles = [s.angle for s in g if s.doa == _OTHER and s.angle is not None]
                why.append("otra dirección" + (f" ({_deg(float(np.median(angles)))})"
                                               if angles else ""))
            described.append(f"{_fmt_t(g[0].start)}–{_fmt_t(g[-1].end)} s: {' y '.join(why)}")
        if described:
            lines.append("saqué " + " · ".join(described))
        if final_sim is not None:
            lines.append(f"lo que quedó vs quien llamó: sim {final_sim:.2f} "
                         f"(mínimo {SPEAKER_MIN_SIMILARITY:g})")
        if full_text:
            lines.append(f"la frase entera decía: «{full_text}»")
        return lines
