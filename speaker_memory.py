"""Memoria de voces: el robot aprende solo quién le habla seguido.

Sin enrolamiento. Cada conversación (de «oye rai» a dormirse) ya viene
etiquetada gratis: durante la ventana despierta speaker_id.py sólo acepta la
voz que lo llamó, así que todo el audio aceptado de una sesión es de UNA
persona. Al cerrar la sesión:

1. Huella de la sesión entera (si juntó >= SPEAKER_LEARN_MIN_S de voz bien
   verificada; una sesión corta es ruidosa y no enseña nada).
2. Se compara con las voces guardadas:
   - sim >= SPEAKER_MERGE_SIM con una voz -> es esa persona: su perfil se
     refuerza (promedio pesado por segundos de voz, con tope para que siga
     adaptándose).
   - sim <  SPEAKER_NEW_SIM con TODAS -> alguien nuevo: `voz_N`.
   - en el medio -> duda: no se aprende (mejor no aprender que mezclar dos
     personas en un perfil, que es el error caro). La sesión queda pendiente:
     si se juntan SPEAKER_NEW_CONFIRM sesiones en duda parecidas ENTRE SÍ
     (>= SPEAKER_MERGE_SIM), son otra persona de voz parecida a una conocida
     y se crea su voz con ellas. Sin esto, alguien así no se aprendería nunca.
3. Cada sesión se anota además en un historial (JSONL, sólo huellas, nunca
   audio) para poder rearmar los grupos si algo sale mal.

Al despertar (`recognize`), el «oye rai» se compara con las voces guardadas:
si coincide con alguien (>= SPEAKER_RECOGNIZE_SIM), speaker_id.py arranca la
conversación con su perfil y no sólo con ~1 s de audio.

Las huellas de mics distintos (normal / array) o de modelos distintos no son
comparables: cada combinación tiene su propia carpeta. Voces vistas en una
sola sesión y no vueltas a ver en SPEAKER_FORGET_DAYS días se borran solas.

Archivos (en SPEAKER_DB_DIR/<mic>_<modelo>/):
  voices.json     perfiles (id, huella, sesiones, segundos, fechas, nombre)
  sessions.jsonl  una línea por sesión: huella, segundos, decisión

Todo es una huella de 192 números por voz/sesión (~1-2 KB). Para borrar todo:
`python speaker_id.py --forget`.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np

from config import (
    MIC_MODE,
    SPEAKER_DB_DIR,
    SPEAKER_FORGET_DAYS,
    SPEAKER_MERGE_SIM,
    SPEAKER_MODEL_NAME,
    SPEAKER_NEW_CONFIRM,
    SPEAKER_NEW_SIM,
    SPEAKER_PROFILE_WEIGHT_S,
    SPEAKER_RECOGNIZE_SIM,
)
from log import dim, info, ok, warn


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def _round(vec: np.ndarray) -> list[float]:
    return [round(float(x), 5) for x in vec]


@dataclass
class Voice:
    id: str
    centroid: list[float]
    sessions: int
    seconds: float
    created: float
    last_seen: float
    name: str = ""

    @property
    def label(self) -> str:
        return f"{self.id} ({self.name})" if self.name else self.id

    def vector(self) -> np.ndarray:
        return np.asarray(self.centroid, dtype=np.float32)


@dataclass
class Learned:
    """Qué se hizo con una sesión (para el log)."""
    decision: str           # "refuerza" | "nueva" | "nueva_de_dudas" | "duda"
    voice: Voice | None
    similarity: float | None


class VoiceMemory:
    """Thread-safe: `recognize` desde el hilo de STT, `learn` desde el hilo
    que cierra sesiones (speaker_id.SpeakerLock)."""

    def __init__(self, directory: str | None = None) -> None:
        model = os.path.splitext(SPEAKER_MODEL_NAME)[0]
        self.dir = directory or os.path.join(SPEAKER_DB_DIR, f"{MIC_MODE}_{model}")
        self._voices_path = os.path.join(self.dir, "voices.json")
        self._sessions_path = os.path.join(self.dir, "sessions.jsonl")
        self._mutex = threading.Lock()
        self._voices: list[Voice] = []
        # Sesiones en duda todavía sin voz: {"emb": [...], "seconds": s, "t": ts}
        self._pending: list[dict] = []
        self._next_id = 1
        self._load()
        self._prune()

    # -- persistencia -------------------------------------------------------

    def _load(self) -> None:
        try:
            with open(self._voices_path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return
        except (OSError, ValueError) as exc:
            warn("SPK", f"no pude leer {self._voices_path} ({exc}): arranco sin voces")
            return
        self._voices = [Voice(**v) for v in data.get("voices", [])]
        self._pending = list(data.get("pending", []))
        self._next_id = int(data.get("next_id", len(self._voices) + 1))

    def _save(self) -> None:
        """Escritura atómica (tmp + rename): un corte de luz no deja el
        archivo a medias."""
        os.makedirs(self.dir, exist_ok=True)
        tmp = self._voices_path + ".tmp"
        data = {"version": 1, "mic_mode": MIC_MODE, "model": SPEAKER_MODEL_NAME,
                "next_id": self._next_id, "voices": [asdict(v) for v in self._voices],
                "pending": self._pending}
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self._voices_path)

    def _append_session(self, record: dict) -> None:
        os.makedirs(self.dir, exist_ok=True)
        with open(self._sessions_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _prune(self) -> None:
        """Voces de una sola sesión no vueltas a ver en SPEAKER_FORGET_DAYS."""
        if SPEAKER_FORGET_DAYS <= 0:
            return
        limit = time.time() - SPEAKER_FORGET_DAYS * 86400
        with self._mutex:
            stale = [v for v in self._voices if v.sessions <= 1 and v.last_seen < limit]
            pending = [p for p in self._pending if p["t"] >= limit]
            if not stale and len(pending) == len(self._pending):
                return
            self._voices = [v for v in self._voices if v not in stale]
            self._pending = pending
            self._save()
            if not stale:
                return
        dim("SPK", f"olvido {len(stale)} voz/voces de una sola sesión "
            f"(>{SPEAKER_FORGET_DAYS:g} días): {', '.join(v.id for v in stale)}")

    # -- consultas ----------------------------------------------------------

    def voices(self) -> list[Voice]:
        with self._mutex:
            return list(self._voices)

    def _best(self, emb: np.ndarray) -> tuple[Voice | None, float]:
        best, best_sim = None, -1.0
        for voice in self._voices:
            sim = float(voice.vector() @ emb)
            if sim > best_sim:
                best, best_sim = voice, sim
        return best, best_sim

    def recognize(self, emb: np.ndarray) -> tuple[Voice | None, float]:
        """¿De quién es esta huella? (voz, sim) si llega a
        SPEAKER_RECOGNIZE_SIM; (None, mejor_sim) si no."""
        with self._mutex:
            voice, sim = self._best(emb)
        if voice is not None and sim >= SPEAKER_RECOGNIZE_SIM:
            return voice, sim
        return None, sim

    # -- aprendizaje --------------------------------------------------------

    def _new_voice(self, emb: np.ndarray, seconds: float, sessions: int, now: float) -> Voice:
        voice = Voice(id=f"voz_{self._next_id}", centroid=_round(_unit(emb)),
                      sessions=sessions, seconds=seconds, created=now, last_seen=now)
        self._next_id += 1
        self._voices.append(voice)
        return voice

    def _from_pending(self, emb: np.ndarray, seconds: float, now: float) -> Voice | None:
        """Sesión en duda: ¿ya van SPEAKER_NEW_CONFIRM parecidas entre sí?
        Entonces es otra persona (de voz parecida a una conocida): voz nueva
        con todas. Si no, queda pendiente."""
        alike = [p for p in self._pending
                 if float(np.asarray(p["emb"], dtype=np.float32) @ emb) >= SPEAKER_MERGE_SIM]
        if len(alike) + 1 < SPEAKER_NEW_CONFIRM:
            self._pending.append({"emb": _round(emb), "seconds": round(seconds, 1),
                                  "t": round(now, 1)})
            self._pending = self._pending[-100:]
            return None
        total = seconds + sum(p["seconds"] for p in alike)
        mixed = emb * seconds + sum(np.asarray(p["emb"], dtype=np.float32) * p["seconds"]
                                    for p in alike)
        self._pending = [p for p in self._pending if p not in alike]
        return self._new_voice(mixed, total, len(alike) + 1, now)

    def learn(self, emb: np.ndarray, seconds: float, recognized: str | None) -> Learned:
        """Suma una sesión cerrada (huella de toda la voz aceptada)."""
        now = time.time()
        with self._mutex:
            best, sim = self._best(emb)
            if best is not None and sim >= SPEAKER_MERGE_SIM:
                # Promedio pesado por segundos; el perfil pesa como mucho
                # SPEAKER_PROFILE_WEIGHT_S para seguir adaptándose (otro mic,
                # resfrío, el robot caminando).
                weight = min(best.seconds, SPEAKER_PROFILE_WEIGHT_S)
                best.centroid = _round(_unit(best.vector() * weight + emb * seconds))
                best.sessions += 1
                best.seconds += seconds
                best.last_seen = now
                result = Learned("refuerza", best, sim)
            elif best is None or sim < SPEAKER_NEW_SIM:
                result = Learned("nueva", self._new_voice(emb, seconds, 1, now),
                                 sim if best is not None else None)
            else:
                voice = self._from_pending(emb, seconds, now)
                result = (Learned("nueva_de_dudas", voice, sim) if voice is not None
                          else Learned("duda", best, sim))
            self._save()
            self._append_session({
                "t": round(now, 1), "seconds": round(seconds, 1),
                "decision": result.decision,
                "voice": result.voice.id if result.voice and result.decision != "duda" else None,
                "nearest": best.id if best else None,
                "sim": None if best is None else round(sim, 3),
                "recognized": recognized, "emb": _round(emb)})
        return result

    @staticmethod
    def report(result: Learned, seconds: float, recognized: str | None) -> None:
        sim = "" if result.similarity is None else f" sim={result.similarity:.2f}"
        if result.decision == "refuerza":
            ok("SPK", f"aprendí: sesión de {seconds:.0f}s = {result.voice.label}{sim} "
               f"({result.voice.sessions} sesiones, {result.voice.seconds:.0f}s de voz)")
        elif result.decision == "nueva":
            ok("SPK", f"aprendí: voz nueva {result.voice.id} ({seconds:.0f}s){sim}")
        elif result.decision == "nueva_de_dudas":
            ok("SPK", f"aprendí: voz nueva {result.voice.id} juntando "
               f"{result.voice.sessions} sesiones en duda parecidas entre sí "
               f"({result.voice.seconds:.0f}s; la conocida más parecida{sim})")
        else:
            info("SPK", f"sesión de {seconds:.0f}s en duda (la más parecida "
                 f"{result.voice.label}{sim}, entre {SPEAKER_NEW_SIM:g} y "
                 f"{SPEAKER_MERGE_SIM:g}): queda pendiente")
        if (recognized and result.voice is not None and result.decision == "refuerza"
                and result.voice.id != recognized):
            warn("SPK", f"al despertar la reconocí como {recognized} pero la sesión "
                 f"entera se parece más a {result.voice.id}")

    # -- mantenimiento (CLI) -------------------------------------------------

    def forget(self, voice_id: str | None = None) -> int:
        """Borra una voz, o todo (perfiles + historial) con None."""
        with self._mutex:
            if voice_id is None:
                n = len(self._voices)
                self._voices, self._pending, self._next_id = [], [], 1
                for path in (self._voices_path, self._sessions_path):
                    if os.path.exists(path):
                        os.remove(path)
                return n
            before = len(self._voices)
            self._voices = [v for v in self._voices if v.id != voice_id]
            if len(self._voices) != before:
                self._save()
            return before - len(self._voices)

    def rename(self, voice_id: str, name: str) -> bool:
        with self._mutex:
            for voice in self._voices:
                if voice.id == voice_id:
                    voice.name = name
                    self._save()
                    return True
        return False

    def sessions(self) -> list[dict]:
        try:
            with open(self._sessions_path, encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            return []

    def print_summary(self) -> None:
        """Voces, similitud entre ellas y decisiones del historial: para
        juzgar si está aprendiendo bien (una persona = una voz)."""
        voices = self.voices()
        sessions = self.sessions()
        with self._mutex:
            pending = len(self._pending)
        print(f"Memoria de voces: {self.dir}")
        if not voices:
            print("  (sin voces todavía)")
        for v in sorted(voices, key=lambda v: -v.seconds):
            print(f"  {v.label:22s} {v.sessions:4d} sesiones  {v.seconds:6.0f}s de voz  "
                  f"última {time.strftime('%Y-%m-%d %H:%M', time.localtime(v.last_seen))}")
        if len(voices) > 1:
            # Dos voces muy parecidas entre sí (> SPEAKER_NEW_SIM) suelen ser
            # la misma persona partida en dos.
            print("\n  Pares de voces más parecidos (alto = posible misma persona):")
            pairs = [(float(a.vector() @ b.vector()), a.id, b.id)
                     for i, a in enumerate(voices) for b in voices[i + 1:]]
            for sim, a, b in sorted(pairs, reverse=True)[:8]:
                print(f"    {a} ~ {b}: {sim:.2f}")
        if sessions:
            counts: dict[str, int] = {}
            for s in sessions:
                counts[s["decision"]] = counts.get(s["decision"], 0) + 1
            print(f"\n  Historial: {len(sessions)} sesiones -> "
                  + ", ".join(f"{k}={n}" for k, n in sorted(counts.items()))
                  + f"; {pending} en duda pendientes")
            print("  Últimas:")
            for s in sessions[-10:]:
                when = time.strftime("%m-%d %H:%M", time.localtime(s["t"]))
                sim = "" if s.get("sim") is None else f" sim={s['sim']:.2f}"
                print(f"    {when} {s['seconds']:5.1f}s {s['decision']:8s} "
                      f"-> {s.get('voice') or s.get('nearest') or '-'}{sim}"
                      + (f" (al despertar: {s['recognized']})" if s.get("recognized") else ""))
