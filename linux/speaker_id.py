"""Verificación de hablante: ¿esta frase la dijo quien me llamó?

El filtro de cercanía de vad.py es de NIVEL y tiene un techo: alguien que
habla bajo cerca del robot y alguien que habla fuerte de lejos llegan con el
mismo RMS. Subir el umbral corta a quien le habla; bajarlo deja pasar el
fondo. Con el ReSpeaker eso lo resuelve la dirección (doa.py); con el mic
común, esto: una huella de la VOZ, que no depende del volumen.

Cómo funciona:

1. Al despertar («oye rai», spotter de audio) main.py le pasa a `lock()` el
   audio de esa frase: es la referencia de la voz de quien llamó.
2. Mientras dure la ventana, cada frase cerrada por el VAD se compara contra
   la referencia (`judge()`, en el hilo de STT, en paralelo con Groq): un
   embedding de voz (un vector de 192 números) por frase y similitud coseno.
   Por debajo de SPEAKER_MIN_SIMILARITY se descarta: es otra voz.
3. Cada frase aceptada se suma a la referencia (`follow()`): el «oye rai»
   solo es ~1 s de voz y la huella sale ruidosa; con más audio mejora.
4. Al dormirse se olvida (`clear()`). Otro «oye rai» que pase el filtro de
   main.py (otra persona toma el foco) fija una referencia nueva.

Modo `text` (sin spotter): la referencia es la frase que dijo «rai».

El embedding lo saca un modelo ONNX (TitaNet-small de NVIDIA NeMo, ~40 MB)
vía sherpa-onnx, local en la Pi. Tarda ~70 ms por segundo de audio en un
núcleo de x86 (en la Pi 5, a medir): corre EN PARALELO con Groq, así que no
suma latencia. Si falta sherpa-onnx o el modelo, queda desactivado con un
aviso y todo sigue como antes.

Se probaron cuatro modelos de sherpa-onnx en LibriSpeech (10 personas): con
CAM++ (3D-Speaker y WeSpeaker) y ResNet34 las voces ajenas daban similitudes
tan altas como la propia con referencias cortas; TitaNet-small separa bien
(ajena: mediana 0.07, p95 0.27; propia con referencia de 1.5 s: p5 0.45).

Calibración: `python speaker_id.py` (ver main() abajo).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

import numpy as np

from config import (
    SAMPLE_RATE,
    SPEAKER_MIN_AUDIO_S,
    SPEAKER_MIN_REF_S,
    SPEAKER_MIN_SIMILARITY,
    SPEAKER_MODEL_PATH,
    SPEAKER_REF_MAX_S,
    SPEAKER_VERIFY_ENABLED,
)
from log import dim, drop, ok, warn


class SpeakerEmbedder:
    """Audio float32 mono 16 kHz -> embedding normalizado (norma 1)."""

    def __init__(self, model_path: str) -> None:
        import sherpa_onnx

        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model_path, num_threads=1, provider="cpu")
        if not config.validate():
            raise ValueError(f"configuración inválida para {model_path}")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.dim = self._extractor.dim

    def embed(self, audio: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(audio, dtype=np.float32))
        stream.input_finished()
        vec = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


def load_embedder() -> SpeakerEmbedder | None:
    """El embedder, o None (con un aviso) si está desactivado o no carga."""
    if not SPEAKER_VERIFY_ENABLED:
        dim("SPK", "verificación de hablante desactivada (SPEAKER_VERIFY_ENABLED=0)")
        return None
    try:
        if not os.path.isfile(SPEAKER_MODEL_PATH):
            raise FileNotFoundError(f"falta {SPEAKER_MODEL_PATH} (ver README)")
        return SpeakerEmbedder(SPEAKER_MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - ImportError, modelo ausente...
        warn("SPK", f"no pude cargar la verificación de hablante ({type(exc).__name__}: "
             f"{exc}); sigo sin ella")
        return None


@dataclass
class Verdict:
    accepted: bool
    similarity: float | None   # None = no se comparó (sin referencia, frase corta)
    why: str


class SpeakerLock:
    """Referencia de voz de quien despertó al robot. Thread-safe: `lock()` y
    `clear()` vienen del hilo de audio / de eventos; `judge()` y `follow()`
    del hilo de STT (el embedding tarda, no frena la captura)."""

    def __init__(self, embedder: SpeakerEmbedder | None) -> None:
        self._embedder = embedder
        self._mutex = threading.Lock()
        self._ref_audio: list[np.ndarray] = []  # [0] = la frase del wake
        self._ref: np.ndarray | None = None     # embedding de _ref_audio (cache)
        self._gen = 0                           # cambia en cada lock()/clear()

    @property
    def enabled(self) -> bool:
        return self._embedder is not None

    def lock(self, audio: np.ndarray | None) -> None:
        """Wake nuevo: la referencia pasa a ser `audio` (la frase del «oye
        rai»). None o muy corto: la toma la primera frase aceptada."""
        if not self.enabled:
            return
        with self._mutex:
            self._gen += 1
            self._ref = None
            seconds = 0.0 if audio is None else len(audio) / SAMPLE_RATE
            if seconds >= SPEAKER_MIN_REF_S:
                self._ref_audio = [audio]
                dim("SPK", f"referencia de voz: la frase del wake ({seconds:.1f}s)")
            else:
                self._ref_audio = []
                dim("SPK", f"«oye rai» muy corto para huella ({seconds:.1f}s < "
                    f"{SPEAKER_MIN_REF_S:g}s): la toma la primera frase")

    def clear(self) -> None:
        with self._mutex:
            self._gen += 1
            self._ref_audio = []
            self._ref = None

    def ref_seconds(self) -> float:
        """Segundos de audio en la referencia (0 = sin referencia)."""
        with self._mutex:
            return sum(len(a) for a in self._ref_audio) / SAMPLE_RATE

    def _reference(self) -> np.ndarray | None:
        """Embedding de toda la referencia (se recalcula si cambió)."""
        with self._mutex:
            if self._ref is not None or not self._ref_audio:
                return self._ref
            gen, audio = self._gen, np.concatenate(self._ref_audio)
        ref = self._embedder.embed(audio)
        with self._mutex:
            if gen == self._gen:
                self._ref = ref
        return ref

    def judge(self, audio: np.ndarray) -> Verdict:
        """¿Es la misma voz que la referencia? Sin referencia o con una frase
        muy corta para sacar huella confiable, se acepta (decide el nivel)."""
        if not self.enabled:
            return Verdict(True, None, "desactivado")
        seconds = len(audio) / SAMPLE_RATE
        if seconds < SPEAKER_MIN_AUDIO_S:
            return Verdict(True, None, f"frase corta ({seconds:.1f}s), no comparo")
        ref = self._reference()
        if ref is None:
            return Verdict(True, None, "sin referencia todavía")
        similarity = float(ref @ self._embedder.embed(audio))
        if similarity >= SPEAKER_MIN_SIMILARITY:
            return Verdict(True, similarity, "misma voz")
        return Verdict(False, similarity, "otra voz")

    def follow(self, audio: np.ndarray) -> None:
        """Frase aceptada de quien me llamó: suma a la referencia. Se queda
        siempre con la frase del wake (la más confiable) y las más nuevas
        hasta SPEAKER_REF_MAX_S."""
        if not self.enabled or len(audio) / SAMPLE_RATE < SPEAKER_MIN_AUDIO_S:
            return
        with self._mutex:
            first = not self._ref_audio
            self._ref_audio.append(audio)
            max_samples = int(SPEAKER_REF_MAX_S * SAMPLE_RATE)
            while (len(self._ref_audio) > 2
                   and sum(len(a) for a in self._ref_audio) > max_samples):
                del self._ref_audio[1]
            self._ref = None
            total = sum(len(a) for a in self._ref_audio) / SAMPLE_RATE
        if first:
            ok("SPK", f"referencia de voz fijada con esta frase ({total:.1f}s)")
        else:
            dim("SPK", f"referencia de voz: {total:.1f}s")

    @staticmethod
    def report(verdict: Verdict, level: float) -> bool:
        """Loguea el veredicto de judge(). True = la frase sigue."""
        if not verdict.accepted:
            drop("SPK", verdict.why, sim=verdict.similarity,
                 minimo=SPEAKER_MIN_SIMILARITY, nivel=level)
            return False
        if verdict.similarity is not None:
            ok("SPK", f"{verdict.why} sim={verdict.similarity:.2f} "
               f"(mínimo {SPEAKER_MIN_SIMILARITY:g})")
        else:
            dim("SPK", verdict.why)
        return True


def main() -> None:
    """Calibración de SPEAKER_MIN_SIMILARITY en vivo, con el mismo camino de
    audio y VAD que main.py (sin wake word ni Groq):

        python speaker_id.py              # mic
        python speaker_id.py a.wav b.wav  # compara archivos (16 kHz mono)

    La PRIMERA frase que cierre el VAD es la referencia: decí «oye rai» como
    al despertarlo. Después, para cada frase imprime la similitud con la
    referencia. Hablá vos varias veces (y bajito) y que hable otra persona:
    SPEAKER_MIN_SIMILARITY va entre el mínimo tuyo y el máximo ajeno. Enter
    vacío no hace falta: Ctrl+C termina e imprime el resumen.
    """
    import sys

    embedder = load_embedder()
    if embedder is None:
        sys.exit(1)

    if len(sys.argv) > 1:
        import wave
        embs = []
        for path in sys.argv[1:]:
            with wave.open(path) as w:
                if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1:
                    sys.exit(f"{path}: tiene que ser mono {SAMPLE_RATE} Hz")
                pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            embs.append(embedder.embed(pcm.astype(np.float32) / 32768.0))
        for i, path in enumerate(sys.argv[2:], start=1):
            print(f"{sys.argv[1]} vs {path}: sim={float(embs[0] @ embs[i]):.3f}")
        return

    from audio_capture import LinuxAudioCapture
    from vad import VoiceActivityDetector

    vad = VoiceActivityDetector()
    capture = LinuxAudioCapture(device_id=None)
    ref: np.ndarray | None = None
    sims: list[float] = []
    print(f"\nDecí «oye rai» (referencia). Umbral actual SPEAKER_MIN_SIMILARITY="
          f"{SPEAKER_MIN_SIMILARITY:g}. Ctrl+C para terminar.\n")
    try:
        for af in capture.frames():
            closed, audio = vad.process_frame(af.pcm, af.rms, af.t)
            if not closed or audio is None:
                continue
            seconds = len(audio) / SAMPLE_RATE
            if ref is None:
                ref = embedder.embed(audio)
                print(f">>> referencia fijada ({seconds:.1f}s). Ahora hablá vos / otra persona.\n")
                continue
            sim = float(ref @ embedder.embed(audio))
            sims.append(sim)
            mark = "PASA " if sim >= SPEAKER_MIN_SIMILARITY else "corta"
            short = "  (corta: en main.py no se compara)" if seconds < SPEAKER_MIN_AUDIO_S else ""
            print(f"{mark} sim={sim:.3f}  {seconds:.1f}s  nivel={vad.last_level:.4f}{short}")
    except KeyboardInterrupt:
        pass
    if sims:
        print(f"\n{len(sims)} frases: min={min(sims):.3f} max={max(sims):.3f} "
              f"mediana={float(np.median(sims)):.3f}")


if __name__ == "__main__":
    main()
