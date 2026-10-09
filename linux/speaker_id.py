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
5. Memoria de voces (speaker_memory.py, SPEAKER_MEMORY): antes de olvidar la
   sesión, su audio verificado se aprende y queda guardado en la Pi. Al
   despertar, si el «oye rai» es de una voz conocida, la referencia arranca
   con su perfil y no sólo con ~1 s de audio.

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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from config import (
    SAMPLE_RATE,
    SPEAKER_LEARN_MIN_S,
    SPEAKER_LEARN_MIN_SIM,
    SPEAKER_MEMORY,
    SPEAKER_MIN_AUDIO_S,
    SPEAKER_MIN_REF_S,
    SPEAKER_MIN_SIMILARITY,
    SPEAKER_MODEL_PATH,
    SPEAKER_PROFILE_WEIGHT_S,
    SPEAKER_RECOGNIZE_SIM,
    SPEAKER_REF_MAX_S,
    SPEAKER_SESSION_MAX_S,
    SPEAKER_VERIFY_ENABLED,
)
from log import dim, drop, event, info, ok, warn
from speaker_memory import Voice, VoiceMemory


class SpeakerEmbedder:
    """Audio float32 mono 16 kHz -> embedding normalizado (norma 1).
    Thread-safe (lo usan el hilo de STT y el que aprende sesiones)."""

    def __init__(self, model_path: str) -> None:
        import sherpa_onnx

        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=model_path, num_threads=1, provider="cpu")
        if not config.validate():
            raise ValueError(f"configuración inválida para {model_path}")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self._mutex = threading.Lock()
        self.dim = self._extractor.dim

    def embed(self, audio: np.ndarray) -> np.ndarray:
        with self._mutex:
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


def load_memory(embedder: SpeakerEmbedder | None) -> VoiceMemory | None:
    """Memoria de voces (speaker_memory.py) según SPEAKER_MEMORY, o None."""
    if embedder is None or SPEAKER_MEMORY == "off":
        return None
    if SPEAKER_MEMORY not in ("on", "observe"):
        warn("CONFIG", f"SPEAKER_MEMORY={SPEAKER_MEMORY!r} desconocido (on|observe|off): uso off")
        return None
    try:
        memory = VoiceMemory()
    except Exception as exc:  # noqa: BLE001 - permisos, disco...
        warn("SPK", f"no pude abrir la memoria de voces ({type(exc).__name__}: {exc}); sigo sin ella")
        return None
    names = ", ".join(v.label for v in memory.voices()) or "ninguna todavía"
    info("SPK", f"memoria de voces ({SPEAKER_MEMORY}): {names}")
    return memory


@dataclass
class Verdict:
    accepted: bool
    similarity: float | None   # None = no se comparó (sin referencia, frase corta)
    why: str


def _seconds(chunks: list[np.ndarray]) -> float:
    return sum(len(a) for a in chunks) / SAMPLE_RATE


class SpeakerLock:
    """Referencia de voz de quien despertó al robot (una "sesión": de «oye
    rai» a dormirse). Thread-safe: `lock()` y `clear()` vienen del hilo de
    audio / de eventos; `judge()` y `follow()` del hilo de STT (el embedding
    tarda, no frena la captura). Aprender la sesión cerrada corre en un hilo
    propio.

    Con memoria de voces (SPEAKER_MEMORY=on): si el «oye rai» es de una voz
    conocida, una frase se acepta si se parece a la sesión O a la sesión
    mezclada con el perfil guardado. El perfil sólo puede sumar aceptaciones
    tuyas, no quitarlas."""

    def __init__(self, embedder: SpeakerEmbedder | None,
                 memory: VoiceMemory | None = None) -> None:
        self._embedder = embedder
        self._memory = memory
        self._learner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="spk-learn")
        self._mutex = threading.Lock()
        # Un solo cálculo de referencia a la vez (el de segundo plano tras el
        # wake y el de la primera frase): el segundo usa el resultado cacheado.
        self._computing = threading.Lock()
        self._gen = 0                            # cambia en cada lock()/clear()
        self._new_session([])

    def _new_session(self, ref_audio: list[np.ndarray]) -> None:
        # Llamar con _mutex tomado (o desde __init__).
        self._gen += 1
        self._ref_audio = ref_audio              # [0] = la frase del wake
        self._learn_audio = list(ref_audio)      # sólo audio verificado
        self._ref: np.ndarray | None = None      # huella de _ref_audio (cache)
        self._voice: Voice | None = None         # voz conocida (memoria)
        self._recognized = False                 # ya se buscó en la memoria

    @property
    def enabled(self) -> bool:
        return self._embedder is not None

    def lock(self, audio: np.ndarray | None) -> None:
        """Wake nuevo: cierra la sesión anterior (y la aprende) y la
        referencia pasa a ser `audio` (la frase del «oye rai»). None o muy
        corto: la toma la primera frase aceptada."""
        if not self.enabled:
            return
        seconds = 0.0 if audio is None else len(audio) / SAMPLE_RATE
        with self._mutex:
            self._finish_session()
            self._new_session([audio] if seconds >= SPEAKER_MIN_REF_S else [])
        if seconds >= SPEAKER_MIN_REF_S:
            dim("SPK", f"referencia de voz: la frase del wake ({seconds:.1f}s)")
            # Huella + reconocimiento ya, en segundo plano: la primera frase
            # no espera y el «te reconozco» sale enseguida en el log.
            self._learner.submit(self._reference)
        else:
            dim("SPK", f"«oye rai» muy corto para huella ({seconds:.1f}s < "
                f"{SPEAKER_MIN_REF_S:g}s): la toma la primera frase")

    def clear(self) -> None:
        """Se durmió: aprende la sesión y la olvida."""
        with self._mutex:
            self._finish_session()
            self._new_session([])

    def close(self) -> None:
        """Al salir: aprende la sesión abierta y espera a que se guarde."""
        self.clear()
        self._learner.shutdown(wait=True)

    def _finish_session(self) -> None:
        # Llamar con _mutex tomado.
        if self._memory is None or not self._learn_audio:
            return
        audio = list(self._learn_audio)
        seconds = _seconds(audio)
        recognized = self._voice.id if self._voice else None
        if seconds < SPEAKER_LEARN_MIN_S:
            event("spk_learn", decision="corta", seconds=round(seconds, 1),
                  recognized=recognized)
            dim("SPK", f"sesión de {seconds:.1f}s de voz verificada: corta para aprender "
                f"(mínimo {SPEAKER_LEARN_MIN_S:g}s)")
            return
        self._learner.submit(self._learn, audio, seconds, recognized)

    def _learn(self, audio: list[np.ndarray], seconds: float, recognized: str | None) -> None:
        try:
            emb = self._embedder.embed(np.concatenate(audio))
            result = self._memory.learn(emb, seconds, recognized)
            event("spk_learn", decision=result.decision, seconds=round(seconds, 1),
                  voice=result.voice.id if result.voice else None,
                  sim=None if result.similarity is None else round(result.similarity, 3),
                  recognized=recognized)
            self._memory.report(result, seconds, recognized)
        except Exception as exc:  # noqa: BLE001 - disco lleno, permisos...
            warn("SPK", f"no pude aprender la sesión ({type(exc).__name__}: {exc})")

    def ref_seconds(self) -> float:
        """Segundos de audio en la referencia (0 = sin referencia)."""
        with self._mutex:
            return _seconds(self._ref_audio)

    def known_voices(self) -> list[Voice]:
        return self._memory.voices() if self._memory is not None else []

    def voice_label(self) -> str | None:
        """Voz conocida de esta sesión (None = desconocida / sin memoria)."""
        with self._mutex:
            return self._voice.label if self._voice else None

    def _reference(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """(huella de la sesión, huella de sesión + perfil guardado o None).
        Se recalcula si cambió; la primera vez busca la voz en la memoria."""
        with self._computing:
            return self._compute_reference()

    def _compute_reference(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._mutex:
            if self._ref is not None or not self._ref_audio:
                return self._ref, self._blend()
            gen, audio = self._gen, np.concatenate(self._ref_audio)
            look_up = self._memory is not None and not self._recognized
        ref = self._embedder.embed(audio)
        voice, sim = self._memory.recognize(ref) if look_up else (None, 0.0)
        with self._mutex:
            if gen != self._gen:  # otra sesión empezó mientras calculaba
                return ref, None
            self._ref = ref
            if look_up:
                self._recognized = True
                self._voice = voice
            blend = self._blend()
        if look_up:
            event("spk_recognize", voice=voice.id if voice else None,
                  best_sim=round(sim, 3) if self._memory.voices() else None,
                  known=len(self._memory.voices()),
                  ref_s=round(len(audio) / SAMPLE_RATE, 2))
            if voice is not None:
                use = ("la uso de referencia" if SPEAKER_MEMORY == "on"
                       else "modo observe: no la uso")
                ok("SPK", f"te reconozco: {voice.label} sim={sim:.2f} "
                   f"({voice.sessions} sesiones); {use}")
            elif self._memory.voices():
                dim("SPK", f"voz desconocida (la más parecida sim={sim:.2f} < "
                    f"{SPEAKER_RECOGNIZE_SIM:g})")
        return ref, blend

    def _blend(self) -> np.ndarray | None:
        # Llamar con _mutex tomado. Sesión + perfil, pesados por segundos.
        if self._voice is None or self._ref is None or SPEAKER_MEMORY != "on":
            return None
        profile = min(self._voice.seconds, SPEAKER_PROFILE_WEIGHT_S)
        mixed = self._voice.vector() * profile + self._ref * _seconds(self._ref_audio)
        return mixed / float(np.linalg.norm(mixed))

    def judge(self, audio: np.ndarray) -> Verdict:
        """¿Es la misma voz que la referencia? Sin referencia o con una frase
        muy corta para sacar huella confiable, se acepta (decide el nivel)."""
        if not self.enabled:
            return Verdict(True, None, "desactivado")
        seconds = len(audio) / SAMPLE_RATE
        if seconds < SPEAKER_MIN_AUDIO_S:
            return Verdict(True, None, f"frase corta ({seconds:.1f}s), no comparo")
        ref, blend = self._reference()
        if ref is None:
            return Verdict(True, None, "sin referencia todavía")
        emb = self._embedder.embed(audio)
        similarity = float(ref @ emb)
        if similarity >= SPEAKER_MIN_SIMILARITY:
            return Verdict(True, similarity, "misma voz")
        if blend is not None:
            with_profile = float(blend @ emb)
            if with_profile >= SPEAKER_MIN_SIMILARITY:
                return Verdict(True, with_profile,
                               f"misma voz por el perfil guardado (sólo sesión {similarity:.2f})")
        return Verdict(False, similarity, "otra voz")

    def similarity(self, audio: np.ndarray) -> float | None:
        """Similitud de un tramo suelto con quien llamó (la mejor entre la
        sesión y sesión + perfil, igual que judge()). None = sin referencia.
        La usa mix_trim.py para juzgar una frase por ventanas."""
        if not self.enabled:
            return None
        ref, blend = self._reference()
        if ref is None:
            return None
        emb = self._embedder.embed(audio)
        sim = float(ref @ emb)
        if blend is not None:
            sim = max(sim, float(blend @ emb))
        return sim

    def follow(self, audio: np.ndarray, similarity: float | None = None,
               trusted: bool = False) -> None:
        """Frase aceptada de quien me llamó: suma a la referencia. Se queda
        siempre con la frase del wake (la más confiable) y las más nuevas
        hasta SPEAKER_REF_MAX_S. Para APRENDER (memoria) sólo cuenta si se
        verificó con similitud alta (>= SPEAKER_LEARN_MIN_SIM) o es la frase
        misma del wake (`trusted`)."""
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
            total = _seconds(self._ref_audio)
            if trusted or (similarity is not None and similarity >= SPEAKER_LEARN_MIN_SIM):
                self._learn_audio.append(audio)
                max_learn = int(SPEAKER_SESSION_MAX_S * SAMPLE_RATE)
                while (len(self._learn_audio) > 1
                       and sum(len(a) for a in self._learn_audio) > max_learn):
                    del self._learn_audio[1]
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

    Memoria de voces (speaker_memory.py; no necesita el mic):

        python speaker_id.py --voices             # qué aprendió y cómo decidió
        python speaker_id.py --rename voz_3 ivan  # ponerle nombre a una voz
        python speaker_id.py --forget voz_3       # borrar una voz
        python speaker_id.py --forget             # borrar TODO (perfiles + historial)

    La PRIMERA frase que cierre el VAD es la referencia: decí «oye rai» como
    al despertarlo. Después, para cada frase imprime la similitud con la
    referencia. Hablá vos varias veces (y bajito) y que hable otra persona:
    SPEAKER_MIN_SIMILARITY va entre el mínimo tuyo y el máximo ajeno. Enter
    vacío no hace falta: Ctrl+C termina e imprime el resumen.
    """
    import sys

    args = sys.argv[1:]
    if args and args[0] in ("--voices", "--forget", "--rename"):
        memory = VoiceMemory()
        if args[0] == "--voices":
            memory.print_summary()
        elif args[0] == "--rename":
            if len(args) != 3:
                sys.exit("uso: python speaker_id.py --rename voz_N nombre")
            print("ok" if memory.rename(args[1], args[2]) else f"no existe {args[1]}")
        elif len(args) == 2:
            print(f"borradas: {memory.forget(args[1])}")
        else:
            answer = input(f"¿Borrar TODAS las voces e historial de {memory.dir}? [s/N] ")
            if answer.strip().lower() in ("s", "si", "sí", "y", "yes"):
                print(f"borradas {memory.forget()} voces y el historial")
        return

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
