"""Wake word: el robot sólo atiende cuando lo llaman por su nombre.

Sin esto, cualquier frase que sobreviva al VAD termina en el LLM — incluida
media conversación ajena que pasó el filtro de cercanía. Con esto el filtro es
explícito: hasta que alguien dice "rai", todo lo transcripto se descarta.

El match es sobre el TEXTO ya transcripto (no un keyword spotter sobre audio):
el pipeline igual transcribe la utterance, así que sale gratis y no agrega
modelos ni dependencias. Se normaliza a minúsculas sin acentos ni puntuación y
se compara contra WAKE_WORDS en las primeras WAKE_SEARCH_WORDS palabras — más
la concatenación de esas palabras, que rescata el "R.A.I." -> "r a i" -> "rai"
que a veces devuelve Whisper.

Una vez despierto queda una ventana de WAKE_WINDOW_S segundos para seguir la
charla sin repetir el nombre; cada frase aceptada la renueva, y también la
renueva el SPEAK_END del orquestador (recién terminó de contestar: lo natural
es que le sigan hablando).

Atención: despertarse no es "escuchar todo lo que pase el VAD durante N
segundos" sino "prestarle atención a QUIEN me llamó". Al despertar se guarda el
nivel de voz de la utterance que traía el "rai" y, mientras dure la ventana,
sólo se aceptan frases que lleguen al menos a ese nivel × ATTENTION_LEVEL_RATIO
(`accepts_level`, consultado por main.py ANTES de transcribir: el fondo de la
sala ni siquiera gasta una llamada a Whisper). La referencia sigue a la persona
con una EMA por frase aceptada, y decir "rai" de nuevo la re-engancha a quien
lo dijo.
"""

from __future__ import annotations

import threading
import time
import unicodedata

from log import log
from config import (
    ATTENTION_FOLLOW_ALPHA,
    ATTENTION_LEVEL_RATIO,
    WAKE_ACK_TEXT,
    WAKE_SEARCH_WORDS,
    WAKE_WINDOW_S,
    WAKE_WORD_ENABLED,
    WAKE_WORDS,
)

_PUNCT_KEEP = "0123456789abcdefghijklmnopqrstuvwxyz "


def normalize(text: str) -> str:
    """minúsculas, sin acentos, sin puntuación, espacios colapsados."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    cleaned = "".join(c if c in _PUNCT_KEEP else " " for c in stripped)
    return " ".join(cleaned.split())


class WakeWord:
    """Estado despierto/dormido + extracción del texto útil de cada frase."""

    def __init__(self) -> None:
        self._words = {normalize(w) for w in WAKE_WORDS if normalize(w)}
        self._lock = threading.Lock()
        self._awake_until = 0.0
        # Nivel (p90 RMS) de la voz a la que le estamos prestando atención.
        self._focus_level = 0.0

    # -- estado ---------------------------------------------------------------

    def is_awake(self) -> bool:
        with self._lock:
            return time.monotonic() < self._awake_until

    def refresh(self) -> None:
        """Renueva la ventana si ya estaba despierto (no lo despierta)."""
        with self._lock:
            if time.monotonic() < self._awake_until:
                self._awake_until = time.monotonic() + WAKE_WINDOW_S

    def _wake(self, level: float | None = None) -> None:
        with self._lock:
            self._awake_until = time.monotonic() + WAKE_WINDOW_S
            if level is not None and level > 0:
                self._focus_level = level

    def _follow(self, level: float) -> None:
        """La persona se movió un poco: la referencia la sigue (EMA)."""
        if level <= 0:
            return
        with self._lock:
            if self._focus_level <= 0:
                self._focus_level = level
            else:
                self._focus_level = ((1.0 - ATTENTION_FOLLOW_ALPHA) * self._focus_level
                                     + ATTENTION_FOLLOW_ALPHA * level)

    def wake_from_audio(self, level: float) -> None:
        """El spotter de audio (wake_spotter.py) reconoció la frase de wake.

        Mismo efecto que encontrar "rai" en el texto: abre la ventana y se
        engancha al nivel de voz de quien lo llamó (`level` = p90 de la
        utterance en curso, 0 si el VAD todavía no abrió una).
        """
        self._wake(level)
        log(f"[WAKE] despierto por audio (foco={self.focus_level():.4f}, "
            f"mínimo={self.min_level():.4f}, ventana {WAKE_WINDOW_S:.0f}s)")

    def sleep(self) -> None:
        with self._lock:
            self._awake_until = 0.0
            self._focus_level = 0.0

    def focus_level(self) -> float:
        with self._lock:
            return self._focus_level

    def min_level(self) -> float:
        """Nivel mínimo que hoy le exigimos a una frase para atenderla
        (0 = sin exigencia: dormido, o atención desactivada)."""
        if ATTENTION_LEVEL_RATIO <= 0 or not WAKE_WORD_ENABLED or not self.is_awake():
            return 0.0
        return self.focus_level() * ATTENTION_LEVEL_RATIO

    def accepts_level(self, level: float) -> bool:
        """¿Vale la pena transcribir una utterance de este nivel?

        Dormido: sí, cualquier cosa que pasó el filtro de cercanía puede ser el
        "rai". Despierto: sólo si suena tan fuerte como quien nos llamó.
        """
        minimum = self.min_level()
        if level >= minimum:
            return True
        log(f"[WAKE] atención: ignorado, más flojo que quien me llamó "
            f"(nivel={level:.4f} < {minimum:.4f}, foco={self.focus_level():.4f})")
        return False

    # -- filtro ---------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> tuple[list[str], list[int]]:
        """Palabras normalizadas + a qué token del texto original pertenece cada
        una. La normalización parte "R.A.I." en tres palabras, así que sin este
        mapeo no se puede recortar el texto original con el índice del match."""
        words: list[str] = []
        owners: list[int] = []
        for i, raw in enumerate(text.split()):
            for word in normalize(raw).split():
                words.append(word)
                owners.append(i)
        return words, owners

    def _find(self, words: list[str]) -> int | None:
        """Índice (en `words`) de la palabra donde termina el wake word."""
        head = words[:WAKE_SEARCH_WORDS]
        for i, word in enumerate(head):
            if word in self._words:
                return i
        # Iniciales deletreadas: "r a i" / "r. a. i." -> "rai".
        for end in range(2, len(head) + 1):
            if "".join(head[:end]) in self._words:
                return end - 1
        return None

    def filter(self, text: str, level: float = 0.0) -> str | None:
        """Qué mandarle al orquestador para esta transcripción.

        `level` es el nivel (p90 RMS) de la utterance de donde salió `text`:
        fija el foco de atención al despertar y lo actualiza mientras dura.
        Devuelve el texto a enviar, o None si hay que ignorar la frase.
        Cuando la frase es sólo el nombre, devuelve WAKE_ACK_TEXT (o None si
        está vacío: despierta en silencio).
        """
        text = text.strip()
        if not text:
            return None
        if not WAKE_WORD_ENABLED:
            return text

        words, owners = self._tokenize(text)
        if not words:
            return None

        index = self._find(words)
        if index is None:
            if self.is_awake():
                self._wake()  # sigue la conversación: renueva la ventana
                self._follow(level)
                log(f"[WAKE] ya despierto, sigue la charla (ventana +{WAKE_WINDOW_S:.0f}s, "
                    f"foco={self.focus_level():.4f})")
                return text
            log(f"[WAKE] dormido, ignorado: {text}")
            return None

        # Despierta Y se engancha al nivel de voz de quien lo llamó.
        self._wake(level)
        # Sacar el nombre y lo que venga antes ("che rai, vení" -> "vení"):
        # al LLM le llega la instrucción sola.
        rest = " ".join(text.split()[owners[index] + 1:]).lstrip(" ,.;:-—").strip()
        if not rest:
            log(f"[WAKE] despierto por «{text}» (sin instrucción, mando ack={WAKE_ACK_TEXT!r})")
            return WAKE_ACK_TEXT or None
        log(f"[WAKE] despierto por «{text}» (foco={self.focus_level():.4f}, "
            f"mínimo={self.min_level():.4f})")
        return rest
