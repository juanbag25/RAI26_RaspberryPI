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
"""

from __future__ import annotations

import threading
import time
import unicodedata

from config import (
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

    # -- estado ---------------------------------------------------------------

    def is_awake(self) -> bool:
        with self._lock:
            return time.monotonic() < self._awake_until

    def refresh(self) -> None:
        """Renueva la ventana si ya estaba despierto (no lo despierta)."""
        with self._lock:
            if time.monotonic() < self._awake_until:
                self._awake_until = time.monotonic() + WAKE_WINDOW_S

    def _wake(self) -> None:
        with self._lock:
            self._awake_until = time.monotonic() + WAKE_WINDOW_S

    def sleep(self) -> None:
        with self._lock:
            self._awake_until = 0.0

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

    def filter(self, text: str) -> str | None:
        """Qué mandarle al orquestador para esta transcripción.

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
                return text
            print(f"[WAKE] dormido, ignorado: {text}", flush=True)
            return None

        self._wake()
        # Sacar el nombre y lo que venga antes ("che rai, vení" -> "vení"):
        # al LLM le llega la instrucción sola.
        rest = " ".join(text.split()[owners[index] + 1:]).lstrip(" ,.;:-—").strip()
        if not rest:
            print(f"[WAKE] despierto por «{text}» (sin instrucción)", flush=True)
            return WAKE_ACK_TEXT or None
        print(f"[WAKE] despierto por «{text}»", flush=True)
        return rest
