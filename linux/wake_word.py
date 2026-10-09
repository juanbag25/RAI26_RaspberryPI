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

Una vez despierto queda una ventana CORTA de WAKE_WINDOW_S segundos para
seguir la charla sin repetir el nombre. La ventana no corre mientras el robot
habla (SPEAK_START -> `hold`) y vuelve a empezar cuando termina (SPEAK_END ->
`refresh`): «Sí, dime» -> la orden enseguida; respuesta -> la réplica
enseguida. Tras mandar una orden se espera hasta REPLY_WAIT_S a que el robot
empiece a contestar (`await_reply`), así la latencia del LLM no lo duerme.

Atención: despertarse no es "escuchar todo lo que pase el VAD durante N
segundos" sino "prestarle atención a QUIEN me llamó". Con el ReSpeaker eso lo
decide la DIRECCIÓN (doa.py, en main.py). Sin array queda un criterio de
NIVEL: mientras dure la ventana sólo se aceptan frases que lleguen a
referencia × ratio (`accepts_level`, consultado por main.py ANTES de
transcribir). La referencia NO es el volumen del «oye rai» — la gente lo dice
fuerte y después baja la voz para la instrucción — sino la primera frase
aceptada después del wake, y la sigue con una EMA.

Cada wake (también uno nuevo estando ya despierto: otra persona toma el foco)
incrementa `wake_seq`; main.py lo usa para avisarle al orquestador, que
contesta «Sí, dime».
"""

from __future__ import annotations

import threading
import time
import unicodedata

from log import dim, drop, fmt, ok
from config import (
    ATTENTION_FOLLOW_ALPHA,
    ATTENTION_LEVEL_RATIO,
    MUTE_TIMEOUT_S,
    REPLY_WAIT_S,
    WAKE_ACK_TEXT,
    WAKE_SEARCH_WORDS,
    WAKE_WINDOW_S,
    WAKE_WORD_ENABLED,
    WAKE_WORDS,
)

_PUNCT_KEEP = "0123456789abcdefghijklmnopqrstuvwxyz "
# Cómo transcribe Whisper el "rai" de un «oye rai» que el spotter ya
# confirmó. NO están en WAKE_WORDS porque, para despertar por texto, "rey"
# suelto daría falsos positivos.
_WAKE_LOOKALIKES = {"rey", "rei", "rail"}


def normalize(text: str) -> str:
    """minúsculas, sin acentos, sin puntuación, espacios colapsados."""
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    cleaned = "".join(c if c in _PUNCT_KEEP else " " for c in stripped)
    return " ".join(cleaned.split())


class WakeWord:
    """Estado despierto/dormido + extracción del texto útil de cada frase."""

    def __init__(self, level_ratio: float = ATTENTION_LEVEL_RATIO) -> None:
        self._words = {normalize(w) for w in WAKE_WORDS if normalize(w)}
        self._lock = threading.Lock()
        self._awake_until = 0.0
        # Nivel (p90 RMS) de la voz a la que le estamos prestando atención.
        # 0 = todavía sin referencia (recién despierto): la toma de la primera
        # frase aceptada.
        self._focus_level = 0.0
        self._level_ratio = level_ratio
        # Wakes que hay que anunciar (cada uno -> evento `awake`).
        self._wake_seq = 0
        # Todos los wakes, anunciados o no (main.py: ¿esta frase despertó?).
        self._wakes = 0

    # -- estado ---------------------------------------------------------------

    def is_awake(self) -> bool:
        with self._lock:
            return time.monotonic() < self._awake_until

    def refresh(self) -> None:
        """Renueva la ventana si ya estaba despierto (no lo despierta)."""
        with self._lock:
            if time.monotonic() < self._awake_until:
                self._awake_until = time.monotonic() + WAKE_WINDOW_S

    def hold(self) -> None:
        """El robot empezó a hablar (SPEAK_START): mientras habla la ventana
        no corre. Si el SPEAK_END se pierde, el mute expira a MUTE_TIMEOUT_S y
        llama a refresh(); el margen extra es por si eso tarda."""
        with self._lock:
            now = time.monotonic()
            if now < self._awake_until:
                self._awake_until = max(self._awake_until,
                                        now + MUTE_TIMEOUT_S + WAKE_WINDOW_S)

    def await_reply(self) -> None:
        """Se le mandó una orden al orquestador: no dormirse mientras el LLM
        piensa y el TTS sintetiza (si contesta, hold/refresh toman la posta)."""
        with self._lock:
            now = time.monotonic()
            if now < self._awake_until:
                self._awake_until = max(self._awake_until, now + REPLY_WAIT_S)

    def announce(self) -> None:
        """Pedir el «Sí, dime» de un wake ya abierto (ver wake_from_audio)."""
        with self._lock:
            self._wake_seq += 1

    def _renew(self) -> None:
        with self._lock:
            self._awake_until = time.monotonic() + WAKE_WINDOW_S

    def _wake_new(self, level: float = 0.0, *, announce: bool = True) -> None:
        """Wake nuevo: ventana nueva y referencia de nivel reiniciada (0 = la
        toma la próxima frase aceptada). `announce`: contarlo en `wake_seq`
        para que el orquestador conteste «Sí, dime»."""
        with self._lock:
            self._awake_until = time.monotonic() + WAKE_WINDOW_S
            self._focus_level = max(0.0, level)
            self._wakes += 1
            if announce:
                self._wake_seq += 1

    def wake_seq(self) -> int:
        with self._lock:
            return self._wake_seq

    def wake_count(self) -> int:
        with self._lock:
            return self._wakes

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

    def wake_from_audio(self, *, announce: bool = True) -> None:
        """El spotter de audio (wake_spotter.py) reconoció la frase de wake.

        Abre (o reinicia) la ventana. La referencia de nivel queda vacía: el
        «oye rai» suele decirse más fuerte que la instrucción que sigue.
        `announce=False`: el «Sí, dime» lo decide main.py un instante después
        (si la persona siguió hablando, no hace falta) con `announce()`.
        """
        self._wake_new(announce=announce)
        ok("WAKE", f"DESPIERTO por audio, ventana {WAKE_WINDOW_S:.0f}s")

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
        if self._level_ratio <= 0 or not WAKE_WORD_ENABLED or not self.is_awake():
            return 0.0
        return self.focus_level() * self._level_ratio

    def accepts_level(self, level: float) -> bool:
        """¿Vale la pena transcribir una utterance de este nivel?

        Dormido o sin referencia todavía: sí. Despierto: sólo si llega a la
        referencia (frases anteriores de esta conversación) × ratio.
        """
        minimum = self.min_level()
        if level >= minimum:
            return True
        drop("WAKE", "más flojo que quien me llamó", nivel=level, minimo=minimum,
             foco=self.focus_level())
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

    def says_name(self, text: str) -> bool:
        """¿La frase empieza llamándolo («rai, ...»)? Sin tocar el estado."""
        return WAKE_WORD_ENABLED and self._find(self._tokenize(text)[0]) is not None

    def strip_wake(self, text: str) -> str:
        """Saca el «oye rai» del principio de una frase que se sabe que
        empezó con él (el spotter lo oyó y la persona siguió hablando).

        Más permisivo que `_find`: acá ya se sabe que el nombre está, así que
        también vale "rey"/"rei" (Whisper oye así "rai"). Si no encuentra el
        nombre, deja el texto como está.
        """
        words, owners = self._tokenize(text)
        names = self._words | _WAKE_LOOKALIKES
        cut = None
        for i, word in enumerate(words[:4]):
            if word in names:
                cut = i
        if cut is None:
            return text.strip()
        return " ".join(text.split()[owners[cut] + 1:]).lstrip(" ,.;:-—").strip()

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
                self._renew()  # sigue la conversación: renueva la ventana
                self._follow(level)
                dim("WAKE", f"sigue la charla, ventana +{WAKE_WINDOW_S:.0f}s "
                    + fmt(foco=self.focus_level()))
                return text
            drop("WAKE", "dormido y no dijo mi nombre", texto=f"«{text}»")
            return None

        # Sacar el nombre y lo que venga antes ("che rai, vení" -> "vení"):
        # al LLM le llega la instrucción sola.
        rest = " ".join(text.split()[owners[index] + 1:]).lstrip(" ,.;:-—").strip()
        # Despierta y se engancha al nivel de esta frase (trae la instrucción).
        # El «Sí, dime» sólo si vino el nombre solo: con instrucción, la
        # respuesta del robot ya es la confirmación (y un `awake` llegando
        # después del texto lo descartaría en el orquestador).
        self._wake_new(level, announce=not rest)
        if not rest:
            ok("WAKE", f"DESPIERTO por «{text}», sin instrucción"
               + (f", mando ack «{WAKE_ACK_TEXT}»" if WAKE_ACK_TEXT else ""))
            return WAKE_ACK_TEXT or None
        ok("WAKE", f"DESPIERTO por «{text}» "
           + fmt(foco=self.focus_level(), minimo=self.min_level()))
        return rest
