import os

from dotenv import load_dotenv

# Los knobs de tuning se pueden pisar desde linux/.env sin tocar código (útil
# para calibrar en la Pi por SSH). main.py ya llama load_dotenv(); acá se
# repite para que mic_level.py y cualquier script suelto vean lo mismo.
load_dotenv()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        print(f"[CONFIG] {name}={raw!r} no es un número: uso {default}")
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on", "si", "sí")


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return default if raw is None else raw.strip()


def _env_list(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Lista separada por comas en .env ("oye rai, oye ray")."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return tuple(item.strip() for item in raw.split(",") if item.strip())


SAMPLE_RATE = 16000
FRAME_MS = 30

# Backend STT: "local" (faster-whisper en CPU) o "groq" (Groq cloud API).
BACKEND = "groq"

# --- Backend local (faster-whisper) ------------------------------------------
# Opciones: "tiny", "base", "small", "medium", "large-v3"
# Debe coincidir con una carpeta models/faster-whisper-<MODEL_SIZE>
MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"

# --- Backend groq ------------------------------------------------------------
# Opciones: "whisper-large-v3", "whisper-large-v3-turbo"
# La API key se lee de la variable de entorno GROQ_API_KEY.
GROQ_MODEL = "whisper-large-v3-turbo"

# --- Común -------------------------------------------------------------------
LANGUAGE = "es"
# Pista de vocabulario para Whisper (prompt / initial_prompt). Sirve para dos
# cosas: que escriba "RAI" y no "rai/ray/rey" (importante para el wake word) y
# que se sesgue al dominio en vez de alucinar.
# Ojo: mantenerlo corto y SIN frases imperativas de ejemplo — con audio flojo
# Whisper tiende a devolver el prompt tal cual, y un "RAI, vení" alucinado sería
# un comando falso.
STT_PROMPT = "Conversación en español con RAI, un perro robot del ITBA."

# --- Control de mute (aviso del orquestador) ----------------------------------
# Puerto donde este script escucha SPEAK_START / SPEAK_END del orquestador
# (= MIC_CTRL_PORT del orchestrator). Mientras el robot habla, el mic se
# silencia para no transcribirse a sí mismo.
CTRL_PORT = 9001
# Auto-desmute si se pierde el SPEAK_END (segundos).
MUTE_TIMEOUT_S = 30.0

VAD_AGGRESSIVENESS = 3
SILENCE_MS = 700
PRE_SPEECH_PADDING_MS = 200
# Duración mínima de voz real dentro de una utterance para enviarla a Whisper.
# Descarta falsos positivos cortos que suelen alucinar "gracias", etc.
MIN_UTTERANCE_MS = int(_env_float("MIN_UTTERANCE_MS", 400))

# --- Foco del micrófono: rechazo de campo lejano -------------------------------
# El mic es omnidireccional: sin filtro, una charla del otro lado de la sala
# entra al VAD y Whisper la transcribe (o directamente alucina). El criterio es
# de ENERGÍA — quien le habla al robot de cerca llega mucho más fuerte que el
# fondo — y se aplica en dos puntos: al abrir la utterance (frame a frame) y al
# cerrarla (sobre el nivel real de toda la utterance).
#
# Piso absoluto para abrir una utterance (audio normalizado [-1, 1]).
RMS_THRESHOLD = _env_float("RMS_THRESHOLD", 0.02)
# Knob PRINCIPAL: nivel que la utterance tiene que alcanzar (percentil 90 de
# sus frames de voz) para contar como "de cerca". Subilo si sigue entrando
# gente de lejos; bajalo si el robot te ignora a vos.
# Calibralo con `python mic_level.py` (imprime p10/p90 y un valor sugerido).
NEAR_RMS_THRESHOLD = _env_float("NEAR_RMS_THRESHOLD", 0.055)
# Además del umbral absoluto: la voz tiene que estar este factor por encima del
# piso de ruido medido en vivo (el murmullo de fondo sube ese piso, así que en
# una sala ruidosa el filtro se endurece solo).
NEAR_SNR_RATIO = _env_float("NEAR_SNR_RATIO", 3.0)
# Frames de voz fuerte CONSECUTIVOS para abrir una utterance (30 ms c/u): evita
# que un golpe o una sílaba lejana abran la ventana.
ONSET_SPEECH_FRAMES = int(_env_float("ONSET_SPEECH_FRAMES", 3))
# Piso de ruido: EMA por frame descartado. Sube rápido, baja lento y está
# topeado para que un ruido fuerte no deje al robot sordo.
NOISE_FLOOR_INIT = 0.005
NOISE_FLOOR_ALPHA = 0.05
NOISE_FLOOR_MAX = _env_float("NOISE_FLOOR_MAX", 0.02)

# --- Wake word ----------------------------------------------------------------
# Con esto activado el robot ignora TODO lo que se transcribe hasta que alguien
# lo llama por su nombre. Después queda "despierto" una ventana de tiempo para
# seguir la conversación sin repetir el nombre en cada frase.
WAKE_WORD_ENABLED = _env_bool("WAKE_WORD_ENABLED", True)
# Cómo se detecta el nombre:
#   "audio": spotter local (Vosk) escuchando "oye rai" todo el tiempo. Dormido
#            no se manda NADA a Groq; al reconocer la frase suena un beep y
#            recién ahí se transcribe. Es el modo "como Siri" (wake_spotter.py).
#   "text":  se transcribe cada frase que pasa el VAD y se busca "rai" en el
#            texto (wake_word.py). Más lento (~2 s) y gasta Groq dormido, pero
#            no necesita el modelo Vosk. Si "audio" no puede arrancar (vosk no
#            instalado, modelo ausente) se cae a este modo solo.
WAKE_MODE = _env_str("WAKE_MODE", "audio").lower()
# Frases que despiertan al robot en modo "audio". Vosk las reconoce con una
# gramática cerrada (todo lo demás es "[unk]"), así que conviene que sean de
# 2+ palabras: "rai" solo es una sílaba y dispararía con cualquier cosa.
# "rai" no es palabra del español y el modelo suele escucharlo como "ray" o
# "rey": por eso están las tres. Agregá variantes con el prefijo que uses
# ("hola rai", "che rai") en .env: WAKE_PHRASES=oye rai,oye ray,hola rai
WAKE_PHRASES = _env_list("WAKE_PHRASES", ("oye rai", "oye ray", "oye rey"))
# Disparar con el resultado PARCIAL del reconocedor (apenas ve la frase) en
# vez de esperar a que Vosk cierre la utterance. Más rápido (~300 ms antes).
WAKE_ON_PARTIAL = _env_bool("WAKE_ON_PARTIAL", True)
# Tras un disparo, ignorar nuevos disparos por este tiempo (la misma frase
# suele aparecer dos veces: parcial y final).
WAKE_COOLDOWN_S = _env_float("WAKE_COOLDOWN_S", 1.5)
# Segundos de audio que puede acumular la cola del spotter si la Pi se
# atrasa; más allá se descartan frames (se pierde un wake, no se traba el mic).
WAKE_QUEUE_S = _env_float("WAKE_QUEUE_S", 3.0)
# Modelo Vosk (carpeta descomprimida). Ver README para descargarlo.
WAKE_MODEL_NAME = _env_str("WAKE_MODEL_NAME", "vosk-model-small-es-0.42")
# Sonido al despertar: "beep" (generado), "none", o ruta a un .wav 16-bit.
WAKE_SOUND = _env_str("WAKE_SOUND", "beep")
WAKE_SOUND_VOLUME = _env_float("WAKE_SOUND_VOLUME", 0.4)
# Parlante para el beep (índice de sounddevice; vacío = default del sistema).
_out = _env_str("AUDIO_OUTPUT_DEVICE", "")
WAKE_SOUND_DEVICE = int(_out) if _out else None
# Variantes con las que Whisper suele escribir "rai". Se comparan en minúsculas,
# sin acentos ni puntuación (y también sobre las iniciales pegadas: "R.A.I." ->
# "r a i" -> "rai").
WAKE_WORDS = (
    "rai", "ray", "rae", "raid", "rai26", "raii",
    "rye", "wry", "rai's",
)
# Sólo se busca el nombre en las primeras N palabras de la frase: "rai vení" sí,
# "el otro día en la clase de rai..." no.
WAKE_SEARCH_WORDS = int(_env_float("WAKE_SEARCH_WORDS", 3))
# Ventana de conversación en segundos: tras despertarlo, cuánto tiempo se le
# puede seguir hablando sin volver a decir "rai". Cada frase aceptada —y cada
# respuesta hablada del robot— la renueva.
WAKE_WINDOW_S = _env_float("WAKE_WINDOW_S", 25.0)
# Si la frase es SÓLO el nombre ("rai"), qué mandarle al orquestador para que
# conteste algo y se note que está escuchando. "" = no mandar nada (sólo abre
# la ventana en silencio). En modo "audio" el beep ya hace de confirmación y
# el "oye rai" solo ni siquiera se transcribe, así que por defecto va vacío.
WAKE_ACK_TEXT = _env_str("WAKE_ACK_TEXT", "rai" if WAKE_MODE == "text" else "")
# Atención: al despertarse, el robot se queda con el NIVEL de la voz que lo
# llamó y, mientras dure la ventana, sólo acepta frases que lleguen al menos a
# ese nivel × ATTENTION_LEVEL_RATIO. Quien está más lejos que quien dijo "rai"
# (el fondo de la sala) queda afuera aunque pase el filtro de cercanía general.
# 1.0 = tan fuerte como el "rai"; 0.5 = la mitad. 0 = desactivado.
ATTENTION_LEVEL_RATIO = _env_float("ATTENTION_LEVEL_RATIO", 0.6)
# El nivel de referencia sigue a la persona (EMA por frase aceptada): si se
# acerca o aleja un poco, la referencia se mueve con ella.
ATTENTION_FOLLOW_ALPHA = 0.3

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "..", "models", f"faster-whisper-{MODEL_SIZE}")
WAKE_MODEL_PATH = os.path.join(_HERE, "..", "models", WAKE_MODEL_NAME)
