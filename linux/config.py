import os

from dotenv import load_dotenv

from log import warn

# Los knobs de tuning se pueden pisar desde linux/.env sin tocar código (útil
# para calibrar en la Pi por SSH). main.py ya llama load_dotenv(); acá se
# repite para que mic_level.py y cualquier script suelto vean lo mismo.
load_dotenv()


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        warn("CONFIG", f"{name}={raw!r} no es un número: uso {default}")
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

# Qué decide "¿esto es voz humana?" (el filtro de cercanía por nivel va
# encima en los dos casos):
#   "silero": red neuronal (silero_vad.py, onnxruntime, local en la Pi). No
#             confunde pasos, motores ni ventiladores con voz. Necesita
#             models/silero_vad.onnx (ver README); si falta, cae a webrtc.
#   "webrtc": webrtcvad, el de siempre (estadístico: el ruido le parece voz).
VAD_ENGINE = _env_str("VAD_ENGINE", "silero").lower()
# Probabilidad mínima de voz para Silero (0-1). Bajala (0.35) si se come el
# arranque de frases dichas bajo; subila (0.6) si todavía abre con ruido.
SILERO_THRESHOLD = _env_float("SILERO_THRESHOLD", 0.5)
SILERO_MODEL_NAME = _env_str("SILERO_MODEL_NAME", "silero_vad.onnx")
# Sólo para VAD_ENGINE=webrtc (0-3).
VAD_AGGRESSIVENESS = 3
SILENCE_MS = 700
# Para CERRAR una utterance cuenta como silencio todo frame que no sea voz
# fuerte: webrtcvad solo, con ruido de fondo (ventiladores, motores, gente)
# dice "voz" casi todo el tiempo y la frase queda abierta para siempre. Un
# frame sigue la utterance si es voz Y su RMS llega a umbral_apertura × este
# factor. 1.0 = hace falta la misma fuerza que para abrir; bajalo si corta
# frases a la mitad cuando alguien baja la voz.
CLOSE_RMS_RATIO = _env_float("CLOSE_RMS_RATIO", 1.0)
# Además, relativo a la VOZ de la propia utterance: un frame sólo la sigue si
# llega a nivel_de_la_frase (p90 de su voz hasta ahora) × este factor. Con el
# robot caminando el ruido de motores/pasos está muy por debajo de quien le
# habla de cerca: eso cuenta como silencio y la frase cierra. Subilo (0.5) si
# el ruido sigue estirando frases; bajalo (0.2) si corta cuando bajás la voz.
# 0 = desactivado.
CONTINUE_LEVEL_RATIO = _env_float("CONTINUE_LEVEL_RATIO", 0.35)
# Tope de duración de una utterance: pasado esto se cierra igual y se evalúa
# (va a Groq si pasa el filtro). Red de seguridad por si el cierre no llega.
MAX_UTTERANCE_MS = int(_env_float("MAX_UTTERANCE_MS", 12000))
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
# Tope del piso. Con el robot caminando (motores, pasos) el fondo sube mucho:
# 0.05 deja que el umbral para abrir llegue a 0.05 × NEAR_SNR_RATIO = 0.15.
# Ojo: con ese ruido hay que hablarle más fuerte/cerca; si te ignora caminando,
# bajá NEAR_SNR_RATIO (2) antes que este tope.
NOISE_FLOOR_MAX = _env_float("NOISE_FLOOR_MAX", 0.05)
# Además el piso se estima SIEMPRE (también con una frase abierta) como el
# percentil 10 del RMS de esta ventana: al hablar siempre hay pausas, así que
# ese mínimo es el ruido. Sin esto, un ruido que arranca de golpe (el robot
# empieza a caminar) abre una frase antes de que el piso lo aprenda.
NOISE_WINDOW_MS = int(_env_float("NOISE_WINDOW_MS", 2000))

# --- Mic array: ReSpeaker USB Mic Array v2.0 (XVF-3000) -----------------------
# Si está conectado se usa solo (si no, el mic mono de siempre). Firmware de 6
# canales: ch0 = audio procesado por el chip (beamforming + supresión de ruido
# + AGC) -> va al spotter y a Whisper; ch1-4 = mics crudos -> de ahí sale el
# NIVEL que usa el filtro de cercanía de vad.py, porque el AGC de ch0 levanta
# a la gente lejana (medido: ganancia 2-5x en 20 s) y rompería ese filtro.
# Ojo: la escala del nivel crudo es ~10 dB más baja que la de ch0 y que la
# del mic viejo: recalibrar NEAR_RMS_THRESHOLD con `python mic_level.py`.
RESPEAKER_ENABLED = _env_bool("RESPEAKER_ENABLED", True)
# Parámetros del DSP que se fijan en cada arranque (el chip los olvida al
# cortarle la alimentación). `python respeaker.py` lista todos. Se pueden
# pisar desde .env: RESPEAKER_PARAMS=AGCMAXGAIN=10,HPFONOFF=2
_RESPEAKER_DEFAULT_PARAMS = {
    # Sin referencia de parlante (el parlante está en la Jetson) el AEC no
    # tiene qué restar: apagado para que no meta artefactos.
    "ECHOONOFF": 0,
    # Pasa-altos 125 Hz: corta retumbe de motores/pasos sin tocar la voz.
    "HPFONOFF": 2,
    "STATNOISEONOFF": 1,      # ventiladores y ruido estacionario
    "NONSTATNOISEONOFF": 1,   # ruido no estacionario
    "AGCONOFF": 1,
}


def _env_params(name: str, default: dict[str, float]) -> dict[str, float]:
    params = dict(default)
    for item in _env_str(name, "").split(","):
        key, sep, value = item.partition("=")
        if not sep:
            continue
        try:
            params[key.strip().upper()] = float(value)
        except ValueError:
            warn("CONFIG", f"{name}: {item!r} no es NOMBRE=número, lo ignoro")
    return params


RESPEAKER_PARAMS = _env_params("RESPEAKER_PARAMS", _RESPEAKER_DEFAULT_PARAMS)
# Lecturas de DoA por segundo (cada una son 2 consultas USB, ~2-25 ms).
DOA_POLL_HZ = _env_float("DOA_POLL_HZ", 20)
# Brillo del anillo de LEDs (0-31). Despierto: el firmware ilumina hacia la
# voz; dormido: apagado.
RESPEAKER_LED_BRIGHTNESS = int(_env_float("RESPEAKER_LED_BRIGHTNESS", 8))

# --- Foco espacial (DoA) ----------------------------------------------------------
# Al despertar, el robot fija la dirección de quien dijo «oye rai» y, mientras
# dure la ventana, sólo atiende frases que vengan de ahí (doa.py). Medido: con
# voz, ~2/3 de las lecturas caen sobre la persona y ~1/3 son reflexiones; por
# eso se exige una FRACCIÓN de lecturas en foco y no que toda la frase lo esté.
#
# Cuánto puede apartarse una lectura del foco y seguir contando como "en foco".
DOA_TOLERANCE_DEG = _env_float("DOA_TOLERANCE_DEG", 35)
# Fracción mínima de lecturas con voz dentro del foco para aceptar la frase.
# Bajala si el robot ignora al que lo llamó; subila si entran otras voces.
DOA_MIN_IN_FOCUS = _env_float("DOA_MIN_IN_FOCUS", 0.4)
# Con menos lecturas con voz que esto, la dirección no decide (se acepta):
# una frase cortita no se pierde por falta de datos.
DOA_MIN_SAMPLES = int(_env_float("DOA_MIN_SAMPLES", 4))
# Ancho del bin del histograma de direcciones (grados).
DOA_BIN_DEG = _env_float("DOA_BIN_DEG", 15)
# El foco sigue a la persona si se mueve (EMA circular por frase aceptada).
DOA_FOLLOW_ALPHA = _env_float("DOA_FOLLOW_ALPHA", 0.3)
# Ángulo del chip que corresponde al FRENTE del robot: lo que se resta para
# loguear/avisar en el marco del robot. Calibrar con `python mic_level.py --doa`
# hablándole de frente.
DOA_FORWARD_OFFSET_DEG = _env_float("DOA_FORWARD_OFFSET_DEG", 0)
# Sectores (marco del robot) donde hay ruido propio fijo: ventiladores,
# motores. Frases que vienen mayormente de ahí se descartan siempre. Formato
# "desde-hasta" separados por coma, cruzar 0° vale: "170-200,350-10".
# Se calibran con `mic_level.py --doa` con el robot prendido y sin nadie.
DOA_BLOCKED_SECTORS = _env_str("DOA_BLOCKED_SECTORS", "")
# Para un SEGUNDO «oye rai» estando despierto (otra persona toma el foco): no
# alcanza con que el spotter lo oiga, tiene que pasar el filtro de cercanía
# general (o venir del foco actual). Si no, un «rai» de fondo se robaría el
# robot. Ventana de audio hacia atrás que se mira para la dirección del wake.
WAKE_DOA_WINDOW_S = _env_float("WAKE_DOA_WINDOW_S", 1.2)

# --- Wake word ----------------------------------------------------------------
# Con esto activado el robot ignora TODO lo que se transcribe hasta que alguien
# lo llama por su nombre. Después queda "despierto" una ventana de tiempo para
# seguir la conversación sin repetir el nombre en cada frase.
WAKE_WORD_ENABLED = _env_bool("WAKE_WORD_ENABLED", True)
# Cómo se detecta el nombre:
#   "audio": spotter local (Vosk) escuchando "oye rai" todo el tiempo. Dormido
#            no se manda NADA a Groq; al reconocer la frase el robot dice
#            «Sí, dime» (por el parlante de la Jetson) y recién ahí se
#            transcribe. Es el modo "como Siri" (wake_spotter.py).
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
# La Pi NO emite sonido: todo lo audible sale del parlante de la Jetson. Cada
# «oye rai» se le avisa al orquestador como evento (mismo socket que el texto,
# prefijo ORCH_EVENT_PREFIX) y él contesta «Sí, dime»; al vencerse la ventana,
# un chime grave. Eventos: awake, asleep, stop, doa:<grados>.
# Debe coincidir con EVENT_PREFIX en orchestrator.py. Whisper nunca devuelve
# texto que empiece así, por eso se puede compartir el socket del texto.
ORCH_EVENT_PREFIX = "@@event:"
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
# Si la frase es SÓLO el nombre ("rai"), qué mandarle al orquestador como
# turno. Por defecto nada: el orquestador ya contesta «Sí, dime» al evento
# `awake`.
WAKE_ACK_TEXT = _env_str("WAKE_ACK_TEXT", "")
# Atención por NIVEL, sólo sin mic array (con el ReSpeaker manda la dirección,
# DoA). La gente dice «hola rai» fuerte y la instrucción más bajo, así que la
# referencia NO es el volumen del wake: es la primera instrucción aceptada
# después (y la sigue con una EMA). Mientras dure la ventana se exige llegar a
# referencia × este factor. 0 = desactivado.
ATTENTION_LEVEL_RATIO = _env_float("ATTENTION_LEVEL_RATIO", 0.25)
# Lo mismo con el ReSpeaker: por defecto apagado, la dirección hace ese trabajo.
ATTENTION_LEVEL_RATIO_ARRAY = _env_float("ATTENTION_LEVEL_RATIO_ARRAY", 0.0)
ATTENTION_FOLLOW_ALPHA = 0.3

# --- Mientras el robot habla ----------------------------------------------------
# "mute": se descarta todo el audio (como siempre).
# "keyword": se sigue descartando todo, salvo una ORDEN DE CORTE
#   (SPEAK_STOP_PHRASES o la frase de wake) que reconoce el spotter, viene de
#   fuera del sector del parlante (DOA_SPEAKER_SECTOR) y suena
#   SPEAK_BARGE_RATIO veces más fuerte que el parlante en ese momento. Manda
#   `stop` al orquestador (y si fue «oye rai», además re-despierta). Requiere
#   WAKE_MODE=audio. Dejar en "mute" hasta probar que no corta solo.
SPEAK_LISTEN_MODE = _env_str("SPEAK_LISTEN_MODE", "mute").lower()
# Dos palabras como WAKE_PHRASES (una sola, "para", aparece en cualquier frase
# del propio robot); "rai" suele salir "ray"/"rey" en Vosk, por eso las variantes.
SPEAK_STOP_PHRASES = _env_list("SPEAK_STOP_PHRASES",
                               ("para rai", "para ray", "basta rai", "basta ray"))
SPEAK_BARGE_RATIO = _env_float("SPEAK_BARGE_RATIO", 2.0)
# Dónde está el parlante del robot visto desde el array (marco del robot,
# mismo formato que DOA_BLOCKED_SECTORS). Vacío = no se chequea dirección.
DOA_SPEAKER_SECTOR = _env_str("DOA_SPEAKER_SECTOR", "")

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "..", "models", f"faster-whisper-{MODEL_SIZE}")
WAKE_MODEL_PATH = os.path.join(_HERE, "..", "models", WAKE_MODEL_NAME)
SILERO_MODEL_PATH = os.path.join(_HERE, "..", "models", SILERO_MODEL_NAME)
