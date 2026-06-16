import os

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

# --- TTS (respuesta hablada) -------------------------------------------------
# Puerto donde esta Pi escucha el texto de respuesta que manda el orquestador.
RESPONSE_PORT = 9001

# Motor de TTS: "edge" (gratis, sin credenciales, necesita internet) o
# "google_cloud" (requiere credenciales + billing).
TTS_ENGINE = "edge"

# Dispositivo de salida (índice de sounddevice). None = default del sistema.
TTS_OUTPUT_DEVICE = None

# --- edge-tts (TTS_ENGINE = "edge") ------------------------------------------
# Voces en español: listalas con  `edge-tts --list-voices | grep es-`
# Sugeridas (Argentina): "es-AR-TomasNeural" (masc), "es-AR-ElenaNeural" (fem).
EDGE_VOICE = "es-AR-TomasNeural"
# Formato de edge-tts para velocidad/tono (porcentaje / Hz, con signo):
EDGE_RATE = "+0%"
EDGE_PITCH = "+0Hz"

# --- Google Cloud TTS (TTS_ENGINE = "google_cloud") --------------------------
# Credenciales: variable GOOGLE_APPLICATION_CREDENTIALS (ruta al JSON).
# Voces: https://cloud.google.com/text-to-speech/docs/voices
TTS_LANGUAGE = "es-US"
TTS_VOICE = "es-US-Neural2-B"   # "" = voz default del idioma
TTS_SAMPLE_RATE = 24000
TTS_SPEAKING_RATE = 1.0

VAD_AGGRESSIVENESS = 3
SILENCE_MS = 700
PRE_SPEECH_PADDING_MS = 200
# Energía RMS mínima (audio normalizado [-1, 1]) para abrir una utterance.
# Filtra ruido ambiente que el VAD por momentos clasifica como voz.
RMS_THRESHOLD = 0.015
# Duración mínima de voz real dentro de una utterance para enviarla a Whisper.
# Descarta falsos positivos cortos que suelen alucinar "gracias", etc.
MIN_UTTERANCE_MS = 300

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "..", "models", f"faster-whisper-{MODEL_SIZE}")
