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
# Energía RMS mínima (audio normalizado [-1, 1]) para abrir una utterance.
# Filtra ruido ambiente que el VAD por momentos clasifica como voz.
RMS_THRESHOLD = 0.015
# Duración mínima de voz real dentro de una utterance para enviarla a Whisper.
# Descarta falsos positivos cortos que suelen alucinar "gracias", etc.
MIN_UTTERANCE_MS = 300

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "..", "models", f"faster-whisper-{MODEL_SIZE}")
