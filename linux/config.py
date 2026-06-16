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

# Google Cloud Text-to-Speech. Las credenciales se leen de la variable de
# entorno GOOGLE_APPLICATION_CREDENTIALS (ruta al JSON de service-account).
# Idioma BCP-47 y voz: ver https://cloud.google.com/text-to-speech/docs/voices
TTS_LANGUAGE = "es-US"
# Dejar "" para que Google elija una voz por defecto del idioma, o fijar una,
# p. ej. "es-US-Neural2-B" / "es-ES-Neural2-A".
TTS_VOICE = "es-US-Neural2-B"
TTS_SAMPLE_RATE = 24000
TTS_SPEAKING_RATE = 1.0
# Dispositivo de salida (índice de sounddevice). None = default del sistema.
TTS_OUTPUT_DEVICE = None

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
