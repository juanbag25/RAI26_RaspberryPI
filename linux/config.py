import os

SAMPLE_RATE = 16000
FRAME_MS = 30

# Opciones: "tiny", "base", "small", "medium", "large-v3"
# Debe coincidir con una carpeta models/faster-whisper-<MODEL_SIZE>
MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"
LANGUAGE = "es"

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
