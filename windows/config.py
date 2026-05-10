import os

SAMPLE_RATE = 16000
FRAME_MS = 30

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_SIZE = os.path.join(_HERE, "..", "models", "faster-whisper-base")
COMPUTE_TYPE = "int8"
LANGUAGE = "es"

VAD_AGGRESSIVENESS = 2
SILENCE_MS = 700
PRE_SPEECH_PADDING_MS = 200
