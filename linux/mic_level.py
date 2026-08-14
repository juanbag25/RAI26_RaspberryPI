"""Medidor de nivel del micrófono (debug).

Imprime el RMS de cada bloque de 100 ms con una barra. Sirve para verificar
que el mic llega hasta acá (WSL o la Pi) y que hablando se supera
RMS_THRESHOLD (config.py): si no lo supera, el VAD nunca abre una utterance
y main.py no transcribe nada.

Uso:
    python mic_level.py              # dispositivo default
    python mic_level.py 3            # índice de sounddevice
"""
import sys

import numpy as np
import sounddevice as sd

from config import RMS_THRESHOLD, SAMPLE_RATE

device = int(sys.argv[1]) if len(sys.argv) > 1 else None
block = SAMPLE_RATE // 10  # 100 ms

print(sd.query_devices())
print(f"\nEscuchando (device={'default' if device is None else device}). "
      f"Hablá y fijate si la barra supera el umbral del VAD ({RMS_THRESHOLD}). "
      f"Ctrl+C para salir.\n")

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                    blocksize=block, device=device) as stream:
    try:
        while True:
            data, _ = stream.read(block)
            a = data.astype(np.float32).ravel() / 32768.0
            rms = float(np.sqrt((a ** 2).mean()))
            bar = "#" * min(60, int(rms * 400))
            flag = "  <-- SUPERA EL UMBRAL" if rms >= RMS_THRESHOLD else ""
            print(f"rms={rms:.4f} |{bar:<60}|{flag}")
    except KeyboardInterrupt:
        pass
