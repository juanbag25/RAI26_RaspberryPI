"""Medidor / calibrador de nivel del micrófono (debug).

Dos usos:

1. Verificar que el mic llega hasta acá (WSL o la Pi): la barra tiene que
   moverse cuando hablás.
2. Calibrar el foco del mic (`NEAR_RMS_THRESHOLD` en config.py), que es el
   umbral que separa "me están hablando a mí" de "hay gente hablando allá".
   Hacé dos pasadas y anotá el p90 de cada una:

     a) Con la sala como es normalmente (gente hablando lejos) pero SIN hablarle
        al robot   -> p90_lejos
     b) Hablándole vos desde donde le hablarías de verdad -> p90_cerca

   Poné NEAR_RMS_THRESHOLD entre los dos, más cerca de p90_lejos:
   aproximadamente `p90_lejos * 2`, y siempre por debajo de `p90_cerca * 0.6`.
   Si el robot te ignora, bajalo; si sigue enganchando charlas ajenas, subilo.

Uso:
    python mic_level.py              # dispositivo default
    python mic_level.py 3            # índice de sounddevice
"""
import sys
import time

import numpy as np
import sounddevice as sd

from config import (
    NEAR_RMS_THRESHOLD,
    NEAR_SNR_RATIO,
    RMS_THRESHOLD,
    SAMPLE_RATE,
)

device = int(sys.argv[1]) if len(sys.argv) > 1 else None
block = SAMPLE_RATE // 10  # 100 ms
WINDOW_S = 10.0            # ventana del resumen
history: list[float] = []
max_samples = int(WINDOW_S * 10)

print(sd.query_devices())
print(f"\nEscuchando (device={'default' if device is None else device}).")
print(f"Umbral de apertura RMS_THRESHOLD={RMS_THRESHOLD} | "
      f"umbral de cercanía NEAR_RMS_THRESHOLD={NEAR_RMS_THRESHOLD} "
      f"(x{NEAR_SNR_RATIO} sobre el ruido)")
print("Ctrl+C para salir y ver el resumen final.\n")


def summary(levels: list[float], title: str) -> None:
    if not levels:
        return
    p10, p50, p90 = (float(np.percentile(levels, q)) for q in (10, 50, 90))
    print(f"--- {title}: p10={p10:.4f} (ambiente)  p50={p50:.4f}  "
          f"p90={p90:.4f} (picos de voz) ---")


with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                    blocksize=block, device=device) as stream:
    last_summary = time.monotonic()
    try:
        while True:
            data, _ = stream.read(block)
            a = data.astype(np.float32).ravel() / 32768.0
            rms = float(np.sqrt((a ** 2).mean()))
            history.append(rms)
            del history[:-max_samples]

            bar = "#" * min(60, int(rms * 400))
            if rms >= NEAR_RMS_THRESHOLD:
                flag = "  <-- CERCA (se transcribe)"
            elif rms >= RMS_THRESHOLD:
                flag = "  <-- voz, pero lejana (se descarta)"
            else:
                flag = ""
            print(f"rms={rms:.4f} |{bar:<60}|{flag}")

            now = time.monotonic()
            if now - last_summary >= 2.0:
                last_summary = now
                summary(history, f"últimos {WINDOW_S:.0f} s")
    except KeyboardInterrupt:
        print()
        summary(history, f"resumen (últimos {WINDOW_S:.0f} s)")
