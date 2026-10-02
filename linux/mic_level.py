"""Medidor / calibrador del micrófono (debug).

Usa el mismo camino de audio que main.py (audio_capture.py): con el ReSpeaker
el nivel es el de los mics CRUDOS (ch1-4, sin AGC), que es lo que mira el
filtro de cercanía; sin array, el del mic mono.

1. Verificar que el mic llega hasta acá (WSL o la Pi): la barra tiene que
   moverse cuando hablás.
2. Calibrar el foco por nivel (`NEAR_RMS_THRESHOLD` en config.py), el umbral
   que separa "me están hablando a mí" de "hay gente hablando allá". Hacé dos
   pasadas y anotá el p90 de cada una:

     a) Con la sala como es normalmente (gente hablando lejos) pero SIN hablarle
        al robot   -> p90_lejos
     b) Hablándole vos desde donde le hablarías de verdad -> p90_cerca

   Poné NEAR_RMS_THRESHOLD entre los dos, más cerca de p90_lejos:
   aproximadamente `p90_lejos * 2`, y siempre por debajo de `p90_cerca * 0.6`.
   Si el robot te ignora, bajalo; si sigue enganchando charlas ajenas, subilo.
3. Con `--doa` (sólo ReSpeaker) muestra además la dirección de la voz en el
   marco del robot y, al salir, un histograma de direcciones con voz:
     a) Hablale DE FRENTE: la dirección tiene que dar ~0°. Si no, poné el valor
        crudo que muestra (`chip=`) en DOA_FORWARD_OFFSET_DEG.
     b) Robot prendido (ventiladores, caminando) y SIN nadie hablando: los
        picos del histograma son ruido propio -> DOA_BLOCKED_SECTORS.
     c) Robot hablando, sin nadie: el pico es el parlante -> DOA_SPEAKER_SECTOR.

Uso:
    python mic_level.py              # mic automático (ReSpeaker si está)
    python mic_level.py 3            # índice de sounddevice
    python mic_level.py --doa        # + dirección (ReSpeaker)
"""
import sys
import time

import numpy as np

from audio_capture import LinuxAudioCapture
from config import (
    DOA_FORWARD_OFFSET_DEG,
    DOA_POLL_HZ,
    NEAR_RMS_THRESHOLD,
    NEAR_SNR_RATIO,
    RMS_THRESHOLD,
)
from doa import to_robot_frame

args = [a for a in sys.argv[1:] if not a.startswith("--")]
show_doa = "--doa" in sys.argv
device = int(args[0]) if args else None
WINDOW_S = 10.0  # ventana del resumen
BLOCK_FRAMES = 3  # 3 frames de 30 ms ~ 100 ms por línea

LinuxAudioCapture.list_devices()
capture = LinuxAudioCapture(device_id=device)
array = None
if show_doa:
    from respeaker import ReSpeaker
    array = ReSpeaker.open() if capture.is_array else None
    if array is None:
        print("\n--doa necesita el ReSpeaker (audio + control USB); sigo sólo con nivel.")
    else:
        array.start_polling(DOA_POLL_HZ)

print(f"\nEscuchando ({'ReSpeaker, nivel de los mics crudos' if capture.is_array else 'mic mono'}).")
print(f"Umbral de apertura RMS_THRESHOLD={RMS_THRESHOLD} | "
      f"umbral de cercanía NEAR_RMS_THRESHOLD={NEAR_RMS_THRESHOLD} "
      f"(x{NEAR_SNR_RATIO} sobre el ruido)")
if array is not None:
    print(f"Dirección en el marco del robot (DOA_FORWARD_OFFSET_DEG={DOA_FORWARD_OFFSET_DEG:g}).")
print("Ctrl+C para salir y ver el resumen final.\n")

history: list[float] = []
max_samples = int(WINDOW_S * 10)
voiced_angles: list[float] = []


def summary(levels: list[float], title: str) -> None:
    if not levels:
        return
    p10, p50, p90 = (float(np.percentile(levels, q)) for q in (10, 50, 90))
    print(f"--- {title}: p10={p10:.4f} (ambiente)  p50={p50:.4f}  "
          f"p90={p90:.4f} (picos de voz) ---")


def doa_histogram(angles: list[float]) -> None:
    if not angles:
        print("--- DoA: ninguna lectura con voz ---")
        return
    counts = np.histogram(np.asarray(angles) % 360, bins=24, range=(0, 360))[0]
    total = counts.sum()
    print(f"--- DoA: {total} lecturas con voz, por sector de 15° (marco del robot) ---")
    for i, c in enumerate(counts):
        if c:
            print(f"  {i * 15:3d}-{i * 15 + 15:3d}°  {c / total:5.1%}  {'#' * int(40 * c / counts.max())}")


try:
    last_summary = time.monotonic()
    block: list[float] = []
    for af in capture.frames():
        rms = af.rms if af.rms is not None else float(
            np.sqrt(np.mean((np.frombuffer(af.pcm, np.int16) / 32768.0) ** 2)))
        block.append(rms)
        if len(block) < BLOCK_FRAMES:
            continue
        rms = float(np.sqrt(np.mean(np.square(block))))
        block.clear()
        history.append(rms)
        del history[:-max_samples]

        bar = "#" * min(50, int(rms * 400))
        if rms >= NEAR_RMS_THRESHOLD:
            flag = "  <-- CERCA"
        elif rms >= RMS_THRESHOLD:
            flag = "  <-- voz lejana"
        else:
            flag = ""
        doa = ""
        if array is not None:
            s = array.latest()
            if s is not None:
                robot = to_robot_frame(s.angle)
                doa = f" {'VOZ' if s.voice else '   '} {robot:3.0f}° (chip={s.angle:3d})"
                if s.voice:
                    voiced_angles.append(robot)
        print(f"rms={rms:.4f}{doa} |{bar:<50}|{flag}")

        now = time.monotonic()
        if now - last_summary >= 2.0:
            last_summary = now
            summary(history, f"últimos {WINDOW_S:.0f} s")
except KeyboardInterrupt:
    print()
    summary(history, f"resumen (últimos {WINDOW_S:.0f} s)")
    if array is not None:
        doa_histogram(voiced_angles)
