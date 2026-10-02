"""Calibración guiada del mic (y del ReSpeaker) en el lugar donde trabaja el robot.

Va diciendo qué hacer en cada paso, mide, y al final escribe los valores en
`linux/.env` (con backup del anterior y confirmación). Correrlo con el array
YA MONTADO en el robot y en la sala real: los ángulos y ruidos dependen de eso.

    ./calibrate_stt.sh               # en la Pi: frena el servicio, calibra, lo levanta
    python calibrate.py              # interactivo (con la venv y el mic libre)
    python calibrate.py --dry-run    # mide y muestra, no escribe .env
    python calibrate.py --yes        # no pregunta antes de guardar

Pasos (cualquiera se saltea con "s" + Enter):

1. Ambiente con el robot prendido, nadie hablando   -> piso de ruido, sectores
                                                       de ventiladores (DoA)
2. Robot caminando, nadie hablando                    -> ruido de marcha, sectores
3. Robot hablando (hacelo hablar desde el orquestador) -> sector del parlante
4. Vos de frente a ~1 m, hablando normal              -> frente del robot,
                                                       nivel "cerca", dispersión
5. Alguien hablando desde el fondo (3+ m)             -> nivel "lejos"
6. «oye rai» 3 veces de frente                        -> prueba del spotter

Qué calibra: NEAR_RMS_THRESHOLD, RMS_THRESHOLD, NEAR_SNR_RATIO,
NOISE_FLOOR_MAX y, con ReSpeaker, DOA_FORWARD_OFFSET_DEG,
DOA_BLOCKED_SECTORS, DOA_SPEAKER_SECTOR y DOA_MIN_IN_FOCUS.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from audio_capture import LinuxAudioCapture
from config import (
    DOA_POLL_HZ,
    NEAR_SNR_RATIO,
    NOISE_FLOOR_MAX,
    DOA_TOLERANCE_DEG,
    FRAME_MS,
    MIC_MODE,
    RESPEAKER_ENABLED,
    RESPEAKER_PARAMS,
)
from doa import ang_diff, dominant_direction

_PER_MIC_KEYS = ("RMS_THRESHOLD", "NEAR_RMS_THRESHOLD", "NEAR_SNR_RATIO", "NOISE_FLOOR_MAX")
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
BIN_DEG = 15          # sectores de ruido: resolución del histograma
SECTOR_MIN_SHARE = 0.12  # un bin es "ruido propio" si junta >= 12 % de las lecturas con voz
SECTOR_MIN_READS = 6
SECTOR_PAD_DEG = 10


# --------------------------------------------------------------------------- IO

def other_listener() -> str | None:
    """¿Otro proceso del STT (python main.py) tiene el mic abierto? Dos
    procesos no pueden abrir el ReSpeaker a la vez: el segundo ni lo ve."""
    me = os.getpid()
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if int(pid) == me:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                argv = [a.decode(errors="replace") for a in fh.read().split(b"\0") if a]
        except OSError:
            continue
        if (argv and os.path.basename(argv[0]).startswith("python")
                and any(os.path.basename(a) == "main.py" for a in argv[1:])):
            return f"PID {pid}: {' '.join(argv)}"
    return None


def say(msg: str = "") -> None:
    print(msg, flush=True)


def title(n: int, total: int, text: str) -> None:
    say()
    say("=" * 70)
    say(f" PASO {n}/{total}: {text}")
    say("=" * 70)


def ask_go(prompt: str = "Enter para empezar, «s» + Enter para saltear este paso: ") -> bool:
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        answer = ""
    return answer not in ("s", "saltear", "skip")


def countdown() -> None:
    for i in (3, 2, 1):
        print(f"\r  arranca en {i}...", end="", flush=True)
        time.sleep(1)
    print("\r  ¡AHORA!            ", flush=True)


# ---------------------------------------------------------------------- medición

@dataclass
class Take:
    """Lo grabado en un paso."""
    rms: list[float] = field(default_factory=list)      # nivel por frame (crudos si hay array)
    pcm: list[bytes] = field(default_factory=list)      # ch0 / mono
    chip_angles: list[int] = field(default_factory=list)  # DOAANGLE crudos con VOICEACTIVITY=1
    doa_reads: int = 0

    def p(self, q: float) -> float:
        return float(np.percentile(self.rms, q)) if self.rms else 0.0


def record(capture: LinuxAudioCapture, array, seconds: float, label: str) -> Take:
    take = Take()
    n_frames = int(seconds * 1000 / FRAME_MS)
    t0 = time.monotonic()
    frames = capture.frames()
    for i, af in enumerate(frames):
        rms = af.rms if af.rms is not None else float(
            np.sqrt(np.mean((np.frombuffer(af.pcm, np.int16) / 32768.0) ** 2)))
        take.rms.append(rms)
        take.pcm.append(af.pcm)
        if i % 5 == 0:
            left = seconds - (time.monotonic() - t0)
            bar = "#" * min(40, int(rms * 400))
            print(f"\r  {label}: {max(0, left):4.1f}s  nivel={rms:.4f} |{bar:<40}|", end="", flush=True)
        if i + 1 >= n_frames:
            break
    frames.close()
    t1 = time.monotonic()
    print("\r" + " " * 78 + "\r", end="")
    if array is not None:
        samples = array.samples_between(t0, t1)
        take.doa_reads = len(samples)
        take.chip_angles = [s.angle for s in samples if s.voice]
    return take


def noise_sectors(chip_angles: list[int], offset: float) -> list[tuple[float, float]]:
    """Bins (marco del robot) que concentran lecturas con voz SIN nadie
    hablando: ruido propio fijo. Bins vecinos se unen; margen de ±PAD."""
    if len(chip_angles) < SECTOR_MIN_READS:
        return []
    robot = (np.asarray(chip_angles, dtype=float) - offset) % 360
    nb = 360 // BIN_DEG
    counts = np.histogram(robot, bins=nb, range=(0, 360))[0]
    hot = [(c >= SECTOR_MIN_SHARE * len(robot) and c >= SECTOR_MIN_READS) for c in counts]
    if not any(hot):
        return []
    sectors: list[list[int]] = []
    for i in range(nb):
        if hot[i]:
            if sectors and sectors[-1][1] == i - 1:
                sectors[-1][1] = i
            else:
                sectors.append([i, i])
    if len(sectors) > 1 and sectors[0][0] == 0 and sectors[-1][1] == nb - 1:
        first = sectors.pop(0)            # une el que cruza 0°
        sectors[-1][1] = first[1] + nb
    return [(((a * BIN_DEG) - SECTOR_PAD_DEG) % 360, (((b + 1) * BIN_DEG) + SECTOR_PAD_DEG) % 360)
            for a, b in sectors]


def fmt_sectors(sectors: list[tuple[float, float]]) -> str:
    return ",".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in sectors)


def overlaps_front(sectors: list[tuple[float, float]], tolerance: float) -> bool:
    from doa import in_sector
    return any(in_sector(a, s) for s in sectors for a in (0, tolerance / 2, -tolerance / 2))


# ------------------------------------------------------------------------ .env

def update_env(path: str, values: dict[str, str]) -> None:
    lines = open(path, encoding="utf-8").read().splitlines() if os.path.exists(path) else []
    pending = dict(values)
    out = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        if key in pending:
            out.append(f"{key}={pending.pop(key)}")
        else:
            out.append(line)
    if pending:
        out.append("")
        out.append(f"# --- calibrate.py {time.strftime('%Y-%m-%d %H:%M')} ---")
        out.extend(f"{k}={v}" for k, v in pending.items())
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


# ------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="no escribir .env")
    ap.add_argument("--yes", action="store_true", help="guardar sin preguntar")
    ap.add_argument("--device", type=int, default=None, help="índice de sounddevice")
    ap.add_argument("--seconds", type=float, default=10.0, help="duración de cada paso")
    args = ap.parse_args()
    secs = args.seconds

    say("CALIBRACIÓN DEL MIC DE RAI")
    say("Hacelo con el mic montado en el robot, en la sala donde va a trabajar.")
    busy = other_listener()
    if busy:
        say(f"\n✖ El STT está corriendo y tiene el mic abierto: {busy}")
        say("  Usá ./calibrate_stt.sh (frena el servicio, calibra y lo vuelve a levantar),")
        say("  o a mano: sudo systemctl stop rai26-stt  ...  sudo systemctl start rai26-stt")
        return 1
    capture = LinuxAudioCapture(device_id=args.device)
    array = None
    if capture.is_array and RESPEAKER_ENABLED:
        from respeaker import ReSpeaker
        array = ReSpeaker.open()
        if array is None:
            say("! ReSpeaker sin control USB (regla udev / pyusb): calibro sólo niveles.")
        else:
            array.apply_params(RESPEAKER_PARAMS)
            array.start_polling(DOA_POLL_HZ)
    say(f"Mic: {'ReSpeaker (nivel de los mics crudos + dirección)' if array else 'ReSpeaker sin DoA' if capture.is_array else 'mono (sólo niveles)'}")
    total = 6 if array else 4

    takes: dict[str, Take] = {}
    step = 0

    step += 1
    title(step, total, "AMBIENTE (robot prendido, NADIE hablando)")
    say("  Prendé el robot (Jetson, ventiladores) y que nadie hable cerca.")
    say("  Mide el ruido de fondo" + (" y de dónde viene el ruido propio del robot." if array else "."))
    if ask_go():
        countdown()
        takes["quiet"] = record(capture, array, secs, "ambiente")

    step += 1
    title(step, total, "ROBOT CAMINANDO (nadie hablando)")
    say("  Hacé caminar al robot en el lugar (o con el teleop) mientras mido.")
    say("  Si no se puede ahora, salteá: el tope de ruido queda en su default.")
    if ask_go():
        countdown()
        takes["walk"] = record(capture, array, secs, "caminando")

    if array:
        step += 1
        title(step, total, "ROBOT HABLANDO (nadie más hablando)")
        say("  Hacé que el robot hable por su parlante (p. ej. preguntale algo largo")
        say("  desde el orquestador, o reproducí audio por el parlante de la Jetson).")
        say("  Sirve para ubicar el parlante (orden de corte, SPEAK_LISTEN_MODE=keyword).")
        if ask_go():
            countdown()
            takes["speaker"] = record(capture, array, secs, "parlante")

    step += 1
    title(step, total, "VOS DE FRENTE, a ~1 m, hablando normal")
    say("  Parate DELANTE del robot (de frente a su cabeza), a la distancia a la")
    say("  que le hablarías, y hablá normal todo el tiempo (contá del 1 al 30).")
    if ask_go():
        countdown()
        takes["near"] = record(capture, array, secs, "cerca")

    step += 1
    title(step, total, "VOZ DE FONDO (3 m o más, como gente que NO le habla)")
    say("  Que alguien hable a volumen normal desde el fondo de la sala, o")
    say("  alejate vos a donde estaría la gente que no le habla al robot.")
    if ask_go():
        countdown()
        takes["far"] = record(capture, array, secs, "lejos")

    wake_hits = None
    if array:
        step += 1
        title(step, total, "PRUEBA DEL WAKE: decí «oye rai» 3 veces, de frente")
        say("  Con ~2 s entre cada una, a tu volumen normal.")
        if ask_go():
            try:
                from wake_spotter import WakeSpotter
                spotter = WakeSpotter()
                countdown()
                take = record(capture, array, max(secs, 9.0), "oye rai")
                for frame in take.pcm:
                    spotter.feed(frame)
                deadline = time.monotonic() + 10
                while spotter.queue_size() and time.monotonic() < deadline:
                    time.sleep(0.1)
                time.sleep(0.3)
                wake_hits = spotter.pop_stats().detections
            except Exception as exc:  # noqa: BLE001 - vosk/modelo ausente
                say(f"  ! no pude probar el spotter: {exc}")

    # ------------------------------------------------------------- cálculo
    say()
    say("=" * 70)
    say(" RESULTADOS")
    say("=" * 70)
    values: dict[str, str] = {}
    notes: list[str] = []

    quiet, walk, near, far = (takes.get(k) for k in ("quiet", "walk", "near", "far"))
    noise = quiet.p(50) if quiet else None
    for name, tk in (("ambiente", quiet), ("caminando", walk)):
        # Ruido estacionario: p90 cerca de p50. Ráfagas (p90 >> p50) = alguien
        # habló o hubo golpes: niveles y sectores de ese paso no son confiables.
        if tk and tk.p(90) > 4 * max(tk.p(50), 1e-4):
            notes.append(f"en '{name}' hubo ráfagas fuertes (¿alguien habló?): p90={tk.p(90):.4f} "
                         f"vs p50={tk.p(50):.4f}; si fue voz, repetí la calibración en silencio "
                         "(los sectores bloqueados podrían ser personas)")
    if quiet:
        say(f"  ambiente:  p50={quiet.p(50):.4f} p90={quiet.p(90):.4f}")
    if walk:
        say(f"  caminando: p50={walk.p(50):.4f} p90={walk.p(90):.4f}")
    if near:
        say(f"  cerca:     p50={near.p(50):.4f} p90={near.p(90):.4f}")
    if far:
        say(f"  lejos:     p50={far.p(50):.4f} p90={far.p(90):.4f}")

    # Frente del robot.
    offset = 0.0
    spread_in = None
    if array and near:
        if len(near.chip_angles) >= 5:
            front = dominant_direction([float(a) for a in near.chip_angles])
            offset = round(front) % 360
            values["DOA_FORWARD_OFFSET_DEG"] = f"{offset:.0f}"
            spread_in = sum(ang_diff(a, front) <= DOA_TOLERANCE_DEG for a in near.chip_angles) / len(near.chip_angles)
            say(f"  frente: el chip marca {front:.0f}° cuando le hablás de frente "
                f"({len(near.chip_angles)} lecturas, {spread_in:.0%} dentro de ±{DOA_TOLERANCE_DEG:.0f}°)")
            # 70 % de lo que logra quien le habla de frente, entre 0.25 y 0.5.
            values["DOA_MIN_IN_FOCUS"] = f"{min(0.5, max(0.25, 0.7 * spread_in)):.2f}"
            if spread_in < 0.4:
                notes.append(f"la dirección se dispersa mucho ({spread_in:.0%} en foco): sala muy "
                             "reverberante o ruido fuerte; el foco por dirección va a ser permisivo")
        else:
            notes.append("paso 'cerca' con muy pocas lecturas de voz del chip: no calibré el frente")

    # Sectores de ruido propio (ambiente + caminando).
    if array and (quiet or walk):
        noise_angles = (quiet.chip_angles if quiet else []) + (walk.chip_angles if walk else [])
        sectors = noise_sectors(noise_angles, offset)
        values["DOA_BLOCKED_SECTORS"] = fmt_sectors(sectors)
        say(f"  ruido propio: {len(noise_angles)} lecturas 'con voz' sin nadie hablando -> "
            f"{fmt_sectors(sectors) or 'ningún sector fijo'}")
        if sectors and overlaps_front(sectors, DOA_TOLERANCE_DEG):
            notes.append("un sector de ruido cae sobre el FRENTE del robot: lo saco "
                         "(bloquearía a quien le habla); revisá el montaje")
            values["DOA_BLOCKED_SECTORS"] = fmt_sectors(
                [s for s in sectors if not overlaps_front([s], DOA_TOLERANCE_DEG)])
    if array:
        spk = takes.get("speaker")
        if spk:
            spk_sec = noise_sectors(spk.chip_angles, offset)
            values["DOA_SPEAKER_SECTOR"] = fmt_sectors(spk_sec)
            say(f"  parlante: {fmt_sectors(spk_sec) or 'no se ubicó (¿sonó?)'}")

    # Niveles.
    if near and near.p(90) < max(0.02, 1.5 * (noise or 0.0)):
        notes.append(f"en 'cerca' casi no hubo voz (p90={near.p(90):.4f}): no calibro niveles "
                     "ni el frente; repetí ese paso hablando todo el tiempo")
        near = None
        for key in ("DOA_FORWARD_OFFSET_DEG", "DOA_MIN_IN_FOCUS"):
            values.pop(key, None)
    if near:
        p90_near = near.p(90)
        # Techo de TODO umbral de nivel: por encima del 60 % de tu voz cercana
        # el robot te ignora. El ruido (ambiente o de marcha) nunca empuja los
        # umbrales sobre esto: si se acerca, se avisa (el trabajo lo tienen que
        # hacer la dirección y el resto del pipeline, no el nivel).
        ceiling = p90_near * 0.6
        if far:
            p90_far = far.p(90)
            if p90_far * 2 <= ceiling:
                near_thr = p90_far * 2
            else:
                near_thr = min(ceiling, float(np.sqrt(p90_far * p90_near)))
                notes.append(f"cerca ({p90_near:.4f}) y lejos ({p90_far:.4f}) están muy parejos: "
                             "el nivel separa poco, el trabajo lo hace la dirección")
        else:
            near_thr = p90_near * 0.5
            notes.append("sin paso 'lejos': NEAR_RMS_THRESHOLD = mitad del nivel cerca")
        if noise is not None and noise * 2 > near_thr:
            # Ambiente quieto: el umbral tiene que quedar sobre él, sin pasar el techo.
            near_thr = min(ceiling, noise * 2)
            if noise * 2 > ceiling:
                notes.append(f"el ambiente quieto ({noise:.4f}) ya está cerca de tu voz: "
                             "acercá el mic a quien habla o alejalo de los ventiladores")
        values["NEAR_RMS_THRESHOLD"] = f"{near_thr:.4f}"
        if noise is not None:
            # Piso absoluto para abrir: holgado sobre el ambiente (mediana:
            # robusta a una voz suelta durante el paso).
            values["RMS_THRESHOLD"] = f"{min(near_thr * 0.6, max(0.005, quiet.p(50) * 3)):.4f}"

        # Con ruido, el umbral efectivo es max(NEAR_RMS_THRESHOLD, piso ×
        # NEAR_SNR_RATIO), y el piso puede subir hasta NOISE_FLOOR_MAX. Ese
        # máximo también tiene que quedar bajo el techo, o caminando te ignora.
        snr = NEAR_SNR_RATIO
        floor_max = NOISE_FLOOR_MAX
        if walk:
            walk_floor = walk.p(50)
            snr = min(3.0, max(1.5, 0.5 * p90_near / max(walk_floor, 1e-4)))
            floor_max = max(0.01, walk_floor * 1.5)
            if walk_floor * snr > ceiling:
                notes.append(f"caminando, el ruido ({walk_floor:.4f}) llega casi a tu voz "
                             f"({p90_near:.4f}): con el robot en marcha hay que hablarle más "
                             "cerca/fuerte (o aislar el mic de la vibración, ver README: «Ruido del robot caminando»)")
        floor_max = min(floor_max, ceiling / snr)
        values["NEAR_SNR_RATIO"] = f"{snr:.1f}"
        values["NOISE_FLOOR_MAX"] = f"{floor_max:.4f}"
        say(f"  techo de los umbrales (60 % de tu voz cercana): {ceiling:.4f}")
    elif "near" not in takes:
        notes.append("sin paso 'cerca' no puedo calibrar los niveles")

    if wake_hits is not None:
        say(f"  wake: el spotter reconoció {wake_hits}/3 «oye rai»")
        if wake_hits < 2:
            notes.append("el spotter reconoce poco el «oye rai»: hablá más claro/cerca o agregá "
                         "variantes a WAKE_PHRASES (mirá `python wake_spotter.py`)")

    if array:
        array.stop_polling()

    # Los umbrales de nivel van por modo de mic (config.MIC_MODE): los del
    # array se guardan con sufijo _ARRAY y no pisan los del mic común.
    if MIC_MODE == "array":
        values = {(f"{k}_ARRAY" if k in _PER_MIC_KEYS else k): v for k, v in values.items()}
    say()
    if values:
        say("Valores calibrados:")
        for k, v in values.items():
            say(f"  {k}={v}")
    for n in notes:
        say(f"  ! {n}")
    if not values:
        say("No hay nada para guardar.")
        return 1
    say()
    if args.dry_run:
        say("(--dry-run: no escribo .env)")
        return 0
    if not args.yes:
        try:
            ok_save = input(f"¿Guardar en {ENV_PATH}? [S/n] ").strip().lower() in ("", "s", "si", "sí", "y")
        except EOFError:
            ok_save = False
        if not ok_save:
            say("No guardé nada.")
            return 0
    if os.path.exists(ENV_PATH):
        backup = f"{ENV_PATH}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(ENV_PATH, backup)
        say(f"Backup del .env anterior: {backup}")
    update_env(ENV_PATH, values)
    say(f"Guardado. Reiniciá main.py para que tome los valores.")
    return 0


if __name__ == "__main__":
    import sounddevice as sd
    try:
        sys.exit(main())
    except sd.PortAudioError as exc:
        say(f"\n✖ No pude abrir el mic: {exc}")
        say("  ¿Está conectado (`arecord -l`)? ¿Lo tiene abierto otro programa (main.py,")
        say("  el servicio rai26-stt)? Usá ./calibrate_stt.sh. No guardé nada.")
        sys.exit(1)
    except KeyboardInterrupt:
        say("\nCancelado: no guardé nada.")
        sys.exit(130)
