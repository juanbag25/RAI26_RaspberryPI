"""Reporte de alimentación de la Pi 5 (`vcgencmd`).

La Pi se alimenta por USB-C desde la salida USB-A del UPS (SunFounder
PiPower). Por ese cable no hay USB-PD ni I2C, así que no hay forma de leer la
batería del UPS desde acá; lo único observable es lo que ve la Pi:

- EXT5V_V: la tensión que realmente le llega. Si baja de ~4.8 V la Pi se va
  a reiniciar o a tirar el USB del mic.
- get_throttled: flags de undervoltage / throttling, ahora y desde el boot.

Uso desde main.py: `report_power()` al arrancar y cada POWER_LOG_S.
Suelto, para probar: `python battery.py`.
"""

from __future__ import annotations

import os
import subprocess

from log import dim, warn

# Cada cuánto re-loguear la alimentación mientras corre (0 = sólo al arrancar).
POWER_LOG_S = float(os.getenv("POWER_LOG_S", "300") or 0)


def _vcgencmd(*args: str) -> str | None:
    try:
        out = subprocess.run(["vcgencmd", *args], capture_output=True, text=True,
                             timeout=2.0)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _pi_power_lines() -> list[str]:
    lines: list[str] = []

    adc = _vcgencmd("pmic_read_adc")
    if adc:
        for raw in adc.splitlines():
            # "EXT5V_V volt(24)=5.10390000V"  (Pi 5)
            if raw.strip().startswith("EXT5V_V") and "=" in raw:
                try:
                    v = float(raw.split("=")[1].rstrip("V"))
                except ValueError:
                    break
                low = "  <-- BAJA (fuente/cable flojo, riesgo de reinicio)" if v < 4.8 else ""
                lines.append(f"entrada 5V real: {v:.2f} V{low}")
                break

    thr = _vcgencmd("get_throttled")
    if thr and "=" in thr:
        value = thr.split("=")[1]
        try:
            bits = int(value, 16)
        except ValueError:
            bits = 0
        flags = []
        if bits & 0x1:
            flags.append("UNDERVOLTAGE AHORA")
        if bits & 0x4:
            flags.append("throttled ahora")
        if bits & 0x8:
            flags.append("límite de temperatura")
        if bits & 0x10000:
            flags.append("hubo undervoltage desde el boot")
        if bits & 0x40000:
            flags.append("hubo throttling desde el boot")
        lines.append(f"throttled={value}" + (f"  [{', '.join(flags)}]" if flags else "  (ok)"))

    return lines


def report_power() -> None:
    """Loguea lo que la Pi sabe de su alimentación (una línea por dato)."""
    lines = _pi_power_lines()
    if not lines:
        dim("POWER", "sin datos de alimentación (¿vcgencmd no disponible?)")
    for line in lines:
        if "<--" in line or "UNDERVOLTAGE" in line or "throttled ahora" in line:
            warn("POWER", line)
        else:
            dim("POWER", line)


if __name__ == "__main__":
    report_power()
