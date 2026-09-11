"""Reporte de alimentación al arrancar: batería del UPS + lo que ve la Pi.

Dos fuentes de verdad, independientes:

1. **El UPS / power HAT** (por I2C). Se autodetecta entre los chips que usan
   los UPS más comunes para Pi 5; no hace falta configurar nada si es uno de
   estos, y si no lo es, `i2cdetect -y 1` dice qué dirección responde:

     - MAX17048/9 @0x36  -> Geekworm X1200/X1201/X1202/X1203 (fuel gauge real:
                            da % de carga y tasa de carga/descarga)
     - INA219   @0x40-45 -> Waveshare UPS HAT (B)/(C) y clones (mide tensión y
                            corriente; el % sale de la tensión, es aproximado)
     - PiSugar 3 @0x57   -> PiSugar 3 / 3 Plus

   Además, si el driver del kernel expone algo en /sys/class/power_supply/
   (PiJuice, UPS con driver propio), también se lee de ahí.

2. **La Pi 5 misma** (`vcgencmd`): la tensión de entrada que realmente le
   llega (EXT5V_V — si baja de ~4.8 V la Pi se va a reiniciar o a tirar el
   USB del mic), los flags de undervoltage/throttling y cuánta corriente
   negoció con la fuente por USB-PD (5 A = fuente oficial; 3 A = fuente
   genérica y los puertos USB quedan limitados a 600 mA).

Uso desde main.py: `report_power(ups)` al arrancar y cada BATTERY_LOG_S.
Suelto, para probar: `python battery.py`.

Requiere `smbus2` (pip) y el I2C habilitado (`sudo raspi-config` ->
Interface Options -> I2C, o `dtparam=i2c_arm=on` en /boot/firmware/config.txt).
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Callable

from log import log

I2C_BUS = int(os.getenv("UPS_I2C_BUS", "1") or 1)
# Forzar una dirección (hex, ej. 0x36) si el autodetect no la encuentra.
_ADDR_OVERRIDE = os.getenv("UPS_I2C_ADDR", "").strip()
# Cada cuánto re-loguear la batería mientras corre (0 = sólo al arrancar).
BATTERY_LOG_S = float(os.getenv("BATTERY_LOG_S", "300") or 0)

try:
    from smbus2 import SMBus  # type: ignore
except ImportError:  # pragma: no cover - en la PC de desarrollo no está
    SMBus = None  # type: ignore


@dataclass
class Ups:
    name: str
    addr: int
    read: Callable[[], str]  # devuelve una línea de estado o levanta OSError


# --- lectores por chip ---------------------------------------------------------

def _read_be16(bus: "SMBus", addr: int, reg: int) -> int:
    hi, lo = bus.read_i2c_block_data(addr, reg, 2)
    return (hi << 8) | lo


def _signed16(raw: int) -> int:
    return raw - 0x10000 if raw & 0x8000 else raw


def _max17048(bus: "SMBus", addr: int) -> Ups | None:
    # VERSION @0x08 = 0x0011 (MAX17048) / 0x0012 (MAX17049). Si no matchea,
    # hay otra cosa en 0x36 y no queremos leer basura como si fuera batería.
    version = _read_be16(bus, addr, 0x08)
    if version not in (0x0011, 0x0012):
        return None

    def read() -> str:
        vcell = _read_be16(bus, addr, 0x02) * 78.125e-6         # V
        soc = _read_be16(bus, addr, 0x04) / 256.0               # %
        crate = _signed16(_read_be16(bus, addr, 0x16)) * 0.208  # %/h, + = carga
        if crate > 0.5:
            state = f"cargando (+{crate:.1f} %/h)"
        elif crate < -0.5:
            state = f"descargando ({crate:.1f} %/h)"
        else:
            state = "estable / sin corriente neta"
        return f"batería {soc:.0f}%  {vcell:.2f} V  {state}"

    return Ups("MAX17048 fuel gauge (Geekworm X120x?)", addr, read)


def _ina219(bus: "SMBus", addr: int) -> Ups | None:
    # Con la config de fábrica (32 V, 320 mV, continuo) el registro de bus
    # voltage ya es válido sin calibrar. Waveshare usa shunt de 0.1 ohm.
    bus_v = (_read_be16(bus, addr, 0x02) >> 3) * 0.004
    if not (2.5 <= bus_v <= 17.0):
        return None

    def read() -> str:
        v = (_read_be16(bus, addr, 0x02) >> 3) * 0.004
        shunt_v = _signed16(_read_be16(bus, addr, 0x01)) * 10e-6
        current = shunt_v / 0.1  # A; convención Waveshare: + carga, - descarga
        # Estimación por tensión (curva lineal, como el ejemplo de Waveshare):
        # 2S (UPS HAT B: 6.0-8.4 V) o 1S (UPS HAT C: 3.0-4.2 V).
        if v > 5.0:
            soc = (v - 6.0) / 2.4 * 100.0
        else:
            soc = (v - 3.0) / 1.2 * 100.0
        soc = max(0.0, min(100.0, soc))
        if current > 0.05:
            state = f"cargando (+{current:.2f} A)"
        elif current < -0.05:
            state = f"descargando ({current:.2f} A)"
        else:
            state = "sin corriente neta"
        return f"batería ~{soc:.0f}% (estimado por tensión)  {v:.2f} V  {state}"

    return Ups("INA219 (Waveshare UPS HAT?)", addr, read)


def _pisugar3(bus: "SMBus", addr: int) -> Ups | None:
    mv = _read_be16(bus, addr, 0x22)
    if not (2500 <= mv <= 4500):
        return None

    def read() -> str:
        mv = _read_be16(bus, addr, 0x22)
        soc = bus.read_byte_data(addr, 0x2A)
        return f"batería {soc}%  {mv / 1000:.2f} V"

    return Ups("PiSugar 3", addr, read)


_PROBES: list[tuple[int, Callable]] = [
    (0x36, _max17048),
    (0x43, _ina219), (0x40, _ina219), (0x41, _ina219), (0x42, _ina219),
    (0x44, _ina219), (0x45, _ina219),
    (0x57, _pisugar3),
]


def probe_ups() -> Ups | None:
    """Busca un UPS conocido en el bus I2C. None si no hay (o no hay smbus2)."""
    if SMBus is None:
        log("[POWER] smbus2 no instalado (pip install smbus2): sin lectura de UPS")
        return None
    try:
        bus = SMBus(I2C_BUS)
    except (OSError, FileNotFoundError) as exc:
        log(f"[POWER] no se pudo abrir /dev/i2c-{I2C_BUS} ({exc}): "
            f"¿I2C habilitado en raspi-config?")
        return None

    probes = _PROBES
    if _ADDR_OVERRIDE:
        forced = int(_ADDR_OVERRIDE, 16)
        probes = [(forced, fn) for _a, fn in _PROBES]

    for addr, fn in probes:
        try:
            ups = fn(bus, addr)
        except OSError:
            continue  # nadie responde en esa dirección
        if ups is not None:
            log(f"[POWER] UPS detectado: {ups.name} en I2C 0x{addr:02x}")
            return ups
    log("[POWER] ningún UPS conocido en I2C (probá `i2cdetect -y 1` y "
        "UPS_I2C_ADDR=0x.. en .env)")
    return None


# --- /sys/class/power_supply -----------------------------------------------------

def _sysfs_batteries() -> list[str]:
    root = "/sys/class/power_supply"
    lines: list[str] = []
    if not os.path.isdir(root):
        return lines
    for name in sorted(os.listdir(root)):
        base = os.path.join(root, name)

        def rd(attr: str) -> str | None:
            try:
                with open(os.path.join(base, attr)) as f:
                    return f.read().strip()
            except OSError:
                return None

        if rd("type") != "Battery":
            continue
        cap, status, volt = rd("capacity"), rd("status"), rd("voltage_now")
        parts = [name]
        if cap:
            parts.append(f"{cap}%")
        if volt:
            parts.append(f"{int(volt) / 1e6:.2f} V")
        if status:
            parts.append(status)
        lines.append("  ".join(parts))
    return lines


# --- la Pi misma (vcgencmd) -----------------------------------------------------

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
                warn = "  <-- BAJA (fuente/cable flojo, riesgo de reinicio)" if v < 4.8 else ""
                lines.append(f"entrada 5V real: {v:.2f} V{warn}")
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

    # Corriente negociada por USB-PD con la fuente (Pi 5). 5000 mA = fuente
    # oficial 27 W; 3000 mA = fuente genérica (USB limitado a 600 mA total).
    try:
        with open("/sys/firmware/devicetree/base/chosen/power/max_current", "rb") as f:
            ma = int.from_bytes(f.read()[:4], "big")
        note = "" if ma >= 5000 else "  (fuente no reconocida como 5A: USB limitado)"
        lines.append(f"fuente negociada: {ma} mA{note}")
    except (OSError, ValueError):
        pass

    return lines


# --- API ----------------------------------------------------------------------

def report_power(ups: Ups | None) -> None:
    """Loguea todo lo que se sepa de la alimentación (una línea por dato)."""
    if ups is not None:
        try:
            log(f"[POWER] {ups.name}: {ups.read()}")
        except OSError as exc:
            log(f"[POWER] error leyendo el UPS ({exc})", err=True)
    for line in _sysfs_batteries():
        log(f"[POWER] power_supply: {line}")
    pi = _pi_power_lines()
    if pi:
        for line in pi:
            log(f"[POWER] Pi: {line}")
    elif ups is None:
        log("[POWER] sin datos de alimentación (ni UPS por I2C ni vcgencmd)")


if __name__ == "__main__":
    report_power(probe_ups())
