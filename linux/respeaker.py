"""Control del ReSpeaker USB Mic Array v2.0 (XMOS XVF-3000) por USB.

El audio llega por ALSA como cualquier mic (audio_capture.py, 6 canales). Este
módulo habla con la OTRA interfaz del dispositivo, la de control (vendor
requests por USB), que es la que da lo que un mic común no tiene:

- Parámetros del DSP (AGC, supresión de ruido, eco, filtro pasa-altos...).
  Se PIERDEN al cortarle la alimentación: `apply_params()` los fija en cada
  arranque (valores en config.RESPEAKER_PARAMS).
- `DOAANGLE`: dirección de la fuente de voz dominante (0-359°) y
  `VOICEACTIVITY`: VAD del chip. Un hilo los lee continuamente y guarda
  muestras con timestamp; doa.py las usa para saber de dónde vino cada frase.
  Ojo (medido): en silencio DOAANGLE queda pegado en el último valor o sigue
  al ruido; sólo valen las lecturas con VOICEACTIVITY=1.
- El anillo de 12 LEDs (protocolo de pixel_ring v2).

Tabla de parámetros y protocolo portados de respeaker/usb_4_mic_array
(`tuning.py`) y respeaker/pixel_ring (`usb_pixel_ring_v2.py`), ambos
Apache-2.0. El `tuning.py` original no anda en Python >= 3.9
(`array.tostring()`); acá se usa `.tobytes()`.

Requiere pyusb y permiso sobre el USB (regla udev `99-respeaker.rules`, ver
README). Uso suelto:

    python respeaker.py            # lee y muestra todos los parámetros
    python respeaker.py doa        # DoA + VAD en vivo
    python respeaker.py NOMBRE     # lee un parámetro
    python respeaker.py NOMBRE V   # escribe un parámetro (hasta el próximo corte)
"""

from __future__ import annotations

import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

from log import dim, err, info, ok, warn

VENDOR_ID = 0x2886
PRODUCT_ID = 0x0018

# name: (id, offset, tipo, max, min, rw/ro, descripción)
PARAMETERS: dict[str, tuple] = {
    "AECFREEZEONOFF": (18, 7, "int", 1, 0, "rw", "Adaptive Echo Canceler updates inhibit (0 adapt, 1 freeze)"),
    "AECNORM": (18, 19, "float", 16, 0.25, "rw", "Limit on norm of AEC filter coefficients"),
    "AECPATHCHANGE": (18, 25, "int", 1, 0, "ro", "AEC Path Change Detection"),
    "RT60": (18, 26, "float", 0.9, 0.25, "ro", "Current RT60 estimate in seconds"),
    "HPFONOFF": (18, 27, "int", 3, 0, "rw", "High-pass filter (0 off, 1 70 Hz, 2 125 Hz, 3 180 Hz)"),
    "RT60ONOFF": (18, 28, "int", 1, 0, "rw", "RT60 estimation for AES"),
    "AECSILENCELEVEL": (18, 30, "float", 1, 1e-09, "rw", "AEC signal detection threshold"),
    "AECSILENCEMODE": (18, 31, "int", 1, 0, "ro", "AEC far-end silence detection status"),
    "AGCONOFF": (19, 0, "int", 1, 0, "rw", "Automatic Gain Control (0 off, 1 on)"),
    "AGCMAXGAIN": (19, 1, "float", 1000, 1, "rw", "Maximum AGC gain factor (31.6 = 30 dB)"),
    "AGCDESIREDLEVEL": (19, 2, "float", 0.99, 1e-08, "rw", "AGC target output power"),
    "AGCGAIN": (19, 3, "float", 1000, 1, "rw", "Current AGC gain factor"),
    "AGCTIME": (19, 4, "float", 1, 0.1, "rw", "AGC ramp time constant (s)"),
    "CNIONOFF": (19, 5, "int", 1, 0, "rw", "Comfort Noise Insertion"),
    "FREEZEONOFF": (19, 6, "int", 1, 0, "rw", "Adaptive beamformer updates (0 adapt, 1 freeze)"),
    "STATNOISEONOFF": (19, 8, "int", 1, 0, "rw", "Stationary noise suppression"),
    "GAMMA_NS": (19, 9, "float", 3, 0, "rw", "Over-subtraction, stationary noise"),
    "MIN_NS": (19, 10, "float", 1, 0, "rw", "Gain floor, stationary noise suppression"),
    "NONSTATNOISEONOFF": (19, 11, "int", 1, 0, "rw", "Non-stationary noise suppression"),
    "GAMMA_NN": (19, 12, "float", 3, 0, "rw", "Over-subtraction, non-stationary noise"),
    "MIN_NN": (19, 13, "float", 1, 0, "rw", "Gain floor, non-stationary noise suppression"),
    "ECHOONOFF": (19, 14, "int", 1, 0, "rw", "Echo suppression"),
    "GAMMA_E": (19, 15, "float", 3, 0, "rw", "Over-subtraction, echo (direct/early)"),
    "GAMMA_ETAIL": (19, 16, "float", 3, 0, "rw", "Over-subtraction, echo (tail)"),
    "GAMMA_ENL": (19, 17, "float", 5, 0, "rw", "Over-subtraction, non-linear echo"),
    "NLATTENONOFF": (19, 18, "int", 1, 0, "rw", "Non-linear echo attenuation"),
    "NLAEC_MODE": (19, 20, "int", 2, 0, "rw", "Non-linear AEC training mode"),
    "SPEECHDETECTED": (19, 22, "int", 1, 0, "ro", "Speech detection status"),
    "FSBUPDATED": (19, 23, "int", 1, 0, "ro", "FSB update decision"),
    "FSBPATHCHANGE": (19, 24, "int", 1, 0, "ro", "FSB path change detection"),
    "TRANSIENTONOFF": (19, 29, "int", 1, 0, "rw", "Transient echo suppression"),
    "VOICEACTIVITY": (19, 32, "int", 1, 0, "ro", "VAD voice activity status"),
    "STATNOISEONOFF_SR": (19, 33, "int", 1, 0, "rw", "Stationary noise suppression for ASR"),
    "NONSTATNOISEONOFF_SR": (19, 34, "int", 1, 0, "rw", "Non-stationary noise suppression for ASR"),
    "GAMMA_NS_SR": (19, 35, "float", 3, 0, "rw", "Over-subtraction, stationary noise, ASR"),
    "GAMMA_NN_SR": (19, 36, "float", 3, 0, "rw", "Over-subtraction, non-stationary noise, ASR"),
    "MIN_NS_SR": (19, 37, "float", 1, 0, "rw", "Gain floor, stationary noise, ASR"),
    "MIN_NN_SR": (19, 38, "float", 1, 0, "rw", "Gain floor, non-stationary noise, ASR"),
    "GAMMAVAD_SR": (19, 39, "float", 1000, 0, "rw", "VAD threshold (dB)"),
    "DOAANGLE": (21, 0, "int", 359, 0, "ro", "DOA angle (orientation depends on mounting)"),
}

_TIMEOUT_MS = 1000

# Comandos del anillo de LEDs (pixel_ring v2).
_LED_TRACE = 0    # el firmware muestra VAD + DoA solo
_LED_MONO = 1
_LED_LISTEN = 2
_LED_THINK = 4
_LED_BRIGHTNESS = 0x20


@dataclass(frozen=True)
class DoaSample:
    t: float        # time.monotonic() de la lectura
    angle: int      # DOAANGLE crudo del chip (0-359)
    voice: bool     # VOICEACTIVITY en ese momento


class ReSpeaker:
    """Interfaz de control. `open()` devuelve None si no está conectado."""

    def __init__(self, dev) -> None:
        import usb.util

        self._dev = dev
        self._usb_util = usb.util
        self._io_lock = threading.Lock()  # los ctrl_transfer no son thread-safe
        self._samples: deque[DoaSample] = deque(maxlen=600)
        self._samples_lock = threading.Lock()
        self._poller: threading.Thread | None = None
        self._running = threading.Event()
        self.poll_hz = 0.0  # tasa real de lectura medida (para el heartbeat)

    @classmethod
    def open(cls) -> "ReSpeaker | None":
        try:
            import usb.core
        except ImportError:
            warn("ARRAY", "pyusb no está instalado (pip install pyusb): sin DoA ni parámetros DSP")
            return None
        try:
            dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
        except Exception as exc:  # noqa: BLE001 - backend libusb ausente, etc.
            warn("ARRAY", f"no pude buscar el ReSpeaker por USB: {exc}")
            return None
        if dev is None:
            return None
        array = cls(dev)
        try:
            array.read("DOAANGLE")
        except Exception as exc:  # noqa: BLE001
            err("ARRAY", f"ReSpeaker encontrado pero no responde por USB: {exc} "
                "(¿permisos? instalá la regla udev 99-respeaker.rules, ver README)")
            return None
        return array

    # -- parámetros -----------------------------------------------------------

    def _ctrl(self, request_type: int, value: int, index: int, data_or_len):
        with self._io_lock:
            return self._dev.ctrl_transfer(request_type, 0, value, index, data_or_len, _TIMEOUT_MS)

    def read(self, name: str) -> float | int:
        pid, offset, kind = PARAMETERS[name][:3]
        cmd = 0x80 | offset | (0x40 if kind == "int" else 0)
        u = self._usb_util
        resp = self._ctrl(u.CTRL_IN | u.CTRL_TYPE_VENDOR | u.CTRL_RECIPIENT_DEVICE, cmd, pid, 8)
        mantissa, exponent = struct.unpack("ii", resp.tobytes())
        return mantissa if kind == "int" else mantissa * (2.0 ** exponent)

    def write(self, name: str, value: float | int) -> None:
        pid, offset, kind, vmax, vmin, access = PARAMETERS[name][:6]
        if access != "rw":
            raise ValueError(f"{name} es de sólo lectura")
        if not vmin <= float(value) <= vmax:
            raise ValueError(f"{name}={value} fuera de rango [{vmin}, {vmax}]")
        if kind == "int":
            payload = struct.pack("iii", offset, int(value), 1)
        else:
            payload = struct.pack("ifi", offset, float(value), 0)
        u = self._usb_util
        self._ctrl(u.CTRL_OUT | u.CTRL_TYPE_VENDOR | u.CTRL_RECIPIENT_DEVICE, 0, pid, payload)

    def apply_params(self, params: dict[str, float]) -> None:
        """Fija los parámetros DSP (no persisten en el dispositivo)."""
        applied = []
        for name, value in params.items():
            try:
                self.write(name, value)
                applied.append(f"{name}={value:g}")
            except Exception as exc:  # noqa: BLE001 - un parámetro malo no frena el resto
                warn("ARRAY", f"no pude fijar {name}={value}: {exc}")
        if applied:
            dim("ARRAY", "DSP: " + " ".join(applied))

    # -- LEDs -----------------------------------------------------------------

    def _led(self, cmd: int, data: list[int] | None = None) -> None:
        u = self._usb_util
        try:
            self._ctrl(u.CTRL_OUT | u.CTRL_TYPE_VENDOR | u.CTRL_RECIPIENT_DEVICE,
                       cmd, 0x1C, data or [0])
        except Exception as exc:  # noqa: BLE001 - los LEDs son cosméticos
            warn("ARRAY", f"LEDs: {exc}")

    def leds_listen(self) -> None:
        """Despierto: el firmware ilumina hacia donde viene la voz."""
        self._led(_LED_LISTEN)

    def leds_think(self) -> None:
        self._led(_LED_THINK)

    def leds_off(self) -> None:
        self._led(_LED_MONO, [0, 0, 0, 0])

    def leds_brightness(self, value: int) -> None:
        self._led(_LED_BRIGHTNESS, [max(0, min(31, int(value)))])

    # -- DoA ------------------------------------------------------------------

    def start_polling(self, hz: float) -> None:
        """Hilo que lee DOAANGLE + VOICEACTIVITY ~`hz` veces por segundo."""
        if self._poller is not None:
            return
        self._running.set()
        self._poller = threading.Thread(target=self._poll, args=(hz,),
                                        name="respeaker-doa", daemon=True)
        self._poller.start()

    def stop_polling(self) -> None:
        """Frena el hilo y espera a que termine la lectura en curso (si no, al
        salir del proceso pyusb libera el dispositivo bajo sus pies)."""
        self._running.clear()
        if self._poller is not None:
            self._poller.join(timeout=1.0)
            self._poller = None

    def _poll(self, hz: float) -> None:
        period = 1.0 / hz if hz > 0 else 0.0
        count, window_start = 0, time.monotonic()
        failures = 0
        while self._running.is_set():
            t0 = time.monotonic()
            try:
                voice = bool(self.read("VOICEACTIVITY"))
                angle = int(self.read("DOAANGLE"))
                failures = 0
            except Exception as exc:  # noqa: BLE001 - USB desconectado, etc.
                failures += 1
                if failures == 1:
                    err("ARRAY", f"no puedo leer el DoA: {exc}")
                if failures >= 20:
                    err("ARRAY", "el ReSpeaker no responde: sigo sin DoA")
                    self._running.clear()
                    return
                time.sleep(0.5)
                continue
            # El timestamp es el del medio de las dos lecturas: el chip
            # calcula el DoA sobre el audio de ese instante.
            sample = DoaSample(t=(t0 + time.monotonic()) / 2, angle=angle, voice=voice)
            with self._samples_lock:
                self._samples.append(sample)
            count += 1
            now = time.monotonic()
            if now - window_start >= 2.0:
                self.poll_hz = count / (now - window_start)
                count, window_start = 0, now
            sleep = period - (now - t0)
            if sleep > 0:
                time.sleep(sleep)

    @property
    def polling(self) -> bool:
        return self._running.is_set()

    def samples_between(self, t0: float, t1: float) -> list[DoaSample]:
        with self._samples_lock:
            return [s for s in self._samples if t0 <= s.t <= t1]

    def latest(self) -> DoaSample | None:
        with self._samples_lock:
            return self._samples[-1] if self._samples else None


def _main(argv: list[str]) -> int:
    array = ReSpeaker.open()
    if array is None:
        err("ARRAY", "no encontré el ReSpeaker (2886:0018). ¿Conectado? `lsusb | grep 2886`")
        return 1
    if not argv:
        for name in sorted(PARAMETERS):
            try:
                value = array.read(name)
            except Exception as exc:  # noqa: BLE001
                value = f"ERROR {exc}"
            access, desc = PARAMETERS[name][5], PARAMETERS[name][6]
            shown = f"{value:.6g}" if isinstance(value, float) else str(value)
            print(f"{name:22s} {shown:>14}  {access}  {desc}")
        return 0
    if argv[0] == "doa":
        info("ARRAY", "DoA en vivo (sólo cuenta con voz=1). Ctrl+C para salir.")
        array.start_polling(20)
        try:
            while True:
                time.sleep(0.25)
                s = array.latest()
                if s:
                    mark = "VOZ" if s.voice else "   "
                    print(f"{mark} {s.angle:3d}°  |{' ' * (s.angle // 10)}^{' ' * (36 - s.angle // 10)}|"
                          f"  {array.poll_hz:4.1f} lecturas/s", flush=True)
        except KeyboardInterrupt:
            return 0
    name = argv[0].upper()
    if name not in PARAMETERS:
        err("ARRAY", f"parámetro desconocido: {name}")
        return 1
    if len(argv) > 1:
        try:
            array.write(name, float(argv[1]))
        except ValueError as exc:
            err("ARRAY", str(exc))
            return 1
        ok("ARRAY", f"{name} = {array.read(name)} (hasta el próximo corte de alimentación)")
    else:
        print(f"{name} = {array.read(name)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(_main(sys.argv[1:]))
    except KeyboardInterrupt:
        pass
