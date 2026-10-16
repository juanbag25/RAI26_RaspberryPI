"""Servidor de control de mute: el orquestador avisa cuándo habla el robot.

El orquestador manda, con el mismo protocolo TCP length-prefixed del resto
del sistema ([uint32 big-endian][payload UTF-8]), dos mensajes:

    SPEAK_START  -> el robot empieza a hablar: silenciar el mic
    SPEAK_END    -> terminó: volver a escuchar (SPEAK_END_TAIL_S después)

`SpeakMute.is_muted()` se consulta desde el loop de captura de main.py. Si un
SPEAK_END se pierde (red, crash del orquestador), el mute expira solo a los
MUTE_TIMEOUT_S segundos para no dejar el robot sordo.

Cada turno del robot (primer SPEAK_START → desmute) deja un evento `speak` en
el log persistente: cuánto habló, cuántos SPEAK_START trajo, si cerró por
SPEAK_END o por timeout, cuánto tardó en arrancar desde lo que la Pi le mandó
(`reply_s`, ver `expect_reply`) y, si hubo orden de corte, cuánto siguió
hablando después (`after_stop_s`, ver `note_stop`).
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Callable

from config import MUTE_TIMEOUT_S, SPEAK_END_TAIL_S
from log import err, event, info, warn

_MAX_MESSAGE_BYTES = 1024


class SpeakMute:
    def __init__(self, timeout_s: float = MUTE_TIMEOUT_S,
                 on_speak_end: Callable[[], None] | None = None,
                 on_speak_start: Callable[[], None] | None = None) -> None:
        self._timeout = timeout_s
        # Se llama cuando el robot empieza a hablar: el wake word congela la
        # ventana de conversación mientras dura.
        self._on_speak_start = on_speak_start
        # Se llama cuando el robot termina de hablar (o expira el mute): lo usa
        # el wake word para renovar la ventana de conversación.
        self._on_speak_end = on_speak_end
        self._muted = threading.Event()
        self._deadline = 0.0
        # True entre el SPEAK_END y el fin de la cola (SPEAK_END_TAIL_S): el
        # vencimiento del deadline es normal, no un SPEAK_END perdido.
        self._ending = False
        # Métricas del turno en curso (evento `speak`).
        self._lock = threading.Lock()
        self._turn_start: float | None = None
        self._starts = 0
        self._stops = 0
        self._first_stop: float | None = None
        self._reply_s: float | None = None
        self._reply_kind: str | None = None
        self._expect: tuple[str, float] | None = None

    # Una respuesta que tarda más que esto ya no se atribuye al envío.
    _REPLY_MAX_S = 30.0

    def expect_reply(self, kind: str) -> None:
        """La Pi le mandó algo al orquestador que debería hacerlo hablar
        ("texto" = transcript, "ack" = wake/«Sí, dime»): el próximo
        SPEAK_START mide la latencia de punta a punta vista desde acá."""
        with self._lock:
            self._expect = (kind, time.monotonic())

    def note_stop(self) -> None:
        """Salió una orden de corte mientras el robot hablaba."""
        with self._lock:
            self._stops += 1
            if self._first_stop is None:
                self._first_stop = time.monotonic()

    def speaking_s(self) -> float | None:
        """Hace cuánto empezó a hablar el robot (None si no está hablando)."""
        start = self._turn_start
        return None if start is None else round(time.monotonic() - start, 2)

    def _turn_begin(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._starts += 1
            if self._turn_start is not None:
                return
            self._turn_start = now
            if self._expect is not None and now - self._expect[1] <= self._REPLY_MAX_S:
                self._reply_kind = self._expect[0]
                self._reply_s = round(now - self._expect[1], 3)
            self._expect = None

    def _turn_end(self, how: str) -> None:
        now = time.monotonic()
        with self._lock:
            if self._turn_start is None:
                return
            event("speak", end=how, dur_s=round(now - self._turn_start, 2),
                  starts=self._starts, reply=self._reply_kind, reply_s=self._reply_s,
                  stops=self._stops,
                  after_stop_s=(None if self._first_stop is None
                                else round(now - self._first_stop, 2)))
            self._turn_start = None
            self._starts = self._stops = 0
            self._first_stop = self._reply_s = self._reply_kind = None

    def is_muted(self) -> bool:
        if not self._muted.is_set():
            return False
        if time.monotonic() > self._deadline:
            self._muted.clear()
            if self._ending:
                info("CTRL", "robot terminó: mic activo")
                self._turn_end("end")
            else:
                warn("CTRL", "SPEAK_END perdido: desmuteo por timeout")
                self._turn_end("timeout")
            self._notify_speak_end()
            return False
        return True

    def _notify_speak_end(self) -> None:
        self._notify(self._on_speak_end, "on_speak_end")

    @staticmethod
    def _notify(callback: Callable[[], None] | None, name: str) -> None:
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:  # noqa: BLE001 - nunca matar el loop de mic
            err("CTRL", f"{name}: {exc}")

    def handle(self, message: str) -> None:
        if message == "SPEAK_START":
            self._ending = False
            self._deadline = time.monotonic() + self._timeout
            self._muted.set()
            self._turn_begin()
            info("CTRL", "robot habla: mic MUTEADO")
            self._notify(self._on_speak_start, "on_speak_start")
        elif message == "SPEAK_END":
            if SPEAK_END_TAIL_S <= 0 or not self._muted.is_set():
                self._muted.clear()
                info("CTRL", "robot terminó: mic activo")
                self._turn_end("end")
                self._notify_speak_end()
                return
            # Sigue muteado un rato: is_muted() desmutea (y avisa) al vencer.
            self._ending = True
            self._deadline = time.monotonic() + SPEAK_END_TAIL_S
        else:
            warn("CTRL", f"mensaje desconocido: {message!r}")


def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _serve_forever(port: int, mute: SpeakMute) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(8)

    while True:
        conn, addr = server.accept()
        with conn:
            header = _recv_exact(conn, 4)
            if header is None:
                warn("CTRL", f"conexión de {addr[0]} cerrada sin header")
                continue
            (length,) = struct.unpack("!I", header)
            if length == 0 or length > _MAX_MESSAGE_BYTES:
                warn("CTRL", f"largo inválido {length} desde {addr[0]}, ignorado")
                continue
            payload = _recv_exact(conn, length)
            if payload is None:
                warn("CTRL", f"conexión de {addr[0]} cortada a mitad del payload")
                continue
            try:
                mute.handle(payload.decode("utf-8", errors="replace"))
            except Exception as exc:  # noqa: BLE001 - nunca matar el server
                err("CTRL", str(exc))


def start_in_background(port: int, mute: SpeakMute) -> threading.Thread:
    thread = threading.Thread(target=_serve_forever, args=(port, mute), daemon=True)
    thread.start()
    return thread
