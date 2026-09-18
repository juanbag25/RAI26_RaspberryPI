"""Servidor de control de mute: el orquestador avisa cuándo habla el robot.

El orquestador manda, con el mismo protocolo TCP length-prefixed del resto
del sistema ([uint32 big-endian][payload UTF-8]), dos mensajes:

    SPEAK_START  -> el robot empieza a hablar: silenciar el mic
    SPEAK_END    -> terminó: volver a escuchar

`SpeakMute.is_muted()` se consulta desde el loop de captura de main.py. Si un
SPEAK_END se pierde (red, crash del orquestador), el mute expira solo a los
MUTE_TIMEOUT_S segundos para no dejar el robot sordo.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Callable

from config import MUTE_TIMEOUT_S
from log import err, info, warn

_MAX_MESSAGE_BYTES = 1024


class SpeakMute:
    def __init__(self, timeout_s: float = MUTE_TIMEOUT_S,
                 on_speak_end: Callable[[], None] | None = None) -> None:
        self._timeout = timeout_s
        # Se llama cuando el robot termina de hablar (o expira el mute): lo usa
        # el wake word para renovar la ventana de conversación.
        self._on_speak_end = on_speak_end
        self._muted = threading.Event()
        self._deadline = 0.0

    def is_muted(self) -> bool:
        if not self._muted.is_set():
            return False
        if time.monotonic() > self._deadline:
            self._muted.clear()
            warn("CTRL", "SPEAK_END perdido: desmuteo por timeout")
            self._notify_speak_end()
            return False
        return True

    def _notify_speak_end(self) -> None:
        if self._on_speak_end is None:
            return
        try:
            self._on_speak_end()
        except Exception as exc:  # noqa: BLE001 - nunca matar el loop de mic
            err("CTRL", f"on_speak_end: {exc}")

    def handle(self, message: str) -> None:
        if message == "SPEAK_START":
            self._deadline = time.monotonic() + self._timeout
            self._muted.set()
            info("CTRL", "robot habla: mic MUTEADO")
        elif message == "SPEAK_END":
            self._muted.clear()
            info("CTRL", "robot terminó: mic activo")
            self._notify_speak_end()
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
