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

from config import MUTE_TIMEOUT_S

_MAX_MESSAGE_BYTES = 1024


class SpeakMute:
    def __init__(self, timeout_s: float = MUTE_TIMEOUT_S) -> None:
        self._timeout = timeout_s
        self._muted = threading.Event()
        self._deadline = 0.0

    def is_muted(self) -> bool:
        if not self._muted.is_set():
            return False
        if time.monotonic() > self._deadline:
            self._muted.clear()
            print("[CTRL] SPEAK_END perdido: desmuteo por timeout")
            return False
        return True

    def handle(self, message: str) -> None:
        if message == "SPEAK_START":
            self._deadline = time.monotonic() + self._timeout
            self._muted.set()
            print("[CTRL] robot hablando: mic muteado")
        elif message == "SPEAK_END":
            self._muted.clear()
            print("[CTRL] robot terminó: mic activo")
        else:
            print(f"[CTRL] mensaje desconocido: {message!r}")


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
        conn, _addr = server.accept()
        with conn:
            header = _recv_exact(conn, 4)
            if header is None:
                continue
            (length,) = struct.unpack("!I", header)
            if length == 0 or length > _MAX_MESSAGE_BYTES:
                continue
            payload = _recv_exact(conn, length)
            if payload is None:
                continue
            try:
                mute.handle(payload.decode("utf-8", errors="replace"))
            except Exception as exc:  # noqa: BLE001 - nunca matar el server
                print(f"[CTRL ERROR] {exc}")


def start_in_background(port: int, mute: SpeakMute) -> threading.Thread:
    thread = threading.Thread(target=_serve_forever, args=(port, mute), daemon=True)
    thread.start()
    return thread
