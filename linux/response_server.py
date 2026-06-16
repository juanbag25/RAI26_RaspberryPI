"""TCP server that receives reply text from the C++ orchestrator.

Mirrors the wire protocol used by `send_to_orchestrator` in main.py:
    [4-byte uint32 length, big-endian][N bytes UTF-8 payload]

Each received message is handed to a callback (typically TtsPlayer.speak).
Runs in its own daemon thread so it doesn't block the capture loop.
"""

from __future__ import annotations

import socket
import struct
import threading
from typing import Callable

_MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _handle_connection(conn: socket.socket, on_text: Callable[[str], None]) -> None:
    with conn:
        header = _recv_exact(conn, 4)
        if header is None:
            return

        (length,) = struct.unpack("!I", header)
        if length == 0 or length > _MAX_MESSAGE_BYTES:
            return

        payload = _recv_exact(conn, length)
        if payload is None:
            return

        text = payload.decode("utf-8", errors="replace")
        print(f"<<< (orchestrator) {text}")
        try:
            on_text(text)
        except Exception as exc:  # noqa: BLE001 - never kill the server
            print(f"[RESPONSE SERVER ERROR] {exc}")


def serve_forever(port: int, on_text: Callable[[str], None]) -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(8)
    print(f"[RESPONSE SERVER] listening on 0.0.0.0:{port}")

    while True:
        conn, _addr = server.accept()
        _handle_connection(conn, on_text)


def start_in_background(port: int, on_text: Callable[[str], None]) -> threading.Thread:
    thread = threading.Thread(
        target=serve_forever,
        args=(port, on_text),
        daemon=True,
    )
    thread.start()
    return thread
