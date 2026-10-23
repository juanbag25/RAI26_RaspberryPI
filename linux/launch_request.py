"""Arranque por voz: «rai, arriba» con el stack de la Jetson sin levantar.

La Pi corre siempre (servicio rai26-stt), aunque el orquestador no esté. Al
oír una frase de LAUNCH_PHRASES (spotter de audio; en WAKE_MODE=text, en el
texto) se le pide el arranque al launcher de la Jetson
(R-AI-026/orchestrator/launcher.py, LAUNCHER_PORT, mismo protocol
length-prefixed). Contesta:

    STARTING / BUSY -> arranca (o ya estaba arrancando): la Pi vuelve a
                       dormirse; no hay orquestador que conteste «Sí, dime».
    RUNNING         -> el orquestador ya corre: la frase vale como «oye rai».

Al terminar, el launcher avisa por el puerto de control (CTRL_PORT):
`ORCH_READY` (levantó: se resetea el backoff del envío y la Pi queda dormida
esperando «oye rai») o `LAUNCH_FAILED` (si fue el preflight, el robot ya
lo dijo por el parlante, con SPEAK_START/END a esta Pi).

Sin launcher (dev_orchestrator.sh apuntando a la PC, Jetson apagada) la
conexión falla, se loguea y la frase queda como un wake común: no rompe nada.
"""

from __future__ import annotations

import socket
import struct
import threading
from typing import Callable

from log import dim, event, info, ok, warn
from wake_word import normalize

CONNECT_TIMEOUT_S = 1.5
REPLY_TIMEOUT_S = 3.0


def _recv_exact(conn: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class LaunchRequest:
    def __init__(self, hosts: list[str], port: int, phrases: tuple[str, ...],
                 on_starting: Callable[[], None] | None = None,
                 on_ready: Callable[[], None] | None = None) -> None:
        self.hosts = hosts
        self.port = port
        self._phrases = tuple(p for p in (normalize(p) for p in phrases) if p)
        self._on_starting = on_starting
        self._on_ready = on_ready
        self._lock = threading.Lock()
        self._in_flight = False

    @property
    def enabled(self) -> bool:
        return bool(self._phrases and self.hosts)

    def describe(self) -> str:
        return ", ".join(f"{h}:{self.port}" for h in self.hosts)

    def matches(self, text: str) -> bool:
        """¿El texto (Whisper, modo texto) contiene una frase de arranque?"""
        padded = f" {normalize(text)} "
        return any(f" {p} " in padded for p in self._phrases)

    def request(self, heard: str) -> None:
        """Pide el arranque en un hilo aparte (no frena el loop de captura)."""
        if not self.enabled:
            return
        with self._lock:
            if self._in_flight:
                return
            self._in_flight = True
        threading.Thread(target=self._run, args=(heard,), name="launch", daemon=True).start()

    def _run(self, heard: str) -> None:
        try:
            host, reply = self._ask()
        finally:
            with self._lock:
                self._in_flight = False
        event("launch", heard=heard, host=host, reply=reply)
        if reply is None:
            warn("LAUNCH", f"«{heard}»: no contestó ningún launcher ({self.describe()}); "
                 "si el orquestador corre, vale como «oye rai»")
        elif reply == "RUNNING":
            dim("LAUNCH", f"«{heard}»: el orquestador ya corre, lo tomo como «oye rai»")
        elif reply in ("STARTING", "BUSY"):
            ok("LAUNCH", f"«{heard}»: {'ARRANCANDO' if reply == 'STARTING' else 'ya estaba arrancando'}"
               f" el stack en {host} (me duermo hasta que esté)")
            if self._on_starting is not None:
                self._on_starting()
        else:
            warn("LAUNCH", f"respuesta desconocida del launcher {host}: {reply!r}")

    def _ask(self) -> tuple[str | None, str | None]:
        data = b"LAUNCH"
        for host in self.hosts:
            try:
                with socket.create_connection((host, self.port), timeout=CONNECT_TIMEOUT_S) as s:
                    s.settimeout(REPLY_TIMEOUT_S)
                    s.sendall(struct.pack("!I", len(data)) + data)
                    header = _recv_exact(s, 4)
                    if header is None:
                        continue
                    (length,) = struct.unpack("!I", header)
                    payload = _recv_exact(s, min(length, 256))
                    if payload is not None:
                        return host, payload.decode("utf-8", errors="replace").strip()
            except OSError as exc:
                dim("LAUNCH", f"launcher {host}:{self.port}: {exc}")
        return None, None

    # -- avisos del launcher por CTRL_PORT (ctrl_server.py) --------------------

    def ready(self) -> None:
        ok("LAUNCH", "el launcher levantó el orquestador: decí «oye rai»")
        event("launch_done", ok=True)
        if self._on_ready is not None:
            self._on_ready()

    def failed(self) -> None:
        warn("LAUNCH", "el arranque falló (ver ~/logs/*_autostart.log en la Jetson)")
        event("launch_done", ok=False)
        info("LAUNCH", "sigo dormido; «rai, arriba» para reintentar")
