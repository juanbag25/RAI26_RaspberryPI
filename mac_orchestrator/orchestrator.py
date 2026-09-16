"""Orquestador mínimo para probar el loop completo en macOS.

Reemplaza, solo para pruebas locales, al orquestador real (R-AI-026/orchestrator
en la Jetson): recibe el texto transcripto del cliente STT (linux/main.py) por
TCP :9000, se lo manda al LLM server (llm-server-RAI-026, POST /v1/process),
y dice la respuesta por voz con `say`, avisando al cliente STT (SPEAK_START/
SPEAK_END por TCP :9001) para que se mutee mientras habla.

Protocolo TCP (igual que linux/main.py y linux/ctrl_server.py):
    [4 bytes big-endian = largo del payload][payload UTF-8]
"""

import os
import socket
import struct
import subprocess
import threading
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8000/v1/process")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "30"))

ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", "9000"))
STT_HOST = os.getenv("STT_HOST", "127.0.0.1")
CTRL_PORT = int(os.getenv("CTRL_PORT", "9001"))

TTS_VOICE = os.getenv("TTS_VOICE", "Mónica")
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))

_history: list[dict[str, str]] = []
_history_lock = threading.Lock()


def _log(msg: str) -> None:
    print(msg, flush=True)


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("conexión cerrada antes de tiempo")
        buf += chunk
    return buf


def _recv_message(conn: socket.socket) -> str:
    header = _recv_exact(conn, 4)
    (length,) = struct.unpack("!I", header)
    payload = _recv_exact(conn, length)
    return payload.decode("utf-8")


def _send_message(sock: socket.socket, text: str) -> None:
    data = text.encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)) + data)


def send_ctrl(message: str) -> None:
    try:
        with socket.create_connection((STT_HOST, CTRL_PORT), timeout=3.0) as s:
            _send_message(s, message)
    except OSError as exc:
        _log(f"[CTRL] no se pudo enviar {message}: {exc}")


def call_llm(transcript: str) -> dict[str, Any]:
    with _history_lock:
        history = list(_history)
    payload = {"transcript": transcript, "history": history}
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["X-API-Key"] = LLM_API_KEY
    resp = httpx.post(LLM_API_URL, json=payload, headers=headers, timeout=LLM_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


def speak(text: str) -> None:
    send_ctrl("SPEAK_START")
    try:
        subprocess.run(["say", "-v", TTS_VOICE, text], check=False)
    finally:
        send_ctrl("SPEAK_END")


def handle_utterance(transcript: str) -> None:
    transcript = transcript.strip()
    if not transcript:
        return
    _log(f"[STT] {transcript!r}")
    try:
        data = call_llm(transcript)
    except Exception as exc:
        _log(f"[LLM] error: {exc}")
        return

    instruction = data.get("instruction", {})
    speech = (instruction.get("speech") or "").strip()
    if not speech:
        _log("[LLM] respuesta sin texto para hablar")
        return

    with _history_lock:
        _history.append({"role": "user", "content": transcript})
        _history.append({"role": "assistant", "content": speech})
        overflow = len(_history) - MAX_HISTORY_TURNS * 2
        if overflow > 0:
            del _history[:overflow]

    _log(f"[LLM] {speech!r}")
    if instruction.get("command"):
        _log(f"[LLM] command (ignorado en esta prueba): {instruction['command']!r}")

    speak(speech)


def _handle_conn(conn: socket.socket, addr: tuple[str, int]) -> None:
    with conn:
        try:
            transcript = _recv_message(conn)
        except (ConnectionError, struct.error, OSError) as exc:
            _log(f"[ORCH] error leyendo mensaje de {addr}: {exc}")
            return
    handle_utterance(transcript)


def main() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", ORCHESTRATOR_PORT))
    server.listen(5)
    _log(
        f"[ORCH] escuchando en :{ORCHESTRATOR_PORT} | LLM={LLM_API_URL} | "
        f"CTRL={STT_HOST}:{CTRL_PORT} | voz={TTS_VOICE}"
    )
    try:
        while True:
            conn, addr = server.accept()
            _handle_conn(conn, addr)
    except KeyboardInterrupt:
        _log("[ORCH] cortado por el usuario")
    finally:
        server.close()


if __name__ == "__main__":
    main()
