"""Métricas de rendimiento de la Pi (rai_perf.py / log.perf): `pi.utterance`,
`pi.reply` y el `turn_id` que viaja al orquestador. Sin mic, sin Groq, sin
orquestador real (uno falso por TCP en localhost).

    python3 -m unittest discover -s tests      (desde la raíz del repo)
"""

from __future__ import annotations

import importlib.util
import json
import os
import queue
import socket
import struct
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


class _Stub(types.ModuleType):
    """Módulo falso para las librerías que no hacen falta en estos tests
    (webrtcvad, groq): cualquier atributo es una clase vacía."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__init__": lambda self, *a, **k: None})


for _name in ("webrtcvad", "groq"):
    if importlib.util.find_spec(_name) is None:
        sys.modules[_name] = _Stub(_name)

os.environ["LOG_COLOR"] = "0"

import log  # noqa: E402
import rai_perf  # noqa: E402


def read_records(directory: str, prefix: str) -> list[dict]:
    rows = []
    for path in sorted(Path(directory).glob(f"{prefix}*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


class LogDirTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = (log._sink, log._perf)
        os.environ.pop("PERF_DIR", None)
        os.environ.pop("PERF_ENABLED", None)
        self.assertEqual(log.persist(self.tmp.name), self.tmp.name)

    def tearDown(self) -> None:
        if log._perf is not None:
            log._perf.close()
        if log._sink is not None and log._sink._file is not None:
            log._sink._file.close()
        log._sink, log._perf = self.saved
        self.tmp.cleanup()

    def perf_records(self, stage: str | None = None) -> list[dict]:
        rows = read_records(self.tmp.name, "perf-")
        return [r for r in rows if stage is None or r["stage"] == stage]


class LogPerfTest(LogDirTestCase):
    def test_perf_goes_to_its_own_file_with_the_common_schema(self) -> None:
        log.perf("pi.wake", accepted=True, reason=None, level=0.01)
        [rec] = self.perf_records()
        self.assertEqual((rec["v"], rec["svc"], rec["stage"]), (1, "stt", "pi.wake"))
        self.assertEqual(rec["run"], log._sink.run)  # mismo run que el log de la Pi
        self.assertIsNone(rec["ms"])
        self.assertTrue(rec["ts"].endswith("Z"))
        # El log de siempre (stt-*) no se mezcla con esto.
        self.assertEqual(read_records(self.tmp.name, "stt-"), [])

    def test_disabled(self) -> None:
        os.environ["PERF_ENABLED"] = "0"
        try:
            log.persist(self.tmp.name)
            log.perf("pi.wake", accepted=True)
        finally:
            os.environ.pop("PERF_ENABLED")
        self.assertEqual(self.perf_records(), [])


class SpeakMuteReplyTest(LogDirTestCase):
    def test_reply_carries_the_turn_of_the_transcript(self) -> None:
        import ctrl_server
        mute = ctrl_server.SpeakMute()
        mute.expect_reply("texto", "abc123")
        time.sleep(0.05)
        mute.handle("SPEAK_START")
        mute.handle("SPEAK_END")
        deadline = time.monotonic() + 3
        while mute.is_muted() and time.monotonic() < deadline:  # cola SPEAK_END_TAIL_S
            time.sleep(0.02)

        [rec] = self.perf_records("pi.reply")
        self.assertEqual(rec["turn"], "abc123")
        self.assertEqual((rec["kind"], rec["end"], rec["starts"], rec["stops"]), ("texto", "end", 1, 0))
        self.assertGreaterEqual(rec["ms"], 50)
        self.assertIsNone(rec["after_stop_ms"])

    def test_turn_without_expected_reply(self) -> None:
        import ctrl_server
        mute = ctrl_server.SpeakMute()
        mute._turn_begin()
        mute._turn_end("timeout")
        [rec] = self.perf_records("pi.reply")
        self.assertEqual((rec["ms"], rec["turn"], rec["kind"], rec["end"]), (None, None, None, "timeout"))


class FakeOrchestrator:
    """Recibe mensajes [uint32 BE][UTF-8] como el orquestador (:9000)."""

    def __init__(self) -> None:
        self.server = socket.socket()
        self.server.bind(("127.0.0.1", 0))
        self.server.listen()
        self.port = self.server.getsockname()[1]
        self.messages: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            with conn:
                size = struct.unpack("!I", conn.recv(4))[0]
                data = b""
                while len(data) < size:
                    data += conn.recv(size - len(data))
                self.messages.put(data.decode())

    def close(self) -> None:
        self.server.close()


class FakeTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text
        self.last_metrics: dict = {}

    def transcribe(self, audio):
        time.sleep(0.02)
        self.last_metrics = {"segs": 1, "segs_drop": 0, "logprob_min": -0.2,
                             "no_speech_max": 0.01, "compression_max": 1.1, "raw_text": None}
        return self.text, 0.93


class FakeWake:
    """Despierto, deja pasar todo."""

    def is_awake(self): return True
    def wake_count(self): return 0
    def window_deadline(self): return 0.0
    def restore_window(self, _deadline): pass
    def filter(self, text, _level=0.0): return text
    def await_reply(self): pass
    def strip_wake(self, text): return text
    def says_name(self, _text): return False


class FakeSpeaker:
    enabled = False

    def follow(self, *_a, **_k): pass


class TranscribeWorkerTest(LogDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        import main
        self.main = main
        self.orch = FakeOrchestrator()
        self.link = main.OrchestratorLink(["127.0.0.1"], self.orch.port)
        self.expected: list[tuple] = []
        self.link.on_reply_expected = lambda kind, turn: self.expected.append((kind, turn))

    def tearDown(self) -> None:
        self.orch.close()
        super().tearDown()

    def run_worker(self, text: str) -> None:
        import numpy as np
        audio_queue: "queue.Queue" = queue.Queue()
        audio = np.zeros(16000, dtype=np.float32)
        worker = threading.Thread(target=self.main.transcribe_worker, daemon=True, args=(
            audio_queue, FakeTranscriber(text), FakeWake(), self.link, self.main.Counters(),
            FakeSpeaker(), False, None, None))
        worker.start()
        audio_queue.put((audio, 0.05, time.monotonic(), False, None, None))
        deadline = time.monotonic() + 5
        while not self.perf_records("pi.utterance") and time.monotonic() < deadline:
            time.sleep(0.02)
        # El None (apagado) recién ahora: si ya está en la cola, el worker
        # sale sin procesar la frase (se queda sólo con la más nueva).
        audio_queue.put(None)
        worker.join(timeout=5)

    def test_sent_utterance_has_one_turn_id_everywhere(self) -> None:
        self.run_worker("decime la hora")

        payload = json.loads(self.orch.messages.get(timeout=3))
        turn = payload["turn_id"]
        self.assertRegex(turn, r"^[0-9a-f]{10}$")
        self.assertEqual(payload["text"], "decime la hora")
        self.assertIn("pi_latency_s", payload)
        self.assertEqual(self.expected, [("texto", turn)])

        [rec] = self.perf_records("pi.utterance")
        self.assertEqual((rec["turn"], rec["outcome"]), (turn, "enviada"))
        self.assertEqual((rec["audio_s"], rec["stt_calls"], rec["conf"]), (1.0, 1, 0.93))
        self.assertGreaterEqual(rec["stt_ms"], 20)
        self.assertGreaterEqual(rec["ms"], rec["stt_ms"])
        self.assertIsNotNone(rec["send_ms"])
        self.assertEqual(rec["chars"], len("decime la hora"))
        self.assertFalse(rec["trimmed"])
        self.assertNotIn("decime", json.dumps(rec))  # sin texto

        [stt] = [r for r in read_records(self.tmp.name, "stt-") if r.get("ev") == "stt"]
        self.assertEqual(stt["turn"], turn)

    def test_empty_transcription_is_recorded_without_sending(self) -> None:
        self.run_worker("")
        self.assertTrue(self.orch.messages.empty())
        [rec] = self.perf_records("pi.utterance")
        self.assertEqual(rec["outcome"], "vacia")
        self.assertIsNone(rec["send_ms"])
        self.assertIsNone(rec["chars"])


class SendTranscriptTest(unittest.TestCase):
    def test_creates_a_turn_id_when_none_is_given(self) -> None:
        import main
        orch = FakeOrchestrator()
        try:
            link = main.OrchestratorLink(["127.0.0.1"], orch.port)
            self.assertTrue(link.send_transcript("levantate", 1.0))
            payload = json.loads(orch.messages.get(timeout=3))
            self.assertRegex(payload["turn_id"], r"^[0-9a-f]{10}$")
        finally:
            orch.close()


class SharedModuleTest(unittest.TestCase):
    def test_copy_matches_the_canonical_one(self) -> None:
        canonical = REPO.parent / "R-AI-026" / "observability" / "rai_perf.py"
        if not canonical.exists():
            self.skipTest("R-AI-026 no está al lado de este repo")
        self.assertEqual((REPO / "rai_perf.py").read_bytes(), canonical.read_bytes(),
                         "rai_perf.py difiere de la copia canónica (R-AI-026/observability)")


if __name__ == "__main__":
    unittest.main()
