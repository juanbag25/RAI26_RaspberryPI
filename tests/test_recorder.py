"""Grabación continua: tope acumulado entre corridas, WAV válidos, índice."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recorder import BUDGET_FILE, INDEX_FILE, WAV_HEADER, Recorder  # noqa: E402

FRAME = b"\x01\x00" * 480  # 30 ms mono int16


def run_session(directory, n_frames, max_bytes, **kw):
    rec = Recorder(directory, max_bytes=max_bytes, run=kw.pop("run", "r"), **kw)
    started = rec.start()
    t = 1000.0
    for _ in range(n_frames):
        rec.push(FRAME, t)
        t += 0.03
        if not rec.active:
            break
    rec.close()
    return rec, started


def wavs(directory):
    return sorted(p for p in os.listdir(directory) if p.endswith(".wav"))


class RecorderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_records_valid_wav_and_counts_bytes(self):
        rec, started = run_session(self.dir, 100, 10**9)
        self.assertTrue(started)
        files = wavs(self.dir)
        self.assertEqual(len(files), 1)
        with wave.open(os.path.join(self.dir, files[0])) as w:
            self.assertEqual((w.getnchannels(), w.getframerate(), w.getnframes()), (1, 16000, 48000))
        budget = json.loads(Path(self.dir, BUDGET_FILE).read_text())
        self.assertEqual(budget["bytes"], 100 * len(FRAME) + WAV_HEADER)
        self.assertEqual(budget["bytes"], os.path.getsize(os.path.join(self.dir, files[0])))
        idx = [json.loads(line) for line in Path(self.dir, INDEX_FILE).read_text().splitlines()]
        self.assertTrue(idx[0]["open"] and idx[0]["sample"] == 0 and idx[0]["t"] == 1000.0)

    def test_limit_is_cumulative_across_sessions(self):
        limit = 50 * len(FRAME) + 2 * WAV_HEADER
        run_session(self.dir, 30, limit, run="a")  # 1ª corrida: 30 frames
        rec, started = run_session(self.dir, 100, limit, run="b")  # 2ª: sólo entran 20 más
        self.assertTrue(started)
        self.assertEqual(rec.stopped_reason, "tope")
        total = sum(os.path.getsize(os.path.join(self.dir, f)) for f in wavs(self.dir))
        self.assertEqual(total, limit)
        rec3, started3 = run_session(self.dir, 10, limit, run="c")  # 3ª: nunca más
        self.assertFalse(started3)
        self.assertEqual(len(wavs(self.dir)), 2)

    def test_deleting_wavs_does_not_reset_the_budget(self):
        limit = 20 * len(FRAME) + WAV_HEADER
        run_session(self.dir, 100, limit)
        for f in wavs(self.dir):
            os.remove(os.path.join(self.dir, f))
        _, started = run_session(self.dir, 10, limit)
        self.assertFalse(started)
        os.remove(os.path.join(self.dir, BUDGET_FILE))  # la única forma de otra tanda
        _, started = run_session(self.dir, 10, limit)
        self.assertTrue(started)

    def test_lost_budget_file_falls_back_to_disk(self):
        run_session(self.dir, 40, 10**9)
        os.remove(os.path.join(self.dir, BUDGET_FILE))
        rec = Recorder(self.dir, max_bytes=10**9)
        self.assertEqual(rec.total, 40 * len(FRAME) + WAV_HEADER)

    def test_rotates_segments(self):
        run_session(self.dir, 100, 10**9, segment_s=1.0)  # 3 s de audio en tramos de 1 s
        self.assertGreaterEqual(len(wavs(self.dir)), 3)

    def test_multichannel(self):
        rec = Recorder(self.dir, max_bytes=10**9, channels=6)
        rec.start()
        rec.push(b"\x00\x00" * 6 * 480, time.time())
        rec.close()
        with wave.open(os.path.join(self.dir, wavs(self.dir)[0])) as w:
            self.assertEqual((w.getnchannels(), w.getnframes()), (6, 480))


if __name__ == "__main__":
    unittest.main()
