"""Ventana de despierto: sólo corre mientras el mic escucha."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wake_word  # noqa: E402
from wake_word import WakeWord  # noqa: E402

WINDOW = wake_word.WAKE_WINDOW_S


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class WakeWindowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = Clock()
        patcher = patch("wake_word.time.monotonic", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.wake = WakeWord()
        self.wake.open_window()

    def advance(self, s: float) -> None:
        self.clock.t += s

    def test_window_does_not_run_while_the_robot_speaks(self) -> None:
        self.advance(WINDOW - 1)
        self.wake.pause("speak")
        self.advance(25)                     # habla largo: más que la ventana
        self.assertTrue(self.wake.is_awake())
        self.wake.resume("speak")
        self.advance(WINDOW - 0.5)           # la ventana vuelve ENTERA
        self.assertTrue(self.wake.is_awake())
        self.advance(1)
        self.assertFalse(self.wake.is_awake())

    def test_silent_action_keeps_it_awake_until_busy_end(self) -> None:
        self.wake.pause("busy")
        self.advance(40)                     # «da 10 pasos»: no se duerme caminando
        self.assertTrue(self.wake.is_awake())
        self.wake.pause("speak")             # el sonido de acción callada
        self.wake.resume("speak")
        self.advance(WINDOW + 1)
        self.assertTrue(self.wake.is_awake())   # sigue la acción
        self.wake.resume("busy")
        self.advance(WINDOW + 0.1)
        self.assertFalse(self.wake.is_awake())

    def test_lost_end_expires_at_its_cap(self) -> None:
        self.wake.pause("busy", max_s=10)
        self.advance(10 + WINDOW - 0.5)
        self.assertTrue(self.wake.is_awake())
        self.advance(1)
        self.assertFalse(self.wake.is_awake())

    def test_pause_does_not_wake_a_sleeping_mic(self) -> None:
        self.advance(WINDOW + 1)
        self.assertFalse(self.wake.is_awake())
        self.wake.pause("speak")
        self.assertFalse(self.wake.is_awake())
        self.wake.resume("speak")
        self.assertFalse(self.wake.is_awake())

    def test_silence_sleeps_even_mid_action(self) -> None:
        self.wake.pause("busy")
        self.wake.sleep()
        self.assertFalse(self.wake.is_awake())
        self.wake.resume("busy")
        self.assertFalse(self.wake.is_awake())

    def test_loose_speak_end_still_refreshes(self) -> None:
        self.advance(WINDOW - 1)
        self.wake.resume("speak")            # SPEAK_END sin START (compat)
        self.advance(WINDOW - 0.5)
        self.assertTrue(self.wake.is_awake())


if __name__ == "__main__":
    unittest.main()
