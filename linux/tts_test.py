"""Standalone TTS smoke test.

Synthesizes a string with the configured TTS engine (config.py TTS_ENGINE) and
plays it on the local speaker. Use it to verify voice/audio output work WITHOUT
running the full STT / orchestrator pipeline.

Default engine is "edge" (edge-tts): free, no API key, no billing — just needs
internet. (For "google_cloud" set GOOGLE_APPLICATION_CREDENTIALS in linux/.env.)

Run it on any machine (your laptop is fine — it'll come out of your laptop's
speaker):

    pip install -r linux/requirements.txt
    python tts_test.py
    python tts_test.py "Hola, soy el perro robot del ITBA"
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

# Levanta linux/.env (busca desde la ubicación de este script hacia arriba), así
# GOOGLE_APPLICATION_CREDENTIALS puede vivir ahí en vez de exportarlo a mano.
load_dotenv()

from tts_player import TtsPlayer

DEFAULT_TEXT = "Hola, soy Lite, el perro robot del ITBA. Encantado de conocerte."


def main() -> None:
    text = " ".join(sys.argv[1:]).strip() or DEFAULT_TEXT
    print(f"Sintetizando: {text!r}")

    player = TtsPlayer()
    player.speak(text)

    print("Listo. Si no escuchaste nada, revisá el dispositivo de salida "
          "(config.py TTS_OUTPUT_DEVICE) y, con edge, que haya internet.")


if __name__ == "__main__":
    main()
