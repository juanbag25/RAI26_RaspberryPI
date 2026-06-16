"""Standalone TTS smoke test.

Synthesizes a string with Google Cloud TTS and plays it on the local speaker.
Use it to verify your Google credentials, voice config and audio output work,
WITHOUT running the full STT / orchestrator pipeline.

Run it on any machine (your laptop is fine — it'll come out of your laptop's
speaker):

    # 1. set credentials (service-account JSON downloaded from GCP)
    #    Linux/macOS:  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
    #    Windows PS:   $env:GOOGLE_APPLICATION_CREDENTIALS="C:\\path\\to\\key.json"
    # 2. run it
    python tts_test.py
    python tts_test.py "Hola, soy el perro robot del ITBA"
"""

from __future__ import annotations

import sys

from tts_player import TtsPlayer

DEFAULT_TEXT = "Hola, soy Lite, el perro robot del ITBA. Encantado de conocerte."


def main() -> None:
    text = " ".join(sys.argv[1:]).strip() or DEFAULT_TEXT
    print(f"Sintetizando: {text!r}")

    player = TtsPlayer()
    player.speak(text)

    print("Listo. Si no escuchaste nada, revisá el dispositivo de salida "
          "(config.py TTS_OUTPUT_DEVICE) y las credenciales de Google.")


if __name__ == "__main__":
    main()
