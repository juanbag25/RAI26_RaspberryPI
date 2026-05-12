from dotenv import load_dotenv

load_dotenv()

from audio_capture import LinuxAudioCapture
from config import BACKEND
from vad import VoiceActivityDetector

if BACKEND == "groq":
    from groq_transcriber import GroqTranscriber as Transcriber
elif BACKEND == "local":
    from transcriber import Transcriber
else:
    raise ValueError(f"Unknown BACKEND: {BACKEND!r} (expected 'local' or 'groq')")


def main() -> None:
    print("Available audio devices:")
    LinuxAudioCapture.list_devices()
    print()

    print(f"Loading transcriber (backend={BACKEND})...")
    transcriber = Transcriber()
    vad = VoiceActivityDetector()
    capture = LinuxAudioCapture()

    print("Listening. Press Ctrl+C to stop.")
    try:
        for frame in capture.frames():
            closed, audio = vad.process_frame(frame)
            if closed and audio is not None:
                text = transcriber.transcribe(audio)
                if text:
                    print(f">>> {text}")
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
