from audio_capture import LinuxAudioCapture
from transcriber import Transcriber
from vad import VoiceActivityDetector


def main() -> None:
    print("Available audio devices:")
    LinuxAudioCapture.list_devices()
    print()

    print("Loading model...")
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
