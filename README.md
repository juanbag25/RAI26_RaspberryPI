# STT Project

Real-time speech-to-text system using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) with a USB conference microphone.

The project is developed and tested on **Windows** first, then ported to a **Raspberry Pi 5** (Linux ARM64) for deployment. Each platform has its own folder with platform-specific code, kept intentionally separate to make differences explicit.

## Project Structure

```
stt-project/
├── windows/              # Windows implementation (development & testing)
│   ├── main.py           # Entry point
│   ├── transcriber.py    # faster-whisper wrapper
│   ├── vad.py            # Voice Activity Detection
│   ├── audio_capture.py  # Audio capture using sounddevice (WASAPI)
│   ├── config.py         # Project constants
│   └── requirements.txt
│
├── linux/                # Raspberry Pi 5 implementation (to be done)
│
├── .gitignore
└── README.md
```

## Hardware

- **Development machine:** Windows 10/11 PC
- **Target deployment:** Raspberry Pi 5 (4 GB or 8 GB) with Raspberry Pi OS 64-bit (Bookworm)
- **Microphone:** USB conference microphone (USB Audio Class compliant)

## Software Stack

- Python 3.10+
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — quantized Whisper inference (`base` model, `int8` on CPU)
- [sounddevice](https://python-sounddevice.readthedocs.io/) — cross-platform audio capture
- [webrtcvad](https://github.com/wiseman/py-webrtcvad) — Voice Activity Detection
- NumPy

## Setup — Windows

### Requirements

- Python 3.10 or higher
- USB microphone connected

### Installation

```powershell
# Clone the repo
git clone https://github.com/YOUR_USER/stt-project.git
cd stt-project

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r windows/requirements.txt
```

### Running

```powershell
python windows/main.py
```

On first run, faster-whisper will download the `base` model (~140 MB) and cache it. Subsequent runs are instant.

The program prints available audio devices on startup. If the default device is not your USB mic, note its ID and modify `windows/main.py` to pass it to `WindowsAudioCapture(device_id=N)`.

You can also list devices anytime with:

```powershell
python -m sounddevice
```

### Usage

Once running, speak into the microphone. The system detects voice activity, accumulates audio until a sustained silence is detected, then transcribes the utterance and prints the result:

```
>>> Hello, this is a test
>>> The transcription works in real time
```

## License

MIT