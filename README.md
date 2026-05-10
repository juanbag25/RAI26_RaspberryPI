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
│   ├── audio_capture.py  # Audio capture using sounddevice
│   ├── config.py         # Project constants
│   └── requirements.txt
│
├── linux/                # Raspberry Pi 5 implementation (to be done)
│
├── models/               # Optional: manually downloaded Whisper weights
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

- Python 3.10 or higher (check with `python --version`)
- USB microphone connected

### Installation

Open PowerShell in the project root:

```powershell
# Clone the repo
git clone https://github.com/YOUR_USER/stt-project.git
cd stt-project

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r windows/requirements.txt
```

### First run

```powershell
python windows/main.py
```

On first run, `faster-whisper` downloads the `base` model from HuggingFace (~140 MB) and caches it under `%USERPROFILE%\.cache\huggingface\hub\`. Subsequent runs load the cached model instantly and are fully offline.

The program prints all available audio devices on startup. If your USB microphone is not the system default, note its ID from the list and edit `windows/main.py` to pass it explicitly:

```python
capture = WindowsAudioCapture(device_id=N)
```

You can also list devices anytime:

```powershell
python -m sounddevice
```

### Usage

Speak into the microphone. The system detects voice activity, accumulates audio until a sustained silence is detected, then transcribes the utterance and prints it:

```
>>> Hola, esto es una prueba
>>> La transcripción funciona en tiempo real
```

Press **Ctrl+C** to stop.

### Tuning

All knobs live in [windows/config.py](windows/config.py):

- `MODEL_SIZE` — `"tiny"`, `"base"`, `"small"`, `"medium"`, `"large-v3"`, or a local folder path
- `LANGUAGE` — language code (`"es"`, `"en"`, etc.)
- `VAD_AGGRESSIVENESS` — webrtcvad strictness, 0 (lenient) to 3 (strict)
- `SILENCE_MS` — sustained silence required to close an utterance
- `PRE_SPEECH_PADDING_MS` — audio buffered before VAD triggers (avoids cutting initial phonemes)

## Troubleshooting

### HuggingFace download fails (`WinError 10054`, connection reset, etc.)

Some networks (corporate firewalls, certain ISPs, VPNs, antivirus with TLS inspection) block the HuggingFace API endpoint. The model files can still be reached from a browser, so download them manually:

1. Open https://huggingface.co/Systran/faster-whisper-base/tree/main in a browser.
2. Create the folder `stt-project/models/faster-whisper-base/` and download these four files into it:
   - `config.json`
   - `model.bin` (~140 MB)
   - `tokenizer.json`
   - `vocabulary.txt`
3. Run `python windows/main.py` again. `config.py` auto-detects the local folder and loads the model from disk instead of contacting HuggingFace.

To use a different model size, repeat the steps from the matching `Systran/faster-whisper-<size>` repo and adjust the folder name (and `MODEL_SIZE` in `config.py` if needed).

## License

MIT
