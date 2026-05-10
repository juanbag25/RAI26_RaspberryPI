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
├── models/               # Whisper model files (downloaded manually, not in git)
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

### 1. Install dependencies

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

### 2. Download the Whisper model

The model weights are **not** bundled in the repo — you download them manually once. By default the project uses the `base` model.

1. Open https://huggingface.co/Systran/faster-whisper-base/tree/main in a browser.
2. Create the folder `stt-project/models/faster-whisper-base/`.
3. Download these four files into it (click each file → click the download icon):
   - `config.json`
   - `model.bin` (~140 MB)
   - `tokenizer.json`
   - `vocabulary.txt`

The final layout should look like:

```
stt-project/
└── models/
    └── faster-whisper-base/
        ├── config.json
        ├── model.bin
        ├── tokenizer.json
        └── vocabulary.txt
```

To use a different model size, download the equivalent files from `Systran/faster-whisper-<size>` (e.g. `tiny`, `small`, `medium`, `large-v3`, `distil-large-v3`) into a sibling folder `models/faster-whisper-<size>/`, then edit `windows/config.py`:

```python
MODEL_SIZE = "small"   # picks models/faster-whisper-small/
```

Approximate sizes (int8): `tiny` ~40 MB, `base` ~140 MB, `small` ~470 MB, `medium` ~1.4 GB, `large-v3` ~3 GB.

### 3. Run

```powershell
python windows/main.py
```

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

Knobs in [windows/config.py](windows/config.py):

- `MODEL_SIZE` — which folder under `models/faster-whisper-<MODEL_SIZE>/` to load
- `LANGUAGE` — language code (`"es"`, `"en"`, etc.)
- `VAD_AGGRESSIVENESS` — webrtcvad strictness, 0 (lenient) to 3 (strict)
- `SILENCE_MS` — sustained silence required to close an utterance
- `PRE_SPEECH_PADDING_MS` — audio buffered before VAD triggers (avoids cutting initial phonemes)

## License

MIT
