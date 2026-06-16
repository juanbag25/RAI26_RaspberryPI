# STT Project — Linux / Raspberry Pi

Linux/ARM64 deployment of the STT project, targeted at a **Raspberry Pi 5** running Raspberry Pi OS 64-bit (Bookworm). Mirrors the Windows code with one platform tweak: ALSA `plughw` routing so USB mics that don't expose 16 kHz natively still work.

For the project overview, model sizes and tuning notes, see the [root README](../README.md).

## Hardware

- Raspberry Pi 5 (4 GB or 8 GB)
- Raspberry Pi OS 64-bit (Bookworm)
- USB conference microphone (USB Audio Class compliant)

## 1. Get the code onto the Pi

Pick whichever fits your workflow.

### Option A — Clone from GitHub (recommended)

On the Pi:

```bash
git clone https://github.com/YOUR_USER/stt-project.git
cd stt-project
```

To pull updates later: `git pull`.

### Option B — Push from the dev machine with `rsync`

Useful when iterating on the code from Windows/macOS and you don't want to round-trip through GitHub. From the dev machine:

```bash
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude 'models' \
  /path/to/stt-project/ pi@raspberrypi.local:~/stt-project/
```

`scp -r ./linux rai26@[ip]:~/stt-project/` also works for a one-shot copy.

## 2. System dependencies

On the Pi:

```bash
sudo apt update
sudo apt install -y python3-venv python3-dev libportaudio2 libasound2-dev ffmpeg
sudo usermod -a -G audio "$USER"
```

Log out and back in (or reboot) so the `audio` group membership takes effect.

## 3. Python environment

```bash
cd ~/stt-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r linux/requirements.txt
```

## 4. Whisper model (local backend only)

Skip this step if you'll use the Groq backend (the default in `linux/config.py`).

Download the four files for the desired model size into `models/faster-whisper-<size>/` — same procedure as Windows. Example for `small`:

```bash
mkdir -p models/faster-whisper-small
cd models/faster-whisper-small
# Download config.json, model.bin, tokenizer.json, vocabulary.txt
# from https://huggingface.co/Systran/faster-whisper-small/tree/main
```

Then in `linux/config.py` set:

```python
BACKEND = "local"
MODEL_SIZE = "small"
```

## 5. Configure the `.env`

Create `linux/.env` (it's already gitignored):

```bash
echo "GROQ_API_KEY=gsk_your_key_here" > linux/.env
```

Only required when `BACKEND = "groq"`.

## 6. Find the microphone

```bash
python -m sounddevice
```

Note the input device ID of your USB mic. If it's not the system default, edit `linux/main.py`:

```python
capture = LinuxAudioCapture(device_id=N)
```

You can also double-check with `arecord -l`.

## 7. Run

```bash
source .venv/bin/activate
python linux/main.py
```

Speak into the mic. The system prints transcribed utterances prefixed with `>>>`. Press **Ctrl+C** to stop.

## TTS — respuesta hablada

Además de transcribir, esta Pi ahora **recibe** la respuesta del orquestador y la
**dice** por el parlante. Flujo:

```
mic → STT → (TCP 9000) orquestador → LLM → (TCP 9001) Pi → TTS → parlante
```

El motor de TTS se elige en `config.py` (`TTS_ENGINE`):

- **`edge`** (default) — edge-tts, voces neuronales de Microsoft Edge. **Gratis,
  sin API key ni billing.** Necesita internet.
- **`google_cloud`** — Google Cloud TTS (requiere credenciales + billing).

### Dependencias

```bash
source .venv/bin/activate
pip install -r linux/requirements.txt
```

(`edge-tts` + `soundfile` ya están en `requirements.txt`. `google-cloud-texttospeech`
está comentado: descomentalo solo si vas a usar ese motor.)

### Configuración

En [`config.py`](config.py):

- `RESPONSE_PORT` (default `9001`) — puerto donde la Pi escucha la respuesta del
  orquestador. Debe coincidir con `HRI_PORT` del orquestador.
- `TTS_ENGINE` — `edge` o `google_cloud`.
- `EDGE_VOICE` — voz de edge-tts (ej. `es-AR-TomasNeural`, `es-AR-ElenaNeural`).
  Listá todas con `edge-tts --list-voices | grep es-`.
- `EDGE_RATE`, `EDGE_PITCH`, `TTS_OUTPUT_DEVICE`.

En `linux/.env`, además de `GROQ_API_KEY`:

```bash
ORCHESTRATOR_IP=<IP de la Jetson en la WiFi>
# Solo si TTS_ENGINE=google_cloud:
# GOOGLE_APPLICATION_CREDENTIALS=/ruta/absoluta/al/key.json
```

(El puerto de salida está fijo en `main.py`: `ORCHESTRATOR_PORT = 9000`.)

### Probar solo el TTS (sin pipeline)

Para escuchar cómo habla con un texto de prueba, **sin** STT ni orquestador.
Con `edge` no hace falta ninguna credencial. Lo podés correr en tu compu y sale
por el parlante de tu compu:

```bash
source .venv/bin/activate
python linux/tts_test.py
python linux/tts_test.py "Hola, soy el perro robot del ITBA"
```

### (Opcional) usar Google Cloud TTS

Poné `TTS_ENGINE = "google_cloud"` en `config.py`, descomentá
`google-cloud-texttospeech` en `requirements.txt` y reinstalá. Necesitás un JSON
de service account y apuntar `GOOGLE_APPLICATION_CREDENTIALS` (en `linux/.env`) a
su ruta absoluta. Voces: https://cloud.google.com/text-to-speech/docs/voices

Mapa completo de IPs/puertos del sistema: ver `docs/NETWORKING.md` en el repo
principal (R-AI-026).

## Tuning

All knobs live in [`linux/config.py`](config.py): `BACKEND`, `MODEL_SIZE`, `LANGUAGE`, `VAD_AGGRESSIVENESS`, `SILENCE_MS`, `PRE_SPEECH_PADDING_MS`, `RMS_THRESHOLD`, `MIN_UTTERANCE_MS`. See the root README for what each one does.

## Troubleshooting

- **`paInvalidSampleRate` when opening the stream** — the code already sets `PA_ALSA_PLUGHW=1` in `audio_capture.py` so PortAudio routes through ALSA's `plug` plugin and gets transparent sample-rate conversion. If you still see this, confirm the mic appears in `arecord -l` and that `libasound2-dev` is installed.
- **Mic not detected** — run `arecord -l`. If empty, check the USB cable and that your user is in the `audio` group (`groups | grep audio`).
- **`GROQ_API_KEY` not found** — make sure `linux/.env` exists and you ran the script from a shell where the venv is activated; `python-dotenv` loads it at import time in `main.py`.
- **Model fails to load (local backend)** — verify `models/faster-whisper-<MODEL_SIZE>/` contains all four files (`config.json`, `model.bin`, `tokenizer.json`, `vocabulary.txt`) and that `MODEL_SIZE` in `config.py` matches the folder name.
- **High CPU / slow transcription** — on a Pi 5, stick to `tiny`, `base` or `small` for the local backend, or use the Groq backend.

## Running headless (optional)

To keep the script running after disconnecting SSH, the simplest options are `tmux` or `screen`:

```bash
sudo apt install -y tmux
tmux new -s stt
source .venv/bin/activate && python linux/main.py
# Detach with Ctrl+B then D. Re-attach later with: tmux attach -t stt
```

For a real service, wrap it in a `systemd` unit — out of scope here.
