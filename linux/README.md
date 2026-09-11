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

Create `linux/.env` (it's already gitignored; see `linux/.env.example`):

```bash
cat > linux/.env <<'EOF'
ORCHESTRATOR_IP=<IP de la máquina del orquestador>
GROQ_API_KEY=gsk_your_key_here
AUDIO_INPUT_DEVICE=
EOF
```

`GROQ_API_KEY` is only required when `BACKEND = "groq"`.

## 6. Find the microphone

```bash
python -m sounddevice
```

Note the input device ID of your USB mic. If it's not the system default, set
it in `linux/.env`:

```bash
AUDIO_INPUT_DEVICE=N
```

You can also double-check with `arecord -l`.

## 7. Run

```bash
source .venv/bin/activate
python linux/main.py
```

Speak into the mic. The system prints transcribed utterances prefixed with `>>>`. Press **Ctrl+C** to stop.

## Respuesta hablada

La respuesta del LLM ya **no** vuelve a esta Pi: la sintetiza y la dice el
orquestador (`R-AI-026/orchestrator`) por los parlantes de la máquina donde
corre (la Jetson en el robot). Esta Pi solo captura audio, transcribe y manda
el texto:

```
mic → STT → (TCP 9000) orquestador → LLM → TTS → parlante del orquestador
 ▲                                          │
 └──────── SPEAK_START/END (TCP 9001) ──────┘   mute del mic mientras habla
```

Mientras el robot habla, el orquestador manda `SPEAK_START`/`SPEAK_END` al
puerto `CTRL_PORT` (default 9001, en `config.py`) y `main.py` descarta el
audio del mic para no transcribir la propia voz del robot. Si el `SPEAK_END`
se pierde, el mute expira solo a los `MUTE_TIMEOUT_S` segundos.

Mapa completo de IPs/puertos del sistema: ver `docs/NETWORKING.md` en el repo
principal (R-AI-026).

## Foco del mic (rechazo de campo lejano)

El mic es omnidireccional y webrtcvad sólo sabe decir "esto es voz humana", no
"esto me lo están diciendo a mí": sin filtro, una charla del otro lado de la
sala abre utterances y Whisper las transcribe (o alucina sobre ellas). Encima
del VAD hay entonces un filtro de **energía** en dos etapas ([`vad.py`](vad.py)),
apoyado en que quien le habla al robot de cerca llega mucho más fuerte que el
fondo:

1. **Al abrir**: hacen falta `ONSET_SPEECH_FRAMES` frames seguidos de voz por
   encima de `max(RMS_THRESHOLD, ruido × NEAR_SNR_RATIO)`.
2. **Al cerrar**: el percentil 90 de los frames de voz de la utterance tiene que
   llegar a `NEAR_RMS_THRESHOLD` y seguir `NEAR_SNR_RATIO` por encima del ruido.
   Una frase que arrancó fuerte (un portazo, una sílaba) pero venía de lejos se
   cae acá y nunca llega a Whisper.

El **piso de ruido se mide en vivo** con los frames descartados (incluye el
murmullo lejano), así que en una sala ruidosa el filtro se endurece solo; está
topeado en `NOISE_FLOOR_MAX` para que un ruido fuerte sostenido no deje sordo al
robot. Cada descarte se loguea con el nivel medido:

```
[VAD] descartada: lejana/floja (nivel=0.0263 < 0.0550, ruido=0.0036)
```

### Calibración

`NEAR_RMS_THRESHOLD` es el knob principal y **depende del mic, su ganancia y la
sala** — hay que calibrarlo una vez en el lugar donde va a laburar el robot:

```bash
python linux/mic_level.py     # o `python linux/mic_level.py N` para elegir mic
```

Hacé dos pasadas y anotá el `p90` que imprime el resumen:

1. Sala como es normalmente (gente hablando lejos) **sin** hablarle al robot →
   `p90_lejos`.
2. Hablándole vos desde donde le hablarías de verdad → `p90_cerca`.

Poné `NEAR_RMS_THRESHOLD` entre los dos, más cerca del primero: arrancá en
`p90_lejos × 2`, siempre por debajo de `p90_cerca × 0.6`. Si el robot te ignora,
bajalo; si sigue enganchando charlas ajenas, subilo. Se puede tocar sin editar
código, desde `linux/.env`:

```bash
NEAR_RMS_THRESHOLD=0.07
NEAR_SNR_RATIO=3
```

## Wake word: "rai"

Segundo filtro, ahora sobre el **texto**: el robot descarta todo lo que
transcribe hasta que alguien lo llama por su nombre.

```
>>> ¿viste el partido de ayer?
[WAKE] dormido, ignorado: ¿viste el partido de ayer?
>>> Rai, vení para acá
[WAKE] despierto por «Rai, vení para acá»     -> se manda "vení para acá"
```

- Se hace sobre la transcripción y no con un keyword spotter de audio: la
  utterance ya pasa por Whisper igual, así que sale gratis y no agrega modelos
  ni dependencias ([`wake_word.py`](wake_word.py)).
- El match ignora mayúsculas, acentos y puntuación, acepta las variantes con las
  que Whisper suele escribirlo (`WAKE_WORDS`: rai, ray, rae, raid…, incluido
  "R.A.I.") y sólo lo busca en las primeras `WAKE_SEARCH_WORDS` palabras.
  Además, `STT_PROMPT` le pasa a Whisper el vocabulario del dominio para que
  escriba "RAI" y no invente.
- El nombre (y lo que venga antes) se recorta: al LLM le llega la instrucción
  sola. Si la frase es sólo "rai", se manda `WAKE_ACK_TEXT` para que conteste y
  se note que está escuchando (poné `""` para que despierte en silencio).
- Después queda despierto `WAKE_WINDOW_S` segundos para seguir la conversación
  sin repetir el nombre. Cada frase aceptada renueva la ventana, y también la
  renueva el `SPEAK_END` del orquestador (acaba de contestar: lo natural es que
  le sigan hablando).

Desde `linux/.env`:

```bash
WAKE_WORD_ENABLED=true    # false = como antes, atiende todo lo que pasa el VAD
WAKE_WINDOW_S=25
```

## Tuning

All knobs live in [`linux/config.py`](config.py): `BACKEND`, `MODEL_SIZE`,
`LANGUAGE`, `STT_PROMPT`, `VAD_AGGRESSIVENESS`, `SILENCE_MS`,
`PRE_SPEECH_PADDING_MS`, `MIN_UTTERANCE_MS`, los del foco del mic
(`RMS_THRESHOLD`, `NEAR_RMS_THRESHOLD`, `NEAR_SNR_RATIO`, `ONSET_SPEECH_FRAMES`,
`NOISE_FLOOR_*`) y los del wake word (`WAKE_*`). Los más usados se pueden pisar
desde `linux/.env` — ver `.env.example`. Para el resto, ver el root README.

## Troubleshooting

- **`paInvalidSampleRate` when opening the stream** — the code already sets `PA_ALSA_PLUGHW=1` in `audio_capture.py` so PortAudio routes through ALSA's `plug` plugin and gets transparent sample-rate conversion. If you still see this, confirm the mic appears in `arecord -l` and that `libasound2-dev` is installed.
- **Mic not detected** — run `arecord -l`. If empty, check the USB cable and that your user is in the `audio` group (`groups | grep audio`).
- **`GROQ_API_KEY` not found** — make sure `linux/.env` exists and you ran the script from a shell where the venv is activated; `python-dotenv` loads it at import time in `main.py`.
- **Model fails to load (local backend)** — verify `models/faster-whisper-<MODEL_SIZE>/` contains all four files (`config.json`, `model.bin`, `tokenizer.json`, `vocabulary.txt`) and that `MODEL_SIZE` in `config.py` matches the folder name.
- **El robot no me escucha a mí** — mirá el log: si aparece `[VAD] descartada:
  lejana/floja`, `NEAR_RMS_THRESHOLD` está muy alto para tu mic (el log imprime
  el nivel medido: poné el umbral debajo de ese valor). Si aparece
  `[WAKE] dormido, ignorado: ...`, transcribió bien pero no le dijiste "rai" —
  o Whisper escribió el nombre de una forma que no está en `WAKE_WORDS`.
- **Sigue enganchando conversaciones ajenas** — subí `NEAR_RMS_THRESHOLD` (y/o
  `NEAR_SNR_RATIO`) con `mic_level.py` en la mano; el wake word tapa el resto.
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
