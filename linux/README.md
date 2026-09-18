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

## 4b. Vosk model (wake word por audio)

`WAKE_MODE=audio` (el default) necesita el modelo chico de Vosk en español
(~40 MB, ~100 MB descomprimido) en `models/`:

```bash
cd ~/stt-project
mkdir -p models && cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip
unzip vosk-model-small-es-0.42.zip && rm vosk-model-small-es-0.42.zip
```

Si falta, `main.py` avisa y arranca en `WAKE_MODE=text` (funciona igual, más
lento). Probalo solo con `python wake_spotter.py` y el beep con
`python wake_sound.py`.

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

## Wake word: "oye rai"

El robot descarta todo hasta que alguien lo llama. Hay dos modos (`WAKE_MODE`):

### Modo `audio` (default): spotter local + beep

Como un celular con Siri: un detector chico corre **siempre** sobre el audio
([`wake_spotter.py`](wake_spotter.py), Vosk con gramática cerrada), y mientras
el robot duerme **no se manda nada a Groq**. Al reconocer "oye rai" suena un
beep ([`wake_sound.py`](wake_sound.py)), el robot se despierta en ~0.3 s y
recién ahí las frases van a Groq.

```
>>> ¿viste el partido de ayer?
[MAIN] dormido: utterance descartada sin transcribir (nivel=0.0812; decí «oye rai»)
>>> oye rai
[SPOT] wake detectado (parcial): «oye ray»
[WAKE] despierto por audio (foco=0.1547, mínimo=0.0928, ventana 25s)
[VAD] utterance descartada: wake por audio (780 ms de voz se pierden)   <- el "oye rai" no se transcribe
   *beep*
>>> vení para acá
[STT] 0.61s >>> Vení para acá.
[NET] enviado al orquestador ...
```

- La frase de wake tiene que sonar *parecido*, no exacto: Vosk sólo puede
  devolver una de `WAKE_PHRASES` o `[unk]`, así que "hola rai" también suele
  disparar como "oye ray". "rai" no es palabra del español y sale como
  "ray"/"rey"; por eso las tres variantes están en el default.
- Decir "oye rai" mientras ya está despierto re-engancha el foco a quien lo
  dijo (y suena el beep otra vez).
- Mientras suena el beep el mic se ignora (~230 ms) para no transcribirse el
  propio beep. Si no hay parlante, `WAKE_SOUND=none` (o se desactiva solo al
  fallar) y el wake funciona igual.
- Además del beep local, cada cambio despierto/dormido se le avisa al
  orquestador (`@@event:awake` / `@@event:asleep` por el mismo socket del
  texto) y **él** hace sonar un chime por el parlante del robot: agudo al
  empezar a escuchar, grave cuando vence la ventana (`WAKE_WINDOW_S`) y se
  duerme. `WAKE_EVENTS_ENABLED=false` lo apaga; el volumen se fija allá
  (`TTS_CHIME_VOLUME` en el `.env` del orquestador).
- El texto que llega a Groq después del beep pasa igual por el filtro de
  texto de abajo: si Whisper escribe "rai vení" se recorta a "vení".

### Modo `text`: sobre la transcripción

Sin modelo extra: cada frase que pasa el VAD se transcribe y se busca "rai" en
el texto ([`wake_word.py`](wake_word.py)). Más lento (~2 s hasta que se entera)
y gasta Groq aunque duerma; es el fallback si Vosk no arranca.

```
>>> ¿viste el partido de ayer?
[WAKE] dormido, ignorado: ¿viste el partido de ayer?
>>> Rai, vení para acá
[WAKE] despierto por «Rai, vení para acá»     -> se manda "vení para acá"
```

- El match ignora mayúsculas, acentos y puntuación, acepta las variantes con las
  que Whisper suele escribirlo (`WAKE_WORDS`: rai, ray, rae, raid…, incluido
  "R.A.I.") y sólo lo busca en las primeras `WAKE_SEARCH_WORDS` palabras.
  Además, `STT_PROMPT` le pasa a Whisper el vocabulario del dominio para que
  escriba "RAI" y no invente.
- El nombre (y lo que venga antes) se recorta: al LLM le llega la instrucción
  sola. Si la frase es sólo "rai", se manda `WAKE_ACK_TEXT` para que conteste y
  se note que está escuchando (poné `""` para que despierte en silencio).
- **Atención a quien lo llamó**: despertarse no es "escuchar todo lo que pase
  el VAD durante N segundos". Al despertar se guarda el nivel de voz del "rai"
  y, mientras dure la ventana, sólo se aceptan frases que lleguen al menos a
  `nivel × ATTENTION_LEVEL_RATIO` (default 0.6) — se chequea **antes** de
  transcribir, así que el fondo de la sala ni gasta Whisper. La referencia
  sigue a la persona (EMA por frase) y decir "rai" de nuevo la re-engancha a
  quien lo dijo. Log: `[WAKE] atención: ignorado, más flojo que quien me llamó`.
- Después queda despierto `WAKE_WINDOW_S` segundos para seguir la conversación
  sin repetir el nombre. Cada frase aceptada renueva la ventana, y también la
  renueva el `SPEAK_END` del orquestador (acaba de contestar: lo natural es que
  le sigan hablando).

Desde `linux/.env`:

```bash
WAKE_WORD_ENABLED=true    # false = como antes, atiende todo lo que pasa el VAD
WAKE_MODE=audio           # audio | text
WAKE_PHRASES=oye rai,oye ray,oye rey   # agregá "hola rai", "che rai"...
WAKE_SOUND=beep           # beep | none | /ruta/ding.wav
AUDIO_OUTPUT_DEVICE=      # parlante para el beep (índice de sounddevice)
WAKE_EVENTS_ENABLED=true  # chime remoto (parlante del orquestador) al despertar/dormirse
WAKE_WINDOW_S=25
ATTENTION_LEVEL_RATIO=0.6 # 0.8 = más cerrado sobre quien lo llamó; 0 = off
```

## Tuning

All knobs live in [`linux/config.py`](config.py): `BACKEND`, `MODEL_SIZE`,
`LANGUAGE`, `STT_PROMPT`, `VAD_AGGRESSIVENESS`, `SILENCE_MS`,
`PRE_SPEECH_PADDING_MS`, `MIN_UTTERANCE_MS`, los del foco del mic
(`RMS_THRESHOLD`, `NEAR_RMS_THRESHOLD`, `NEAR_SNR_RATIO`, `ONSET_SPEECH_FRAMES`,
`NOISE_FLOOR_*`) y los del wake word (`WAKE_*`, `AUDIO_OUTPUT_DEVICE`). Los más
usados se pueden pisar desde `linux/.env` — ver `.env.example`. Para el resto,
ver el root README.

## Logs y diagnóstico

Todas las líneas salen con timestamp (`[HH:MM:SS.mmm]`) y sin buffer, así que
sirven igual por SSH, `tmux` o redirigidas a un archivo. Cada etapa del
pipeline deja rastro:

```
[AUDIO] stream abierto (latencia=32 ms)
[AUDIO] primer frame recibido del mic: el stream funciona
[VAD] utterance ABIERTA (rms=0.0812 >= umbral=0.0200, ruido=0.0041)
[VAD] utterance CERRADA y aceptada: 1230 ms de voz, 2130 ms totales, nivel=0.0790 -> a transcribir
[MAIN] utterance encolada para STT (cola=1)
[STT] transcribiendo 2.1s de audio...
[STT] 0.84s >>> Rai, vení para acá
[WAKE] despierto por «Rai, vení para acá»
[NET] enviado al orquestador 192.168.1.50:9000 (13 B en 0.01s): «vení para acá»
```

Además, cada `LOG_HEARTBEAT_S` segundos (default 10) sale un resumen `[HB]`:

```
[HB] frames=333 muteados=0 voz=41 voz>umbral=38 | rms max=0.0912 media=0.0060 ruido=0.0044 umbral_abrir=0.0200 cerca=0.055 | utt abiertas=1 ok=1 desc=0 en_utt=no | cola_stt=0 stt=1 vacías=0 enviadas=1 fallidas=0 | mute=no wake=despierto
```

Cómo leerlo cuando "se queda escuchando y no pasa nada":

| Síntoma en `[HB]` | Significa | Qué tocar |
|---|---|---|
| `frames=0` / `<-- SIN AUDIO DEL MIC` | PortAudio no entrega audio | mic equivocado (`AUDIO_INPUT_DEVICE`), cable, `arecord -l` |
| `rms max` ≈ 0.000x aunque hables | el mic está pero casi mudo | subir ganancia (`alsamixer`), otro device |
| `voz=0` aunque hables | webrtcvad no ve voz | mic/sample rate raro; probar `VAD_AGGRESSIVENESS` más bajo |
| `voz>0` pero `voz>umbral=0` | la voz llega floja | bajar `RMS_THRESHOLD` / subir ganancia; `LOG_DEBUG=1` muestra cada frame |
| `abiertas>0` pero `ok=0` | se abren y se descartan | mirar los `[VAD] descartada:` (corta → `MIN_UTTERANCE_MS`; lejana → `NEAR_RMS_THRESHOLD`) |
| `ok>0` pero `stt` no crece | el hilo de STT está trabado | mirar `[STT ERROR]` (Groq / API key / red) |
| `vacías` crece | Groq devuelve "" | `[STT ERROR]` arriba, o audio inaudible |
| `[WAKE] dormido, ignorado` | transcribió pero no dijiste "rai" | `WAKE_WORD_ENABLED=false` para probar |
| `fallidas` crece | no llega al orquestador | `[NET ERROR]`: IP/puerto/firewall |
| `mute=SÍ` todo el tiempo | se perdió un `SPEAK_END` | expira solo a los `MUTE_TIMEOUT_S`; revisar el orquestador |

## Alimentación

Al arrancar (y cada `POWER_LOG_S` segundos, default 300) `main.py` loguea lo
que la Pi ve de su alimentación ([`battery.py`](battery.py)); también se puede
correr suelto con `python linux/battery.py`:

```
[POWER] Pi: entrada 5V real: 5.08 V
[POWER] Pi: throttled=0x0  (ok)
```

- `entrada 5V real` (`EXT5V_V`, Pi 5): si baja de 4.8 V la Pi avisa y es
  probable que se reinicie o suelte el USB del mic.
- `throttled`: flags de undervoltage/throttling, ahora y desde el boot.

La Pi se alimenta por USB-C desde la salida USB-A del UPS (SunFounder
PiPower). Por ese cable no hay USB-PD (la Pi siempre asume 900 mA de fuente,
no importa) ni I2C, así que no se puede leer el % de batería del UPS desde la
Pi. Si aparece `hubo undervoltage desde el boot` seguido, el cable USB-A→C
cae bajo carga: probar cable más corto/grueso.

## Troubleshooting

- **`paInvalidSampleRate` when opening the stream** — the code already sets `PA_ALSA_PLUGHW=1` in `audio_capture.py` so PortAudio routes through ALSA's `plug` plugin and gets transparent sample-rate conversion. If you still see this, confirm the mic appears in `arecord -l` and that `libasound2-dev` is installed.
- **Mic not detected** — run `arecord -l`. If empty, check the USB cable and that your user is in the `audio` group (`groups | grep audio`).
- **Escucha pero nunca transcribe nada** — mirá la línea `[HB]` y la tabla de
  la sección *Logs y diagnóstico*: dice en qué etapa se queda (mic, VAD,
  umbral, STT, wake word o red).
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
- **No despierta con "oye rai"** — mirá `oyó=...` en la línea `[HB]`: es lo último que Vosk entendió. Si dice `'oye'` y nunca `'oye ray'`, probá `LOG_DEBUG=1` y `python wake_spotter.py`, hablá más cerca, o agregá la variante que veas a `WAKE_PHRASES`. Si `desc=` (frames descartados del spotter) crece, la Pi no da abasto.
- **Despierta solo** — sacá variantes de `WAKE_PHRASES` (dejá sólo `oye rai,oye ray`) o subí `WAKE_COOLDOWN_S`. Si hay un parlante cerca del mic, bajá `WAKE_SOUND_VOLUME`.
- **No suena el beep** — `python wake_sound.py`; si falla, elegí el parlante con `AUDIO_OUTPUT_DEVICE` (mismo listado que el mic al arrancar) o `WAKE_SOUND=none`.

## Probar con tu PC como orquestador (Tailscale)

Para desarrollar el orquestador en tu compu sin tocar el robot: la Pi entra a
tu tailnet y `dev_orchestrator.sh` le apunta el STT a tu PC por esa red.

### Una vez: registrar la Pi en Tailscale

En la admin console de Tailscale generá una auth key (Settings → Keys →
Generate auth key; conviene *reusable* y con tag si usás ACLs). Después, en la
terminal de la Pi:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --auth-key tskey-auth-XXXXXXXXXXXX --hostname rai-pi
tailscale ip -4      # IP 100.x.y.z de la Pi: va en el orquestador para el mute (puerto 9001)
tailscale status     # tiene que aparecer tu PC
```

En la PC: Tailscale instalado y logueado en la misma tailnet, y el orquestador
escuchando en `0.0.0.0:9000` (no en `127.0.0.1`). Si el orquestador corre en
WSL2, la PC recibe en Windows pero WSL2 no lo ve: o instalás Tailscale dentro
de WSL, o activás `networkingMode=mirrored` en `%UserProfile%\.wslconfig`.
Windows Firewall va a preguntar la primera vez que Python escuche: permitir.

### Cada vez: apuntar la Pi a tu PC

```bash
./linux/dev_orchestrator.sh mi-pc          # hostname de Tailscale, o la IP 100.x.y.z
```

Frena el servicio systemd (`STT_SERVICE`, default `rai26-stt`) o cualquier
`main.py` suelto, y corre `main.py` en primer plano con `ORCHESTRATOR_IP`
pisado por variable de entorno (el `.env` queda intacto). **Ctrl+C** vuelve
a levantar el servicio si estaba corriendo. Para forzar la vuelta a la Jetson:

```bash
./linux/dev_orchestrator.sh --restore
```

Al arrancar imprime la IP Tailscale de la Pi y avisa si nada escucha en
`IP:9000` (orquestador apagado o firewall).

## Running headless (optional)

To keep the script running after disconnecting SSH, the simplest options are `tmux` or `screen`:

```bash
sudo apt install -y tmux
tmux new -s stt
source .venv/bin/activate && python linux/main.py
# Detach with Ctrl+B then D. Re-attach later with: tmux attach -t stt
```

For a real service, wrap it in a `systemd` unit — out of scope here.
