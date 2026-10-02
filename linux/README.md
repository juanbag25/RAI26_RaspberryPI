# STT Project — Linux / Raspberry Pi

Linux/ARM64 deployment of the STT project, targeted at a **Raspberry Pi 5** running Raspberry Pi OS 64-bit (Bookworm). Mirrors the Windows code with one platform tweak: ALSA `plughw` routing so USB mics that don't expose 16 kHz natively still work.

For the project overview, model sizes and tuning notes, see the [root README](../README.md).

## Hardware

- Raspberry Pi 5 (4 GB or 8 GB)
- Raspberry Pi OS 64-bit (Bookworm)
- **ReSpeaker USB Mic Array v2.0** (XMOS XVF-3000, 4 mics, firmware de 6
  canales) — ver [Mic array](#mic-array-respeaker-usb-mic-array-v20). Sin él
  funciona con cualquier mic USB (mono), sin foco por dirección.
- **Sin parlante**: la Pi no emite sonido. Todo lo audible (respuestas, «Sí,
  dime», chimes) sale por el parlante de la Jetson (orquestador).

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

Con el ReSpeaker, además, permiso sobre su interfaz de control USB (DoA,
parámetros DSP, LEDs) sin root:

```bash
sudo cp linux/99-respeaker.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger
sudo usermod -a -G plugdev "$USER"   # y volver a loguearse
python linux/respeaker.py            # tiene que listar los parámetros
```

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
lento). Probalo solo con `python wake_spotter.py`.

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

## Mic array: ReSpeaker USB Mic Array v2.0

Si está conectado, `main.py` lo detecta solo (`RESPEAKER_ENABLED=false` lo
ignora). Firmware de 6 canales, 16 kHz (verificado: viene así de fábrica):

| Canal | Qué es | Para qué se usa |
|---|---|---|
| ch0 | audio procesado por el chip: beamforming + supresión de ruido + AGC | spotter de «oye rai» y Whisper (se transcribe mejor que los crudos) |
| ch1-4 | los 4 mics crudos | **nivel** del filtro de cercanía (sin AGC: el AGC de ch0 subía 2-5x a la gente lejana) |
| ch5 | lo que el array reproduce por su jack | nada (siempre 0: el parlante está en la Jetson) |

Por la interfaz de control USB ([`respeaker.py`](respeaker.py)):

- **Parámetros DSP**, fijados en cada arranque porque el chip los olvida al
  cortarle la corriente (`RESPEAKER_PARAMS`): eco apagado (no hay referencia
  del parlante), pasa-altos a 125 Hz (retumbe de motores), supresión de ruido
  estacionaria y no estacionaria, AGC. `python respeaker.py` lista todos;
  `python respeaker.py NOMBRE VALOR` prueba uno al vuelo.
- **Dirección de la voz (DoA)** a ~20 lecturas/s, con el VAD del chip.
- **LEDs**: apagados dormido; despierto, el firmware ilumina hacia la voz.

### Foco por dirección

Al oír «oye rai» el robot fija la **dirección** de quien lo dijo, y mientras
dure la ventana sólo atiende frases que vengan de ahí ([`doa.py`](doa.py)):

```
SPOT  ✓ oí «oye ray»
WAKE  ✓ DESPIERTO por audio, ventana 25s
DOA   ✓ foco en 32° (14 lecturas con voz)
>>> (otra persona, del otro lado)  ¿y eso qué es?
DOA   ✗ DESCARTADO fuera de foco  dir=205° foco=32°±35 en_foco=8% lecturas=17
>>> (la persona del foco, más bajo)  vení para acá
STT   · «Vení para acá.» (0.58s)
>>> (la otra persona)  oye rai
DOA   ✓ NUEVO FOCO dir 32°→205° (12 lecturas)
```

Medido con el array real: cuando hay voz, ~2/3 de las lecturas caen a pocos
grados de la persona y ~1/3 son reflexiones de la sala; en silencio el ángulo
queda pegado al último valor. Por eso sólo se usan lecturas con voz y se exige
una **fracción** dentro del foco (`DOA_MIN_IN_FOCUS`, 40 %), no que toda la
frase lo esté. Una frase con menos de `DOA_MIN_SAMPLES` lecturas no se juzga
(se acepta).

- **Traspaso**: otro «oye rai» estando despierto pasa el foco a quien lo dijo
  (el robot contesta «Sí, dime» de nuevo), siempre que suene cerca (filtro de
  cercanía general) o venga del foco actual. Un «rai» lejano no se lo roba.
- **No hace falta hablar tan fuerte como el «oye rai»**: con el array no hay
  exigencia de nivel relativa al wake (`ATTENTION_LEVEL_RATIO_ARRAY=0`); sin
  array, la referencia es la primera instrucción, no el wake.
- **Ruido propio del robot**: los ventiladores y motores están fijos respecto
  del array. Frases que vienen mayormente de `DOA_BLOCKED_SECTORS` se
  descartan siempre (y no despiertan).

### Calibración en el robot (montado, en el lugar donde trabaja)

**Automática (recomendada):**

```bash
./calibrate_stt.sh                   # --dry-run para sólo mirar, --yes para no confirmar
```

`calibrate_stt.sh` (raíz del repo) frena el servicio `rai26-stt` —que tiene
el mic abierto: dos procesos no pueden usar el ReSpeaker a la vez—, corre
`linux/calibrate.py` con la venv (`.venv/bin/python`; el `python` del sistema
no tiene las dependencias) y vuelve a levantar el servicio al terminar, aunque
se cancele. Pide la clave de `sudo` para frenar/levantar el servicio.

Te va diciendo qué hacer en 6 pasos de ~10 s (cualquiera se saltea con «s»):
ambiente con el robot prendido y nadie hablando, robot caminando, robot
hablando, vos de frente a ~1 m, alguien hablando desde el fondo, y «oye rai»
3 veces. Con eso calcula y escribe en `linux/.env` (backup del anterior en
`.env.bak-<fecha>`, gitignoreado): `DOA_FORWARD_OFFSET_DEG`,
`DOA_BLOCKED_SECTORS`, `DOA_SPEAKER_SECTOR`, `DOA_MIN_IN_FOCUS`,
`NEAR_RMS_THRESHOLD`, `RMS_THRESHOLD`, `NEAR_SNR_RATIO` y `NOISE_FLOOR_MAX`.
Avisa si algo salió raro (alguien habló en un paso de silencio, cerca y lejos
suenan parecido, el spotter no reconoce el «oye rai»). Sin ReSpeaker calibra
sólo los niveles.

Ningún umbral de nivel puede quedar por encima del **60 % de tu voz
cercana** (paso «de frente»): ni el ruido del ambiente ni el de marcha los
empujan más arriba, porque el robot dejaría de escucharte. Si el ruido llega
a ese techo, la calibración avisa en vez de subir el umbral.

**A mano**, si querés ver los números en vivo:

```bash
python linux/mic_level.py --doa
```

1. Hablale **de frente**: si la dirección no da ~0°, poné el valor `chip=` en
   `DOA_FORWARD_OFFSET_DEG`.
2. Robot prendido (ventiladores; y caminando) **sin nadie hablando**, Ctrl+C:
   los picos del histograma son ruido propio -> `DOA_BLOCKED_SECTORS`
   (ej. `170-200,350-10`).
3. Recalibrá `NEAR_RMS_THRESHOLD` como dice *Foco del mic* (abajo): el nivel
   ahora es el de los mics crudos, ~10 dB más bajo que el del mic viejo.
4. Prueba: A dice «oye rai» y una instrucción más bajo -> se acepta; B, desde
   otro lado, sin wake -> `DOA ✗ fuera de foco`; B dice «oye rai» -> `NUEVO
   FOCO`.

```bash
DOA_FORWARD_OFFSET_DEG=0
DOA_BLOCKED_SECTORS=
DOA_TOLERANCE_DEG=35      # cuánto puede apartarse una lectura del foco
DOA_MIN_IN_FOCUS=0.4      # bajalo si te ignora; subilo si entran otras voces
RESPEAKER_PARAMS=AGCMAXGAIN=10,HPFONOFF=3   # pisa parámetros DSP
```

### Ruido del robot caminando

El ruido de marcha (pasos, motores, vibración del cuerpo) es casi tan fuerte
como una voz cercana, así que **subir umbrales no lo resuelve**: si el umbral
supera al ruido, también supera a la persona. Lo que sí ayuda, de mayor a
menor impacto:

1. **Montaje del mic**: gran parte del ruido de marcha entra como vibración
   por la estructura, no por el aire. El array va sobre goma/espuma (no
   atornillado rígido al cuerpo), lo más arriba posible y lejos de motores y
   ventiladores.
2. **Filtros del chip** (sin tocar código, `RESPEAKER_PARAMS` en `.env`):
   pasa-altos más alto `HPFONOFF=3` (180 Hz, corta golpes graves de los
   pasos) y más supresión de ruido no estacionario `GAMMA_NN=1.5`, `MIN_NN=0.2`.
   Probalo con `python respeaker.py NOMBRE VALOR` antes de dejarlo fijo.
3. **Dirección**: el ruido de marcha no viene de un punto fijo, así que sus
   lecturas de DoA se dispersan y no llegan al `DOA_MIN_IN_FOCUS` del foco.
4. **Hablarle cerca**: con el robot en marcha la diferencia la hace la
   distancia; la calibración avisa si el ruido llega a tu voz.

### Orden de corte mientras el robot habla (experimental)

Por default, mientras el robot habla el mic se descarta entero
(`SPEAK_LISTEN_MODE=mute`). Con `SPEAK_LISTEN_MODE=keyword` el spotter sigue
escuchando **sólo** «para rai» / «basta rai» (`SPEAK_STOP_PHRASES`) o «oye
rai»: si viene de fuera del sector del parlante (`DOA_SPEAKER_SECTOR`, se mide
con `mic_level.py --doa` mientras el robot habla) y suena `SPEAK_BARGE_RATIO`
veces más fuerte que el parlante, le manda `stop` al orquestador, que se calla
(y con «oye rai», además contesta «Sí, dime»). Dejarlo en `mute` hasta probar
que no se corta solo.

## Foco del mic (rechazo de campo lejano)

Este filtro corre siempre, con o sin array (con el ReSpeaker, sobre el nivel
de los mics crudos). El mic es omnidireccional y webrtcvad sólo sabe decir "esto es voz humana", no
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
VAD   ✗ DESCARTADO lejana/floja  nivel=0.0263 umbral=0.0550 ruido=0.0036
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

### Modo `audio` (default): spotter local + «Sí, dime»

Como un celular con Siri: un detector chico corre **siempre** sobre el audio
([`wake_spotter.py`](wake_spotter.py), Vosk con gramática cerrada), y mientras
el robot duerme **no se manda nada a Groq**. Al reconocer "oye rai" la Pi le
avisa al orquestador (`@@event:awake`) y el robot contesta **«Sí, dime»** por
el parlante de la Jetson (frase pre-generada, ~0.6 s; la Pi está muteada
mientras suena). Recién ahí las frases van a Groq.

```
>>> ¿viste el partido de ayer?
VAD   ✓ voz 1200 ms nivel=0.0812 ruido=0.0041
WAKE  ✗ DESCARTADO dormido: no transcribo hasta oír «oye rai»  nivel=0.0812
>>> oye rai
SPOT  ✓ oí «oye ray»
WAKE  ✓ DESPIERTO por audio, ventana 25s
VAD   ✗ DESCARTADO era el «oye rai», no hace falta transcribirlo  voz_ms=780
   (robot: «Sí, dime»)
>>> vení para acá
VAD   ✓ voz 900 ms nivel=0.1490 ruido=0.0041
STT   · «Vení para acá.» (0.61s)
NET   ✓ enviado «Vení para acá.» (0.01s)
```

- La frase de wake tiene que sonar *parecido*, no exacto: Vosk sólo puede
  devolver una de `WAKE_PHRASES` o `[unk]`, así que "hola rai" también suele
  disparar como "oye ray". "rai" no es palabra del español y sale como
  "ray"/"rey"; por eso las tres variantes están en el default.
- Decir "oye rai" mientras ya está despierto re-engancha el foco a quien lo
  dijo (y el robot contesta «Sí, dime» otra vez): otra persona puede tomar la
  palabra sin esperar a que se duerma.
- Eventos al orquestador (mismo socket del texto, prefijo `@@event:`):
  `awake` en **cada** wake, `asleep` cuando vence la ventana (`WAKE_WINDOW_S`;
  el orquestador hace sonar un chime grave), `doa:<grados>` con la dirección
  fijada y `stop` (orden de corte). Todo lo que suena manda
  `SPEAK_START`/`SPEAK_END`, así el mic no se escucha a sí mismo.
- El texto que llega a Groq después del wake pasa igual por el filtro de
  texto de abajo: si Whisper escribe "rai vení" se recorta a "vení".

### Modo `text`: sobre la transcripción

Sin modelo extra: cada frase que pasa el VAD se transcribe y se busca "rai" en
el texto ([`wake_word.py`](wake_word.py)). Más lento (~2 s hasta que se entera)
y gasta Groq aunque duerma; es el fallback si Vosk no arranca.

```
>>> ¿viste el partido de ayer?
WAKE  ✗ DESCARTADO dormido y no dijo mi nombre  texto=«¿viste el partido de ayer?»
>>> Rai, vení para acá
WAKE  ✓ DESPIERTO por «Rai, vení para acá» foco=0.0790 minimo=0.0474
NET   ✓ enviado «vení para acá» (0.01s)
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
  quien lo dijo. Log: `WAKE ✗ DESCARTADO más flojo que quien me llamó`.
- Después queda despierto `WAKE_WINDOW_S` segundos para seguir la conversación
  sin repetir el nombre. Cada frase aceptada renueva la ventana, y también la
  renueva el `SPEAK_END` del orquestador (acaba de contestar: lo natural es que
  le sigan hablando).

Desde `linux/.env`:

```bash
WAKE_WORD_ENABLED=true    # false = como antes, atiende todo lo que pasa el VAD
WAKE_MODE=audio           # audio | text
WAKE_PHRASES=oye rai,oye ray,oye rey   # agregá "hola rai", "che rai"...
WAKE_WINDOW_S=25
ATTENTION_LEVEL_RATIO=0.25      # sin array: nivel mínimo relativo a la conversación; 0 = off
ATTENTION_LEVEL_RATIO_ARRAY=0   # con array manda la dirección
```

## Tuning

All knobs live in [`linux/config.py`](config.py): `BACKEND`, `MODEL_SIZE`,
`LANGUAGE`, `STT_PROMPT`, `VAD_AGGRESSIVENESS`, `SILENCE_MS`,
`PRE_SPEECH_PADDING_MS`, `MIN_UTTERANCE_MS`, los del foco del mic
(`RMS_THRESHOLD`, `NEAR_RMS_THRESHOLD`, `NEAR_SNR_RATIO`, `ONSET_SPEECH_FRAMES`,
`NOISE_FLOOR_*`), los del wake word (`WAKE_*`, `ATTENTION_*`), los del array
(`RESPEAKER_*`, `DOA_*`) y los de mientras habla (`SPEAK_*`). Los más
usados se pueden pisar desde `linux/.env` — ver `.env.example`. Para el resto,
ver el root README.

## Logs y diagnóstico

Cada línea es un evento: hora, **etapa** (siempre del mismo color) y un símbolo
que dice qué pasó. Sin buffer, así que sirve igual por SSH, `tmux` o redirigido
a un archivo.

```
HH:MM:SS.mmm ETAPA  símbolo mensaje
```

| Símbolo | Significa |
|---|---|
| `✓` verde | la frase pasó esta etapa (o algo se hizo bien) |
| `✗ DESCARTADO <razón>` amarillo | la frase **no sigue**; al lado, sólo los valores que explican la razón |
| `!` amarillo | aviso |
| `✖` rojo | error |
| `·` | info |
| gris | contexto (heartbeat, estado) |

Etapas, en el orden en que una frase las recorre: `AUDIO` (mic) → `SPOT`
(spotter "oye rai") → `VAD` (voz + cercanía) → `WAKE` (despierto/dormido y
foco) → `STT` (Groq) → `NET` (orquestador). `CTRL` son los SPEAK_START/END del
orquestador y `HB` el heartbeat.

Una frase que llega hasta el robot se ve así:

```
AUDIO ✓ primer frame del mic: escuchando
SPOT  ✓ oí «oye rai»
WAKE  ✓ DESPIERTO por audio, 25s foco=0.0812 minimo=0.0487
VAD   ▶ voz rms=0.0812 abre=0.0200
VAD   ✓ voz 1230 ms nivel=0.0790 ruido=0.0041
STT   · «vení para acá» (0.84s)
WAKE  sigue la charla, ventana +25s foco=0.0790
NET   ✓ enviado «vení para acá» (0.01s)
```

### Por qué NO escuchó

Todo descarte sale como `✗ DESCARTADO <razón>` en la etapa que lo decidió, con
los números que lo explican. Razones posibles:

| Línea | Qué pasó | Qué tocar |
|---|---|---|
| `VAD ✗ DESCARTADO muy corta  voz_ms=150 minimo_ms=400` | un golpe, una sílaba | `MIN_UTTERANCE_MS` |
| `VAD ✗ DESCARTADO lejana/floja  nivel=… umbral=… ruido=…` | voz de fondo, no de cerca | `NEAR_RMS_THRESHOLD` por debajo del `nivel` medido (o hablar más cerca) |
| `VAD ✗ DESCARTADO el robot empezó a hablar` | llegó `SPEAK_START` a mitad de frase | nada: es lo esperado |
| `VAD ✗ DESCARTADO era el «oye rai»…` | la frase de wake no se transcribe | nada |
| `WAKE ✗ DESCARTADO dormido: no transcribo hasta oír «oye rai»` | modo audio, nadie lo llamó | decir la frase de wake; ver `spotter oyó` en `HB` |
| `WAKE ✗ DESCARTADO más flojo que quien me llamó  nivel=… minimo=… foco=…` | otra persona más lejos que quien dijo "rai" | `ATTENTION_LEVEL_RATIO` |
| `WAKE ✗ DESCARTADO dormido y no dijo mi nombre  texto=«…»` | modo texto: transcribió pero no empezó con "rai" | `WAKE_WORDS` si Whisper escribió el nombre raro |
| `STT ✗ DESCARTADO Groq no devolvió texto` | error de Groq (línea `✖` arriba) o audio inaudible | API key / red / ganancia |
| `STT ✗ DESCARTADO eco del prompt de Whisper` | Whisper repitió `STT_PROMPT`: ruido | nada |
| `NET ✖ …` | transcribió bien pero no llegó al orquestador | IP / puerto / firewall |

`LOG_DEBUG=1` agrega lo que por defecto no sale: cada frame de voz que no llega
al umbral de apertura, los parciales de Vosk, el detalle de cada envío TCP.

### Heartbeat

Cada `LOG_HEARTBEAT_S` segundos (default 10) sale una línea `HB` en gris con lo
que vio el mic en esa ventana; sale en amarillo (`!`) si detecta un problema:

```
HB    voz 1.2s, fuerte 0.9s · ruido=0.0044 abre=0.0200 cerca=0.0550 · utt ok=1 desc=0 · total stt=1 env=1 · despierto foco=0.0790 minimo=0.0474 · spotter oyó «oye rai»
```

| Síntoma en `HB` | Significa | Qué tocar |
|---|---|---|
| `SIN AUDIO DEL MIC` | PortAudio no entrega audio | mic equivocado (`AUDIO_INPUT_DEVICE`), cable, `arecord -l` |
| `pocos frames` | el loop de captura se traba | mirar `AUDIO ! input overflow`, CPU |
| `sin voz (rms max 0.000x)` aunque hables | el mic está pero casi mudo | subir ganancia (`alsamixer`), otro device |
| `voz 1.2s, fuerte 0.0s` | webrtcvad ve voz pero no llega a `abre` | bajar `RMS_THRESHOLD` / subir ganancia; `LOG_DEBUG=1` muestra cada frame |
| `utt ok=0 desc=N` | se abren y se descartan | mirar los `✗ DESCARTADO` de arriba |
| `stt` no crece con `ok>0` | el hilo de STT está trabado | mirar `STT ✖` (Groq / API key / red) |
| `FALLIDAS=N` | no llega al orquestador | `NET ✖`: IP/puerto/firewall |
| `MUTE` todo el tiempo | se perdió un `SPEAK_END` | expira solo a los `MUTE_TIMEOUT_S`; revisar el orquestador |
| `spotter oyó «oye»` y nunca la frase completa | Vosk no reconoce la variante | `LOG_DEBUG=1`, `python wake_spotter.py`, agregar variante a `WAKE_PHRASES` |
| `spotter atrasado` | la Pi no da abasto | CPU |

Colores: se activan solos si stdout es una terminal. `LOG_COLOR=1` los fuerza
(útil con `| tee`), `LOG_COLOR=0` los apaga.

## Alimentación

Al arrancar (y cada `POWER_LOG_S` segundos, default 300) `main.py` loguea lo
que la Pi ve de su alimentación ([`battery.py`](battery.py)); también se puede
correr suelto con `python linux/battery.py`:

```
POWER entrada 5V real: 5.08 V
POWER throttled=0x0  (ok)
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
- **Escucha pero nunca transcribe nada** — mirá la línea `HB` y la tabla de
  la sección *Logs y diagnóstico*: dice en qué etapa se queda (mic, VAD,
  umbral, STT, wake word o red).
- **`GROQ_API_KEY` not found** — make sure `linux/.env` exists and you ran the script from a shell where the venv is activated; `python-dotenv` loads it at import time in `main.py`.
- **Model fails to load (local backend)** — verify `models/faster-whisper-<MODEL_SIZE>/` contains all four files (`config.json`, `model.bin`, `tokenizer.json`, `vocabulary.txt`) and that `MODEL_SIZE` in `config.py` matches the folder name.
- **El robot no me escucha a mí** — buscá la línea `✗ DESCARTADO`: dice la
  razón. `lejana/floja` → `NEAR_RMS_THRESHOLD` está muy alto para tu mic (poné
  el umbral debajo del `nivel` que imprime). `dormido y no dijo mi nombre` →
  transcribió bien pero no le dijiste "rai", o Whisper escribió el nombre de
  una forma que no está en `WAKE_WORDS`.
- **Sigue enganchando conversaciones ajenas** — subí `NEAR_RMS_THRESHOLD` (y/o
  `NEAR_SNR_RATIO`) con `mic_level.py` en la mano; el wake word tapa el resto.
- **High CPU / slow transcription** — on a Pi 5, stick to `tiny`, `base` or `small` for the local backend, or use the Groq backend.
- **No despierta con "oye rai"** — mirá `spotter oyó «…»` en la línea `HB`: es lo último que Vosk entendió. Si dice `«oye»` y nunca `«oye ray»`, probá `LOG_DEBUG=1` y `python wake_spotter.py`, hablá más cerca, o agregá la variante que veas a `WAKE_PHRASES`. Si aparece `spotter atrasado`, la Pi no da abasto.
- **Despierta solo** — sacá variantes de `WAKE_PHRASES` (dejá sólo `oye rai,oye ray`) o subí `WAKE_COOLDOWN_S`.
- **`DOA ✗ fuera de foco` contra quien lo llamó** — bajá `DOA_MIN_IN_FOCUS` (0.3) o subí `DOA_TOLERANCE_DEG`; mirá con `mic_level.py --doa` cuánto se dispersa el ángulo en esa sala.
- **`DoA CAÍDO` en el heartbeat / `ReSpeaker ... no responde por USB`** — permisos: falta la regla udev (sección 2) o el grupo `plugdev`. `python respeaker.py` lo confirma.
- **El ReSpeaker no aparece como 6 canales** — `arecord -D plughw:<card>,0 --dump-hw-params -d 1 /dev/null` tiene que decir `CHANNELS: 6`; si dice 1, tiene el firmware de 1 canal (flashear el de 6 con `dfu.py` de `respeaker/usb_4_mic_array`).

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
