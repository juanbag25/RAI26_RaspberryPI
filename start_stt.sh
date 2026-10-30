#!/bin/bash
# Lo ejecuta el servicio systemd rai26-stt (ExecStart): trae lo último de
# GitHub, actualiza dependencias y arranca el STT con la venv. Sin red sigue
# con la copia local.
cd "$(dirname "$(readlink -f "$0")")" || exit 1
git pull --ff-only || echo "[start_stt] git pull falló, sigo con la copia local"
source .venv/bin/activate
pip install -q -r requirements.txt || true
exec python main.py
