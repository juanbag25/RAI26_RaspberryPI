#!/bin/bash
# Calibración del mic en la Pi: frena el servicio del STT (que tiene el mic
# abierto), corre linux/calibrate.py con la venv y vuelve a levantar el
# servicio al terminar, aunque se cancele con Ctrl+C. Los argumentos pasan
# tal cual a calibrate.py (--dry-run, --yes, --seconds N).
#
#   ./calibrate_stt.sh
#   ./calibrate_stt.sh --dry-run
SERVICE="${STT_SERVICE:-rai26-stt}"
cd "$(dirname "$(readlink -f "$0")")" || exit 1

if [ ! -x .venv/bin/python ]; then
    echo "✖ falta la venv en $(pwd)/.venv (ver linux/README.md, sección 3)"
    exit 1
fi

restart=0
if systemctl is-active --quiet "$SERVICE"; then
    echo "· frenando el servicio $SERVICE (tiene el mic abierto)..."
    sudo systemctl stop "$SERVICE" || exit 1
    restart=1
fi

cleanup() {
    if [ "$restart" = 1 ]; then
        echo "· levantando de nuevo $SERVICE..."
        sudo systemctl start "$SERVICE" && echo "✓ $SERVICE corriendo (con los valores nuevos)"
    fi
}
trap cleanup EXIT

cd linux && ../.venv/bin/python calibrate.py "$@"
