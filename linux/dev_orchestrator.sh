#!/usr/bin/env bash
# Apuntar el STT de la Pi a OTRO orquestador (p. ej. tu PC por Tailscale) sin
# tocar .env ni el servicio de producción.
#
#   ./linux/dev_orchestrator.sh <ip-o-hostname-tailscale>   # frena el servicio, corre main.py contra esa IP
#   ./linux/dev_orchestrator.sh --restore                   # vuelve a levantar el servicio (Jetson)
#
# Con Ctrl+C sale y, si el servicio estaba corriendo, lo vuelve a levantar solo.
#
# Qué hace:
#   1. Resuelve el destino (un hostname de Tailscale se traduce con `tailscale ip`).
#   2. Frena lo que esté corriendo: el servicio systemd (STT_SERVICE, default "stt")
#      si existe, y cualquier `linux/main.py` suelto (tmux, nohup).
#   3. Corre `python linux/main.py` en primer plano con ORCHESTRATOR_IP pisado
#      por variable de entorno (main.py hace load_dotenv() sin override, así
#      que el .env no se toca).
#
# Ojo: el orquestador de la PC también tiene que poder llegar a la Pi (puerto
# CTRL_PORT=9001, mensajes SPEAK_START/SPEAK_END). Configurale la IP de
# Tailscale de la Pi, que este script imprime al arrancar.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
# STT_SERVICE se puede fijar en linux/.env (STT_SERVICE=nombre-del-servicio).
if [ -z "${STT_SERVICE:-}" ] && [ -f "$HERE/.env" ]; then
    STT_SERVICE="$(grep -E '^STT_SERVICE=' "$HERE/.env" | tail -n1 | cut -d= -f2- | tr -d '"' | tr -d "'" || true)"
fi
SERVICE="${STT_SERVICE:-stt}"
PORT="${ORCHESTRATOR_PORT:-9000}"
CTRL_PORT="${CTRL_PORT:-9001}"

log() { printf '[dev] %s\n' "$*"; }
die() { printf '[dev] ERROR: %s\n' "$*" >&2; exit 1; }

service_exists() {
    command -v systemctl >/dev/null 2>&1 \
        && systemctl list-unit-files --type=service 2>/dev/null | grep -q "^${SERVICE}\.service"
}

service_active() {
    service_exists && systemctl is-active --quiet "$SERVICE"
}

stop_running() {
    if service_active; then
        log "frenando servicio ${SERVICE}.service"
        sudo systemctl stop "$SERVICE"
    fi
    # main.py suelto (tmux, nohup, otra terminal), lanzado desde cualquier
    # cwd: "python linux/main.py" o "python main.py". También mic_level.py y
    # wake_spotter.py, que agarran el mic igual.
    local pids
    pids="$(pgrep -f 'python[0-9.]* .*(main|mic_level|wake_spotter)\.py' || true)"
    if [ -n "$pids" ]; then
        log "matando main.py suelto (pids: $(echo "$pids" | tr '\n' ' '))"
        # sudo: si lo lanzó un servicio como root, kill sin sudo falla callado.
        # shellcheck disable=SC2086
        sudo kill $pids 2>/dev/null || true
        sleep 1
    fi
    # Si el puerto de control sigue tomado, hay OTRO main.py que no vimos
    # (servicio con otro nombre). Mejor frenar acá que arrancar y explotar.
    local owner
    owner="$(sudo ss -ltnp 2>/dev/null | grep ":${CTRL_PORT} " || true)"
    if [ -n "$owner" ]; then
        printf '%s\n' "$owner" >&2
        die "el puerto ${CTRL_PORT} sigue ocupado por el proceso de arriba. Si es un servicio," \
            "frenalo y/o poné su nombre: STT_SERVICE=<nombre> $0 ... (o en linux/.env)"
    fi
}

restore() {
    if service_exists; then
        log "levantando servicio ${SERVICE}.service (vuelve a la Jetson)"
        sudo systemctl start "$SERVICE"
        systemctl --no-pager --lines=0 status "$SERVICE" || true
    else
        log "no hay servicio ${SERVICE}.service; nada que restaurar" \
            "(si lo corrías en tmux, levantalo a mano)"
    fi
}

[ $# -ge 1 ] || die "uso: $0 <ip-o-hostname-tailscale> | --restore"

if [ "$1" = "--restore" ]; then
    stop_running
    restore
    exit 0
fi

TARGET="$1"
IP="$TARGET"
if ! [[ "$TARGET" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    command -v tailscale >/dev/null 2>&1 || die "tailscale no instalado y '$TARGET' no es una IP"
    IP="$(tailscale ip -4 "$TARGET" 2>/dev/null || true)"
    [ -n "$IP" ] || die "no pude resolver '$TARGET' en Tailscale (¿está en la misma tailnet? 'tailscale status')"
    log "$TARGET -> $IP"
fi

if command -v tailscale >/dev/null 2>&1; then
    MY_IP="$(tailscale ip -4 2>/dev/null || true)"
    [ -n "$MY_IP" ] && log "IP Tailscale de esta Pi: $MY_IP  <- ponela en el orquestador para el mute (puerto 9001)"
fi

# Aviso temprano si el orquestador no escucha todavía (no es fatal: main.py
# reintenta en cada frase).
if command -v nc >/dev/null 2>&1; then
    if nc -z -w 2 "$IP" "$PORT" 2>/dev/null; then
        log "orquestador responde en $IP:$PORT"
    else
        log "AVISO: nada escucha en $IP:$PORT todavía (¿orquestador levantado? ¿firewall de la PC?)"
    fi
fi

WAS_ACTIVE=0
service_active && WAS_ACTIVE=1
stop_running

cleanup() {
    trap - EXIT INT TERM
    if [ "$WAS_ACTIVE" = 1 ]; then
        echo
        restore
    fi
}
trap cleanup EXIT INT TERM

cd "$ROOT"
# shellcheck disable=SC1091
[ -f .venv/bin/activate ] && source .venv/bin/activate
log "corriendo linux/main.py contra $IP:$PORT (Ctrl+C para volver)"
ORCHESTRATOR_IP="$IP" python linux/main.py
