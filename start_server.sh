#!/usr/bin/env bash
# ExoNit server - manual / development launcher (Linux/macOS).
# For a permanent install use the systemd unit ops/systemd/exonit.service.
# All configuration comes from .env (copy .env.example to .env and edit it first).
set -euo pipefail
cd "$(dirname "$0")"
if [ -d venv ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi
if [ -f .env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
else
    echo "warning: no .env found; copy .env.example to .env and edit it" >&2
fi
exec python3 run_nightly.py serve --workers "${EXONIT_SERVER_WORKERS:-1}"
