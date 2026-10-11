#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python3 -u worker.py --config config.json
