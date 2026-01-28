#!/bin/bash
# Alfred startup script
cd /home/aialfred/alfred
source venv/bin/activate
export PYTHONPATH=/home/aialfred/alfred

echo "Starting Alfred..."
python -m uvicorn core.api.main:app --host 0.0.0.0 --port 8400 --workers 1
