#!/bin/bash
# Alfred startup script
cd /home/aialfred/alfred
source venv/bin/activate
export PYTHONPATH=/home/aialfred/alfred

# Build React frontend if package.json exists
if [ -f frontend/package.json ]; then
    echo "Building frontend..."
    cd frontend
    npm run build 2>&1 | tail -5
    cd ..
    echo "Frontend build complete."
fi

echo "Starting Alfred..."
python -m uvicorn core.api.main:app --host 0.0.0.0 --port 8400 --workers 1
