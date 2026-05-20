#!/bin/bash
# Local dev startup script for EcoTrack AI Backend
set -e

echo "EcoTrack AI Backend - Local Dev"

# Create virtual env if missing
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt -q

# Copy .env if it doesn't exist
if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "Created .env from .env.example. Edit JWT_SECRET before deploying."
fi

# Initialize DB if needed
flask --app app init-db 2>/dev/null || python3 -c "
from app import app, db
with app.app_context():
    db.create_all()
    print('DB initialized.')
"

echo "Starting Flask on http://localhost:5000"
python3 app.py
