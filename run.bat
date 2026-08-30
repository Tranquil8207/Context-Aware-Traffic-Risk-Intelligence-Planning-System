@echo off
cd /d "%~dp0"
echo Starting catris at http://localhost:8000
start "" "http://localhost:8000"
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
