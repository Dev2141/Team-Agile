# ThreatSense — Quick Start Script
# Run this from the ThreatSense folder in PowerShell

Write-Host ""
Write-Host "╔══════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     ThreatSense AI-DVR Launcher      ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$root = Split-Path -Parent $MyInvocation.MyCommand.Path

# Check venv
if (-not (Test-Path "$root\.venv")) {
    Write-Host "[SETUP] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv "$root\.venv"
    Write-Host "[SETUP] Installing dependencies..." -ForegroundColor Yellow
    & "$root\.venv\Scripts\pip.exe" install -r "$root\requirements.txt" -q
    Write-Host "[SETUP] Done." -ForegroundColor Green
}

# Check Ollama
Write-Host "[CHECK] Testing Ollama at localhost:11434..." -ForegroundColor Yellow
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/tags" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
    Write-Host "[OK]    Ollama is running." -ForegroundColor Green
} catch {
    Write-Host "[WARN]  Ollama not detected. Start it with: ollama serve" -ForegroundColor Red
}

Write-Host ""
Write-Host "Starting services..." -ForegroundColor Cyan
Write-Host ""

# Start FastAPI backend in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; .\.venv\Scripts\uvicorn.exe api:app --host 0.0.0.0 --port 8100 --reload" -WindowStyle Normal

Start-Sleep -Seconds 2

# Start Streamlit in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$root'; .\.venv\Scripts\streamlit.exe run app.py --server.port 8501" -WindowStyle Normal

Start-Sleep -Seconds 3

Write-Host "╔══════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  Dashboard  → http://localhost:8501          ║" -ForegroundColor Green
Write-Host "║  VLM UI     → http://localhost:8100          ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Remember: Ollama must be running → ollama serve" -ForegroundColor Yellow
Write-Host ""
