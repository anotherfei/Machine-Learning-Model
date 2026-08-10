$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Setting up Spindle Condition Monitoring for local Windows execution..."

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found in PATH. Install Python first."
}
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm was not found in PATH. Install Node.js first."
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

$python = (Resolve-Path ".venv\Scripts\python.exe").Path
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt

Push-Location frontend
npm install
Pop-Location

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Edit it before production mode; mock mode can run immediately." -ForegroundColor Yellow
}

Write-Host "Local setup complete." -ForegroundColor Green
Write-Host "For the web demo: .\start_project.ps1 -Mock"
Write-Host "For production: configure .env + artifacts, then run .\start_project.ps1"
