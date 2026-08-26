param(
    [switch]$Mock,
    [switch]$NoFrontend,
    [int]$ApiPort = 8000,
    [Alias("Days")]
    [ValidateRange(0, 3650)]
    [int]$BackfillDays = 0
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Local virtual environment not found. Run .\setup_local.ps1 first."
}

if (-not $Mock -and -not (Test-Path ".env")) {
    throw ".env not found. Copy .env.example to .env and configure PostgreSQL, or run .\start_project.ps1 -Mock for the temporary demo database."
}

$launcherArgs = @("local_launcher.py", "--api-port", "$ApiPort")
if ($Mock) { $launcherArgs += "--mock" }
if ($NoFrontend) { $launcherArgs += "--no-frontend" }
if ($BackfillDays -gt 0) {
    if ($Mock) {
        throw "-BackfillDays is available only in production mode. Demo history is already seeded."
    }
    $launcherArgs += @("--backfill-days", "$BackfillDays")
}

& $python @launcherArgs
exit $LASTEXITCODE
