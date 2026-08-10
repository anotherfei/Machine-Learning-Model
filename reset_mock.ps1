$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$db = Join-Path $PSScriptRoot "mock_demo.db"
@($db, "$db-wal", "$db-shm") | ForEach-Object {
    if (Test-Path $_) { Remove-Item $_ -Force }
}

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Local virtual environment not found. Run .\setup_local.ps1 first."
}

& $python -c "from api.mock_main import initialize_mock_database; print('Mock database ready:', initialize_mock_database(reset=True))"
