param([string]$VersionPython = "3.12")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & py "-$VersionPython" -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "No se pudo crear el entorno. Verifica: py -$VersionPython --version" }
}
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Error al actualizar pip" }
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Error al instalar requirements.txt" }
& ".\.venv\Scripts\python.exe" -m src.main
