$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path ".venv")) {
    if (Get-Command py -ErrorAction SilentlyContinue) { py -3 -m venv .venv }
    else { python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw "Python virtual environment creation failed." }
}
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
pip install -r backend\requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Backend dependency installation failed." }
python backend\prepare_dataset.py
if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed." }
python backend\train_model.py
if ($LASTEXITCODE -ne 0) { throw "Model training failed." }
Push-Location frontend
npm install
if ($LASTEXITCODE -ne 0) { throw "Frontend dependency installation failed." }
Pop-Location
Write-Host "Setup complete. Start the app with .\scripts\start_project.bat"
