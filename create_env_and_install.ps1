# Creates a venv on F: and installs required packages with cache/temp redirected to F:
param(
    [string]$EnvPath = "F:\Thesis\Experimentation\envs\nasbench_env",
    [string]$CacheDir = "F:\Thesis\Experimentation\pip_cache",
    [string]$TempDir = "F:\Thesis\Experimentation\tmp"
)

Write-Host "Creating folders..."
New-Item -ItemType Directory -Force -Path $EnvPath | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

# Set environment variables for this process so pip uses F: for cache and temp
$env:PIP_CACHE_DIR = $CacheDir
$env:TEMP = $TempDir
$env:TMP = $TempDir

Write-Host "Creating virtual environment at $EnvPath"
python -m venv $EnvPath

if (-Not (Test-Path "$EnvPath\Scripts\Activate.ps1")) {
    Write-Error "Virtual environment creation failed or Python not found. Ensure 'python' on PATH points to a valid Python installation."
    exit 1
}

Write-Host "Activating virtual environment..."
# Activate the venv for the rest of the script
& "$EnvPath\Scripts\Activate.ps1"

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing lightweight dependencies (numpy, h5py, psutil)..."
pip install --no-warn-script-location numpy h5py psutil

Write-Host "Installing CPU-only PyTorch (recommended for local unpacking). This may take a few minutes."
# CPU-only wheels from PyTorch: uses the official cpu index
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio --no-warn-script-location

Write-Host "All done. To use the environment run:`n`n    $Env:VIRTUAL_ENV = '$EnvPath'`n    & '$EnvPath\Scripts\Activate.ps1'`n
Or in a new PowerShell session:`n    & '$EnvPath\Scripts\Activate.ps1'"
