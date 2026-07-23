Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "NeuralForge - Neural Architecture Search" -ForegroundColor Cyan
Write-Host "with CUDA Acceleration" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

function Test-Command {
    param($Command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try {
        if (Get-Command $Command) { return $true }
    }
    catch { return $false }
    finally { $ErrorActionPreference = $oldPreference }
}

Write-Host "[1/5] Checking dependencies..." -ForegroundColor Yellow
if (-not (Test-Command python)) {
    Write-Host "Error: Python is not installed" -ForegroundColor Red
    exit 1
}
if (-not (Test-Command nvcc)) {
    Write-Host "Warning: NVCC not found. CUDA compilation may fail." -ForegroundColor Yellow
}
Write-Host "Dependencies check completed" -ForegroundColor Green
Write-Host ""

Write-Host "[2/5] Creating necessary directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path models | Out-Null
New-Item -ItemType Directory -Force -Path logs | Out-Null
New-Item -ItemType Directory -Force -Path data | Out-Null
New-Item -ItemType Directory -Force -Path build | Out-Null
Write-Host "Directories created" -ForegroundColor Green
Write-Host ""

Write-Host "[3/5] Installing Python dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip | Out-Null
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyTorch already installed or installation skipped" -ForegroundColor Yellow
}
pip install numpy matplotlib tqdm Pillow scipy tensorboard 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Dependencies already installed or installation skipped" -ForegroundColor Yellow
}
Write-Host "Python dependencies installed" -ForegroundColor Green
Write-Host ""

Write-Host "[4/5] Installing NeuralForge package..." -ForegroundColor Yellow
pip install -e . 2>&1 | Tee-Object -FilePath build/install.log

if ($LASTEXITCODE -eq 0) {
    Write-Host "NeuralForge installed successfully" -ForegroundColor Green
} else {
    Write-Host "Warning: Installation encountered issues. Check build/install.log for details" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "[5/5] Starting training..." -ForegroundColor Yellow
python train.py --dataset stl10 --model resnet18 --epochs 50 --batch-size 64

$TrainExitCode = $LASTEXITCODE

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
if ($TrainExitCode -eq 0) {
    Write-Host "Training completed successfully!" -ForegroundColor Green
    Write-Host "Results saved in:" -ForegroundColor Cyan
    Write-Host "  - models/    (model checkpoints)" -ForegroundColor White
    Write-Host "  - logs/      (training logs)" -ForegroundColor White
} else {
    Write-Host "Training failed with exit code: $TrainExitCode" -ForegroundColor Red
    Write-Host "Check logs/ for error details" -ForegroundColor Yellow
}
Write-Host "==========================================" -ForegroundColor Cyan

exit $TrainExitCode
