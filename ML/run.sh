#!/bin/bash

echo "=========================================="
echo "NeuralForge - Neural Architecture Search"
echo "with CUDA Acceleration"
echo "=========================================="
echo ""

check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "Error: $1 is not installed"
        return 1
    fi
    return 0
}

echo "[1/5] Checking dependencies..."
check_command python3 || exit 1
check_command nvcc || echo "Warning: NVCC not found. CUDA compilation may fail."
echo "Dependencies check completed"
echo ""

echo "[2/5] Creating necessary directories..."
mkdir -p models
mkdir -p logs
mkdir -p data
mkdir -p build
echo "Directories created"
echo ""

echo "[3/5] Installing Python dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || echo "PyTorch already installed or installation skipped"
pip install numpy matplotlib tqdm Pillow scipy tensorboard || echo "Dependencies already installed or installation skipped"
echo "Python dependencies installed"
echo ""

echo "[4/5] Installing NeuralForge package..."
pip install -e . 2>&1 | tee build/install.log

if [ $? -eq 0 ]; then
    echo "NeuralForge installed successfully"
else
    echo "Warning: Installation encountered issues. Check build/install.log for details"
fi
echo ""

echo "[5/5] Starting training..."
python3 train.py --dataset stl10 --model resnet18 --epochs 50 --batch-size 64

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Results saved in:"
    echo "  - models/    (model checkpoints)"
    echo "  - logs/      (training logs)"
else
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "Check logs/ for error details"
fi
echo "=========================================="

exit $TRAIN_EXIT_CODE
