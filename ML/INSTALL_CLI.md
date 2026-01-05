# Installing NeuralForge CLI

This guide explains how to install NeuralForge so you can use the `NeuralForgeAI` command from anywhere.

## Installation Steps

### Install via PIP
```bash
pip install NeuralForgeAI
```

### Option 1: Install in Editable Mode (Recommended for Development)

This allows you to make changes to the code and use them immediately:

```bash
# From the NeuralForge directory
pip install -e .
```

### Option 2: Regular Installation

```bash
pip install .
```

### Option 3: Install without CUDA Extensions (CPU-only)

If you don't have CUDA or want a faster install:

```bash
pip install --no-build-isolation -e .
```

## Verify Installation

After installation, verify the commands are available:

```bash
# Check if commands are installed
NeuralForgeAI --help
neuralforge --help
neuralforge-train --help
neuralforge-test --help
neuralforge-gui --help
neuralforge-nas --help
```

## Usage Examples

Once installed, you can use NeuralForge from anywhere on your system:

### Basic Usage
```bash
# Train on CIFAR-10 with ResNet18
NeuralForgeAI --dataset cifar10 --model resnet18 --epochs 50 --batch-size 64

# Train on STL-10
NeuralForgeAI --dataset stl10 --model resnet18 --epochs 100 --batch-size 64

# Train on MNIST
NeuralForgeAI --dataset mnist --model simple --epochs 20
```

### Advanced Usage
```bash
# Customize optimizer and scheduler
NeuralForgeAI --dataset cifar100 --model resnet18 --epochs 100 \
              --batch-size 128 --lr 0.001 --optimizer adamw \
              --scheduler cosine --device cuda

# Use a config file
NeuralForgeAI --config my_config.json

# Synthetic dataset for quick testing
NeuralForgeAI --dataset synthetic --num-samples 1000 --epochs 5
```

## Available Commands

After installation, these commands will be available globally:

| Command | Description |
|---------|-------------|
| `NeuralForgeAI` | Main training command (same as `neuralforge`) |
| `neuralforge` | Training command |
| `neuralforge-train` | Explicit training command |
| `neuralforge-test` | Test trained models |
| `neuralforge-gui` | Launch GUI interface |
| `neuralforge-nas` | Neural Architecture Search |

## Troubleshooting

### Command not found after installation

If you get "command not found" after installation, try:

1. **Check if pip's bin directory is in PATH:**
   ```bash
   # On Linux/Mac
   echo $PATH | grep pip
   
   # On Windows
   echo %PATH%
   ```

2. **Find where pip installs scripts:**
   ```bash
   pip show neuralforgeai
   python -m site --user-base
   ```

3. **Run directly with Python:**
   ```bash
   python -m neuralforge.cli.train --help
   ```

### Import errors

If you get import errors, make sure PyTorch is installed:
```bash
pip install torch torchvision
```

### CUDA compilation errors

If CUDA compilation fails:
1. Install without CUDA extensions (CPU-only mode)
2. Or ensure you have CUDA Toolkit 11.0+ and compatible compiler installed

## Alternative: Use Without Installation

You can also run NeuralForge without installing it as a package:

```bash
# From the NeuralForge directory
python train.py --dataset cifar10 --model resnet18 --epochs 50 --batch-size 64
```

## Uninstalling

To uninstall NeuralForge:

```bash
pip uninstall neuralforgeai
```
