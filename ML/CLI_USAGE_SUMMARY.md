# NeuralForge CLI - Quick Reference

## Installation

```bash
# Install the package
pip install -e .

# Verify installation
NeuralForgeAI --help
```

## Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `NeuralForgeAI` | Train neural networks | `NeuralForgeAI --dataset cifar10 --model resnet18 --epochs 50` |
| `neuralforge` | Same as NeuralForgeAI | `neuralforge --dataset stl10 --model resnet18` |
| `neuralforge-train` | Explicit training | `neuralforge-train --dataset mnist --epochs 20` |
| `neuralforge-test` | Test models | `neuralforge-test --help` |
| `neuralforge-gui` | Launch GUI | `neuralforge-gui` |
| `neuralforge-nas` | Architecture search | `neuralforge-nas --help` |

## Quick Examples

### Basic Training
```bash
# CIFAR-10 with ResNet18
NeuralForgeAI --dataset cifar10 --model resnet18 --epochs 50 --batch-size 64

# STL-10 with custom settings
NeuralForgeAI --dataset stl10 --model resnet18 --epochs 100 --lr 0.001 --batch-size 64

# MNIST quick test
NeuralForgeAI --dataset mnist --model simple --epochs 10
```

### Advanced Usage
```bash
# Full customization
NeuralForgeAI --dataset cifar100 --model resnet18 --epochs 100 \
              --batch-size 128 --lr 0.001 --optimizer adamw \
              --scheduler cosine --device cuda --seed 42

# Using config file
NeuralForgeAI --config my_config.json

# Synthetic data for testing
NeuralForgeAI --dataset synthetic --num-samples 1000 --epochs 5
```

## Common Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | synthetic | Dataset name (cifar10, mnist, stl10, etc.) |
| `--model` | str | simple | Model architecture (simple, resnet18, efficientnet, vit) |
| `--epochs` | int | 50 | Number of training epochs |
| `--batch-size` | int | 32 | Batch size for training |
| `--lr` | float | 0.001 | Learning rate |
| `--optimizer` | str | adamw | Optimizer (adamw, adam, sgd) |
| `--scheduler` | str | cosine | LR scheduler (cosine, onecycle, none) |
| `--device` | str | auto | Device (cuda, cpu) |
| `--seed` | int | 42 | Random seed |

## Supported Datasets

- `cifar10` - CIFAR-10 (60K images, 10 classes, 32x32)
- `cifar100` - CIFAR-100 (60K images, 100 classes, 32x32)
- `mnist` - MNIST (70K images, 10 classes, 28x28)
- `fashion_mnist` - Fashion-MNIST (70K images, 10 classes, 28x28)
- `stl10` - STL-10 (13K images, 10 classes, 96x96)
- `tiny_imagenet` - Tiny ImageNet (200 classes, 64x64)
- `synthetic` - Synthetic data for testing

## Comparison: CLI vs Python Script

### Using CLI (After pip install)
```bash
# Use from anywhere
NeuralForgeAI --dataset stl10 --model resnet18 --epochs 50 --batch-size 64
```

**Pros:**
- ✅ Use from any directory
- ✅ Clean, simple syntax
- ✅ No need to write Python code
- ✅ Easy to integrate in scripts/workflows

### Using Python Script (Traditional)
```bash
# Must be in NeuralForge directory
python train.py --dataset stl10 --model resnet18 --epochs 50 --batch-size 64
```

**Pros:**
- ✅ Works without installation
- ✅ Easy to modify for custom needs

## Getting Help

```bash
# Show all available options
NeuralForgeAI --help

# Get help for specific commands
neuralforge-train --help
neuralforge-test --help
neuralforge-nas --help
```

## Documentation

- **[README.md](README.md)** - Overview and features
- **[INSTALL_CLI.md](INSTALL_CLI.md)** - Detailed installation guide
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide with examples
- **[DOCUMENTATION.md](DOCUMENTATION.md)** - Complete API reference
- **[DATASETS.md](DATASETS.md)** - Dataset information

## Troubleshooting

### Command not found
If `NeuralForgeAI` is not recognized:
1. Make sure you installed the package: `pip install -e .`
2. Check pip's scripts are in PATH
3. Use full Python path: `python -m neuralforge.cli.train`

### Import errors
Install required dependencies:
```bash
pip install torch torchvision numpy matplotlib tqdm pillow scipy tensorboard
```

### CUDA issues
For CPU-only installation:
```bash
pip install --no-build-isolation -e .
```
