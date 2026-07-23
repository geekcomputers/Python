# NeuralForge Documentation

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
   - [Command-Line Interface (CLI)](#command-line-interface-cli-usage)
   - [Python API](#python-api-usage)
3. [Architecture](#architecture)
4. [CUDA Kernels](#cuda-kernels)
5. [Neural Architecture Search](#neural-architecture-search)
6. [Training](#training)
7. [API Reference](#api-reference)

📚 **Quick Links:**
- [CLI Usage Summary](CLI_USAGE_SUMMARY.md) - Quick reference for CLI commands
- [Installation Guide](INSTALL_CLI.md) - Detailed installation instructions

## Installation

### Requirements
- Python 3.8+
- CUDA Toolkit 11.0+
- PyTorch 2.0+
- GCC/G++ 7.0+ (Linux) or MSVC 2019+ (Windows)

### Quick Install

**Option 1: Install as Package (Recommended)**
```bash
# Clone repository
git clone https://github.com/yourusername/neuralforge.git
cd neuralforge

# Install in editable mode (for development)
pip install -e .

# Or install normally
pip install .
```

**Option 2: Quick Setup Script**

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

**Windows:**
```powershell
.\run.ps1
```

**Option 3: Manual Install**
```bash
pip install torch torchvision numpy matplotlib tqdm Pillow scipy tensorboard
python setup.py install
```

After installation, you'll have access to these command-line tools:
- `NeuralForgeAI` - Main training command
- `neuralforge` - Alternative training command
- `neuralforge-train` - Explicit training command
- `neuralforge-test` - Model testing tool
- `neuralforge-gui` - GUI interface
- `neuralforge-nas` - Neural Architecture Search

## Quick Start

### Command-Line Interface (CLI) Usage

After installing NeuralForge, you can use it as a command-line tool:

#### Basic Examples
```bash
# Train on CIFAR-10 with ResNet18
NeuralForgeAI --dataset cifar10 --model resnet18 --epochs 50 --batch-size 64

# Train on STL-10 with custom learning rate
NeuralForgeAI --dataset stl10 --model resnet18 --epochs 100 --lr 0.001 --batch-size 64

# Train on MNIST with simple model
NeuralForgeAI --dataset mnist --model simple --epochs 20 --batch-size 128

# Train with specific optimizer and scheduler
NeuralForgeAI --dataset cifar100 --model resnet18 --epochs 100 \
              --optimizer adamw --scheduler cosine --lr 0.001
```

#### Available Arguments
```
--dataset          Dataset to use (cifar10, mnist, stl10, fashion_mnist, etc.)
--model            Model architecture (simple, resnet18, efficientnet, vit)
--epochs           Number of training epochs (default: 50)
--batch-size       Batch size (default: 32)
--lr               Learning rate (default: 0.001)
--optimizer        Optimizer (adamw, adam, sgd) (default: adamw)
--scheduler        LR scheduler (cosine, onecycle, none) (default: cosine)
--device           Device to use (cuda, cpu) (default: auto-detect)
--seed             Random seed (default: 42)
--num-samples      Number of samples for synthetic dataset (default: 5000)
--num-classes      Number of classes for synthetic dataset (default: 10)
--config           Path to config JSON file
```

#### Get Help
```bash
NeuralForgeAI --help
neuralforge --help
neuralforge-train --help
```

### Python API Usage

You can also use NeuralForge as a Python library:

#### Basic Training
```python
import torch
from neuralforge import Trainer, Config
from neuralforge.data.dataset import SyntheticDataset, DataLoaderBuilder
from neuralforge.models.resnet import ResNet18

config = Config()
config.batch_size = 32
config.epochs = 100

train_dataset = SyntheticDataset(num_samples=10000, num_classes=10)
val_dataset = SyntheticDataset(num_samples=2000, num_classes=10)

loader_builder = DataLoaderBuilder(config)
train_loader = loader_builder.build_train_loader(train_dataset)
val_loader = loader_builder.build_val_loader(val_dataset)

model = ResNet18(num_classes=10)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

trainer = Trainer(model, train_loader, val_loader, optimizer, criterion, config)
trainer.train()
```

### Command Line Training
```bash
python train.py --model resnet18 --batch-size 32 --epochs 50 --lr 0.001
```

## Architecture

### Project Structure
```
NeuralForge/
├── src/
│   ├── cuda/              # CUDA kernels
│   │   ├── kernels.cu     # Basic operations
│   │   ├── matmul.cu      # Matrix multiplication
│   │   ├── activations.cu # Activation functions
│   │   └── optimizers.cu  # Optimizer kernels
│   ├── cpp/               # C++ extensions
│   │   ├── extension.cpp  # PyBind11 bindings
│   │   └── operators.cpp  # Operator implementations
│   └── python/neuralforge/
│       ├── nn/            # Neural network modules
│       ├── optim/         # Optimizers and schedulers
│       ├── data/          # Data loading and augmentation
│       ├── nas/           # Neural architecture search
│       ├── utils/         # Utilities
│       └── models/        # Pre-built models
├── models/                # Saved model checkpoints
├── logs/                  # Training logs
└── examples/              # Example scripts
```

## CUDA Kernels

### Custom CUDA Operations

NeuralForge implements optimized CUDA kernels for:

#### Matrix Operations
- Tiled matrix multiplication
- Batched matrix multiplication
- Transpose operations
- GEMM with alpha/beta scaling

#### Activation Functions
- ReLU, LeakyReLU, ELU, SELU
- GELU, Swish, Mish
- Sigmoid, Tanh
- Softmax, LogSoftmax

#### Optimizers
- SGD with momentum
- Adam, AdamW
- LAMB (Layer-wise Adaptive Moments)
- RMSprop, AdaGrad

#### Normalization
- Batch Normalization
- Layer Normalization
- Group Normalization

### Using CUDA Kernels
```python
import neuralforge_cuda

a = torch.randn(1024, 1024).cuda()
b = torch.randn(1024, 1024).cuda()

c = neuralforge_cuda.matmul(a, b, use_tiled=True)

x = torch.randn(100, 1000).cuda()
y = neuralforge_cuda.gelu_forward(x)
```

## Neural Architecture Search

### Evolutionary Search

```python
from neuralforge.nas import SearchSpace, EvolutionarySearch, ProxyEvaluator

search_config = {'num_layers': 15, 'num_blocks': 4}
search_space = SearchSpace(search_config)

evaluator = ProxyEvaluator(device='cuda')

evolution = EvolutionarySearch(
    search_space=search_space,
    evaluator=evaluator,
    population_size=20,
    generations=50,
    mutation_rate=0.1
)

best_architecture = evolution.search()
model = search_space.build_model(best_architecture, num_classes=10)
```

### Architecture Components

The search space includes:
- **Layer types:** conv3x3, conv5x5, conv7x7, depthwise, bottleneck, identity
- **Activations:** ReLU, GELU, SiLU, Mish
- **Pooling:** Max, Average, None
- **Channels:** 32, 64, 128, 256, 512

## Training

### Configuration

```python
from neuralforge import Config

config = Config()
config.batch_size = 64
config.epochs = 100
config.learning_rate = 0.001
config.weight_decay = 0.0001
config.optimizer = "adamw"
config.scheduler = "cosine"
config.use_amp = True
config.grad_clip = 1.0

config.save('config.json')
config = Config.load('config.json')
```

### Data Augmentation

```python
from neuralforge.data.augmentation import RandAugment, MixUp, CutMix

rand_aug = RandAugment(n=2, m=9)
mixup = MixUp(alpha=0.2, num_classes=1000)
cutmix = CutMix(alpha=1.0, num_classes=1000)
```

### Custom Models

```python
import torch.nn as nn
from neuralforge.nn import ConvBlock, ResidualBlock, SEBlock

class CustomModel(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = ConvBlock(3, 64, kernel_size=7, stride=2)
        self.res1 = ResidualBlock(64)
        self.se = SEBlock(64)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.se(x)
        x = self.fc(x.mean([2, 3]))
        return x
```

## API Reference

### Core Classes

#### Trainer
Main training class with support for:
- Automatic mixed precision
- Gradient clipping
- Learning rate scheduling
- Checkpointing
- TensorBoard logging

```python
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    config=config,
    scheduler=scheduler
)
```

#### Config
Configuration management:
- JSON serialization
- Parameter validation
- Default values

### Optimizers

#### AdamW
```python
from neuralforge.optim import AdamW
optimizer = AdamW(params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
```

#### LAMB
```python
from neuralforge.optim import LAMB
optimizer = LAMB(params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)
```

### Schedulers

#### CosineAnnealingWarmRestarts
```python
from neuralforge.optim import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
```

#### OneCycleLR
```python
from neuralforge.optim import OneCycleLR
scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=1000)
```

### Utilities

#### Logger
```python
from neuralforge.utils import Logger
logger = Logger(log_dir='./logs', name='training')
logger.info("Training started")
logger.log_metrics({'loss': 0.5, 'acc': 95.0}, step=100)
```

#### MetricsTracker
```python
from neuralforge.utils import MetricsTracker
metrics = MetricsTracker()
metrics.update({'train_loss': 0.5, 'val_loss': 0.6})
metrics.save('metrics.json')
```

## Performance Tips

1. **Use Mixed Precision Training**
   ```python
   config.use_amp = True
   ```

2. **Enable Gradient Clipping**
   ```python
   config.grad_clip = 1.0
   ```

3. **Optimize Data Loading**
   ```python
   config.num_workers = 4
   config.pin_memory = True
   ```

4. **Use Custom CUDA Kernels**
   - Automatically used when available
   - Significant speedup for large models

5. **Batch Size Tuning**
   - Start with 32-64
   - Increase until OOM
   - Use gradient accumulation if needed

## Examples

See `examples/` directory for:
- Custom training loops
- Neural architecture search
- Transfer learning
- Multi-GPU training
- Custom data loaders

## License

MIT License - See LICENSE file for details
