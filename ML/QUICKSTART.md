# NeuralForge - Quick Start Guide

📚 **Additional Resources:**
- [CLI Usage Summary](CLI_USAGE_SUMMARY.md) - Quick reference for all CLI commands
- [Installation Guide](INSTALL_CLI.md) - Detailed installation and troubleshooting

## 🚀 Get Started in 3 Steps

### Step 1: Install NeuralForge
```bash
# Install from source
pip install -e .

# Or use setup script
.\run.ps1  # Windows
./run.sh   # Linux/Mac
```
This will:
- ✓ Install dependencies
- ✓ Compile CUDA extensions (if available)
- ✓ Create command-line tools

### Step 2: Train on Real Data

**Using CLI Tool (Recommended):**
```bash
NeuralForgeAI --dataset cifar10 --model resnet18 --epochs 20 --batch-size 128
```

**Using Python Script:**
```bash
python train.py --dataset cifar10 --model resnet18 --epochs 20 --batch-size 128
```

### Step 3: Test Your Model
```bash
python tests/test_model.py --dataset cifar10 --mode interactive
```

---

## 💡 Using NeuralForge as a Library

After `pip install`, you can use NeuralForge from anywhere:

```bash
# Simple usage - just specify dataset and model
NeuralForgeAI --dataset stl10 --model resnet18 --epochs 50 --batch-size 64

# Advanced usage - customize everything
NeuralForgeAI --dataset cifar100 --model resnet18 --epochs 100 \
              --batch-size 128 --lr 0.001 --optimizer adamw \
              --scheduler cosine --device cuda

# Quick test on MNIST
NeuralForgeAI --dataset mnist --model simple --epochs 10
```

**All available CLI commands:**
- `NeuralForgeAI` - Main training command (alias for `neuralforge`)
- `neuralforge` - Training command
- `neuralforge-train` - Training command (explicit)
- `neuralforge-test` - Model testing
- `neuralforge-gui` - Launch GUI interface
- `neuralforge-nas` - Neural Architecture Search

---

## 📊 Available Datasets

| Dataset | Size | Classes | Image Size | Download |
|---------|------|---------|------------|----------|
| **CIFAR-10** | 60K | 10 | 32x32 | ~170 MB |
| **CIFAR-100** | 60K | 100 | 32x32 | ~170 MB |
| **MNIST** | 70K | 10 | 28x28 | ~12 MB |
| **Fashion-MNIST** | 70K | 10 | 28x28 | ~30 MB |
| **STL-10** | 13K | 10 | 96x96 | ~2.5 GB |
| **Synthetic** | Custom | Custom | Custom | None |

---

## 🎯 Common Use Cases

### Quick Test (5 minutes)
```bash
# Small synthetic dataset for testing (CLI)
NeuralForgeAI --dataset synthetic --num-samples 1000 --epochs 5

# Or using Python script
python train.py --dataset synthetic --num-samples 1000 --epochs 5
python tests/test_model.py --dataset synthetic --mode random --samples 20
```

### CIFAR-10 Classification (30 minutes)
```bash
# Train ResNet18 on CIFAR-10 (CLI)
NeuralForgeAI --dataset cifar10 --model resnet18 --epochs 50 --batch-size 64

# Or using Python script
python train.py --dataset cifar10 --model resnet18 --epochs 20 --batch-size 128 --lr 0.001

# Test the trained model
python tests/test_model.py --dataset cifar10 --mode interactive
```

### MNIST Digit Recognition (10 minutes)
```bash
# Train simple CNN on MNIST
python train.py --dataset mnist --model simple --epochs 10 --batch-size 64

# Test accuracy
python tests/test_model.py --dataset mnist --mode accuracy
```

### Fashion-MNIST (20 minutes)
```bash
# Train ResNet on Fashion-MNIST
python train.py --dataset fashion_mnist --model resnet18 --epochs 20 --batch-size 128

# Interactive testing
python tests/test_model.py --dataset fashion_mnist --mode interactive
```

---

## 🎮 Interactive Testing Mode

After training, test your model interactively:

```bash
python tests/test_model.py --dataset cifar10 --mode interactive
```

### Available Commands:

**Test random samples:**
```
>>> random 10
Testing 10 random samples...
 1. ✓ True: cat      | Pred: cat      | Conf: 85.3%
 2. ✓ True: dog      | Pred: dog      | Conf: 92.1%
 3. ✗ True: bird     | Pred: plane    | Conf: 68.2%
...
```

**Test specific sample:**
```
>>> sample 42
Sample #42
True Label:      cat
Predicted:       cat
Confidence:      89.5%
Status:          ✓ Correct

Top-3 Predictions:
  1. cat             89.5%
  2. dog             7.3%
  3. deer            2.1%
```

**Full accuracy:**
```
>>> accuracy
Calculating per-class accuracy...
Per-class Accuracy:
  airplane       : 87.2% (872/1000)
  automobile     : 91.5% (915/1000)
  bird           : 82.3% (823/1000)
...
Overall Accuracy: 86.50%
```

**Test your own image:**
```
>>> image my_cat.jpg
Custom Image: my_cat.jpg
Predicted:       cat
Confidence:      94.3%
```

---

## 🔧 Training Options

### Basic Training
```bash
python train.py --dataset cifar10 --epochs 50
```

### Advanced Configuration
```bash
python train.py \
    --dataset cifar10 \
    --model resnet18 \
    --batch-size 128 \
    --epochs 100 \
    --lr 0.001 \
    --device cuda \
    --seed 42
```

### Available Models
- `simple` - Lightweight CNN (fast, good for testing)
- `resnet18` - ResNet-18 (best accuracy)
- `efficientnet` - EfficientNet B0
- `vit` - Vision Transformer

---

## 📁 File Structure After Training

```
NeuralForge/
├── models/
│   ├── best_model.pt          # Best validation model
│   ├── final_model.pt          # Final epoch model
│   └── checkpoint_epoch_X.pt   # Periodic checkpoints
├── logs/
│   ├── training_*.log          # Training logs
│   ├── neuralforge_*.log       # Detailed logs
│   ├── config.json             # Saved config
│   └── metrics.json            # Training metrics
└── data/
    ├── cifar-10-batches-py/    # Downloaded datasets
    └── ...
```

---

## 🎓 Training Examples

### Example 1: Fast Test
```bash
# 5-minute test run
python train.py --dataset synthetic --num-samples 500 --epochs 3
python tests/test_model.py --dataset synthetic --mode random
```

### Example 2: CIFAR-10 Full Training
```bash
# Full CIFAR-10 training (~1 hour on RTX 3060 Ti)
python train.py --dataset cifar10 --model resnet18 --epochs 100 --batch-size 128
python tests/test_model.py --dataset cifar10 --mode accuracy
```

### Example 3: Multiple Datasets
```bash
# Train on different datasets
python train.py --dataset mnist --epochs 10
python train.py --dataset fashion_mnist --epochs 20
python train.py --dataset cifar10 --epochs 50

# Compare results
python tests/test_model.py --dataset mnist --mode accuracy
python tests/test_model.py --dataset fashion_mnist --mode accuracy
python tests/test_model.py --dataset cifar10 --mode accuracy
```

---

## 💡 Tips & Tricks

### 1. Monitor Training
Watch the logs in real-time:
```bash
# Windows PowerShell
Get-Content ./logs/*.log -Wait -Tail 20

# Linux/Mac
tail -f ./logs/*.log
```

### 2. Resume Training
Models are automatically checkpointed. Load them:
```python
trainer.load_checkpoint('./models/checkpoint_epoch_50.pt')
trainer.train()
```

### 3. Adjust Batch Size
If you get OOM errors:
```bash
# Reduce batch size
python train.py --dataset cifar10 --batch-size 64
python train.py --dataset cifar10 --batch-size 32
```

### 4. Quick Experiments
Use synthetic data for fast experiments:
```bash
python train.py --dataset synthetic --num-samples 100 --epochs 2
```

### 5. Save Memory
Reduce workers if low on RAM:
```python
config.num_workers = 2  # Default is 4
```

---

## 🐛 Troubleshooting

### "CUDA out of memory"
→ Reduce batch size: `--batch-size 32`

### "Dataset not found"
→ Will auto-download on first run

### "Model not found"
→ Train first: `python train.py --dataset cifar10 --epochs 5`

### Slow training
→ Check GPU usage: `nvidia-smi`
→ Increase num_workers in config

---

## 📈 Expected Results

### CIFAR-10 (after 50 epochs)
- Training Accuracy: ~85-90%
- Validation Accuracy: ~80-85%
- Time: ~30-40 minutes (RTX 3060 Ti)

### MNIST (after 10 epochs)
- Training Accuracy: ~99%
- Validation Accuracy: ~98-99%
- Time: ~5 minutes

### Fashion-MNIST (after 20 epochs)
- Training Accuracy: ~92-95%
- Validation Accuracy: ~90-92%
- Time: ~10 minutes

---

## 🎉 Next Steps

1. **Try Neural Architecture Search:**
   ```bash
   python examples/neural_architecture_search.py
   ```

2. **Custom Training:**
   ```bash
   python examples/train_cifar10.py
   ```

3. **Experiment with Models:**
   ```bash
   python train.py --dataset cifar10 --model efficientnet
   ```

4. **Build Your Own Model:**
   See `DOCUMENTATION.md` for API reference

---

## 🤝 Need Help?

- Check `DOCUMENTATION.md` for full API reference
- See `DATASETS.md` for dataset details
- Review `FEATURES.md` for capabilities
- Run `python tests/quick_test.py` for validation

Happy training! 🚀
