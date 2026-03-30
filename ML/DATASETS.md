# NeuralForge - Dataset Guide

## 📊 Supported Datasets

NeuralForge supports **10 datasets** ranging from small (12 MB) to very large (155 GB)!

## Small Datasets (Quick Training)

### 1. CIFAR-10
- **Size:** 60,000 images (50,000 train + 10,000 test)
- **Image Size:** 32x32 RGB
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Download Size:** ~170 MB

```bash
python train.py --dataset cifar10 --model resnet18 --epochs 50 --batch-size 128
```

### 2. CIFAR-100
- **Size:** 60,000 images (50,000 train + 10,000 test)
- **Image Size:** 32x32 RGB
- **Classes:** 100 fine-grained categories
- **Download Size:** ~170 MB

```bash
python train.py --dataset cifar100 --model resnet18 --epochs 100 --batch-size 128
```

### 3. MNIST
- **Size:** 70,000 images (60,000 train + 10,000 test)
- **Image Size:** 28x28 grayscale
- **Classes:** 10 (digits 0-9)
- **Download Size:** ~12 MB

```bash
python train.py --dataset mnist --model simple --epochs 20 --batch-size 64
```

### 4. Fashion-MNIST
- **Size:** 70,000 images (60,000 train + 10,000 test)
- **Image Size:** 28x28 grayscale
- **Classes:** 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Download Size:** ~30 MB

```bash
python train.py --dataset fashion_mnist --model resnet18 --epochs 30 --batch-size 128
```

### 5. STL-10
- **Size:** 13,000 images (5,000 train + 8,000 test)
- **Image Size:** 96x96 RGB
- **Classes:** 10 (airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck)
- **Download Size:** ~2.5 GB

```bash
python train.py --dataset stl10 --model resnet18 --epochs 50 --batch-size 64
```

### 6. Synthetic (Testing)
- **Size:** Configurable (default: 5,000 train + 1,000 test)
- **Image Size:** 224x224 RGB (configurable)
- **Classes:** Configurable (default: 10)
- **Download Size:** N/A (generated on-the-fly)

```bash
python train.py --dataset synthetic --num-samples 10000 --num-classes 100
```

---

## Medium Datasets (1-5 GB)

### 7. Tiny ImageNet
- **Size:** 120,000 images (100,000 train + 10,000 val + 10,000 test)
- **Image Size:** 64x64 RGB
- **Classes:** 200 (subset of ImageNet)
- **Download Size:** ~237 MB
- **Auto-download:** ✅ Yes

```bash
python train.py --dataset tiny_imagenet --model resnet18 --epochs 50 --batch-size 128
```

### 8. Food-101
- **Size:** 101,000 images (75,750 train + 25,250 test)
- **Image Size:** Variable (resized to 224x224)
- **Classes:** 101 food categories
- **Download Size:** ~5 GB
- **Auto-download:** ✅ Yes

```bash
python train.py --dataset food101 --model resnet18 --epochs 30 --batch-size 64
```

### 9. Caltech-256
- **Size:** 30,607 images
- **Image Size:** Variable (resized to 224x224)
- **Classes:** 257 object categories
- **Download Size:** ~1.2 GB
- **Auto-download:** ✅ Yes

```bash
python train.py --dataset caltech256 --model resnet18 --epochs 50
```

### 10. Oxford-IIIT Pets
- **Size:** 7,349 images (3,680 train + 3,669 test)
- **Image Size:** Variable (resized to 224x224)
- **Classes:** 37 pet breeds (25 dogs, 12 cats)
- **Download Size:** ~800 MB
- **Auto-download:** ✅ Yes

```bash
python train.py --dataset oxford_pets --model resnet18 --epochs 40
```

---

## Large Datasets (100+ GB)

### 11. ImageNet (ILSVRC2012)
- **Size:** 1.3 million training images, 50,000 validation
- **Image Size:** Variable (resized to 224x224)
- **Classes:** 1000
- **Download Size:** ~155 GB (train) + ~6.3 GB (val)
- **Auto-download:** ❌ Manual download required

**Manual Download Instructions:**
1. Register at https://image-net.org/
2. Download ILSVRC2012 dataset
3. Extract to `./data/imagenet/train/` and `./data/imagenet/val/`
4. Run training:

```bash
python train.py --dataset imagenet --model resnet18 --epochs 90 --batch-size 256
```

**Expected Structure:**
```
data/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ... (1000 folders)
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ... (1000 folders)
```

## 📁 Dataset Storage

All datasets are automatically downloaded to `./data/` directory:
```
data/
├── cifar-10-batches-py/       (~170 MB)
├── cifar-100-python/          (~170 MB)
├── MNIST/                     (~12 MB)
├── FashionMNIST/              (~30 MB)
├── stl10_binary/              (~2.5 GB)
├── tiny-imagenet-200/         (~237 MB)
├── food-101/                  (~5 GB)
├── caltech256/                (~1.2 GB)
├── oxford-iiit-pet/           (~800 MB)
└── imagenet/                  (~161 GB, manual)
    ├── train/
    └── val/
```

**Total auto-download: ~9.5 GB**  
**With ImageNet: ~170 GB**

## Quick Test

Test dataset loading:
```bash
python tests/quick_test.py
```

## 🚀 Performance Benchmarks

### Training Speed & Results (RTX 3060 Ti)

| Dataset | Size | Classes | Model | Batch | Epoch Time | Expected Acc | Total Time |
|---------|------|---------|-------|-------|------------|--------------|------------|
| **MNIST** | 12 MB | 10 | Simple | 64 | ~5s | ~99% | ~1 min |
| **Fashion-MNIST** | 30 MB | 10 | ResNet18 | 128 | ~10s | ~92% | ~3 min |
| **CIFAR-10** | 170 MB | 10 | ResNet18 | 128 | ~9s | **85-90%** | ~8 min |
| **CIFAR-100** | 170 MB | 100 | ResNet18 | 128 | ~9s | ~70% | ~8 min |
| **STL-10** | 2.5 GB | 10 | ResNet18 | 64 | ~45s | ~75% | ~30 min |
| **Tiny ImageNet** | 237 MB | 200 | ResNet18 | 128 | ~15s | ~60% | ~12 min |
| **Oxford Pets** | 800 MB | 37 | ResNet18 | 64 | ~8s | ~85% | ~6 min |
| **Caltech-256** | 1.2 GB | 257 | ResNet18 | 64 | ~10s | ~70% | ~8 min |
| **Food-101** | 5 GB | 101 | ResNet18 | 64 | ~45s | ~75% | ~30 min |
| **ImageNet** | 161 GB | 1000 | ResNet18 | 256 | ~20 min | ~70% | ~30 hours |

### 🏆 Your Recent Results
- **CIFAR-10**: 85.35% validation accuracy in 50 epochs! (8 minutes)

## Using Custom Datasets

Create a custom dataset class:

```python
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # Load your data here
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
```

## Data Augmentation

All real datasets come with pre-configured augmentation:

**Training Augmentation (CIFAR-10):**
- Random crop with padding
- Random horizontal flip
- Normalization

**Training Augmentation (MNIST/Fashion-MNIST):**
- Basic normalization

**Additional Augmentation:**
```python
from neuralforge.data.augmentation import RandAugment, MixUp, CutMix

rand_aug = RandAugment(n=2, m=9)
mixup = MixUp(alpha=0.2)
cutmix = CutMix(alpha=1.0)
```

## Testing Your Model

After training, test interactively:

```bash
python tests/test_model.py --dataset cifar10 --mode interactive
```

Interactive commands:
- `random 20` - Test 20 random samples
- `sample 100` - Test specific sample
- `accuracy` - Calculate full test accuracy
- `image cat.jpg` - Test your own image

## Dataset Classes

**CIFAR-10:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

**CIFAR-100:** 100 classes across 20 superclasses (aquatic mammals, fish, flowers, food, etc.)

**MNIST:** Digits 0-9

**Fashion-MNIST:** T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

**STL-10:** airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck

**Tiny ImageNet:** 200 classes from ImageNet (subset with smaller images)

**Food-101:** 101 food categories (apple pie, pizza, sushi, etc.)

**Caltech-256:** 257 object categories (musical instruments, vehicles, animals, etc.)

**Oxford Pets:** 37 pet breeds (25 dog breeds + 12 cat breeds)

**ImageNet:** 1000 classes (animals, objects, vehicles, etc.)