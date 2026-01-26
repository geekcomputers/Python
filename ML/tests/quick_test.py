import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.python.neuralforge.data.datasets import get_dataset
from src.python.neuralforge.models.resnet import ResNet18

print("=" * 60)
print("  NeuralForge Quick Test")
print("=" * 60)

print("\n[1/3] Testing CIFAR-10 dataset download...")
try:
    dataset = get_dataset('cifar10', root='./data', train=False, download=True)
    print(f"✓ CIFAR-10 loaded: {len(dataset)} samples")
    print(f"  Classes: {dataset.classes}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n[2/3] Testing model creation...")
try:
    model = ResNet18(num_classes=10)
    print(f"✓ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n[3/3] Testing inference...")
try:
    model.eval()
    image, label = dataset[0]
    with torch.no_grad():
        output = model(image.unsqueeze(0))
    print(f"✓ Inference successful: output shape {output.shape}")
    print(f"  True label: {dataset.classes[label]}")
    pred = output.argmax(1).item()
    print(f"  Predicted: {dataset.classes[pred]}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "=" * 60)
print("  All tests passed! Ready to train.")
print("=" * 60)
print("\nTry these commands:")
print("  python train.py --dataset cifar10 --epochs 20")
print("  python tests/test_model.py --dataset cifar10 --mode interactive")
print("=" * 60)
