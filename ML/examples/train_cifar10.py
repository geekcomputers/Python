import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from src.python.neuralforge import Trainer, Config
from src.python.neuralforge.data.datasets import get_dataset
from src.python.neuralforge.data.dataset import DataLoaderBuilder
from src.python.neuralforge.models.resnet import ResNet18
from src.python.neuralforge.optim.optimizers import AdamW
from src.python.neuralforge.optim.schedulers import CosineAnnealingWarmRestarts

def main():
    print("Training ResNet18 on CIFAR-10")
    
    config = Config()
    config.batch_size = 128
    config.epochs = 100
    config.learning_rate = 0.001
    config.num_classes = 10
    config.image_size = 32
    config.model_name = "resnet18_cifar10"
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Downloading CIFAR-10 dataset...")
    train_dataset = get_dataset('cifar10', root='./data', train=True, download=True)
    val_dataset = get_dataset('cifar10', root='./data', train=False, download=True)
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    
    loader_builder = DataLoaderBuilder(config)
    train_loader = loader_builder.build_train_loader(train_dataset)
    val_loader = loader_builder.build_val_loader(val_dataset)
    
    model = ResNet18(num_classes=10, in_channels=3)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        scheduler=scheduler
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Model saved to: ./models/best_model.pt")
    print(f"\nTest the model:")
    print(f"  python tests/test_model.py --dataset cifar10 --mode interactive")

if __name__ == '__main__':
    main()
