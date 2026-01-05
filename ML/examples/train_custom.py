import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from src.python.neuralforge import Trainer, Config
from src.python.neuralforge.data.dataset import SyntheticDataset, DataLoaderBuilder
from src.python.neuralforge.models.resnet import ResNet18
from src.python.neuralforge.optim.optimizers import AdamW
from src.python.neuralforge.optim.schedulers import CosineAnnealingWarmRestarts

def main():
    config = Config()
    config.batch_size = 64
    config.epochs = 100
    config.learning_rate = 0.001
    config.num_classes = 100
    config.model_name = "resnet18_custom"
    
    train_dataset = SyntheticDataset(num_samples=10000, num_classes=100)
    val_dataset = SyntheticDataset(num_samples=2000, num_classes=100)
    
    loader_builder = DataLoaderBuilder(config)
    train_loader = loader_builder.build_train_loader(train_dataset)
    val_loader = loader_builder.build_val_loader(val_dataset)
    
    model = ResNet18(num_classes=100)
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
    
    trainer.train()
    
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")

if __name__ == '__main__':
    main()
