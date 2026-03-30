import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import random
import numpy as np

from src.python.neuralforge import nn as nf_nn
from src.python.neuralforge import optim as nf_optim
from src.python.neuralforge.trainer import Trainer
from src.python.neuralforge.config import Config
from src.python.neuralforge.data.dataset import SyntheticDataset, DataLoaderBuilder
from src.python.neuralforge.data.datasets import get_dataset, get_num_classes
from src.python.neuralforge.data.transforms import get_transforms
from src.python.neuralforge.models.resnet import ResNet18
from src.python.neuralforge.utils.logger import Logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_simple_model(num_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
        
        nn.Flatten(),
        nn.Linear(128, num_classes)
    )

def main():
    parser = argparse.ArgumentParser(description='NeuralForge Training')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'resnet18', 'efficientnet', 'vit'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-samples', type=int, default=5000, help='Number of synthetic samples')
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='synthetic', 
                       choices=['synthetic', 'cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'stl10',
                               'tiny_imagenet', 'imagenet', 'food101', 'caltech256', 'oxford_pets'],
                       help='Dataset to use')
    args = parser.parse_args()
    
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
        config.batch_size = args.batch_size
        config.epochs = args.epochs
        config.learning_rate = args.lr
        config.device = args.device
        config.num_classes = args.num_classes
        config.seed = args.seed
    
    set_seed(config.seed)
    
    logger = Logger(config.log_dir, "training")
    logger.info("=" * 80)
    logger.info("NeuralForge Training Framework")
    logger.info("=" * 80)
    logger.info(f"Configuration:\n{config}")
    
    if args.dataset == 'synthetic':
        logger.info("Creating synthetic dataset...")
        train_dataset = SyntheticDataset(
            num_samples=args.num_samples,
            num_classes=config.num_classes,
            image_size=config.image_size,
            channels=3
        )
        
        val_dataset = SyntheticDataset(
            num_samples=args.num_samples // 5,
            num_classes=config.num_classes,
            image_size=config.image_size,
            channels=3
        )
    else:
        logger.info(f"Downloading and loading {args.dataset} dataset...")
        config.num_classes = get_num_classes(args.dataset)
        
        train_dataset = get_dataset(args.dataset, root=config.data_path, train=True, download=True)
        val_dataset = get_dataset(args.dataset, root=config.data_path, train=False, download=True)
        
        if args.dataset in ['mnist', 'fashion_mnist']:
            config.image_size = 28
        elif args.dataset in ['cifar10', 'cifar100']:
            config.image_size = 32
        elif args.dataset == 'tiny_imagenet':
            config.image_size = 64
        elif args.dataset == 'stl10':
            config.image_size = 96
        elif args.dataset in ['imagenet', 'food101', 'caltech256', 'oxford_pets']:
            config.image_size = 224
    
    loader_builder = DataLoaderBuilder(config)
    train_loader = loader_builder.build_train_loader(train_dataset)
    val_loader = loader_builder.build_val_loader(val_dataset)
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    logger.info(f"Creating model: {args.model}")
    if args.model == 'simple':
        model = create_simple_model(config.num_classes)
    elif args.model == 'resnet18':
        model = ResNet18(num_classes=config.num_classes)
    else:
        model = create_simple_model(config.num_classes)
    
    logger.log_model_summary(model)
    
    criterion = nn.CrossEntropyLoss()
    
    if config.optimizer.lower() == 'adamw':
        optimizer = nf_optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    
    if config.scheduler == 'cosine':
        scheduler = nf_optim.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
    elif config.scheduler == 'onecycle':
        scheduler = nf_optim.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=config.epochs * len(train_loader)
        )
    else:
        scheduler = None
    
    logger.info(f"Optimizer: {config.optimizer}")
    logger.info(f"Scheduler: {config.scheduler}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        scheduler=scheduler,
        device=config.device
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training completed successfully!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    config.save(os.path.join(config.log_dir, 'config.json'))
    logger.info(f"Configuration saved to {os.path.join(config.log_dir, 'config.json')}")

if __name__ == '__main__':
    main()
