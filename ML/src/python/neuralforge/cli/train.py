import argparse
import sys
import torch
import torch.nn as nn
import random
import numpy as np

from neuralforge.trainer import Trainer
from neuralforge.config import Config
from neuralforge.data.datasets import get_dataset, get_num_classes
from neuralforge.data.dataset import SyntheticDataset, DataLoaderBuilder
from neuralforge.models.resnet import ResNet18
from neuralforge.optim.optimizers import AdamW
from neuralforge.optim.schedulers import CosineAnnealingWarmRestarts, OneCycleLR
from neuralforge.utils.logger import Logger

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
    parser = argparse.ArgumentParser(
        description='NeuralForge - Train neural networks with CUDA acceleration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neuralforge --dataset cifar10 --epochs 50
  neuralforge --dataset mnist --model simple --batch-size 64
  neuralforge --dataset stl10 --model resnet18 --epochs 100 --lr 0.001
  neuralforge --dataset tiny_imagenet --batch-size 128 --epochs 200
        """
    )
    
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--model', type=str, default='simple', 
                       choices=['simple', 'resnet18', 'efficientnet', 'vit'],
                       help='Model architecture')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       help='Dataset (cifar10, mnist, stl10, tiny_imagenet, etc.)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device (cuda/cpu)')
    parser.add_argument('--num-samples', type=int, default=5000, help='Number of synthetic samples')
    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes (for synthetic)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--optimizer', type=str, default='adamw', 
                       choices=['adamw', 'adam', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'onecycle', 'none'],
                       help='Learning rate scheduler')
    
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
        config.optimizer = args.optimizer
        config.scheduler = args.scheduler
        
        # Set paths relative to current working directory (not package directory)
        import os
        cwd = os.getcwd()
        config.model_dir = os.path.join(cwd, "models")
        config.log_dir = os.path.join(cwd, "logs")
        config.data_path = os.path.join(cwd, "data")
    
    set_seed(config.seed)
    
    logger = Logger(config.log_dir, "training")
    logger.info("=" * 80)
    logger.info("NeuralForge Training Framework")
    logger.info("=" * 80)
    logger.info(f"Configuration:\n{config}")
    
    dataset_aliases = {
        'cifar-10': 'cifar10', 'cifar_10': 'cifar10',
        'cifar-100': 'cifar100', 'cifar_100': 'cifar100',
        'fashion-mnist': 'fashion_mnist', 'fashionmnist': 'fashion_mnist',
        'stl-10': 'stl10', 'stl_10': 'stl10',
        'tiny-imagenet': 'tiny_imagenet', 'tinyimagenet': 'tiny_imagenet',
        'food-101': 'food101', 'food_101': 'food101',
        'caltech-256': 'caltech256', 'caltech_256': 'caltech256',
        'oxford-pets': 'oxford_pets', 'oxfordpets': 'oxford_pets',
    }
    
    dataset_name = dataset_aliases.get(args.dataset.lower(), args.dataset.lower())
    
    if dataset_name == 'synthetic':
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
        logger.info(f"Downloading and loading {dataset_name} dataset...")
        config.num_classes = get_num_classes(dataset_name)
        
        train_dataset = get_dataset(dataset_name, root=config.data_path, train=True, download=True)
        val_dataset = get_dataset(dataset_name, root=config.data_path, train=False, download=True)
        
        if dataset_name in ['mnist', 'fashion_mnist']:
            config.image_size = 28
        elif dataset_name in ['cifar10', 'cifar100']:
            config.image_size = 32
        elif dataset_name == 'tiny_imagenet':
            config.image_size = 64
        elif dataset_name == 'stl10':
            config.image_size = 96
        elif dataset_name in ['imagenet', 'food101', 'caltech256', 'oxford_pets']:
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
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    
    scheduler = None
    if config.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    elif config.scheduler == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=config.learning_rate, total_steps=config.epochs * len(train_loader))
    
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

if __name__ == '__main__':
    main()
