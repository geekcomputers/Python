import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
from typing import Optional, Callable, Tuple, List
import numpy as np

class ImageDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        split: str = 'train'
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        
        self.samples = []
        self.class_to_idx = {}
        self._load_dataset()
    
    def _load_dataset(self):
        split_dir = os.path.join(self.root, self.split)
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Dataset directory not found: {split_dir}")
        
        classes = sorted([d for d in os.listdir(split_dir) 
                         if os.path.isdir(os.path.join(split_dir, d))])
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label

class SyntheticDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 10000,
        num_classes: int = 10,
        image_size: int = 224,
        channels: int = 3
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.channels = channels
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = torch.randn(self.channels, self.image_size, self.image_size)
        label = idx % self.num_classes
        return image, label

class MemoryDataset(Dataset):
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        assert len(data) == len(labels)
        self.data = data
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.data[idx], self.labels[idx]

class DataLoaderBuilder:
    def __init__(self, config):
        self.config = config
    
    def build_train_loader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
            persistent_workers=self.config.num_workers > 0
        )
    
    def build_val_loader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False,
            persistent_workers=self.config.num_workers > 0
        )
    
    def build_test_loader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=False
        )

class CachedDataset(Dataset):
    def __init__(self, dataset: Dataset, cache_size: int = 1000):
        self.dataset = dataset
        self.cache_size = cache_size
        self.cache = {}
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if idx in self.cache:
            return self.cache[idx]
        
        item = self.dataset[idx]
        
        if len(self.cache) < self.cache_size:
            self.cache[idx] = item
        
        return item

class MultiScaleDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        scales: List[int] = [224, 256, 288, 320]
    ):
        self.dataset = dataset
        self.scales = scales
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        
        scale = np.random.choice(self.scales)
        resize = transforms.Resize((scale, scale))
        image = resize(image)
        
        return image, label

class PrefetchDataset(Dataset):
    def __init__(self, dataset: Dataset, prefetch_size: int = 100):
        self.dataset = dataset
        self.prefetch_size = prefetch_size
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]