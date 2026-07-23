from torchvision import transforms
import torch
from typing import List, Tuple

def get_transforms(image_size: int = 224, is_training: bool = True, mean=None, std=None):
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    if is_training:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

class RandomMixup:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, batch):
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1.0
        
        batch_size = batch[0].size(0)
        index = torch.randperm(batch_size)
        
        mixed_input = lam * batch[0] + (1 - lam) * batch[0][index, :]
        y_a, y_b = batch[1], batch[1][index]
        
        return mixed_input, y_a, y_b, lam

class RandomCutmix:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, batch):
        images, labels = batch
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 1.0
        
        _, _, H, W = images.shape
        cut_rat = torch.sqrt(1.0 - lam)
        cut_w = (W * cut_rat).int()
        cut_h = (H * cut_rat).int()
        
        cx = torch.randint(W, (1,)).item()
        cy = torch.randint(H, (1,)).item()
        
        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)
        
        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return images, labels, labels[index], lam

class GaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

class RandomGaussianBlur:
    def __init__(self, kernel_size: int = 5, sigma: Tuple[float, float] = (0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, img):
        return transforms.GaussianBlur(self.kernel_size, self.sigma)(img)

def get_strong_augmentation(image_size: int = 224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=23)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
