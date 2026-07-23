import torch.nn as nn
from ..nn.convolution import EfficientNetBlock

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        self.blocks = nn.Sequential(
            EfficientNetBlock(32, 16, 3, 1, 1),
            EfficientNetBlock(16, 24, 3, 2, 6),
            EfficientNetBlock(24, 24, 3, 1, 6),
            EfficientNetBlock(24, 40, 5, 2, 6),
            EfficientNetBlock(40, 40, 5, 1, 6),
            EfficientNetBlock(40, 80, 3, 2, 6),
            EfficientNetBlock(80, 80, 3, 1, 6),
            EfficientNetBlock(80, 112, 5, 1, 6),
            EfficientNetBlock(112, 112, 5, 1, 6),
            EfficientNetBlock(112, 192, 5, 2, 6),
            EfficientNetBlock(192, 192, 5, 1, 6),
            EfficientNetBlock(192, 320, 3, 1, 6),
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x