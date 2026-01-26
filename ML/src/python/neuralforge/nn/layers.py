import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 use_bn=True, activation='relu', drop_rate=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'mish':
            self.activation = nn.Mish(inplace=True)
        else:
            self.activation = nn.Identity()
        
        self.dropout = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, drop_rate=0.0):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size, padding=kernel_size // 2, drop_rate=drop_rate)
        self.conv2 = ConvBlock(channels, channels, kernel_size, padding=kernel_size // 2, activation='none')
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual
        x = self.activation(x)
        return x

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        mid_channels = out_channels // expansion
        
        self.conv1 = ConvBlock(in_channels, mid_channels, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1)
        self.conv3 = ConvBlock(mid_channels, out_channels, kernel_size=1, padding=0, activation='none')
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x + residual
        x = self.activation(x)
        return x

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBlock(in_channels, hidden_dim, kernel_size=1, padding=0))
        
        layers.extend([
            ConvBlock(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, activation='relu'),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, growth_rate * 4, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(growth_rate * 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=3, padding=1, bias=False)
        
        self.dropout = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()
    
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, drop_rate=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate, drop_rate))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        se = self.squeeze(x).view(b, c)
        se = self.excitation(se).view(b, c, 1, 1)
        return x * se.expand_as(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
