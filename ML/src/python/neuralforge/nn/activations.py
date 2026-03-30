import torch
import torch.nn as nn
import torch.nn.functional as F

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * torch.pow(x, 3))))

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class HardSwish(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * F.hardtanh(x + 3, 0.0, 6.0) / 6.0

class HardSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return F.relu6(x + 3.0) / 6.0

class FReLU(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.bn = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        tx = self.bn(self.conv(x))
        return torch.max(x, tx)

class GLU(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        a, b = x.chunk(2, dim=self.dim)
        return a * torch.sigmoid(b)

class ReGLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)

class GEGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = GELU()
    
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * self.gelu(b)

class SiLU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x * torch.sigmoid(x)

class ELU(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))

class SELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946
    
    def forward(self, x):
        return self.scale * torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))

class PReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_parameters) * init)
    
    def forward(self, x):
        return torch.where(x > 0, x, self.weight * x)

class LeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope)

class Softplus(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta
    
    def forward(self, x):
        return F.softplus(x, self.beta)
