import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, base_scheduler=None, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        
        if self.base_scheduler is not None:
            return self.base_scheduler.get_last_lr()
        
        return self.base_lrs
    
    def step(self, epoch=None):
        if self.last_epoch < self.warmup_epochs:
            super().step(epoch)
        elif self.base_scheduler is not None:
            self.base_scheduler.step(epoch)

class CosineAnnealingWarmRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.T_i = T_0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        
        self.last_epoch = math.floor(epoch)
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class OneCycleLR(_LRScheduler):
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, anneal_strategy='cos',
                 div_factor=25.0, final_div_factor=1e4, last_epoch=-1):
        self.max_lr = max_lr if isinstance(max_lr, list) else [max_lr] * len(optimizer.param_groups)
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = [lr / self.div_factor for lr in self.max_lr]
        self.min_lr = [lr / self.final_div_factor for lr in self.max_lr]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        step_num = self.last_epoch
        
        if step_num > self.total_steps:
            return self.min_lr
        
        if step_num <= self.pct_start * self.total_steps:
            pct = step_num / (self.pct_start * self.total_steps)
            return [initial + (maximum - initial) * pct 
                    for initial, maximum in zip(self.initial_lr, self.max_lr)]
        else:
            pct = (step_num - self.pct_start * self.total_steps) / ((1 - self.pct_start) * self.total_steps)
            
            if self.anneal_strategy == 'cos':
                return [minimum + (maximum - minimum) * (1 + math.cos(math.pi * pct)) / 2
                        for minimum, maximum in zip(self.min_lr, self.max_lr)]
            else:
                return [maximum - (maximum - minimum) * pct
                        for minimum, maximum in zip(self.min_lr, self.max_lr)]

class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, power=1.0, last_epoch=-1):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        decay_factor = ((1.0 - self.last_epoch / self.total_iters) / (1.0 - (self.last_epoch - 1) / self.total_iters)) ** self.power
        return [group['lr'] * decay_factor for group in self.optimizer.param_groups]

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=0.0, eta_min=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            return [self.eta_min + (base_lr - self.eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]

class ExponentialWarmup(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, gamma=0.9, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        
        return [base_lr * self.gamma ** (self.last_epoch - self.warmup_epochs) for base_lr in self.base_lrs]