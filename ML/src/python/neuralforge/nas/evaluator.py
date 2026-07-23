import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import time
from typing import Tuple
from .search_space import SearchSpace, Architecture

class ModelEvaluator:
    def __init__(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        epochs: int = 5,
        quick_eval: bool = True
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.quick_eval = quick_eval
    
    def evaluate(self, architecture: Architecture, search_space: SearchSpace) -> Tuple[float, float]:
        try:
            model = search_space.build_model(architecture)
            model = model.to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            if self.quick_eval:
                accuracy = self._quick_evaluate(model, criterion, optimizer)
            else:
                accuracy = self._full_evaluate(model, criterion, optimizer)
            
            complexity = search_space.estimate_complexity(architecture)
            params = complexity['params']
            flops = complexity['flops']
            
            param_penalty = params / 1e7
            flop_penalty = flops / 1e9
            
            fitness = accuracy - 0.1 * param_penalty - 0.05 * flop_penalty
            
            return fitness, accuracy
        
        except Exception as e:
            print(f"Error evaluating architecture: {e}")
            return 0.0, 0.0
    
    def _quick_evaluate(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        model.train()
        
        num_batches = min(50, len(self.train_loader))
        
        for epoch in range(self.epochs):
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                if batch_idx >= num_batches:
                    break
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        
        num_val_batches = min(20, len(self.val_loader))
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                if batch_idx >= num_val_batches:
                    break
                
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return accuracy
    
    def _full_evaluate(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        for epoch in range(self.epochs):
            model.train()
            
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        return accuracy

class ProxyEvaluator:
    def __init__(self, device: str = 'cuda'):
        self.device = device
    
    def evaluate(self, architecture: Architecture, search_space: SearchSpace) -> Tuple[float, float]:
        model = search_space.build_model(architecture)
        model = model.to(self.device)
        
        complexity = search_space.estimate_complexity(architecture)
        params = complexity['params']
        flops = complexity['flops']
        
        num_layers = len([g for g in architecture.genome if g.get('type') != 'pooling'])
        
        estimated_accuracy = 60.0 + torch.rand(1).item() * 20.0
        estimated_accuracy = min(95.0, estimated_accuracy - params / 1e8)
        
        fitness = estimated_accuracy - 0.1 * (params / 1e7) - 0.05 * (flops / 1e9)
        
        return fitness, estimated_accuracy