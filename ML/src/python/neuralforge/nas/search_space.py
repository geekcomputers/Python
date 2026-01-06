import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional
import random
import numpy as np

class Architecture:
    def __init__(self, genome: List[int]):
        self.genome = genome
        self.fitness = 0.0
        self.accuracy = 0.0
        self.params = 0
        self.flops = 0
    
    def __repr__(self):
        return f"Architecture(fitness={self.fitness:.4f}, acc={self.accuracy:.2f}%, params={self.params})"

class SearchSpace:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        self.layer_types = ['conv3x3', 'conv5x5', 'conv7x7', 'depthwise', 'bottleneck', 'identity']
        self.activation_types = ['relu', 'gelu', 'silu', 'mish']
        self.pooling_types = ['max', 'avg', 'none']
        self.channels = [32, 64, 128, 256, 512]
        
        self.num_layers = config.get('num_layers', 20)
        self.num_blocks = config.get('num_blocks', 5)
    
    def random_architecture(self) -> Architecture:
        genome = []
        
        for block_idx in range(self.num_blocks):
            num_layers_in_block = random.randint(2, 5)
            
            for layer_idx in range(num_layers_in_block):
                layer_gene = {
                    'type': random.choice(self.layer_types),
                    'channels': random.choice(self.channels),
                    'activation': random.choice(self.activation_types),
                    'use_bn': random.choice([True, False]),
                    'dropout': random.uniform(0.0, 0.3),
                }
                genome.append(layer_gene)
            
            pooling_gene = {
                'type': 'pooling',
                'pooling_type': random.choice(self.pooling_types),
            }
            genome.append(pooling_gene)
        
        return Architecture(genome)
    
    def build_model(self, architecture: Architecture, input_channels: int = 3, num_classes: int = 1000) -> nn.Module:
        layers = []
        current_channels = input_channels
        
        for gene in architecture.genome:
            if gene.get('type') == 'pooling':
                if gene['pooling_type'] == 'max':
                    layers.append(nn.MaxPool2d(2))
                elif gene['pooling_type'] == 'avg':
                    layers.append(nn.AvgPool2d(2))
            else:
                layer_type = gene['type']
                out_channels = gene['channels']
                activation = gene['activation']
                use_bn = gene['use_bn']
                dropout = gene['dropout']
                
                if layer_type == 'conv3x3':
                    layers.append(nn.Conv2d(current_channels, out_channels, 3, padding=1))
                elif layer_type == 'conv5x5':
                    layers.append(nn.Conv2d(current_channels, out_channels, 5, padding=2))
                elif layer_type == 'conv7x7':
                    layers.append(nn.Conv2d(current_channels, out_channels, 7, padding=3))
                elif layer_type == 'depthwise':
                    layers.append(nn.Conv2d(current_channels, current_channels, 3, padding=1, groups=current_channels))
                    layers.append(nn.Conv2d(current_channels, out_channels, 1))
                elif layer_type == 'bottleneck':
                    mid_channels = out_channels // 4
                    layers.append(nn.Conv2d(current_channels, mid_channels, 1))
                    if use_bn:
                        layers.append(nn.BatchNorm2d(mid_channels))
                    layers.append(self._get_activation(activation))
                    layers.append(nn.Conv2d(mid_channels, mid_channels, 3, padding=1))
                    if use_bn:
                        layers.append(nn.BatchNorm2d(mid_channels))
                    layers.append(self._get_activation(activation))
                    layers.append(nn.Conv2d(mid_channels, out_channels, 1))
                elif layer_type == 'identity':
                    if current_channels != out_channels:
                        layers.append(nn.Conv2d(current_channels, out_channels, 1))
                    else:
                        layers.append(nn.Identity())
                
                if use_bn and layer_type != 'bottleneck':
                    layers.append(nn.BatchNorm2d(out_channels))
                
                if layer_type != 'bottleneck':
                    layers.append(self._get_activation(activation))
                
                if dropout > 0:
                    layers.append(nn.Dropout2d(dropout))
                
                current_channels = out_channels
        
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(current_channels, num_classes))
        
        model = nn.Sequential(*layers)
        return model
    
    def _get_activation(self, activation: str) -> nn.Module:
        if activation == 'relu':
            return nn.ReLU(inplace=True)
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'silu':
            return nn.SiLU(inplace=True)
        elif activation == 'mish':
            return nn.Mish(inplace=True)
        else:
            return nn.ReLU(inplace=True)
    
    def mutate(self, architecture: Architecture, mutation_rate: float = 0.1) -> Architecture:
        new_genome = []
        
        for gene in architecture.genome:
            if random.random() < mutation_rate:
                if gene.get('type') == 'pooling':
                    gene = gene.copy()
                    gene['pooling_type'] = random.choice(self.pooling_types)
                else:
                    gene = gene.copy()
                    gene['type'] = random.choice(self.layer_types)
                    gene['channels'] = random.choice(self.channels)
                    gene['activation'] = random.choice(self.activation_types)
            
            new_genome.append(gene)
        
        return Architecture(new_genome)
    
    def crossover(self, parent1: Architecture, parent2: Architecture) -> Architecture:
        min_len = min(len(parent1.genome), len(parent2.genome))
        crossover_point = random.randint(1, min_len - 1)
        
        child_genome = parent1.genome[:crossover_point] + parent2.genome[crossover_point:]
        
        return Architecture(child_genome)
    
    def estimate_complexity(self, architecture: Architecture, input_size: int = 224) -> Dict[str, float]:
        total_params = 0
        total_flops = 0
        current_channels = 3
        current_size = input_size
        
        for gene in architecture.genome:
            if gene.get('type') == 'pooling':
                current_size = current_size // 2
            else:
                out_channels = gene['channels']
                
                if gene['type'] in ['conv3x3', 'conv5x5', 'conv7x7']:
                    kernel_size = int(gene['type'][-3])
                    params = current_channels * out_channels * kernel_size * kernel_size
                    flops = params * current_size * current_size
                elif gene['type'] == 'depthwise':
                    params = current_channels * 9 + current_channels * out_channels
                    flops = current_channels * 9 * current_size * current_size + current_channels * out_channels * current_size * current_size
                elif gene['type'] == 'bottleneck':
                    mid_channels = out_channels // 4
                    params = current_channels * mid_channels + mid_channels * 9 + mid_channels * out_channels
                    flops = (current_channels * mid_channels + mid_channels * 9 + mid_channels * out_channels) * current_size * current_size
                
                total_params += params
                total_flops += flops
                current_channels = out_channels
        
        return {'params': total_params, 'flops': total_flops}