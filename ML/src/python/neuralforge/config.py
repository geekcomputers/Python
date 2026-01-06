import json
import os
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class Config:
    model_name: str = "neuralforge_model"
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    
    data_path: str = "./data"
    num_workers: int = 4
    pin_memory: bool = True
    
    model_dir: str = "./models"
    log_dir: str = "./logs"
    checkpoint_freq: int = 10
    
    use_amp: bool = True
    device: str = "cuda"
    seed: int = 42
    
    nas_enabled: bool = False
    nas_population_size: int = 20
    nas_generations: int = 50
    nas_mutation_rate: float = 0.1
    
    image_size: int = 224
    num_classes: int = 1000
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def __str__(self) -> str:
        return json.dumps(asdict(self), indent=2)