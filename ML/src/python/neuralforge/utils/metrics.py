import json
import os
from typing import Dict, List, Any
import numpy as np

class MetricsTracker:
    def __init__(self):
        self.metrics = []
        self.best_metrics = {}
    
    def update(self, metrics: Dict[str, Any]):
        self.metrics.append(metrics.copy())
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key not in self.best_metrics:
                    self.best_metrics[key] = value
                else:
                    if 'loss' in key.lower():
                        self.best_metrics[key] = min(self.best_metrics[key], value)
                    else:
                        self.best_metrics[key] = max(self.best_metrics[key], value)
    
    def get_history(self, key: str) -> List[Any]:
        return [m.get(key) for m in self.metrics if key in m]
    
    def get_latest(self, key: str) -> Any:
        for m in reversed(self.metrics):
            if key in m:
                return m[key]
        return None
    
    def get_best(self, key: str) -> Any:
        return self.best_metrics.get(key)
    
    def get_average(self, key: str, last_n: int = None) -> float:
        history = self.get_history(key)
        if not history:
            return 0.0
        
        if last_n is not None:
            history = history[-last_n:]
        
        return np.mean([v for v in history if v is not None])
    
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'metrics': self.metrics,
            'best_metrics': self.best_metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metrics = data.get('metrics', [])
        self.best_metrics = data.get('best_metrics', {})
    
    def summary(self) -> str:
        lines = ["=" * 50, "Metrics Summary", "=" * 50]
        
        for key, value in self.best_metrics.items():
            latest = self.get_latest(key)
            if isinstance(value, float):
                lines.append(f"{key}: best={value:.4f}, latest={latest:.4f}")
            else:
                lines.append(f"{key}: best={value}, latest={latest}")
        
        lines.append("=" * 50)
        return "\n".join(lines)

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop

class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    def update(self, predictions: np.ndarray, targets: np.ndarray):
        for pred, target in zip(predictions, targets):
            self.matrix[target, pred] += 1
    
    def reset(self):
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
    
    def compute_metrics(self) -> Dict[str, float]:
        tp = np.diag(self.matrix)
        fp = np.sum(self.matrix, axis=0) - tp
        fn = np.sum(self.matrix, axis=1) - tp
        tn = np.sum(self.matrix) - (tp + fp + fn)
        
        accuracy = np.sum(tp) / np.sum(self.matrix) if np.sum(self.matrix) > 0 else 0.0
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        return {
            'accuracy': accuracy,
            'precision': np.mean(precision),
            'recall': np.mean(recall),
            'f1_score': np.mean(f1_score)
        }
    
    def get_matrix(self) -> np.ndarray:
        return self.matrix

def accuracy(predictions, targets):
    correct = (predictions == targets).sum()
    total = len(targets)
    return 100.0 * correct / total if total > 0 else 0.0

def top_k_accuracy(output, target, k=5):
    with torch.no_grad():
        maxk = min(k, output.size(1))
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / target.size(0)).item()
