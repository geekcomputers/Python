import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Optional

def plot_training_curves(
    metrics_tracker,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    train_loss = metrics_tracker.get_history('train_loss')
    val_loss = metrics_tracker.get_history('val_loss')
    train_acc = metrics_tracker.get_history('train_acc')
    val_acc = metrics_tracker.get_history('val_acc')
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    if train_loss:
        axes[0].plot(train_loss, label='Train Loss', linewidth=2)
    if val_loss:
        axes[0].plot(val_loss, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if train_acc:
        axes[1].plot(train_acc, label='Train Accuracy', linewidth=2)
    if val_acc:
        axes[1].plot(val_acc, label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.close()

def plot_learning_rate(
    lr_history: List[float],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 5)
):
    plt.figure(figsize=figsize)
    plt.plot(lr_history, linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate plot saved to {save_path}")
    
    plt.close()

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
):
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    if class_names:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.close()

def visualize_architecture(architecture, save_path: Optional[str] = None):
    layer_types = [gene.get('type', 'unknown') for gene in architecture.genome]
    layer_counts = {}
    
    for layer_type in layer_types:
        layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
    
    plt.figure(figsize=(10, 6))
    plt.bar(layer_counts.keys(), layer_counts.values())
    plt.xlabel('Layer Type')
    plt.ylabel('Count')
    plt.title('Architecture Layer Distribution')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Architecture visualization saved to {save_path}")
    
    plt.close()

def plot_nas_history(
    history: List[Dict],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    generations = [h['generation'] for h in history]
    best_fitness = [h['best_fitness'] for h in history]
    avg_fitness = [h['avg_fitness'] for h in history]
    best_accuracy = [h['best_accuracy'] for h in history]
    avg_accuracy = [h['avg_accuracy'] for h in history]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    axes[0].plot(generations, best_fitness, label='Best Fitness', linewidth=2, marker='o')
    axes[0].plot(generations, avg_fitness, label='Avg Fitness', linewidth=2, marker='s')
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Fitness')
    axes[0].set_title('NAS Fitness Evolution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(generations, best_accuracy, label='Best Accuracy', linewidth=2, marker='o')
    axes[1].plot(generations, avg_accuracy, label='Avg Accuracy', linewidth=2, marker='s')
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('NAS Accuracy Evolution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"NAS history plot saved to {save_path}")
    
    plt.close()

def plot_gradient_flow(named_parameters, save_path: Optional[str] = None):
    ave_grads = []
    max_grads = []
    layers = []
    
    for n, p in named_parameters:
        if p.requires_grad and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu().item())
            max_grads.append(p.grad.abs().max().cpu().item())
    
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c", label="max gradient")
    plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.5, lw=1, color="b", label="mean gradient")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=max(max_grads) * 1.1)
    plt.xlabel("Layers")
    plt.ylabel("Gradient")
    plt.title("Gradient Flow")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gradient flow plot saved to {save_path}")
    
    plt.close()