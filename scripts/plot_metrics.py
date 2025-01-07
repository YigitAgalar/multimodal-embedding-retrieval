"""
Plot training and validation metrics.
"""
import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np

def load_metrics(metrics_dir):
    """Load metrics from all epoch files."""
    metrics_files = sorted(Path(metrics_dir).glob("metrics_epoch_*.json"),
                         key=lambda x: int(x.stem.split('_')[-1]))
    
    train_metrics = {
        'loss': [], 'i2t_accuracy': [], 't2i_accuracy': [],
        'i2t_recall@1': [], 't2i_recall@1': [],
        'i2t_recall@5': [], 't2i_recall@5': [],
        'i2t_recall@10': [], 't2i_recall@10': []
    }
    val_metrics = {k: [] for k in train_metrics.keys()}
    
    for f in metrics_files:
        with open(f) as file:
            data = json.load(file)
            for k in train_metrics.keys():
                train_metrics[k].append(data['train'][k])
                val_metrics[k].append(data['validation'][k])
    
    return train_metrics, val_metrics

def plot_metrics(train_metrics, val_metrics, save_dir):
    """Create plots for different metrics."""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_metrics['loss']) + 1)
    
    # Plot settings
    plt.style.use('bmh')  # Using bmh style instead of seaborn
    
    # 1. Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics['loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_metrics['loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_metrics['i2t_accuracy'], 'b-', label='Train Image→Text', linewidth=2)
    plt.plot(epochs, train_metrics['t2i_accuracy'], 'g-', label='Train Text→Image', linewidth=2)
    plt.plot(epochs, val_metrics['i2t_accuracy'], 'b--', label='Val Image→Text', linewidth=2)
    plt.plot(epochs, val_metrics['t2i_accuracy'], 'g--', label='Val Text→Image', linewidth=2)
    plt.title('Training and Validation Accuracy', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Recall plots
    for k in [1, 5, 10]:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_metrics[f'i2t_recall@{k}'], 'b-', 
                label=f'Train Image→Text R@{k}', linewidth=2)
        plt.plot(epochs, train_metrics[f't2i_recall@{k}'], 'g-', 
                label=f'Train Text→Image R@{k}', linewidth=2)
        plt.plot(epochs, val_metrics[f'i2t_recall@{k}'], 'b--', 
                label=f'Val Image→Text R@{k}', linewidth=2)
        plt.plot(epochs, val_metrics[f't2i_recall@{k}'], 'g--', 
                label=f'Val Text→Image R@{k}', linewidth=2)
        plt.title(f'Training and Validation Recall@{k}', fontsize=14, pad=15)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(f'Recall@{k}', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'recall_{k}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Combined recall plot
    plt.figure(figsize=(12, 8))
    colors = ['b', 'g', 'r']
    for i, k in enumerate([1, 5, 10]):
        plt.plot(epochs, val_metrics[f'i2t_recall@{k}'], f'{colors[i]}--', 
                label=f'Image→Text R@{k}', linewidth=2)
        plt.plot(epochs, val_metrics[f't2i_recall@{k}'], f'{colors[i]}:', 
                label=f'Text→Image R@{k}', linewidth=2)
    plt.title('Validation Recall Metrics', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'recall_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Load metrics
    train_metrics, val_metrics = load_metrics('outputs')
    
    # Create plots
    plot_metrics(train_metrics, val_metrics, 'assets/plots') 