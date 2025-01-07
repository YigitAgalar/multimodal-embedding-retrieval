"""
Main training script for multimodal embedding model.
"""
import tensorflow as tf
from pathlib import Path
import json
from tqdm import tqdm

from derin_ogrenme.data.dataset import MultimodalDataset
from derin_ogrenme.models.encoders import MultimodalEmbedding
from derin_ogrenme.models.training import MultimodalTrainer

def train(
    # Data parameters
    dataset_name: str = "ashraq/fashion-product-images-small",
    cache_dir: str = "data/cache",
    batch_size: int = 32,
    max_samples: int = 1000,  # Using 1000 samples
    
    # Model parameters
    embedding_dim: int = 128,  # Smaller embedding dimension
    
    # Training parameters
    num_epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0001,
    
    # Output parameters
    output_dir: str = "outputs",
    model_name: str = "basic_multimodal_embedding"
):
    """Train a basic multimodal embedding model."""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save training config
    config = {
        "dataset_name": dataset_name,
        "batch_size": batch_size,
        "max_samples": max_samples,
        "embedding_dim": embedding_dim,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Load dataset
    print("\nLoading dataset...")
    print(f"Using {max_samples} samples with batch size {batch_size}")
    dataset = MultimodalDataset(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        batch_size=batch_size,
        max_samples=max_samples
    )
    
    train_ds, val_ds, test_ds = dataset.get_train_val_test_splits()
    
    # Create model and trainer
    print("\nInitializing enhanced model...")
    print(f"Embedding dimension: {embedding_dim}")
    model = MultimodalEmbedding(embedding_dim=embedding_dim)
    trainer = MultimodalTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    
    # Training loop
    print("\nStarting training...")
    print(f"Training for {num_epochs} epochs")
    best_val_loss = float('inf')
    patience = 7  # Increased patience for complex model
    no_improve = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        train_metrics = []
        train_bar = tqdm(train_ds, desc="Training")
        
        for batch in train_bar:
            loss, metrics = trainer.train_step(batch)
            train_metrics.append(metrics)
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': f"{loss.numpy():.4f}",
                'i2t_acc': f"{metrics['i2t_accuracy'].numpy():.4f}",
                't2i_acc': f"{metrics['t2i_accuracy'].numpy():.4f}"
            })
        
        # Compute average training metrics
        train_avg = {
            k: float(tf.reduce_mean([m[k] for m in train_metrics]))
            for k in train_metrics[0].keys()
        }
        
        # Validation
        val_metrics = []
        val_bar = tqdm(val_ds, desc="Validation")
        
        for batch in val_bar:
            metrics = trainer.evaluate_step(batch)
            val_metrics.append(metrics)
            
            # Update progress bar
            val_bar.set_postfix({
                'loss': f"{metrics['loss'].numpy():.4f}",
                'i2t_acc': f"{metrics['i2t_accuracy'].numpy():.4f}",
                't2i_acc': f"{metrics['t2i_accuracy'].numpy():.4f}"
            })
        
        # Compute average validation metrics
        val_avg = {
            k: float(tf.reduce_mean([m[k] for m in val_metrics]))
            for k in val_metrics[0].keys()
        }
        
        # Save metrics
        metrics_file = output_dir / f"metrics_epoch_{epoch+1}.json"
        with open(metrics_file, "w") as f:
            json.dump({
                "train": train_avg,
                "validation": val_avg
            }, f, indent=2)
        
        # Save best model and check for early stopping
        if val_avg['loss'] < best_val_loss:
            best_val_loss = val_avg['loss']
            model_path = output_dir / f"{model_name}_best.weights.h5"
            model.save_weights(str(model_path))
            print(f"\nSaved best model to {model_path}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break
        
        # Print epoch summary
        print("\nTraining metrics:")
        for k, v in train_avg.items():
            print(f"{k}: {v:.4f}")
        
        print("\nValidation metrics:")
        for k, v in val_avg.items():
            print(f"{k}: {v:.4f}")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = []
    test_bar = tqdm(test_ds, desc="Testing")
    
    for batch in test_bar:
        metrics = trainer.evaluate_step(batch)
        test_metrics.append(metrics)
    
    # Compute and save test metrics
    test_avg = {
        k: float(tf.reduce_mean([m[k] for m in test_metrics]))
        for k in test_metrics[0].keys()
    }
    
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_avg, f, indent=2)
    
    print("\nTest metrics:")
    for k, v in test_avg.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    train() 