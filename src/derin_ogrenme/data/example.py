"""
Example script demonstrating the usage of the multimodal dataset.
"""
from derin_ogrenme.data.dataset import MultimodalDataset
from derin_ogrenme.utils.data_utils import visualize_batch, print_dataset_info

def main():
    # Initialize dataset
    print("Loading and preprocessing dataset...")
    dataset = MultimodalDataset(
        dataset_name="ashraq/fashion-product-images-small",
        cache_dir="data/cache",
        val_size=0.1,
        test_size=0.1,
        batch_size=32,
        max_samples=1000  # Limit to 1000 samples
    )
    
    # Load raw dataset first to show statistics
    raw_dataset = dataset.load_data()
    print("\nDataset splits:")
    for split, ds in raw_dataset.items():
        print(f"{split}: {len(ds)} examples")
    
    print("\nDataset features:", raw_dataset["train"].features)
    
    # Get preprocessed TensorFlow datasets
    print("\nConverting to TensorFlow datasets...")
    train_ds, val_ds, test_ds = dataset.get_train_val_test_splits()
    
    # Print information about the datasets
    print("\nTraining Dataset:")
    print_dataset_info(train_ds)
    
    print("\nValidation Dataset:")
    print_dataset_info(val_ds)
    
    print("\nTest Dataset:")
    print_dataset_info(test_ds)
    
    # Visualize a batch of training data
    print("\nVisualizing a batch of training data...")
    for batch in train_ds.take(1):
        visualize_batch(batch)

if __name__ == "__main__":
    main() 