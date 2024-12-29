"""
Example code for visualizing the fashion product dataset.
Copy and paste these code blocks into your Jupyter notebook as needed.
"""

# Block 1: Imports
"""
import sys
sys.path.append('..')

from src.derin_ogrenme.data.dataset import MultimodalDataset
from src.derin_ogrenme.utils.data_utils import visualize_batch, print_dataset_info
import matplotlib.pyplot as plt
"""

# Block 2: Initialize Dataset
"""
# Initialize dataset with a small number of samples
dataset = MultimodalDataset(
    dataset_name="ashraq/fashion-product-images-small",
    cache_dir="../data/cache",
    val_size=0.1,
    test_size=0.1,
    batch_size=4,  # Small batch size for better visualization
    max_samples=20  # Just load 20 samples for quick visualization
)

# Load raw dataset to see what's available
raw_dataset = dataset.load_data()

# Print dataset statistics
print("Dataset splits:")
for split, ds in raw_dataset.items():
    print(f"{split}: {len(ds)} examples")
"""

# Block 3: Get Preprocessed Data
"""
# Get the preprocessed datasets
train_ds, val_ds, test_ds = dataset.get_train_val_test_splits()

# Print information about the training dataset
print("Training Dataset Information:")
print_dataset_info(train_ds)
"""

# Block 4: Visualize Images
"""
# Display multiple batches of images with their descriptions
print("Showing fashion product images with their descriptions...\\n")

for i, batch in enumerate(train_ds.take(3)):
    print(f"Batch {i+1}:")
    visualize_batch(batch)
    plt.show()
    print("\\n" + "-"*50 + "\\n")
"""

# Block 5: Show Raw Sample
"""
# Display a raw sample from the dataset
sample = raw_dataset['train'][0]
print("Product Name:", sample['productDisplayName'])
print("Category:", sample['masterCategory'])
print("Color:", sample['baseColour'])
print("Gender:", sample['gender'])
print("Season:", sample['season'])
""" 