"""
Dataset module for multimodal embedding search model.
"""
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from datasets import load_dataset, Dataset
import tensorflow as tf
from PIL import Image
import io

# Enable mixed precision for better GPU performance
tf.keras.mixed_precision.set_global_policy('mixed_float16')

class MultimodalDataset:
    def __init__(
        self,
        dataset_name: str = "ashraq/fashion-product-images-small",
        cache_dir: str = "data/cache",
        val_size: float = 0.1,
        test_size: float = 0.1,
        batch_size: int = 32,
        max_samples: int = None,
        seed: int = 42
    ):
        self.dataset_name = dataset_name
        self.cache_dir = Path(cache_dir)
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.seed = seed
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up CUDA-optimized configuration
        self.autotune = tf.data.AUTOTUNE
        
    def load_data(self) -> Dataset:
        """Load dataset from Hugging Face hub."""
        # Load the dataset
        dataset = load_dataset(
            self.dataset_name,
            cache_dir=str(self.cache_dir)
        )
        
        # Get the training split
        if isinstance(dataset, dict):
            train_data = dataset['train']
        else:
            train_data = dataset
        
        # Shuffle and limit size if specified
        if self.max_samples:
            print(f"\nLimiting dataset to {self.max_samples} samples")
            train_data = train_data.shuffle(seed=self.seed)
            train_data = train_data.select(range(min(self.max_samples, len(train_data))))
        
        # Split into train/val/test
        test_size = int(len(train_data) * self.test_size)
        val_size = int(len(train_data) * self.val_size)
        
        # Create splits
        splits = train_data.train_test_split(
            test_size=test_size + val_size,
            seed=self.seed
        )
        
        # Further split the test portion into validation and test
        val_test_splits = splits['test'].train_test_split(
            test_size=test_size / (test_size + val_size),
            seed=self.seed
        )
        
        # Create final dataset dictionary
        dataset = {
            'train': splits['train'],
            'validation': val_test_splits['train'],
            'test': val_test_splits['test']
        }
        
        # Print split sizes
        print("\nDataset split sizes:")
        for split_name, split_data in dataset.items():
            print(f"{split_name}: {len(split_data)} examples")
        
        return dataset
    
    def preprocess_image(self, image) -> tf.Tensor:
        """Preprocess image for the model."""
        try:
            # If image is already a PIL Image
            if isinstance(image, Image.Image):
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                # Convert to numpy array
                image = np.array(image)
            # If image is bytes
            elif isinstance(image, bytes):
                # Open and convert to RGB
                image = Image.open(io.BytesIO(image))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = np.array(image)
            # If image is numpy array
            elif isinstance(image, np.ndarray):
                # Ensure it's RGB
                if len(image.shape) == 2:  # Grayscale
                    image = np.stack([image] * 3, axis=-1)
                elif len(image.shape) == 3 and image.shape[-1] == 1:  # Grayscale
                    image = np.concatenate([image] * 3, axis=-1)
                elif len(image.shape) == 3 and image.shape[-1] == 4:  # RGBA
                    image = image[..., :3]
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Convert to tensor and normalize
            image = tf.convert_to_tensor(image, dtype=tf.float32)
            if tf.reduce_max(image) > 1.0:
                image = image / 255.0
            
            # Resize to target size
            image = tf.image.resize(image, (224, 224))
            
            return image
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return tf.zeros((224, 224, 3), dtype=tf.float32)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for the model."""
        if not isinstance(text, str):
            text = str(text)
        return text.lower().strip()
    
    def preprocess_function(self, examples):
        """Preprocess function to be applied to the dataset."""
        result = {
            "image": [self.preprocess_image(img) for img in examples["image"]],
            "text": [self.preprocess_text(text) for text in examples["productDisplayName"]]
        }
        return result
    
    def prepare_tf_dataset(self, dataset: Dataset, shuffle: bool = False) -> tf.data.Dataset:
        """Prepare TensorFlow dataset with CUDA optimization."""
        tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
        
        if shuffle:
            tf_dataset = tf_dataset.shuffle(
                buffer_size=10000,
                seed=self.seed,
                reshuffle_each_iteration=True
            )
        
        # Apply preprocessing with GPU acceleration
        tf_dataset = tf_dataset.map(
            self.preprocess_function,
            num_parallel_calls=self.autotune
        )
        
        # Optimize performance
        tf_dataset = tf_dataset.cache()
        tf_dataset = tf_dataset.batch(self.batch_size)
        tf_dataset = tf_dataset.prefetch(self.autotune)
        
        return tf_dataset
    
    def collate_fn(self, examples):
        """Collate function for batching."""
        images = tf.stack([example["image"] for example in examples])
        texts = [example["text"] for example in examples]
        return {"image": images, "text": texts}
    
    def get_train_val_test_splits(
        self
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Get train/validation/test splits."""
        # Load and split the dataset
        datasets = self.load_data()
        
        # Convert each split to TensorFlow dataset
        print("\nProcessing training set...")
        train_tf = self.prepare_tf_dataset(datasets['train'], shuffle=True)
        print("Processing validation set...")
        val_tf = self.prepare_tf_dataset(datasets['validation'], shuffle=False)
        print("Processing test set...")
        test_tf = self.prepare_tf_dataset(datasets['test'], shuffle=False)
        
        return train_tf, val_tf, test_tf 