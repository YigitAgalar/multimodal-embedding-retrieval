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
from transformers import DataCollatorWithPadding
from tqdm.auto import tqdm

class MultimodalDataset:
    def __init__(
        self,
        dataset_name: str = "ashraq/fashion-product-images-small",
        cache_dir: str = "data/cache",
        val_size: float = 0.1,
        test_size: float = 0.1,
        batch_size: int = 32,
        max_samples: int = 1000,
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
        
    def load_data(self) -> Dataset:
        """Load dataset from Hugging Face hub."""
        dataset = load_dataset(
            self.dataset_name,
            cache_dir=str(self.cache_dir)
        )
        
        # If dataset only has 'train' split, create validation and test splits
        if len(dataset) == 1 and 'train' in dataset:
            # Limit dataset size
            if self.max_samples:
                dataset = dataset.shuffle(seed=self.seed)
                dataset['train'] = dataset['train'].select(range(min(self.max_samples, len(dataset['train']))))
            
            # First split off the test set
            dataset = dataset['train'].train_test_split(
                test_size=self.test_size,
                seed=self.seed
            )
            # Then split the remaining train into train and validation
            train_val = dataset['train'].train_test_split(
                test_size=self.val_size/(1-self.test_size),
                seed=self.seed
            )
            # Create a new dataset dictionary with all splits
            dataset = {
                'train': train_val['train'],
                'validation': train_val['test'],
                'test': dataset['test']
            }
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
            
            # Ensure image is float32 and in [0, 1]
            image = tf.convert_to_tensor(image, dtype=tf.float32)
            if tf.reduce_max(image) > 1.0:
                image = image / 255.0
                
            # Resize
            image = tf.image.resize(image, (224, 224))
            
            return image
            
        except Exception as e:
            print(f"Error processing image: {e}")
            # Return a blank image in case of error
            return tf.zeros((224, 224, 3), dtype=tf.float32)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for the model."""
        # Basic text preprocessing
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
        """Convert a Hugging Face dataset to a TensorFlow dataset."""
        # Apply preprocessing with progress bar
        dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Preprocessing dataset",
            load_from_cache_file=True
        )
        
        # Convert to TensorFlow dataset
        tf_dataset = dataset.to_tf_dataset(
            columns=["image", "text"],
            shuffle=shuffle,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn
        )
        
        return tf_dataset.prefetch(tf.data.AUTOTUNE)
    
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