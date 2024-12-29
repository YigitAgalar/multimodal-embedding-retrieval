"""
Utility functions for data handling and visualization.
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def decode_text(text_tensor):
    """Helper function to decode text tensor to string."""
    try:
        if isinstance(text_tensor, tf.Tensor):
            text = text_tensor.numpy()
            if isinstance(text, bytes):
                return text.decode('utf-8')
            return str(text)
        return str(text_tensor)
    except Exception as e:
        print(f"Error decoding text: {e}")
        print(f"Text tensor type: {type(text_tensor)}")
        print(f"Text tensor: {text_tensor}")
        return "Error decoding text"

def process_text(text):
    """Process text for display, handling both scalar and vector tensors."""
    try:
        if isinstance(text, tf.Tensor):
            # Handle scalar tensor
            if not text.shape:  # Scalar tensor
                decoded = decode_text(text)
            else:  # Vector tensor
                decoded = decode_text(text[0])
        else:
            decoded = str(text)
        
        # Truncate if too long
        return decoded[:50] + "..."
    except Exception as e:
        print(f"Error processing text: {e}")
        return "Error processing text"

def visualize_batch(batch_data, num_samples: int = 4):
    """Visualize a batch of image-text pairs."""
    try:
        # Debug information
        print("Batch data type:", type(batch_data))
        print("Batch data keys:", batch_data.keys())
        print("Images type:", type(batch_data["image"]))
        print("Texts type:", type(batch_data["text"]))
        
        images = batch_data["image"]
        texts = batch_data["text"]
        
        # Convert EagerTensor to numpy if needed
        if isinstance(images, tf.Tensor):
            images = images.numpy()
        
        plt.figure(figsize=(15, 5))
        for i in range(min(num_samples, len(images))):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(images[i])
            
            # Process text safely
            if isinstance(texts, (list, tuple, np.ndarray)):
                title = process_text(texts[i])
            else:
                # Handle scalar tensor
                title = process_text(texts)
            
            plt.title(title, wrap=True)
            plt.axis("off")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in visualize_batch: {e}")
        print(f"Batch data structure: {batch_data}")

def print_dataset_info(dataset: tf.data.Dataset):
    """Print information about the dataset."""
    try:
        for batch in dataset.take(1):
            print("\nDataset batch structure:")
            print(f"Batch type: {type(batch)}")
            print(f"Batch keys: {batch.keys()}")
            print(f"Image shape: {batch['image'].shape}")
            print(f"Text shape: {batch['text'].shape if hasattr(batch['text'], 'shape') else 'no shape'}")
            print("\nFirst few texts:")
            if isinstance(batch['text'], tf.Tensor):
                if batch['text'].shape.rank > 0:  # Vector tensor
                    for i, text in enumerate(batch['text'][:3]):
                        print(f"Text {i}: {decode_text(text)}")
                else:  # Scalar tensor
                    print(f"Text: {decode_text(batch['text'])}")
            else:
                for i, text in enumerate(batch['text'][:3]):
                    print(f"Text {i}: {decode_text(text)}")
    except Exception as e:
        print(f"Error in print_dataset_info: {e}")
        print(f"Dataset structure: {dataset}") 