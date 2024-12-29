"""
Encoder models for image and text processing.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

class ImageEncoder(Model):
    """Very basic CNN-based image encoder."""
    
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        
        # Simple CNN model with single conv layer
        self.image_model = tf.keras.Sequential([
            # Single conv block
            layers.Conv2D(16, 3, activation='relu', padding='same'),
            layers.MaxPooling2D(2),
            
            # Flatten and project
            layers.Flatten(),
            layers.Dense(embedding_dim),
        ])
    
    def call(self, images):
        return tf.nn.l2_normalize(self.image_model(images), axis=-1)

class TextEncoder(Model):
    """Basic text encoder with just embedding layer."""
    
    def __init__(self, embedding_dim: int = 128, vocab_size: int = 10000, max_length: int = 100):
        super().__init__()
        
        self.max_length = max_length
        
        # Text processing layers - just embedding and pooling
        self.text_model = tf.keras.Sequential([
            # Basic embedding layer
            layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
            
            # Global pooling to get fixed-size representation
            layers.GlobalAveragePooling1D(),
        ])
        
        # Simple tokenizer
        self.vectorize_layer = layers.TextVectorization(
            max_tokens=vocab_size,
            output_mode='int',
            output_sequence_length=max_length
        )
        
        # Initialize with some dummy data
        self.vectorize_layer.adapt(['dummy text'])
    
    def call(self, texts):
        # Preprocess texts
        if isinstance(texts, tf.Tensor):
            texts = tf.strings.lower(texts)
            texts = tf.strings.regex_replace(texts, r'[^\w\s]', '')
        
        # Vectorize texts
        sequences = self.vectorize_layer(texts)
        
        # Get embeddings
        embeddings = self.text_model(sequences)
        
        return tf.nn.l2_normalize(embeddings, axis=-1)

class MultimodalEmbedding(Model):
    """Combined model for basic image-text embedding."""
    
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        
        # Initialize basic encoders
        self.image_encoder = ImageEncoder(embedding_dim)
        self.text_encoder = TextEncoder(embedding_dim)
        
        # Temperature parameter for similarity scaling
        self.temperature = tf.Variable(0.07, trainable=True)
    
    def call(self, batch):
        # Encode images and texts
        image_embeddings = self.image_encoder(batch['image'])
        text_embeddings = self.text_encoder(batch['text'])
        
        return {
            'image_embeddings': image_embeddings,
            'text_embeddings': text_embeddings,
            'temperature': self.temperature
        }
    
    def compute_similarity(self, image_embeddings, text_embeddings):
        """Compute similarity matrix between image and text embeddings."""
        # Compute cosine similarity
        similarity = tf.matmul(image_embeddings, text_embeddings, transpose_b=True)
        
        # Scale by temperature
        similarity = similarity / self.temperature
        
        return similarity 