"""
Encoder models for image and text processing.
"""
import tensorflow as tf

class ImageEncoder(tf.keras.Model):
    """Basic CNN-based image encoder with a few conv layers."""
    
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        
        # Simple CNN model with multiple conv layers
        self.image_model = tf.keras.Sequential([
            # First conv block
            tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(2),
            
            # Second conv block
            tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(2),
            
            # Third conv block
            tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D(2),
            
            # Dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(embedding_dim),
        ])
    
    def call(self, images):
        return tf.nn.l2_normalize(self.image_model(images), axis=-1)

class TextEncoder(tf.keras.Model):
    """Basic text encoder with embedding and conv layers."""
    
    def __init__(self, embedding_dim: int = 128, vocab_size: int = 10000, max_length: int = 100):
        super().__init__()
        
        self.max_length = max_length
        
        # Text processing layers
        self.text_model = tf.keras.Sequential([
            # Embedding layer
            tf.keras.layers.Embedding(vocab_size, 256, input_length=max_length),
            
            # 1D convolutions for feature extraction
            tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(256, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling1D(2),
            
            # Global pooling and dense layers
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(embedding_dim),
        ])
        
        # Simple tokenizer
        self.vectorize_layer = tf.keras.layers.TextVectorization(
            max_tokens=vocab_size,
            output_mode='int',
            output_sequence_length=max_length
        )
        
        # Initialize with dummy data
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

class MultimodalEmbedding(tf.keras.Model):
    """Combined model for image-text embedding."""
    
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        
        # Initialize encoders
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
        """Compute cosine similarity matrix between image and text embeddings."""
        # Normalize embeddings (although they should already be normalized)
        image_embeddings = tf.nn.l2_normalize(image_embeddings, axis=-1)
        text_embeddings = tf.nn.l2_normalize(text_embeddings, axis=-1)
        
        # Compute cosine similarity
        similarity = tf.matmul(image_embeddings, text_embeddings, transpose_b=True)
        
        # Scale by temperature
        similarity = similarity / self.temperature
        
        return similarity 