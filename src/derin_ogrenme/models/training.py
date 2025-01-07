"""
Training pipeline and loss functions for multimodal embedding model.
"""
import tensorflow as tf
from typing import Dict, Tuple
import numpy as np

# Configure TensorFlow for GPU optimization
tf.config.optimizer.set_jit(True)  # Enable XLA optimization
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)

@tf.function(jit_compile=True)
def contrastive_loss(similarity):
    """Compute contrastive loss for image-text pairs."""
    # Create labels (diagonal indices)
    batch_size = tf.shape(similarity)[0]
    labels = tf.range(batch_size, dtype=tf.int32)
    
    # Compute cross entropy loss in both directions
    # Image to text
    loss_i2t = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=similarity
    )
    
    # Text to image
    loss_t2i = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=tf.transpose(similarity)
    )
    
    # Average both directions
    return (loss_i2t + loss_t2i) / 2.0

@tf.function(jit_compile=True)
def compute_recall_at_k(similarity, labels, k):
    """Compute recall@k for both image-to-text and text-to-image."""
    batch_size = tf.shape(similarity)[0]
    k = tf.minimum(k, batch_size)
    
    # Image to text
    i2t_topk = tf.math.top_k(similarity, k=k)
    i2t_correct = tf.reduce_any(
        tf.equal(tf.cast(i2t_topk.indices, tf.int32), 
                tf.expand_dims(labels, -1)),
        axis=-1
    )
    i2t_recall = tf.reduce_mean(tf.cast(i2t_correct, tf.float32))
    
    # Text to image
    t2i_topk = tf.math.top_k(tf.transpose(similarity), k=k)
    t2i_correct = tf.reduce_any(
        tf.equal(tf.cast(t2i_topk.indices, tf.int32),
                tf.expand_dims(labels, -1)),
        axis=-1
    )
    t2i_recall = tf.reduce_mean(tf.cast(t2i_correct, tf.float32))
    
    return (i2t_recall + t2i_recall) / 2.0

class MultimodalTrainer:
    """Training manager for multimodal embedding model."""
    
    def __init__(
        self,
        model: tf.keras.Model,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01
    ):
        self.model = model
        
        # Use AMP (Automatic Mixed Precision) optimizer
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            jit_compile=True
        )
        self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    def compute_metrics(
        self,
        similarity: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """Compute retrieval metrics."""
        # Get the size of the batch
        batch_size = tf.shape(similarity)[0]
        
        # Ground truth labels (diagonal)
        labels = tf.range(batch_size, dtype=tf.int32)
        
        # Compute accuracy for both directions
        # Image to text
        i2t_preds = tf.cast(tf.argmax(similarity, axis=1), tf.int32)
        i2t_acc = tf.reduce_mean(tf.cast(i2t_preds == labels, tf.float32))
        
        # Text to image
        t2i_preds = tf.cast(tf.argmax(similarity, axis=0), tf.int32)
        t2i_acc = tf.reduce_mean(tf.cast(t2i_preds == labels, tf.float32))
        
        # Initialize metrics dictionary
        metrics = {
            "i2t_accuracy": i2t_acc,
            "t2i_accuracy": t2i_acc,
        }
        
        # Compute recall@k for k=1,5,10
        for k in [1, 5, 10]:
            i2t_recall, t2i_recall = compute_recall_at_k(similarity, labels, k)
            metrics[f"i2t_recall@{k}"] = i2t_recall
            metrics[f"t2i_recall@{k}"] = t2i_recall
        
        return metrics
    
    @tf.function(jit_compile=True)
    def train_step(
        self,
        batch: Dict[str, tf.Tensor]
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Optimized training step with XLA and AMP."""
        with tf.GradientTape() as tape:
            # Forward pass
            image_features = self.model.encode_image(batch["image"])
            text_features = self.model.encode_text(batch["text"])
            
            # Compute similarity
            similarity = tf.matmul(image_features, text_features, transpose_b=True)
            similarity = similarity * tf.exp(self.model.logit_scale)
            
            # Compute loss
            loss = contrastive_loss(similarity)
            scaled_loss = self.optimizer.get_scaled_loss(loss)
        
        # Compute gradients
        scaled_gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Compute metrics
        metrics = self.compute_metrics(similarity)
        metrics["loss"] = loss
        
        return loss, metrics
    
    @tf.function
    def evaluate_step(
        self,
        batch: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:
        """Single evaluation step."""
        # Forward pass
        outputs = self.model(batch)
        similarity = self.model.compute_similarity(
            outputs['image_embeddings'],
            outputs['text_embeddings']
        )
        
        # Compute loss and metrics
        loss = contrastive_loss(similarity)
        loss = tf.reduce_mean(loss)
        metrics = self.compute_metrics(similarity)
        metrics['loss'] = loss
        
        return metrics 