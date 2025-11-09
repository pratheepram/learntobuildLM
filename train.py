"""
Training script for the TinySLM decoder-only Transformer model.

This script implements the training loop for the GPT-style model using MLX.
It handles data loading, model initialization, training, validation, and checkpointing.
"""

import time
from typing import Tuple
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from model import GPT, vocab_size, block_size, n_layer, n_head, n_embd
from mlx.utils import tree_flatten, tree_unflatten, tree_map
import math
import os
from loguru import logger

# Import all configuration from config.py (single source of truth)
from config import (
    DATA_PATH,
    MODEL_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    STEPS_PER_EPOCH,
    VAL_STEPS,
    SAVE_EVERY,
    LOG_INTERVAL,
    GRADIENT_ACCUMULATION_STEPS,
    USE_EARLY_STOPPING,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    SAVE_BEST_MODEL,
)

def get_batch(data: np.memmap, batch_size: int, block_size: int) -> Tuple[mx.array, mx.array]:
    """
    Sample a random batch of training data from the memory-mapped dataset.
    
    Args:
        data: Memory-mapped numpy array containing tokenized data
        batch_size: Number of sequences in the batch
        block_size: Length of each sequence (context window)
    
    Returns:
        Tuple of (x, y) where:
        - x: Input sequences of shape (batch_size, block_size)
        - y: Target sequences (shifted by 1) of shape (batch_size, block_size)
    """
    data_size = len(data)
    
    # Select random starting points for our batches
    # Ensure we don't go out of bounds
    max_start = max(0, data_size - block_size)
    if max_start <= 0:
        raise ValueError(f"Data size ({data_size}) is smaller than block_size ({block_size})")
    
    ix = np.random.randint(0, max_start, (batch_size,))
    
    # Create the x (input) and y (target) batches
    # x is the context, y is the context shifted by 1 (next token prediction)
    # Create 2D numpy arrays first, then convert to MLX arrays
    x_np = np.array([data[i:i+block_size] for i in ix])
    y_np = np.array([data[i+1:i+block_size+1] for i in ix])
    x = mx.array(x_np)
    y = mx.array(y_np)
    return x, y

def loss_fn(model: GPT, x: mx.array, y: mx.array) -> mx.array:
    """
    Calculate the cross-entropy loss for next-token prediction.
    
    This must be a separate function for nn.value_and_grad to work properly.
    It computes the loss between model predictions and target tokens.
    
    Args:
        model: The GPT model instance
        x: Input token sequences of shape (batch_size, block_size)
        y: Target token sequences of shape (batch_size, block_size)
    
    Returns:
        Scalar loss value (mean cross-entropy)
    """
    logits = model(x)
    B, T, C = logits.shape
    
    # Reshape for cross_entropy
    # (B, T, C) -> (B*T, C) - flatten batch and sequence dimensions
    logits = logits.reshape(B * T, C)
    # (B, T) -> (B*T,) - flatten target tokens
    y = y.reshape(B * T)
    
    return nn.losses.cross_entropy(logits, y, reduction="mean")

def evaluate_ppl(model: GPT, val_data: np.memmap) -> Tuple[float, float]:
    """
    Calculate the Perplexity and loss on the validation set.
    
    Perplexity is the exponentiation of the cross-entropy loss.
    Lower perplexity indicates better model performance.
    PPL = e^(loss)
    
    Args:
        model: The GPT model instance
        val_data: Memory-mapped validation dataset
    
    Returns:
        Tuple of (perplexity, validation_loss)
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    
    logger.info(f"Evaluating on {VAL_STEPS} validation batches...")
    for step in range(VAL_STEPS):
        x, y = get_batch(val_data, BATCH_SIZE, block_size)
        loss = loss_fn(model, x, y)
        total_loss += loss.item()
        
        if (step + 1) % 10 == 0:
            logger.debug(f"Validation step {step + 1}/{VAL_STEPS}")
    
    avg_val_loss = total_loss / VAL_STEPS
    perplexity = math.exp(avg_val_loss)
    
    model.train()  # Set model back to training mode
    return perplexity, avg_val_loss

def save_checkpoint(model: GPT, optimizer: optim.Optimizer, epoch: int) -> None:
    """
    Save a complete checkpoint: model weights AND optimizer state.
    
    This is critical for resuming training. The optimizer state contains
    momentum and other state that is essential for proper training continuation.
    Failing to save the optimizer state will destroy training convergence.
    
    Args:
        model: The GPT model instance
        optimizer: The optimizer instance (contains state to save)
        epoch: Current epoch number (used in filename)
    """
    logger.info(f"Saving checkpoint at epoch {epoch}...")
    
    # 1. Save model weights
    model_file = os.path.join(MODEL_PATH, f"tinyslm_model_epoch_{epoch}.npz")
    model.save_weights(model_file)
    logger.debug(f"Model weights saved to {model_file}")
    
    # 2. Save optimizer state
    # This is CRITICAL. Failing to save this will destroy
    # training convergence upon resumption.
    opt_file = os.path.join(MODEL_PATH, f"tinyslm_optimizer_epoch_{epoch}.safetensors")
    state_dict = tree_flatten(optimizer.state)
    mx.save_safetensors(opt_file, dict(state_dict))
    logger.debug(f"Optimizer state saved to {opt_file}")
    
    logger.info(f"Checkpoint saved to {MODEL_PATH}")

def main() -> None:
    """
    Main training function.
    
    Orchestrates the entire training process:
    1. Loads tokenized data from binary files
    2. Initializes model and optimizer
    3. Runs training loop with validation
    4. Saves checkpoints periodically
    """
    logger.info("=" * 60)
    logger.info("Starting TinySLM Training")
    logger.info("=" * 60)
    
    # Log training configuration
    logger.info("Training Configuration:")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS} (with gradient accumulation)")
    logger.info(f"  Learning rate: {LEARNING_RATE}")
    logger.info(f"  Weight decay: {WEIGHT_DECAY}")
    logger.info(f"  Epochs: {NUM_EPOCHS}")
    logger.info(f"  Steps per epoch: {STEPS_PER_EPOCH}")
    logger.info(f"  Validation steps: {VAL_STEPS}")
    logger.info(f"  Save every: {SAVE_EVERY} epochs")
    logger.info(f"  Log interval: {LOG_INTERVAL} steps")
    logger.info(f"  Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    
    # Log model configuration
    logger.info("Model Configuration:")
    logger.info(f"  Vocabulary size: {vocab_size}")
    logger.info(f"  Embedding dimension: {n_embd}")
    logger.info(f"  Number of layers: {n_layer}")
    logger.info(f"  Number of heads: {n_head}")
    logger.info(f"  Block size (context length): {block_size}")
    
    # Load data using numpy.memmap
    logger.info("Loading data...")
    train_file = os.path.join(DATA_PATH, 'train.bin')
    val_file = os.path.join(DATA_PATH, 'val.bin')
    
    if not os.path.exists(train_file):
        logger.error(f"Training data file not found: {train_file}")
        raise FileNotFoundError(f"Training data file not found: {train_file}")
    if not os.path.exists(val_file):
        logger.error(f"Validation data file not found: {val_file}")
        raise FileNotFoundError(f"Validation data file not found: {val_file}")
    
    train_data = np.memmap(train_file, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_file, dtype=np.uint16, mode='r')
    
    train_tokens = len(train_data)
    val_tokens = len(val_data)
    logger.info(f"Loaded training data: {train_tokens:,} tokens")
    logger.info(f"Loaded validation data: {val_tokens:,} tokens")

    # 1. Setup Model and Optimizer
    logger.info("Initializing model and optimizer...")
    model = GPT()
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 2. Materialize model parameters
    # This forces MLX to initialize the model's weights.
    mx.eval(model.parameters())
    
    # Count model parameters
    # Recursively count all parameters in the model
    def count_params(params):
        """Recursively count parameters in a nested structure."""
        total = 0
        if isinstance(params, dict):
            for v in params.values():
                total += count_params(v)
        elif isinstance(params, (list, tuple)):
            for item in params:
                total += count_params(item)
        elif hasattr(params, 'size'):
            total += params.size
        return total
    
    total_params = count_params(model.parameters())
    logger.info(f"Model initialized with {total_params:,} parameters")
    
    # Memory warning (after model initialization)
    estimated_memory_mb = (
        (BATCH_SIZE * block_size * block_size * n_head * 4 * 4) / (1024 * 1024) +  # Attention matrices with gradients
        (BATCH_SIZE * block_size * n_embd * 4 * 4) / (1024 * 1024) +  # Embeddings
        (total_params * 4 * 3) / (1024 * 1024)  # Model weights + gradients + optimizer state
    )
    logger.warning(f"⚠️  Estimated peak memory usage: ~{estimated_memory_mb:.0f}MB")
    logger.warning(f"⚠️  If system crashes, reduce BATCH_SIZE to 4 or reduce block_size in model.py")
    
    # 3. Create the value_and_grad function
    # This transforms loss_fn into a new function that
    # returns both the loss and the gradients.
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    logger.info("Training setup complete")

    # --- The Training Loop ---
    logger.info("=" * 60)
    logger.info("Starting training loop...")
    logger.info("=" * 60)
    
    # Early stopping state
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    if USE_EARLY_STOPPING:
        logger.info(f"Early stopping enabled:")
        logger.info(f"  Patience: {EARLY_STOPPING_PATIENCE} epochs")
        logger.info(f"  Min delta: {EARLY_STOPPING_MIN_DELTA}")
        logger.info(f"  Save best model: {SAVE_BEST_MODEL}")
    else:
        logger.info(f"Early stopping disabled - training for full {NUM_EPOCHS} epochs")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_start_time = time.time()
        total_epoch_loss = 0.0
        best_loss = float('inf')

        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        logger.info(f"{'='*60}")

        accumulated_grads = None
        accumulation_count = 0
        
        for step in range(STEPS_PER_EPOCH):
            step_start_time = time.time()
            
            x, y = get_batch(train_data, BATCH_SIZE, block_size)
            
            # 4. Calculate loss and gradients
            # This builds the computation graph.
            loss, grads = loss_and_grad_fn(model, x, y)
            
            # Gradient accumulation: accumulate gradients over multiple steps
            # This allows us to use smaller batch sizes while maintaining effective batch size
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                # Accumulate gradients
                accumulated_grads = tree_map(
                    lambda a, b: a + b,
                    accumulated_grads,
                    grads
                )
            
            accumulation_count += 1
            loss_value = loss.item()
            total_epoch_loss += loss_value
            
            # Only update weights after accumulating enough gradients
            if accumulation_count >= GRADIENT_ACCUMULATION_STEPS:
                # Normalize accumulated gradients
                accumulated_grads = tree_map(
                    lambda g: g / GRADIENT_ACCUMULATION_STEPS,
                    accumulated_grads
                )
                
                # 5. Update model parameters
                # This adds the optimizer update to the graph.
                optimizer.update(model, accumulated_grads)
                
                # 6. CRITICAL: Evaluate computation
                # MLX is lazy. This forces the execution of all
                # previous steps (loss, grads, update).
                mx.eval(model.parameters(), optimizer.state)
                
                # Reset accumulation
                accumulated_grads = None
                accumulation_count = 0
            
            # Track best loss for this epoch
            if loss_value < best_loss:
                best_loss = loss_value
            
            # Log progress at regular intervals
            if (step + 1) % LOG_INTERVAL == 0 or step == 0:
                step_time = (time.time() - step_start_time) * 1000  # in ms
                progress = 100 * (step + 1) / STEPS_PER_EPOCH
                avg_loss_so_far = total_epoch_loss / (step + 1)
                logger.info(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                    f"Step {step + 1}/{STEPS_PER_EPOCH} ({progress:.1f}%) | "
                    f"Loss: {loss_value:.4f} | "
                    f"Avg Loss: {avg_loss_so_far:.4f} | "
                    f"Time: {step_time:.2f}ms"
                )
        
        # Handle remaining accumulated gradients at end of epoch
        if accumulated_grads is not None and accumulation_count > 0:
            # Normalize and apply remaining gradients
            accumulated_grads = tree_map(
                lambda g: g / accumulation_count,
                accumulated_grads
            )
            optimizer.update(model, accumulated_grads)
            mx.eval(model.parameters(), optimizer.state)
            logger.debug(f"Applied remaining {accumulation_count} accumulated gradient steps")
        
        # --- End of Epoch ---
        epoch_time = time.time() - epoch_start_time
        avg_epoch_loss = total_epoch_loss / STEPS_PER_EPOCH
        
        # Evaluate perplexity and validation loss
        logger.info("Running validation...")
        val_ppl, val_loss = evaluate_ppl(model, val_data)
        
        # Early stopping logic
        improved = False
        if USE_EARLY_STOPPING:
            # Check if validation loss improved
            improvement = best_val_loss - val_loss
            if improvement > EARLY_STOPPING_MIN_DELTA:
                # Validation loss improved
                improved = True
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                
                logger.info(f"✓ Validation loss improved: {best_val_loss:.4f} (improvement: {improvement:.4f})")
                
                # Save best model if enabled
                if SAVE_BEST_MODEL:
                    logger.info("Saving best model checkpoint...")
                    model_file = os.path.join(MODEL_PATH, "tinyslm_model_best.npz")
                    opt_file = os.path.join(MODEL_PATH, "tinyslm_optimizer_best.safetensors")
                    model.save_weights(model_file)
                    state_dict = tree_flatten(optimizer.state)
                    mx.save_safetensors(opt_file, dict(state_dict))
                    logger.info(f"Best model saved to {MODEL_PATH}")
            else:
                # Validation loss did not improve
                patience_counter += 1
                logger.info(
                    f"✗ Validation loss did not improve: {val_loss:.4f} "
                    f"(best: {best_val_loss:.4f}, patience: {patience_counter}/{EARLY_STOPPING_PATIENCE})"
                )
                
                # Check if we should stop
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    logger.warning(f"\n{'='*60}")
                    logger.warning("Early stopping triggered!")
                    logger.warning(f"Validation loss did not improve for {EARLY_STOPPING_PATIENCE} epochs")
                    logger.warning(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
                    logger.warning(f"Stopping training at epoch {epoch + 1}")
                    logger.warning(f"{'='*60}\n")
                    break
        
        # Log epoch summary
        logger.info(f"\n{'─'*60}")
        logger.info(f"Epoch {epoch + 1} Summary:")
        logger.info(f"  Time: {epoch_time:.2f}s ({epoch_time/60:.2f} minutes)")
        logger.info(f"  Average Train Loss: {avg_epoch_loss:.4f}")
        logger.info(f"  Best Train Loss: {best_loss:.4f}")
        logger.info(f"  Validation Loss: {val_loss:.4f}")
        logger.info(f"  Validation Perplexity: {val_ppl:.4f}")
        if USE_EARLY_STOPPING:
            logger.info(f"  Best Val Loss: {best_val_loss:.4f} (epoch {best_epoch})")
            logger.info(f"  Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        logger.info(f"  Steps per second: {STEPS_PER_EPOCH/epoch_time:.2f}")
        logger.info(f"{'─'*60}\n")
        
        # Save checkpoint
        if (epoch + 1) % SAVE_EVERY == 0 or epoch == NUM_EPOCHS - 1:
            save_checkpoint(model, optimizer, epoch + 1)
            
    logger.info("=" * 60)
    logger.info("Training finished!")
    if USE_EARLY_STOPPING:
        logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
        if SAVE_BEST_MODEL:
            logger.info("Best model checkpoint available at: tinyslm_model_best.npz")
    logger.info("=" * 60)
    
    # Final save
    logger.info("Saving final checkpoint...")
    save_checkpoint(model, optimizer, "final")
    logger.info("Training complete!")

if __name__ == "__main__":
    main()