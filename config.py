"""
Training and Model Configuration File

This file contains all hyperparameters and settings for training the TinySLM model.
Each parameter includes detailed documentation about:
- What it controls
- How increasing it affects training
- How decreasing it affects training
- Memory/compute implications
- Recommended values for different hardware

IMPORTANT: This is the single source of truth for all training parameters.
Modify values here and import them in train.py and model.py.
"""

from typing import Final

# ============================================================================
# MODEL ARCHITECTURE PARAMETERS
# ============================================================================
# These parameters define the structure and size of the neural network.
# Changing these will fundamentally alter the model's capacity and memory usage.

# --- Vocabulary Size ---
vocab_size: Final[int] = 8192
"""
Number of unique tokens in the vocabulary.

WHAT IT DOES:
- Defines the size of the token embedding table and output projection layer
- Each token in your dataset maps to an integer ID in range [0, vocab_size)
- Determines the number of possible outputs the model can predict

INCREASING vocab_size:
  ✓ Allows model to represent more unique tokens/words
  ✓ Better for diverse vocabularies (e.g., code, multilingual text)
  ✗ Increases memory: +vocab_size * n_embd * 4 bytes (embedding table)
  ✗ Increases parameters: ~vocab_size * n_embd parameters
  ✗ Slower training (larger output layer)
  ✗ May require more data to learn all tokens effectively

DECREASING vocab_size:
  ✓ Reduces memory usage significantly
  ✓ Faster training (smaller output layer)
  ✓ Fewer parameters to learn
  ✗ May lose expressiveness if vocabulary is too small
  ✗ Rare words might be mapped to <UNK> token
  ✗ Lower quality for specialized domains

RECOMMENDED:
  - For English text: 5,000-10,000
  - For code: 8,000-16,000
  - For multilingual: 16,000-32,000
  - Must match the tokenizer used in data/prepare.py
"""

# --- Embedding Dimension ---
n_embd: Final[int] = 448
"""
The dimension of token and position embeddings (hidden size).

WHAT IT DOES:
- Size of the vector representation for each token
- All transformer layers operate on vectors of this size
- Determines the "width" of the model (vs. n_layer which is "depth")

INCREASING n_embd:
  ✓ More expressive representations (model can learn richer features)
  ✓ Better performance on complex tasks
  ✗ QUADRATIC memory increase: attention matrices scale as n_embd²
  ✗ More parameters: ~n_embd² per attention layer
  ✗ Slower training (more compute per layer)
  ✗ Memory formula: batch_size * block_size² * n_head * 4 bytes

DECREASING n_embd:
  ✓ Significantly reduces memory usage (quadratic savings)
  ✓ Faster training
  ✓ Fewer parameters
  ✗ Less expressive (may struggle with complex patterns)
  ✗ May need more layers (n_layer) to compensate

RECOMMENDED:
  - Tiny (16GB RAM): 256-384
  - Small (24GB RAM): 384-512
  - Medium (32GB+ RAM): 512-768
  - Must be divisible by n_head
"""

# --- Number of Layers ---
n_layer: Final[int] = 10
"""
Number of transformer blocks (depth of the model).

WHAT IT DOES:
- Each layer applies self-attention + MLP
- More layers = deeper model = more sequential processing
- Determines how many "hops" of reasoning the model can do

INCREASING n_layer:
  ✓ Deeper reasoning (model can learn more complex patterns)
  ✓ Better performance on tasks requiring long-range dependencies
  ✗ LINEAR memory increase: each layer stores activations
  ✗ More parameters: ~(n_embd² * 12) per layer
  ✗ Slower training (more forward/backward passes)
  ✗ Harder to train (vanishing gradients, need better initialization)
  ✗ Risk of overfitting if data is limited

DECREASING n_layer:
  ✓ Reduces memory usage (fewer activation stores)
  ✓ Faster training
  ✓ Fewer parameters
  ✓ Easier to train (fewer gradient issues)
  ✗ Less capacity (may struggle with complex patterns)
  ✗ Shorter reasoning chains

RECOMMENDED:
  - Tiny: 6-8 layers
  - Small: 8-12 layers
  - Medium: 12-24 layers
  - Large: 24-48+ layers
"""

# --- Number of Attention Heads ---
n_head: Final[int] = 8
"""
Number of parallel attention heads in multi-head attention.

WHAT IT DOES:
- Each head learns to attend to different aspects of the sequence
- Allows model to capture multiple types of relationships simultaneously
- Head dimension = n_embd / n_head

INCREASING n_head:
  ✓ More parallel attention patterns (richer representations)
  ✓ Better for complex relationships
  ✗ More compute (each head processes independently)
  ✗ Memory: attention matrices scale with n_head
  ✗ Diminishing returns beyond 8-16 heads
  ✗ Must keep n_embd divisible by n_head

DECREASING n_head:
  ✓ Less compute and memory
  ✓ Faster attention computation
  ✗ Fewer attention patterns (less expressive)
  ✗ May struggle with diverse relationships

RECOMMENDED:
  - Standard: 8-16 heads
  - Small models: 4-8 heads
  - Large models: 16-32 heads
  - Must divide n_embd evenly (e.g., n_embd=448, n_head=8 works: 448/8=56)
"""

# --- Block Size (Context Length) ---
block_size: Final[int] = 256
"""
Maximum sequence length (context window) the model can process.

WHAT IT DOES:
- Maximum number of tokens the model can see at once
- Determines how much context the model can use for predictions
- Sequences longer than this must be truncated or split

INCREASING block_size:
  ✓ Longer context (model sees more history)
  ✓ Better for long documents, code, conversations
  ✗ QUADRATIC memory increase: attention is O(block_size²)
  ✗ Memory formula: batch_size * block_size² * n_head * 4 bytes
  ✗ Much slower training (attention scales quadratically)
  ✗ May need more data to learn long-range patterns

DECREASING block_size:
  ✓ Dramatically reduces memory (quadratic savings)
  ✓ Much faster training
  ✗ Shorter context (model sees less history)
  ✗ May lose long-range dependencies
  ✗ Poor for long documents or code

RECOMMENDED:
  - Tiny (16GB): 128-256
  - Small (24GB): 256-512
  - Medium (32GB+): 512-1024
  - Large: 1024-2048+
  - Must match or exceed your typical sequence length
"""

# --- Dropout Rate ---
dropout: Final[float] = 0.1
"""
Probability of randomly zeroing activations during training (regularization).

WHAT IT DOES:
- Prevents overfitting by randomly dropping activations
- Forces model to learn robust features (not rely on specific neurons)
- Only active during training (disabled during inference)

INCREASING dropout:
  ✓ Stronger regularization (reduces overfitting)
  ✓ Better generalization
  ✗ May underfit (model becomes too conservative)
  ✗ Slower convergence (model learns less per step)
  ✗ Too high (>0.5) can prevent learning

DECREASING dropout:
  ✓ Faster convergence
  ✓ Model learns more aggressively
  ✗ Higher risk of overfitting
  ✗ May memorize training data instead of generalizing

RECOMMENDED:
  - Standard: 0.1-0.2
  - Small models: 0.0-0.1 (less prone to overfitting)
  - Large models: 0.1-0.3 (more prone to overfitting)
  - If validation loss >> training loss: increase dropout
  - If both losses are high: decrease dropout
"""

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================
# These parameters control how the model learns from data.

# --- Batch Size ---
BATCH_SIZE: Final[int] = 8
"""
Number of sequences processed in parallel per training step.

WHAT IT DOES:
- Larger batches = more stable gradients (less noise)
- Determines memory usage per step
- Affects gradient quality and training speed

INCREASING BATCH_SIZE:
  ✓ More stable gradients (less variance)
  ✓ Better GPU/accelerator utilization
  ✓ Can use higher learning rates
  ✗ LINEAR memory increase: batch_size * block_size * n_embd * 4 bytes
  ✗ Attention memory: batch_size * block_size² * n_head (QUADRATIC in block_size)
  ✗ May need to adjust learning rate (larger batches often need higher LR)
  ✗ Slower per-step (but may need fewer steps)

DECREASING BATCH_SIZE:
  ✓ Less memory usage
  ✓ More frequent updates (faster per-step)
  ✓ Better for small datasets
  ✗ Noisier gradients (training may be less stable)
  ✗ Underutilizes hardware (slower overall)
  ✗ May need lower learning rate

RECOMMENDED:
  - 16GB Mac Mini M4: 4-8
  - 24GB: 8-16
  - 32GB+: 16-32
  - If OOM errors: reduce to 4 or use gradient accumulation
  - Memory scales with: batch_size * block_size² * n_head
"""

# --- Learning Rate ---
LEARNING_RATE: Final[float] = 3e-4
"""
Step size for parameter updates (how big steps the optimizer takes).

WHAT IT DOES:
- Controls how much the model changes per update
- Too high: training unstable, loss explodes
- Too low: training too slow, may get stuck in local minima

INCREASING LEARNING_RATE:
  ✓ Faster convergence (fewer steps to reach good loss)
  ✓ May escape bad local minima
  ✗ Risk of training instability (loss NaN, exploding gradients)
  ✗ May overshoot optimal parameters
  ✗ Can cause loss to oscillate or diverge

DECREASING LEARNING_RATE:
  ✓ More stable training
  ✓ Finer-grained optimization
  ✗ Slower convergence (needs more steps)
  ✗ May get stuck in poor local minima
  ✗ Wastes compute if too conservative

RECOMMENDED:
  - Standard: 1e-4 to 3e-4
  - With warmup: start at 1e-6, ramp to 3e-4
  - Large batches: try 1e-3 to 3e-3
  - Small batches: try 1e-4 to 1e-3
  - If loss explodes: reduce by 10x
  - If loss plateaus: try 2-3x increase
"""

# --- Weight Decay ---
WEIGHT_DECAY: Final[float] = 0.01
"""
L2 regularization strength (penalizes large parameter values).

WHAT IT DOES:
- Prevents overfitting by keeping parameters small
- Acts as implicit constraint on model complexity
- Helps generalization

INCREASING WEIGHT_DECAY:
  ✓ Stronger regularization (reduces overfitting)
  ✓ Smaller parameters (more stable)
  ✗ May underfit (model too constrained)
  ✗ Slower learning (parameters shrink too aggressively)

DECREASING WEIGHT_DECAY:
  ✓ Faster learning
  ✓ Less constraint on parameters
  ✗ Higher risk of overfitting
  ✗ Parameters may grow too large

RECOMMENDED:
  - Standard: 0.01-0.1
  - Small models: 0.001-0.01
  - Large models: 0.01-0.1
  - If validation loss >> training loss: increase
  - If both losses high: decrease or set to 0
"""

# --- Number of Epochs ---
NUM_EPOCHS: Final[int] = 20
"""
Total number of times to iterate through the entire training dataset.

WHAT IT DOES:
- One epoch = one full pass through training data
- More epochs = more learning opportunities
- Determines total training time

INCREASING NUM_EPOCHS:
  ✓ More learning (model sees data more times)
  ✓ Better final performance (if not overfitting)
  ✗ Longer training time
  ✗ Risk of overfitting (if validation loss stops improving)
  ✗ Diminishing returns (early stopping often better)

DECREASING NUM_EPOCHS:
  ✓ Faster training
  ✓ Less risk of overfitting
  ✗ May underfit (model hasn't learned enough)
  ✗ Lower final performance

RECOMMENDED:
  - Start with 10-20 epochs
  - Use early stopping (stop if validation loss doesn't improve)
  - Monitor validation loss: stop when it plateaus or increases
  - Large datasets: fewer epochs needed
  - Small datasets: more epochs may help
"""

# --- Steps Per Epoch ---
STEPS_PER_EPOCH: Final[int] = 2000
"""
Number of training steps (batches) to process per epoch.

WHAT IT DOES:
- Controls how many gradient updates per epoch
- Total training steps = NUM_EPOCHS * STEPS_PER_EPOCH
- Determines epoch duration

INCREASING STEPS_PER_EPOCH:
  ✓ More gradient updates per epoch
  ✓ Better convergence (more learning per epoch)
  ✗ Longer epochs
  ✗ May overfit if too many steps

DECREASING STEPS_PER_EPOCH:
  ✓ Faster epochs
  ✓ More frequent validation checks
  ✗ Fewer updates per epoch
  ✗ May need more epochs to converge

RECOMMENDED:
  - Calculate based on dataset size: dataset_size / (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
  - Typical: 1000-5000 steps per epoch
  - Should be enough to see most of the data each epoch
  - If dataset is small, may need to repeat samples
"""

# --- Validation Steps ---
VAL_STEPS: Final[int] = 50
"""
Number of validation batches to evaluate per validation check.

WHAT IT DOES:
- How many batches to use for computing validation metrics
- More steps = more accurate validation score (but slower)
- Used to monitor overfitting

INCREASING VAL_STEPS:
  ✓ More accurate validation metrics
  ✓ Better estimate of true performance
  ✗ Slower validation (delays training)
  ✗ Diminishing returns (50-100 usually sufficient)

DECREASING VAL_STEPS:
  ✓ Faster validation
  ✓ More frequent training updates
  ✗ Less accurate metrics (more variance)
  ✗ May miss overfitting signals

RECOMMENDED:
  - Standard: 50-100 steps
  - Small validation sets: use all available batches
  - Large validation sets: 50-200 is usually enough
  - Should be enough to get stable perplexity estimate
"""

# --- Gradient Accumulation Steps ---
GRADIENT_ACCUMULATION_STEPS: Final[int] = 8
"""
Number of steps to accumulate gradients before updating parameters.

WHAT IT DOES:
- Simulates larger batch size without using more memory
- Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
- Allows training with small batches on memory-constrained hardware

INCREASING GRADIENT_ACCUMULATION_STEPS:
  ✓ Larger effective batch size (more stable gradients)
  ✓ Better gradient estimates
  ✓ Can train with smaller BATCH_SIZE (saves memory)
  ✗ Slower updates (waits for accumulation)
  ✗ Delayed learning (updates less frequent)

DECREASING GRADIENT_ACCUMULATION_STEPS:
  ✓ More frequent updates
  ✓ Faster training progress
  ✗ Smaller effective batch size
  ✗ Noisier gradients (may need to reduce learning rate)

RECOMMENDED:
  - Use when BATCH_SIZE is limited by memory
  - Effective batch size should be 64-256 for stable training
  - Example: BATCH_SIZE=8, GRADIENT_ACCUMULATION_STEPS=8 → effective batch=64
  - If memory allows, prefer larger BATCH_SIZE over accumulation
"""

# --- Save Every N Epochs ---
SAVE_EVERY: Final[int] = 2
"""
Save a checkpoint every N epochs.

WHAT IT DOES:
- Controls checkpoint frequency
- More frequent = can resume from more points, but uses more disk space
- Checkpoints include model weights AND optimizer state

INCREASING SAVE_EVERY:
  ✓ Less disk space usage
  ✓ Faster training (fewer I/O operations
  ✗ Fewer recovery points (if training crashes)
  ✗ May lose more progress

DECREASING SAVE_EVERY:
  ✓ More recovery points
  ✓ Can compare checkpoints at different stages
  ✗ More disk space usage
  ✗ Slower training (more I/O)

RECOMMENDED:
  - Standard: 2-5 epochs
  - Long training: 5-10 epochs
  - Short experiments: every epoch
  - Always save final checkpoint
"""

# --- Log Interval ---
LOG_INTERVAL: Final[int] = 100
"""
Log training progress every N steps.

WHAT IT DOES:
- Controls how often to print loss and metrics
- More frequent = better visibility, but more I/O

INCREASING LOG_INTERVAL:
  ✓ Less I/O overhead
  ✓ Cleaner logs
  ✗ Less visibility into training progress
  ✗ Harder to debug issues

DECREASING LOG_INTERVAL:
  ✓ Better visibility
  ✓ Easier to spot problems early
  ✗ More I/O overhead
  ✗ Verbose logs

RECOMMENDED:
  - Standard: 50-200 steps
  - Fast training: 100-500 steps
  - Slow training: 10-50 steps
  - Should log at least a few times per epoch
"""

# --- Early Stopping ---
USE_EARLY_STOPPING: Final[bool] = True
"""
Enable early stopping to prevent overfitting.

WHAT IT DOES:
- Monitors validation loss/perplexity after each epoch
- Stops training if validation loss doesn't improve for PATIENCE epochs
- Saves the best model checkpoint automatically
- Prevents wasting compute on overfitting

INCREASING (enabling) USE_EARLY_STOPPING:
  ✓ Prevents overfitting (stops when validation stops improving)
  ✓ Saves compute time (stops unnecessary training)
  ✓ Automatically saves best model
  ✗ May stop too early if validation loss is noisy
  ✗ Need to tune PATIENCE parameter

DECREASING (disabling) USE_EARLY_STOPPING:
  ✓ Trains for full NUM_EPOCHS regardless
  ✓ Good for experiments where you want to see full training curve
  ✗ May waste compute on overfitting
  ✗ Need to manually monitor and stop

RECOMMENDED:
  - Enable for production training (saves time and prevents overfitting)
  - Disable for experiments/exploration
  - Works best with stable validation metrics (use enough VAL_STEPS)
"""

EARLY_STOPPING_PATIENCE: Final[int] = 3
"""
Number of epochs to wait for validation improvement before stopping.

WHAT IT DOES:
- If validation loss doesn't improve (decrease) for PATIENCE epochs, training stops
- Higher patience = more tolerance for temporary plateaus
- Lower patience = stops faster, but may stop too early

INCREASING EARLY_STOPPING_PATIENCE:
  ✓ More tolerance for temporary plateaus
  ✓ Less likely to stop too early
  ✗ May continue training when already overfitting
  ✗ Wastes more compute if model has truly converged

DECREASING EARLY_STOPPING_PATIENCE:
  ✓ Stops faster (saves compute)
  ✓ More aggressive early stopping
  ✗ May stop too early if validation is noisy
  ✗ May interrupt training during temporary plateaus

RECOMMENDED:
  - Standard: 3-5 epochs
  - Noisy validation: 5-7 epochs
  - Stable validation: 2-3 epochs
  - Very long training: 5-10 epochs
  - If training stops too early: increase patience
  - If overfitting occurs: decrease patience
"""

EARLY_STOPPING_MIN_DELTA: Final[float] = 0.0
"""
Minimum change in validation loss to qualify as an improvement.

WHAT IT DOES:
- Validation loss must decrease by at least MIN_DELTA to count as improvement
- Prevents stopping on tiny, insignificant improvements
- Set to 0.0 to accept any decrease, or higher (e.g., 0.001) for meaningful improvements

INCREASING EARLY_STOPPING_MIN_DELTA:
  ✓ Only counts meaningful improvements
  ✓ Less sensitive to noise
  ✗ May be too strict (miss small but real improvements)
  ✗ May stop too early if improvements are small but consistent

DECREASING EARLY_STOPPING_MIN_DELTA:
  ✓ More sensitive to any improvement
  ✓ Continues training on small improvements
  ✗ May continue on insignificant improvements
  ✗ More sensitive to validation noise

RECOMMENDED:
  - Standard: 0.0 (accept any decrease)
  - For loss: 0.0-0.001
  - For perplexity: 0.01-0.1 (since perplexity is larger scale)
  - If validation is noisy: use 0.001-0.01
  - If validation is stable: use 0.0
"""

SAVE_BEST_MODEL: Final[bool] = True
"""
Save the best model checkpoint (lowest validation loss) separately.

WHAT IT DOES:
- Tracks the best validation loss seen so far
- Saves a special "best" checkpoint when new best is found
- Useful for resuming from the best model, not just the latest

INCREASING (enabling) SAVE_BEST_MODEL:
  ✓ Always have access to best model
  ✓ Can resume from best checkpoint
  ✗ Uses extra disk space (one additional checkpoint)

DECREASING (disabling) SAVE_BEST_MODEL:
  ✓ Saves disk space
  ✗ May lose best model if training continues and overfits
  ✗ Latest checkpoint may not be the best

RECOMMENDED:
  - Enable for production training
  - Disable if disk space is limited
  - Best checkpoint saved as: tinyslm_model_best.npz
"""

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

DATA_PATH: Final[str] = "./data/"
"""
Path to directory containing train.bin and val.bin files.

WHAT IT DOES:
- Location of preprocessed tokenized data
- Must contain train.bin and val.bin (created by data/prepare.py)
- Data is loaded using numpy.memmap for memory efficiency
"""

MODEL_PATH: Final[str] = "./"
"""
Path to directory for saving model checkpoints.

WHAT IT DOES:
- Where model weights and optimizer state are saved
- Checkpoints saved as: tinyslm_model_epoch_{N}.npz and tinyslm_optimizer_epoch_{N}.safetensors
- Should have sufficient disk space (each checkpoint ~100-500MB depending on model size)
"""

# ============================================================================
# MEMORY ESTIMATION
# ============================================================================
"""
Approximate memory usage calculation:

1. Model Parameters:
   - Embeddings: vocab_size * n_embd * 4 bytes
   - Attention layers: n_layer * (n_embd² * 4 * 3) * 4 bytes  (QKV projections)
   - MLP layers: n_layer * (n_embd * 4 * n_embd * 2) * 4 bytes
   - Total: ~16-17M parameters * 4 bytes = ~64-68MB

2. Activations (per batch):
   - Attention matrices: BATCH_SIZE * block_size² * n_head * 4 bytes
   - Embeddings: BATCH_SIZE * block_size * n_embd * 4 bytes
   - MLP activations: BATCH_SIZE * block_size * n_embd * 4 * 4 bytes

3. Gradients: Same size as parameters (another 64-68MB)

4. Optimizer state (AdamW): 2x parameters (momentum + variance = 128-136MB)

5. Total Peak Memory:
   - Model: ~64MB
   - Gradients: ~64MB
   - Optimizer: ~128MB
   - Activations: ~2-4GB (depends heavily on BATCH_SIZE and block_size)
   - Total: ~2.5-4.5GB for typical settings

CRITICAL: Attention memory scales as BATCH_SIZE * block_size² * n_head
If you get OOM errors, reduce BATCH_SIZE first, then block_size.
"""

