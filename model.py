"""
Decoder-only Transformer model (GPT-style) implementation using MLX.

This module implements a small, efficient Transformer model designed for training
on resource-constrained hardware (e.g., 16GB Mac Mini M4). The architecture
follows the GPT design with causal self-attention, feed-forward networks, and
layer normalization.

Model Architecture:
- Token and position embeddings
- Stack of N transformer blocks (each with attention + MLP)
- Final layer normalization
- Language model head (vocabulary projection)

Parameter Count: ~16-17M parameters
"""

import mlx.core as mx
import mlx.nn as nn
import math

# Import model hyperparameters from config.py (single source of truth)
# See config.py for detailed documentation on each parameter
from config import (
    vocab_size,
    n_embd,
    n_layer,
    n_head,
    block_size,
    dropout,
)

class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention mechanism for decoder-only transformers.
    
    This is the core attention mechanism that allows tokens to "look at" and
    "attend" to other tokens in the sequence. The "causal" nature ensures that
    a token at position `i` can only look at tokens at positions < `i`,
    preventing it from "cheating" by seeing future tokens. This is essential
    for autoregressive language modeling.
    
    Architecture:
    - Combined QKV projection (optimization: single linear layer)
    - Multi-head attention with scaled dot-product
    - Causal masking (upper triangular mask)
    - Output projection with residual dropout
    """
    
    def __init__(self, n_embd: int, n_head: int) -> None:
        """
        Initialize the causal self-attention layer.
        
        Args:
            n_embd: Embedding dimension (must be divisible by n_head)
            n_head: Number of attention heads
        """
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # Key, Query, Value projections in one single linear layer.
        # This is an optimization: computes Q, K, V together as (n_embd, 3 * n_embd)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        
        # Output projection: projects concatenated head outputs back to n_embd
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        # Regularization: dropout applied to attention scores and residual connection
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through the causal self-attention layer.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_embd)
        
        Returns:
            Output tensor of shape (batch_size, sequence_length, n_embd)
        """
        B, T, C = x.shape  # Batch size, Sequence length, Embedding dim

        # 1. Calculate Q, K, V from input
        # (B, T, C) -> (B, T, 3 * C)
        qkv = self.c_attn(x)
        
        # Split into Query, Key, Value
        # (B, T, 3 * C) -> 3 x (B, T, C)
        q, k, v = mx.split(qkv, 3, axis=2)

        # 2. Reshape for Multi-Head Attention
        # Split embedding dimension across heads
        # (B, T, C) -> (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        # 3. Scaled Dot-Product Attention
        # Compute attention scores: Q @ K^T / sqrt(head_dim)
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(self.head_dim))

        # 4. Apply Causal Mask
        # This is the most important part of a "decoder".
        # It creates a triangular mask for the upper-right half
        # of the (T, T) score matrix, setting future positions to -inf.
        # This prevents tokens from attending to future tokens.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        scores = scores + mask  # Broadcasting adds the mask to each head/batch

        # 5. Normalize and apply values
        # Softmax converts scores to probabilities, then apply dropout
        scores = mx.softmax(scores, axis=-1)
        scores = self.attn_dropout(scores)

        # Apply attention weights to values
        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        output = (scores @ v)

        # 6. Reshape and project
        # Concatenate heads and project back to embedding dimension
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        output = output.transpose(0, 2, 1, 3).reshape(B, T, C)
        output = self.resid_dropout(self.c_proj(output))
        
        return output

class MLP(nn.Module):
    """
    Feed-Forward Network (MLP) component of the transformer block.
    
    This provides the "thinking" time for the model. It introduces
    non-linearity and allows the model to process the information
    gathered by the attention layer. It is a simple two-layer
    perceptron with a 4x expansion in the middle, following the
    standard transformer architecture.
    
    Architecture:
    - Expand: n_embd -> 4 * n_embd
    - Activation: GELU
    - Project: 4 * n_embd -> n_embd
    - Dropout for regularization
    """
    
    def __init__(self, n_embd: int) -> None:
        """
        Initialize the MLP layer.
        
        Args:
            n_embd: Embedding dimension
        """
        super().__init__()
        # Standard MLP expands 4x, as per the "Attention Is All You Need" paper.
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)      # Expansion layer
        self.c_proj = nn.Linear(4 * n_embd, n_embd)    # Projection layer
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_embd)
        
        Returns:
            Output tensor of shape (batch_size, sequence_length, n_embd)
        """
        x = self.c_fc(x)           # Expand: (B, T, n_embd) -> (B, T, 4*n_embd)
        x = nn.gelu(x)             # GELU activation (Gaussian Error Linear Unit)
        x = self.c_proj(x)         # Project: (B, T, 4*n_embd) -> (B, T, n_embd)
        x = self.dropout(x)        # Apply dropout for regularization
        return x

class Block(nn.Module):
    """
    Transformer block: the fundamental building unit of the model.
    
    A Transformer is simply a stack of N identical blocks.
    Each block contains:
    - One causal self-attention layer
    - One feed-forward MLP layer
    - Residual connections (x + ...)
    - Layer normalization (pre-norm architecture)
    
    This uses a "Pre-LayerNorm" architecture, which is more stable
    than post-norm during training. The residual connections allow
    gradients to flow directly through the network, enabling the
    training of very deep models.
    """
    
    def __init__(self, n_embd: int, n_head: int) -> None:
        """
        Initialize the transformer block.
        
        Args:
            n_embd: Embedding dimension
            n_head: Number of attention heads
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)                    # Layer norm before attention
        self.attn = CausalSelfAttention(n_embd, n_head)     # Causal self-attention
        self.ln_2 = nn.LayerNorm(n_embd)                    # Layer norm before MLP
        self.mlp = MLP(n_embd)                              # Feed-forward network

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, n_embd)
        
        Returns:
            Output tensor of shape (batch_size, sequence_length, n_embd)
        """
        # The residual connection (x + ...) is crucial.
        # It allows gradients to flow directly through the network,
        # enabling the training of very deep models.
        
        # Pre-normalization: Norm -> Attention -> Add
        # Apply layer norm, then attention, then add residual
        x = x + self.attn(self.ln_1(x))
        
        # Pre-normalization: Norm -> MLP -> Add
        # Apply layer norm, then MLP, then add residual
        x = x + self.mlp(self.ln_2(x))
        
        return x

class GPT(nn.Module):
    """
    Full GPT-style decoder-only Transformer model.
    
    This class assembles all the components into the final
    decoder-only Transformer architecture. It implements a
    language model that predicts the next token in a sequence.
    
    Architecture:
    1. Token embeddings (vocab_size -> n_embd)
    2. Position embeddings (block_size -> n_embd)
    3. Stack of N transformer blocks
    4. Final layer normalization
    5. Language model head (n_embd -> vocab_size)
    
    The model uses weight tying between the token embedding
    and language model head, which saves parameters and
    improves performance.
    """
    
    def __init__(self) -> None:
        """
        Initialize the GPT model with all components.
        """
        super().__init__()
        
        # Token and Position embeddings
        # wte = "word token embedding" (maps token IDs to vectors)
        self.wte = nn.Embedding(vocab_size, n_embd)
        # wpe = "word position embedding" (maps position indices to vectors)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)

        # Stack of N Transformer Blocks
        # Each block contains attention + MLP with residual connections
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])

        # Final LayerNorm before the output head
        # This stabilizes the activations before the final projection
        self.ln_f = nn.LayerNorm(n_embd)

        # The Language Model Head
        # This is a linear layer that projects the final embedding
        # vector back to the size of the vocabulary.
        # The output values are "logits" (unnormalized probabilities).
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight Tying: Share weights between token embedding and LM head
        # This is a common practice that:
        # - Saves parameters (reduces model size)
        # - Improves performance (regularization effect)
        # - Asserts that token-to-vector is inverse of vector-to-token
        self.wte.weight = self.lm_head.weight

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass through the GPT model.
        
        Args:
            x: Input token IDs of shape (batch_size, sequence_length)
               Each element is a token ID in range [0, vocab_size)
        
        Returns:
            Logits tensor of shape (batch_size, sequence_length, vocab_size)
            These are unnormalized probabilities for next-token prediction.
        
        Raises:
            AssertionError: If sequence length exceeds block_size
        """
        B, T = x.shape
        assert T <= block_size, (
            f"Cannot forward sequence of length {T}, "
            f"maximum is {block_size}"
        )

        # 1. Get Token and Position Embeddings
        # Token embeddings: map token IDs to dense vectors
        # x (B, T) -> (B, T, C)
        tok_emb = self.wte(x)
        
        # Position embeddings: encode position information
        # pos (T,) -> pos_emb (T, C)
        pos = mx.arange(0, T, dtype=mx.int32)
        pos_emb = self.wpe(pos)

        # 2. Combine embeddings (with broadcasting)
        # Add token and position embeddings, then apply dropout
        # (B, T, C) + (T, C) -> (B, T, C) via broadcasting
        x = self.dropout(tok_emb + pos_emb)

        # 3. Pass through N Transformer blocks
        # Each block applies attention and MLP with residual connections
        x = self.blocks(x)

        # 4. Final normalization
        # Stabilize activations before final projection
        x = self.ln_f(x)

        # 5. Get Logits
        # Project final embeddings to vocabulary space
        # (B, T, C) -> (B, T, vocab_size)
        logits = self.lm_head(x)
        
        return logits