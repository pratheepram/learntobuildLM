# Learn to Build LM

An educational project for building a Small Language Model (SLM) from scratch using Apple's MLX framework. This decoder-only Transformer is designed to train on a Mac Mini (M4) with 16GB unified memory, making it accessible for learning without expensive cloud compute.

## What This Project Does

This repository implements a complete GPT-style language model that can:
- Learn to generate coherent text stories from the TinyStories dataset
- Train efficiently on Apple Silicon using MLX
- Demonstrate transformer architecture fundamentals (attention, MLP, layer normalization)
- Provide hands-on experience with language model training and inference

## Model Architecture

The model is a small decoder-only Transformer with:
- **8 transformer blocks** (layers)
- **384 embedding dimensions**
- **8 attention heads**
- **256 context length**
- **8,192 vocabulary size**
- **~16-17M parameters** (designed for 16GB Mac Mini)

Key components implemented from scratch:
- `CausalSelfAttention`: Multi-head self-attention with causal masking
- `MLP`: Feed-forward network with GELU activation
- `Block`: Transformer block with residual connections and layer normalization
- `GPT`: Complete language model with token/position embeddings and output head

## Prerequisites

- Python 3.12
- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB+ unified memory recommended

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd learntobuildLM
```

2. Create and activate virtual environment:
```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

### Step 1: Download the Dataset

Download the TinyStories dataset from Hugging Face:
```bash
python data/download.py
```

This creates:
- `data/TinyStories-train.txt` (~1.92GB)
- `data/TinyStories-valid.txt` (~0.12GB)

### Step 2: Train the Tokenizer

Train a BPE (Byte-Pair Encoding) tokenizer on the dataset:
```bash
python data/tokenizer.py
```

This creates `data/tinystories_tokenizer.json` with a vocabulary of 8,192 tokens.

### Step 3: Prepare Training Data

Tokenize the text files and convert them to binary format for efficient training:
```bash
python data/prepare.py
```

This creates:
- `data/train.bin` (tokenized training data)
- `data/val.bin` (tokenized validation data)

### Step 4: Train the Model

Start training the language model:
```bash
python train.py
```

The training script will:
- Load the tokenized data using memory-mapped files
- Train for 10 epochs with validation after each epoch
- Save checkpoints every 2 epochs
- Display training loss and validation perplexity

**Note:** Adjust `BATCH_SIZE` in `train.py` based on your available memory (default: 32 for 16GB).

### Step 5: Generate Text

Generate text from the trained model:
```bash
python generate.py --prompt "Once upon a time," --max_new_tokens 100
```

You can customize:
- `--prompt`: Starting text for generation
- `--max_new_tokens`: Number of tokens to generate

## Project Structure

```
learntobuildLM/
├── model.py              # GPT model implementation (attention, MLP, blocks)
├── train.py              # Training loop with MLX optimizers
├── generate.py           # Text generation script
├── data/
│   ├── download.py       # Download TinyStories dataset
│   ├── tokenizer.py      # Train BPE tokenizer
│   └── prepare.py        # Tokenize and prepare data
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Key Features

- **MLX-Native**: Built entirely with Apple's MLX framework (no PyTorch/TensorFlow)
- **Memory Efficient**: Uses `numpy.memmap` for large datasets to avoid OOM errors
- **Educational**: Clean, well-commented code following transformer architecture principles
- **Hardware Optimized**: Designed specifically for Apple Silicon with 16GB unified memory
- **Complete Pipeline**: From dataset download to text generation

## Model Configuration

The model hyperparameters are defined in `model.py`:
- `vocab_size = 8192`
- `n_embd = 384`
- `n_layer = 8`
- `n_head = 8`
- `block_size = 256`
- `dropout = 0.1`

These can be adjusted, but note that larger models will require more memory and training time.

## Training Configuration

Training settings in `train.py`:
- `BATCH_SIZE = 32` (adjust for your memory)
- `LEARNING_RATE = 1e-3`
- `NUM_EPOCHS = 10`
- `STEPS_PER_EPOCH = 1000`

## Notes

- The model uses lazy evaluation in MLX - always call `mx.eval()` after optimizer updates
- Checkpoints save both model weights and optimizer state (required for resuming training)
- The tokenizer uses BPE with special tokens: `[UNK]`, `<PAD>`, `<BOS>`, `<EOS>`
- Data is stored as `uint16` to save memory (vocab_size 8192 fits in 16 bits)

## Learning Resources

This project is inspired by:
- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)
- The [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) for small model training
- Apple's [MLX documentation](https://ml-explore.github.io/mlx/)

## License

[Add your license here]

