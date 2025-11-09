import os
import numpy as np
from tokenizers import Tokenizer
from loguru import logger

# --- Configuration ---
TOKENIZER_PATH = "./data/tinystories_tokenizer.json"
INPUT_FILES = {
    "train": "./data/TinyStories-train.txt",
    "val": "./data/TinyStories-valid.txt"
}
OUTPUT_FILES = {
    "train": "./data/train.bin",
    "val": "./data/val.bin"
}
# We use uint16 since our vocab_size (8192) fits comfortably.
# This saves 50% of memory compared to the default int32/int64.
DATA_TYPE = np.uint16
CHUNK_SIZE = 1024 * 1024  # Read 1MB of text at a time

LOG_INTERVAL = 10_000_000  # Log progress every N tokens
# ---

def tokenize_and_pack(input_file_path, output_file_path, tokenizer):
    """
    Reads a large text file, tokenizes it in chunks, and writes the
    token IDs to a binary memmap file.
    Uses a two-pass approach: first count tokens, then write to memmap.
    """
    if not os.path.exists(input_file_path):
        logger.error(f"Input file not found: {input_file_path}")
        return

    logger.info(f"Processing {input_file_path} -> {output_file_path}...")

    # Pass 1: Count total tokens
    logger.info("Pass 1: Counting tokens...")
    token_count = 0
    last_log = 0
    with open(input_file_path, 'r', encoding='utf-8') as f:
        while True:
            text_chunk = f.read(CHUNK_SIZE)
            if not text_chunk:
                break
            tokens = tokenizer.encode(text_chunk).ids
            token_count += len(tokens)
            if token_count - last_log >= LOG_INTERVAL:
                logger.info(f"Counted {token_count:,} tokens so far...")
                last_log = token_count
    logger.info(f"Total tokens: {token_count:,}")

    # Pass 2: Create memmap with exact size and write tokens
    logger.info("Pass 2: Writing tokens to memmap...")
    memmap_file = np.memmap(output_file_path, dtype=DATA_TYPE, mode='w+', shape=(token_count,))
    
    token_idx = 0
    last_log = 0
    with open(input_file_path, 'r', encoding='utf-8') as f:
        while True:
            text_chunk = f.read(CHUNK_SIZE)
            if not text_chunk:
                break
            
            tokens = tokenizer.encode(text_chunk).ids
            num_tokens = len(tokens)
            if num_tokens == 0:
                continue
            
            # Write the tokens to the memmap
            memmap_file[token_idx : token_idx + num_tokens] = tokens
            token_idx += num_tokens
            
            if token_idx - last_log >= LOG_INTERVAL:
                logger.info(f"Written {token_idx:,} / {token_count:,} tokens ({100 * token_idx // token_count}%)...")
                last_log = token_idx
    
    # Flush changes to disk
    memmap_file.flush()
    del memmap_file  # Close the memmap
    logger.info(f"Finished. Total tokens: {token_count:,}")


if __name__ == "__main__":
    logger.info(f"Loading tokenizer from {TOKENIZER_PATH}...")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    # Add special tokens if they are not already part of the vocab
    # This is a common gotcha.
    tokenizer.add_special_tokens(["<PAD>", "<UNK>", "<BOS>", "<EOS>"])

    # Process both train and validation files
    for split, path in INPUT_FILES.items():
        output_path = OUTPUT_FILES[split]
        tokenize_and_pack(path, output_path, tokenizer)
    
    logger.info("All files processed.")