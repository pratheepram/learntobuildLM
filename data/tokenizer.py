import glob
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from loguru import logger

if __name__ == "__main__":
    # 1. Find the raw text files to train on.
    # Assumes the TinyStories-train.txt and TinyStories-valid.txt
    # have been downloaded into a './data/' directory.
    files = glob.glob("./data/*.txt")
    logger.info(f"Found {len(files)} text file(s) to train on: {files}")

    # 2. Instantiate a new, untrained Tokenizer
    # We initialize a new Tokenizer with a BPE model.
    # [UNK] is the token used for any out-of-vocabulary words.
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # 3. Set Normalizer and Pre-tokenizer
    # Normalization (e.g., lowercase, strip accents)
    # We keep this simple for the TinyStories dataset.
    # Sequence() requires a list of normalizers - empty list means no normalization
    tokenizer.normalizer = normalizers.Sequence([])

    # Pre-tokenization splits the text into "words" first.
    # The BPE algorithm will then learn merges *within* these words.
    # Whitespace() is the simplest: it just splits on spaces.
    tokenizer.pre_tokenizer = Whitespace()

    # 4. Instantiate the Trainer
    # We define our target vocabulary size and special tokens.
    # A small vocab size is suitable for this simple dataset.
    vocab_size = 8192
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "<PAD>", "<BOS>", "<EOS>"]
    )

    # 5. Train the Tokenizer
    # The train method takes the list of files and the trainer.
    # This will take a minute or two.
    logger.info(f"Training tokenizer with vocab size {vocab_size} on {files}...")
    tokenizer.train(files, trainer)
    logger.info("Training complete.")

    # 6. Save the Tokenizer
    # This saves the trained vocabulary and merge rules to a JSON file.
    # This file IS our tokenizer.
    output_path = "./data/tinystories_tokenizer.json"
    tokenizer.save(output_path)
    logger.info(f"Tokenizer saved to {output_path}")