# This project uses the roneneldan/TinyStories dataset from Hugging Face. You must create a simple data/download.py script using the datasets library to download it.   
# (Note: A simple load_dataset("roneneldan/TinyStories") and saving the 'train' and 'validation' splits to .txt files is all that is needed.)
# After running your download script, you should have:
# data/TinyStories-train.txt (~1.92GB)
# data/TinyStories-valid.txt (~0.12GB)

from datasets import load_dataset
import os
from loguru import logger

if __name__ == "__main__":
    logger.info("Downloading TinyStories dataset...")
    DATA_PATH = "data/"

    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    TRAIN_FILE = os.path.join(DATA_PATH, "TinyStories-train.txt")
    VALID_FILE = os.path.join(DATA_PATH, "TinyStories-valid.txt")

    dataset = load_dataset("roneneldan/TinyStories")

    # Write train split to file
    logger.info(f"Writing train split to {TRAIN_FILE}...")
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for example in dataset["train"]:
            f.write(example["text"] + "\n")

    # Write validation split to file
    logger.info(f"Writing validation split to {VALID_FILE}...")
    with open(VALID_FILE, "w", encoding="utf-8") as f:
        for example in dataset["validation"]:
            f.write(example["text"] + "\n")

    logger.info("Download complete.")

