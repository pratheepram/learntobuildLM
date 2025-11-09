"""
Text generation script for the trained TinySLM model.

This script loads a trained model checkpoint and generates text
autoregressively using the model's predictions.
"""

import mlx.core as mx
from model import GPT, block_size
from tokenizers import Tokenizer
import argparse
from loguru import logger

# --- Configuration ---
MODEL_FILE = "./tinyslm_model_epoch_final.npz" # Path to your trained model
TOKENIZER_FILE = "./data/tinystories_tokenizer.json"
# ---

def generate(model, tokenizer, prompt_text, max_new_tokens=50):
    """
    Performs autoregressive text generation.
    """
    model.eval() # Set model to evaluation mode
    
    # Encode the prompt text to token IDs
    try:
        bos_id = tokenizer.token_to_id("<BOS>")
    except:
        try:
            bos_id = tokenizer.token_to_id("<s>")
        except:
            bos_id = None
        
    prompt_ids = tokenizer.encode(prompt_text).ids
    
    # Prepend the BOS (Beginning of Sequence) token if it exists
    if bos_id is not None:
        prompt_ids = [bos_id] + prompt_ids
        
    prompt = mx.array([prompt_ids])
    
    # Track the last decoded text to only print new characters
    # This ensures proper spacing between tokens
    last_decoded_text = prompt_text
    print(prompt_text, end="", flush=True)

    # The autoregressive loop
    for _ in range(max_new_tokens):
        # 1. Get logits
        # We only need to pass the last `block_size` tokens as context
        logits = model(prompt[:, -block_size:])
        
        # 2. Get the logits for the very last token
        # (B, T, C) -> (B, C)
        logits = logits[:, -1, :]
        
        # 3. Sample the next token
        # `mx.random.categorical` samples from the distribution.
        # This adds randomness. For deterministic output, use mx.argmax(logits, axis=-1)
        next_tok = mx.random.categorical(logits)
        
        # 4. Append the new token to the prompt
        # This is the "autoregressive" step.
        prompt = mx.concatenate([prompt, next_tok.reshape(1, 1)], axis=1)
        
        # 5. Check for EOS token and stop if found
        try:
            eos_id = tokenizer.token_to_id("<EOS>")
            if next_tok.item() == eos_id:
                break
        except:
            try:
                eos_id = tokenizer.token_to_id("</s>")
                if next_tok.item() == eos_id:
                    break
            except:
                pass # No EOS token
        
        # 6. Decode the entire sequence so far to get proper spacing
        # Decode all tokens (excluding BOS if present) to preserve spacing
        all_token_ids = prompt[0].tolist()
        if bos_id is not None and len(all_token_ids) > 0 and all_token_ids[0] == bos_id:
            # Remove BOS token for decoding
            tokens_to_decode = all_token_ids[1:]
        else:
            tokens_to_decode = all_token_ids
        
        # Decode the full sequence to get proper spacing
        current_decoded_text = tokenizer.decode(tokens_to_decode)
        
        # Only print the new characters (the difference)
        new_text = current_decoded_text[len(last_decoded_text):]
        if new_text:
            print(new_text, end="", flush=True)
            last_decoded_text = current_decoded_text

    print() # Newline at the end
    model.train() # Set model back to training mode

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from a TinyMLX model")
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Once upon a time,", 
        help="The prompt to start generation from"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=50, 
        help="Number of new tokens to generate"
    )
    args = parser.parse_args()

    logger.info("Loading model and tokenizer...")
    # 1. Load the trained model
    model = GPT()
    model.load_weights(MODEL_FILE)
    logger.info(f"Model loaded from {MODEL_FILE}")
    
    # 2. Load the tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    logger.info(f"Tokenizer loaded from {TOKENIZER_FILE}")
    
    logger.info(f"Generating {args.max_new_tokens} tokens from prompt: '{args.prompt}'")
    logger.info("=" * 60)
    generate(model, tokenizer, args.prompt, args.max_new_tokens)
    logger.info("=" * 60)
    logger.info("Generation complete!")