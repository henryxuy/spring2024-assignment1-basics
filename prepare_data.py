#!/usr/bin/env python3
"""
Utility script to prepare text data for training.

This script:
1. Loads a trained BPE tokenizer
2. Encodes text data to token IDs
3. Saves the result as a memory-mapped numpy array for efficient training
"""

import argparse
import numpy as np
from pathlib import Path
import json


def load_tokenizer(vocab_path: str, merges_path: str):
    """
    Load a BPE tokenizer from vocab and merges files.
    
    Args:
        vocab_path: Path to vocab.json file
        merges_path: Path to merges.txt file
    
    Returns:
        BPE tokenizer
    """
    try:
        from cs336_bpe.bpe_encoder import get_tokenizer
    except ImportError:
        raise ImportError(
            "Could not import BPE tokenizer. Make sure cs336_bpe is installed."
        )
    
    # Load vocab
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_dict = json.load(f)
    
    # Convert keys from strings to integers
    vocab = {int(k): bytes(v, 'utf-8') if isinstance(v, str) else v 
             for k, v in vocab_dict.items()}
    
    # Load merges
    merges = []
    with open(merges_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    token1 = bytes(parts[0], 'utf-8') if isinstance(parts[0], str) else parts[0]
                    token2 = bytes(parts[1], 'utf-8') if isinstance(parts[1], str) else parts[1]
                    merges.append((token1, token2))
    
    tokenizer = get_tokenizer(vocab, merges)
    return tokenizer


def encode_text_file(
    input_path: str,
    output_path: str,
    vocab_path: str,
    merges_path: str,
    chunk_size: int = 10000,
):
    """
    Encode a text file to token IDs and save as numpy array.
    
    Args:
        input_path: Path to input text file
        output_path: Path to output .npy file
        vocab_path: Path to vocab.json
        merges_path: Path to merges.txt
        chunk_size: Number of lines to process at once (for memory efficiency)
    """
    print(f"Loading tokenizer from {vocab_path} and {merges_path}...")
    tokenizer = load_tokenizer(vocab_path, merges_path)
    
    print(f"Encoding {input_path}...")
    all_token_ids = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = []
        line_count = 0
        
        for line in f:
            lines.append(line)
            line_count += 1
            
            if len(lines) >= chunk_size:
                # Process chunk
                text = ''.join(lines)
                token_ids = tokenizer.encode(text)
                all_token_ids.extend(token_ids)
                lines = []
                
                if line_count % 100000 == 0:
                    print(f"Processed {line_count:,} lines, {len(all_token_ids):,} tokens so far...")
        
        # Process remaining lines
        if lines:
            text = ''.join(lines)
            token_ids = tokenizer.encode(text)
            all_token_ids.extend(token_ids)
    
    print(f"Total tokens: {len(all_token_ids):,}")
    
    # Convert to numpy array and save
    print(f"Saving to {output_path}...")
    token_array = np.array(all_token_ids, dtype=np.int32)
    np.save(output_path, token_array)
    
    print(f"Done! Saved {len(token_array):,} tokens to {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / (1024**2):.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare text data for training by encoding to token IDs"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input text file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output .npy file",
    )
    parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="Path to vocab.json file",
    )
    parser.add_argument(
        "--merges",
        type=str,
        required=True,
        help="Path to merges.txt file",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="Number of lines to process at once (default: 10000)",
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    if not Path(args.vocab).exists():
        print(f"Error: Vocab file not found: {args.vocab}")
        return
    
    if not Path(args.merges).exists():
        print(f"Error: Merges file not found: {args.merges}")
        return
    
    # Ensure output directory exists
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Encode
    encode_text_file(
        input_path=args.input,
        output_path=args.output,
        vocab_path=args.vocab,
        merges_path=args.merges,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()

