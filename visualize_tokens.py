#!/usr/bin/env python3
"""
Minimal NanoGPT Token Visualizer with Training Data Validation

Loads a NanoGPT checkpoint and creates word cloud visualizations of token embeddings.
Can validate which tokens are from training data vs. inherited from GPT-2.
"""

import os
from dotenv import load_dotenv

load_dotenv()
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import tiktoken


def load_checkpoint(checkpoint_path):
    """Load a NanoGPT checkpoint and return model info."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Get model configuration
    model_args = checkpoint["model_args"]
    vocab_size = model_args["vocab_size"]
    n_embd = model_args["n_embd"]

    print(f"Model vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {n_embd}")

    # Extract token embeddings from model state
    model_state = checkpoint["model"]
    embeddings = model_state["transformer.wte.weight"]  # [vocab_size, n_embd]

    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings, vocab_size, model_args


def create_vocabulary_mapping(vocab_size, nanogpt_path=None):
    """Create a mapping from model indices to token strings."""
    print(f"Creating vocabulary mapping for {vocab_size} tokens...")

    meta_path = os.environ.get("NANOGPT_META_PATH", "./meta_word.pkl")
    if os.path.exists(meta_path):
        try:
            import pickle

            meta = pickle.load(open(meta_path, "rb"))
            if "itos" in meta:
                print("Using word tokenizer vocabulary from meta_word.pkl")
                vocab = (
                    meta["itos"]
                    if isinstance(meta["itos"], dict)
                    else {i: str(token) for i, token in enumerate(meta["itos"])}
                )
                print(f"Mapped {len(vocab)} word tokens")
                print("Sample token mappings:")
                for i in range(min(10, len(vocab))):
                    print(f"  {i} -> {vocab[i]}")
                return vocab
        except Exception as e:
            print(f"Failed to load meta_word.pkl: {e}")

    if vocab_size >= 50000:
        print("Using standard GPT-2 vocabulary")
        enc = tiktoken.get_encoding("gpt2")
        gpt2_vocab_size = enc.n_vocab
        print(f"GPT-2 vocabulary size: {gpt2_vocab_size}")

        vocab = {}
        for i in range(vocab_size):
            if i < gpt2_vocab_size:
                try:
                    vocab[i] = enc.decode([i])
                except:
                    vocab[i] = f"<decode_error_{i}>"
            else:
                vocab[i] = f"<extra_token_{i}>"

        print(
            f"Mapped {len(vocab)} tokens ({gpt2_vocab_size} standard + {vocab_size - gpt2_vocab_size} extra)"
        )
        return vocab

    print("Using generic token IDs")
    return {i: f"token_{i}" for i in range(vocab_size)}


def analyze_training_data(training_data_path, vocab):
    print(f"\n📚 Analyzing training data: {training_data_path}")

    if not os.path.exists(training_data_path):
        print(f"❌ Training data not found: {training_data_path}")
        return set()

    with open(training_data_path, "r", encoding="utf-8") as f:
        training_text = f.read()

    print(f"📝 Training text length: {len(training_text)} characters")
    print(f"📝 First 200 chars: {repr(training_text[:200])}")

    meta_path = os.environ.get("NANOGPT_META_PATH", "./meta_word.pkl")
    if os.path.exists(meta_path):
        try:
            import pickle

            meta = pickle.load(open(meta_path, "rb"))
            if "tokenizer" in meta:
                print("Using word tokenizer from meta_word.pkl")
                tokenizer = meta["tokenizer"]
                training_tokens = tokenizer.encode(training_text)
                training_vocab_indices = set(training_tokens)
                print(f"✅ Found {len(training_vocab_indices)} unique training tokens")
                return training_vocab_indices
        except Exception as e:
            print(f"Failed to use word tokenizer: {e}")

    try:
        enc = tiktoken.get_encoding("gpt2")
        training_token_ids = enc.encode(
            training_text, allowed_special={"<|endoftext|>"}
        )
        training_vocab_indices = set(training_token_ids)
        print(f"✅ Found {len(training_vocab_indices)} unique training tokens")
        return training_vocab_indices

    except Exception as e:
        print(f"❌ Failed to analyze training data: {e}")
        return set()


def create_word_cloud(
    embeddings, vocab, dimension, output_dir, training_tokens=None, top_n=30
):
    print(f"Creating word cloud for dimension {dimension}")

    dim_values = embeddings[:, dimension].numpy()
    pos_indices = np.argsort(-dim_values)[:top_n]
    pos_scores = dim_values[pos_indices]
    neg_indices = np.argsort(dim_values)[:top_n]
    neg_scores = -dim_values[neg_indices]

    word_frequencies = {}
    word_colors = {}

    for idx, score in zip(pos_indices, pos_scores):
        if score > 0:
            token_str = str(vocab.get(idx, f"token_{idx}"))
            word_frequencies[token_str] = float(score)
            word_colors[token_str] = "#000000"

    for idx, score in zip(neg_indices, neg_scores):
        if dim_values[idx] < 0:
            token_str = str(vocab.get(idx, f"token_{idx}"))
            if token_str not in word_frequencies:
                word_frequencies[token_str] = float(score)
                word_colors[token_str] = "#CC0000"

    if not word_frequencies:
        print(f"No valid tokens for dimension {dimension}")
        return

    print(
        f"Dimension {dimension}: {len([c for c in word_colors.values() if c == '#000000'])} positive, {len([c for c in word_colors.values() if c == '#CC0000'])} negative"
    )

    def color_func(word, **kwargs):
        return word_colors.get(word, "#000000")

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=len(word_frequencies),
        color_func=color_func,
        prefer_horizontal=1.0,
    ).generate_from_frequencies(word_frequencies)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Dimension {dimension} - Black: Positive, Red: Negative", fontsize=14)
    output_path = os.path.join(output_dir, f"dimension_{dimension}.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python visualize_tokens.py <checkpoint.pt> [training_data.txt]")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    training_data_path = sys.argv[2] if len(sys.argv) == 3 else None

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    embeddings, vocab_size, model_args = load_checkpoint(checkpoint_path)
    nanogpt_path = os.environ.get("NANOGPT_PATH")
    vocab = create_vocabulary_mapping(vocab_size, nanogpt_path)

    training_tokens = None
    if training_data_path:
        training_tokens = analyze_training_data(training_data_path, vocab)
        unused_tokens = set(vocab.keys()) - training_tokens
        if unused_tokens:
            print("\n🚨 WARNING: Some vocab tokens do not appear in training data")
            for tid in sorted(list(unused_tokens)[:10]):
                print(f"  {tid}: {vocab.get(tid, f'token_{tid}')}")
            print(f"  ... and {len(unused_tokens) - 10} more (if any)")

    output_dir = "token_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    n_embd = embeddings.shape[1]
    dimensions_to_visualize = min(5, n_embd)

    print(f"Creating visualizations for {dimensions_to_visualize} dimensions...")

    for dim in range(dimensions_to_visualize):
        create_word_cloud(embeddings, vocab, dim, output_dir)

    print(f"\nDone! Visualizations saved to: {output_dir}/")
    print(f"Created {dimensions_to_visualize} word cloud images")


if __name__ == "__main__":
    main()
