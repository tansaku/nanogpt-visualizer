#!/usr/bin/env python3
"""
Attention Pattern Visualizer for NanoGPT

Generates a heatmap of attention weights for a given probe sentence,
showing which words attend to which other words in the first attention layer.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
from dotenv import load_dotenv

load_dotenv()


def softmax(x):
    """Compute softmax along the last axis."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def load_checkpoint(checkpoint_path):
    """Load a NanoGPT checkpoint and return key model components."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model_args = checkpoint["model_args"]
    model_state = checkpoint["model"]

    token_embeddings = model_state["transformer.wte.weight"]
    pos_embeddings = model_state["transformer.wpe.weight"]

    print(f"Token embeddings: {token_embeddings.shape}")
    print(f"Positional embeddings: {pos_embeddings.shape}")

    return token_embeddings, pos_embeddings, model_args, model_state


def load_tokenizer():
    """Load the word tokenizer from meta file."""
    meta_path = os.environ.get("NANOGPT_META_PATH", "./meta_word.pkl")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            if "itos" in meta and "stoi" in meta:
                print("Using word tokenizer from meta_word.pkl")
                return meta["stoi"], meta["itos"]
        except Exception as e:
            print(f"Failed to load meta_word.pkl: {e}")
    return None, None


def load_q_k_matrices(model_state, model_args, layer_idx=0):
    """Load Q and K weight matrices from a specific attention layer."""
    print(f"Loading Q/K matrices from layer {layer_idx}")

    n_embd = model_args["n_embd"]
    attention_key = f"transformer.h.{layer_idx}.attn.c_attn.weight"

    if attention_key not in model_state:
        print(f"Error: Attention layer {layer_idx} not found")
        return None, None

    c_attn_weight = model_state[attention_key]
    print(f"Found combined attention matrix: {c_attn_weight.shape}")

    # Handle transposed format if needed
    if c_attn_weight.shape[0] == 3 * n_embd and c_attn_weight.shape[1] == n_embd:
        c_attn_weight = c_attn_weight.T  # Now [n_embd, 3*n_embd]

    if c_attn_weight.shape[1] == 3 * n_embd:
        W_q = c_attn_weight[:, :n_embd].numpy()
        W_k = c_attn_weight[:, n_embd : 2 * n_embd].numpy()
        print(f"Extracted Q matrix: {W_q.shape}, K matrix: {W_k.shape}")
        return W_q, W_k
    else:
        print(f"Unexpected c_attn shape: {c_attn_weight.shape}")
        return None, None


def tokenize_sentence(sentence, stoi):
    """Tokenize a sentence using the word tokenizer."""
    words = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
    token_ids = [stoi[word] for word in words if word in stoi]
    # Filter words to only those in vocab
    words_in_vocab = [word for word in words if word in stoi]
    return words_in_vocab, token_ids


def get_combined_representations(token_ids, token_embeddings, pos_embeddings):
    """Get combined token and positional embeddings for a sequence."""
    combined_reprs = []
    for i, token_id in enumerate(token_ids):
        token_emb = token_embeddings[token_id].numpy()
        pos_emb = pos_embeddings[i].numpy()
        combined_reprs.append(token_emb + pos_emb)
    return np.array(combined_reprs)


def create_attention_heatmap(attention_weights, words, output_dir, model_name):
    """Create and save a heatmap of attention weights."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_weights,
        xticklabels=words,
        yticklabels=words,
        cmap="viridis",
        annot=True,
        fmt=".4f",
        linewidths=0.5,
    )
    plt.title(f"Attention Pattern - {model_name}", fontsize=16)
    plt.xlabel("Key (attended to)", fontsize=12)
    plt.ylabel("Query (attending from)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "attention_pattern.png")
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved attention heatmap to {output_path}")
    return output_path


def generate_html_page(output_dir, model_name, probe_sentence, image_path):
    """Generate a simple HTML page to display the heatmap."""
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Attention Pattern - {model_name}</title>
    <style>
        body {{ font-family: sans-serif; margin: 2em; background: #f9f9f9; }}
        .container {{ max-width: 1000px; margin: auto; background: white; padding: 2em; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        p {{ color: #555; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Attention Pattern</h1>
        <p><strong>Model:</strong> {model_name}</p>
        <p><strong>Probe Sentence:</strong> "{probe_sentence}"</p>
        <p>This heatmap shows the attention weights from the first attention layer.
        Each row represents a "query" word, and each column represents a "key" word.
        The value in a cell indicates how much attention the query word pays to the key word.</p>
        <img src="{os.path.basename(image_path)}" alt="Attention Pattern Heatmap">
    </div>
</body>
</html>"""
    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"Generated HTML page: {html_path}")


def main():
    # Configuration
    checkpoint_path = os.environ.get("MODEL")
    probe_sentence = os.environ.get("PROBE_SENTENCE", "knock knock whos there bob")
    layer_idx = 0

    if not checkpoint_path:
        print("Error: MODEL environment variable not set")
        sys.exit(1)

    # Load model, tokenizer
    token_embeddings, pos_embeddings, model_args, model_state = load_checkpoint(
        checkpoint_path
    )
    stoi, itos = load_tokenizer()
    if not stoi:
        sys.exit(1)

    # Prepare sentence data
    words, token_ids = tokenize_sentence(probe_sentence, stoi)
    combined_reprs = get_combined_representations(
        token_ids, token_embeddings, pos_embeddings
    )

    # Load Q/K matrices
    W_q, W_k = load_q_k_matrices(model_state, model_args, layer_idx)
    if W_q is None:
        sys.exit(1)

    # Compute attention
    n_head = model_args.get("n_head", 1)  # Default to 1 if not specified
    d_k = model_args["n_embd"] // n_head

    queries = combined_reprs @ W_q
    keys = combined_reprs @ W_k
    scores = (queries @ keys.T) / np.sqrt(d_k)
    attention_weights = softmax(scores)

    # Create visualization
    model_dir = os.path.basename(os.path.dirname(checkpoint_path))
    output_dir = os.path.join("visualizations", model_dir, "attention_pattern")
    os.makedirs(output_dir, exist_ok=True)

    image_path = create_attention_heatmap(
        attention_weights, words, output_dir, model_dir
    )
    generate_html_page(output_dir, model_dir, probe_sentence, image_path)

    print("\nDone! View the attention pattern at:")
    print(f"{output_dir}/index.html")


if __name__ == "__main__":
    main()
