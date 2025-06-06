#!/usr/bin/env python3
"""
Q/K/V Decomposition Visualizer for NanoGPT

Decomposes Q, K, and V transformation dimensions into vocabulary word contributions.
Shows what each dimension represents in terms of vocabulary words.
"""

import os
import sys
import torch
import numpy as np
import pickle
from dotenv import load_dotenv
import re
from PIL import Image, ImageDraw, ImageFont
from wordcloud import WordCloud

load_dotenv()


def load_checkpoint(checkpoint_path):
    """Load a NanoGPT checkpoint and return model info."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_args = checkpoint["model_args"]
    model_state = checkpoint["model"]

    token_embeddings = model_state["transformer.wte.weight"]  # [vocab_size, n_embd]
    pos_embeddings = model_state["transformer.wpe.weight"]  # [block_size, n_embd]

    print(f"Token embeddings: {token_embeddings.shape}")
    print(f"Positional embeddings: {pos_embeddings.shape}")

    return token_embeddings, pos_embeddings, model_args, model_state


def load_tokenizer():
    """Load the word tokenizer from meta file."""
    meta_path = os.environ.get("NANOGPT_META_PATH", "./meta_word.pkl")
    if os.path.exists(meta_path):
        try:
            meta = pickle.load(open(meta_path, "rb"))
            if "itos" in meta and "stoi" in meta:
                print("Using word tokenizer from meta_word.pkl")
                return meta["stoi"], meta["itos"]
        except Exception as e:
            print(f"Failed to load meta_word.pkl: {e}")
    return None, None


def load_qkv_matrices(model_state, model_args, layer_idx=0):
    """Load Q, K, V weight matrices from the first attention layer."""
    print(f"Loading Q/K/V matrices from layer {layer_idx}")

    n_embd = model_args["n_embd"]
    attention_key = f"transformer.h.{layer_idx}.attn.c_attn.weight"

    if attention_key not in model_state:
        print(f"Error: Attention layer {layer_idx} not found")
        return None, None, None

    c_attn_weight = model_state[attention_key]
    print(f"Found attention matrix: {c_attn_weight.shape}")

    # Handle transposed format if needed
    if c_attn_weight.shape[0] == 3 * n_embd and c_attn_weight.shape[1] == n_embd:
        print("Detected transposed QKV format [3*n_embd, n_embd]")
        c_attn_weight = c_attn_weight.T  # Now [n_embd, 3*n_embd]

    if c_attn_weight.shape[1] == 3 * n_embd:
        # Extract Q, K, V matrices
        W_q = c_attn_weight[:, :n_embd].numpy()
        W_k = c_attn_weight[:, n_embd : 2 * n_embd].numpy()
        W_v = c_attn_weight[:, 2 * n_embd :].numpy()
        print(f"Extracted Q: {W_q.shape}, K: {W_k.shape}, V: {W_v.shape}")
        return W_q, W_k, W_v
    else:
        print(f"Unexpected c_attn shape: {c_attn_weight.shape}")
        return None, None, None


def get_word_representation(word, stoi, token_embeddings, pos_embeddings, position=0):
    """Get the complete representation of a word (token + positional)."""
    if word not in stoi:
        print(f"Warning: '{word}' not in vocabulary")
        return None

    token_id = stoi[word]

    # Get token embedding
    token_emb = token_embeddings[token_id].numpy()  # [n_embd]

    # Get positional embedding
    if position < pos_embeddings.shape[0]:
        pos_emb = pos_embeddings[position].numpy()  # [n_embd]
    else:
        pos_emb = np.zeros_like(token_emb)

    # Combined representation
    combined_repr = token_emb + pos_emb  # [n_embd]

    print(f"Word '{word}' representation:")
    print(f"  Token embedding range: [{token_emb.min():.3f}, {token_emb.max():.3f}]")
    print(f"  Positional embedding range: [{pos_emb.min():.3f}, {pos_emb.max():.3f}]")
    print(f"  Combined range: [{combined_repr.min():.3f}, {combined_repr.max():.3f}]")

    return combined_repr


def compute_transformed_activations(word_repr, W_matrix):
    """Compute Q, K, or V activations for the word representation."""
    activations = word_repr @ W_matrix  # [n_embd] @ [n_embd, n_embd] = [n_embd]
    return activations


def compute_vocab_contributions_for_dim(token_embeddings, W_matrix, dim, itos):
    """Compute how much each vocabulary word contributes to a specific output dimension."""
    q_weights_for_dim = W_matrix[:, dim]

    vocab_contributions = {}

    for token_id in range(token_embeddings.shape[0]):
        if token_id < len(itos):
            word = itos[token_id]

            # Get token embedding for this word
            token_emb = token_embeddings[token_id].numpy()  # [n_embd]

            # Calculate contribution: sum over input dims of (embedding[input_dim] * q_weight[input_dim])
            contribution = np.sum(token_emb * q_weights_for_dim)

            vocab_contributions[word] = float(contribution)

    print(f"Computed contributions for {len(vocab_contributions)} vocabulary words")

    # Show top contributors
    sorted_contribs = sorted(
        vocab_contributions.items(), key=lambda x: abs(x[1]), reverse=True
    )
    print(f"Top 10 contributors (by absolute value):")
    for i, (word, contrib) in enumerate(sorted_contribs[:10]):
        print(f"  {i+1:2d}. {word:15s}: {contrib:8.4f}")

    return vocab_contributions


def create_dimension_wordcloud(
    vocab_contributions, activation, dim, output_dir, word, matrix_name
):
    """Create a wordcloud for a dimension showing vocabulary contributions."""

    print(f"\n=== DEBUG {matrix_name} Dimension {dim} ===")
    print(f"{matrix_name} activation: {activation:.6f}")
    print(f"Total vocab contributions computed: {len(vocab_contributions)}")

    # Show contribution statistics
    all_contribs = list(vocab_contributions.values())
    if all_contribs:
        print(f"Contribution range: [{min(all_contribs):.6f}, {max(all_contribs):.6f}]")
        print(f"Contribution mean: {np.mean(all_contribs):.6f}")
        print(
            f"Non-zero contributions: {sum(1 for c in all_contribs if abs(c) > 1e-10)}"
        )

    # Start with a very low threshold, we can always increase it later
    min_contribution_threshold = 1e-6  # Very permissive threshold
    filtered_contributions = {
        word: contrib
        for word, contrib in vocab_contributions.items()
        if abs(contrib) >= min_contribution_threshold
    }

    print(
        f"After filtering (threshold {min_contribution_threshold}): {len(filtered_contributions)} words"
    )

    # If still too few words, take the top contributors regardless of threshold
    if len(filtered_contributions) < 20:
        print(
            f"Too few words ({len(filtered_contributions)}), taking top 50 contributors regardless of threshold"
        )
        sorted_contribs = sorted(
            vocab_contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )
        filtered_contributions = dict(sorted_contribs[:50])
        print(f"Now have {len(filtered_contributions)} words")

    if not filtered_contributions:
        print(f"ERROR: No contributions found for {matrix_name} dimension {dim}!")
        # Create a placeholder image
        placeholder_img = Image.new("RGB", (800, 640), "lightgray")
        draw = ImageDraw.Draw(placeholder_img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        text = (
            f"No contributions\n{matrix_name} Dim {dim}\nActivation: {activation:.6f}"
        )
        draw.text((50, 300), text, fill="black", font=font)

        output_path = os.path.join(
            output_dir, f"{matrix_name.lower()}_dim_{dim:02d}_{word}.png"
        )
        placeholder_img.save(output_path, "PNG")
        return output_path

    # Combine all contributions into a single dictionary for word cloud generation
    word_frequencies = {
        word: abs(contrib) for word, contrib in filtered_contributions.items()
    }

    # Define a color function to distinguish positive (black) and negative (red)
    def color_func(word, **kwargs):
        contrib = filtered_contributions.get(word, 0)
        return "black" if contrib >= 0 else "red"

    # Create a single wordcloud for the dimension
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=len(word_frequencies),
        color_func=color_func,
        relative_scaling=0.5,
        min_font_size=6,
    ).generate_from_frequencies(word_frequencies)

    wordcloud_img = wordcloud.to_image()

    # Improved opacity calculation - ensure we always have reasonable visibility
    if abs(activation) < 1e-6:
        opacity = 0.3  # Minimum opacity for very weak activations
        print(f"Very weak activation, using minimum opacity: {opacity}")
    else:
        # Use a more reasonable scale - find the max activation across all dimensions for normalization
        # For now, just use a simple absolute scaling
        opacity = min(1.0, abs(activation) * 10)  # Scale up weak activations
        opacity = max(0.3, opacity)  # Ensure minimum visibility
        print(f"Computed opacity: {opacity}")

    # Apply opacity to the wordcloud image
    if wordcloud_img.mode != "RGBA":
        wordcloud_img = wordcloud_img.convert("RGBA")

    alpha = wordcloud_img.split()[-1]
    alpha = alpha.point(lambda p: int(p * opacity))
    wordcloud_img.putalpha(alpha)

    # Create final image with label and stats
    wordcloud_width, wordcloud_height = wordcloud_img.size
    label_height = 100
    final_width = wordcloud_width
    final_height = wordcloud_height + label_height

    final_img = Image.new("RGB", (final_width, final_height), "white")

    # Paste wordcloud (with opacity) onto the final image
    final_img.paste(wordcloud_img, (0, 0), wordcloud_img)

    # Add labels
    draw = ImageDraw.Draw(final_img)
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Title
    title = f"{matrix_name} Dimension {dim} - Word: {word}"
    draw.text((10, wordcloud_height + 10), title, fill="black", font=font_large)

    max_abs_contrib = (
        max(abs(c) for c in filtered_contributions.values())
        if filtered_contributions
        else 0
    )

    # Stats
    stats_text = f"Activation: {activation:.4f} | Opacity: {opacity:.2f} | Max contrib: {max_abs_contrib:.4f}"
    draw.text((10, wordcloud_height + 40), stats_text, fill="gray", font=font_small)

    positive_contributions = {w: c for w, c in filtered_contributions.items() if c > 0}
    negative_contributions = {w: c for w, c in filtered_contributions.items() if c < 0}

    # Contribution counts
    contrib_text = f"Pos: {len(positive_contributions)} words | Neg: {len(negative_contributions)} words | Total: {len(filtered_contributions)}/{len(vocab_contributions)}"
    draw.text((10, wordcloud_height + 60), contrib_text, fill="gray", font=font_small)

    # Color indicator for overall activation direction
    color = "green" if activation >= 0 else "red"
    activation_text = (
        f"Overall: {'Positive' if activation >= 0 else 'Negative'} Activation"
    )
    draw.text((10, wordcloud_height + 80), activation_text, fill=color, font=font_small)

    # Save
    output_path = os.path.join(
        output_dir, f"{matrix_name.lower()}_dim_{dim:02d}_{word}.png"
    )
    final_img.save(output_path, "PNG")

    print(f"Saved {matrix_name} dimension {dim} wordcloud: {output_path}")
    print(f"=== END DEBUG {matrix_name} Dimension {dim} ===\n")
    return output_path


def create_overview_grid(output_paths, word, n_embd, output_dir, matrix_name):
    """Create a grid overview of all dimension wordclouds."""

    if not output_paths:
        return None

    # Calculate grid dimensions
    grid_cols = int(np.ceil(np.sqrt(n_embd)))
    grid_rows = int(np.ceil(n_embd / grid_cols))

    # Load first image to get size
    first_img = Image.open(output_paths[0])
    img_width, img_height = first_img.size

    # Scale down for overview
    scale_factor = 0.3
    thumb_width = int(img_width * scale_factor)
    thumb_height = int(img_height * scale_factor)

    # Calculate canvas size
    spacing = 10
    canvas_width = grid_cols * thumb_width + (grid_cols - 1) * spacing + 2 * spacing
    canvas_height = (
        grid_rows * thumb_height + (grid_rows - 1) * spacing + 2 * spacing + 60
    )

    # Create canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

    # Add title
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        title_font = ImageFont.load_default()

    title = f"{matrix_name} Decomposition Overview - Word: {word}"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (canvas_width - title_width) // 2
    draw.text((title_x, 10), title, fill="black", font=title_font)

    # Place thumbnails
    for i, path in enumerate(output_paths):
        if i >= n_embd:
            break

        row = i // grid_cols
        col = i % grid_cols

        x = spacing + col * (thumb_width + spacing)
        y = 60 + row * (thumb_height + spacing)

        # Load and resize image
        img = Image.open(path)
        thumb = img.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)

        canvas.paste(thumb, (x, y))

        # Add dimension label
        dim_label = f"D{i}"
        label_x = x + thumb_width // 2 - 10
        label_y = y + thumb_height - 20
        draw.text((label_x, label_y), dim_label, fill="white", font=title_font)

    # Save overview
    overview_path = os.path.join(
        output_dir, f"{matrix_name.lower()}_decomposition_overview_{word}.png"
    )
    canvas.save(overview_path, "PNG")

    print(f"Saved overview: {overview_path}")
    return overview_path


def main():
    # Get configuration
    checkpoint_path = os.environ.get("MODEL")
    target_word = os.environ.get("TARGET_WORD", "knock")
    position = int(os.environ.get("WORD_POSITION", "0"))

    if not checkpoint_path:
        print("Error: MODEL environment variable not set")
        sys.exit(1)

    print(f"Analyzing word: '{target_word}' at position {position}")

    # Load model and tokenizer
    token_embeddings, pos_embeddings, model_args, model_state = load_checkpoint(
        checkpoint_path
    )
    stoi, itos = load_tokenizer()

    if stoi is None:
        print("Failed to load tokenizer")
        sys.exit(1)

    # Load Q, K, V matrices
    W_q, W_k, W_v = load_qkv_matrices(model_state, model_args, layer_idx=0)
    if W_q is None:
        print("Failed to load Q/K/V matrices")
        sys.exit(1)

    # Get word representation (token + positional)
    word_repr = get_word_representation(
        target_word, stoi, token_embeddings, pos_embeddings, position
    )
    if word_repr is None:
        sys.exit(1)

    # Create master output directory
    model_dir = os.path.basename(os.path.dirname(checkpoint_path))
    output_dir_base = os.path.join(
        "visualizations", model_dir, f"qkv_decomposition_{target_word}"
    )
    os.makedirs(output_dir_base, exist_ok=True)

    print(f"\nGenerating Q/K/V decomposition visualizations...")
    print(f"Output directory: {output_dir_base}")

    all_overviews = {}

    matrices = {"Q": W_q, "K": W_k, "V": W_v}

    for matrix_name, W_matrix in matrices.items():
        print(f"\n{'='*20} PROCESSING {matrix_name} MATRIX {'='*20}")

        # Compute activations for this transformation
        activations = compute_transformed_activations(word_repr, W_matrix)

        # Create subdirectory for this matrix type
        output_dir = os.path.join(output_dir_base, matrix_name.lower())
        os.makedirs(output_dir, exist_ok=True)

        # Generate wordcloud for each dimension
        n_embd = model_args["n_embd"]
        output_paths = []

        for dim in range(n_embd):
            print(f"\nProcessing {matrix_name} dimension {dim}/{n_embd-1}")

            # Compute vocabulary contributions for this dimension
            vocab_contributions = compute_vocab_contributions_for_dim(
                token_embeddings, W_matrix, dim, itos
            )

            # Create wordcloud with opacity based on activation
            output_path = create_dimension_wordcloud(
                vocab_contributions,
                activations[dim],
                dim,
                output_dir,
                target_word,
                matrix_name,
            )

            if output_path:
                output_paths.append(output_path)

        # Create overview grid for this matrix
        overview_path = create_overview_grid(
            output_paths, target_word, n_embd, output_dir, matrix_name
        )
        if overview_path:
            all_overviews[matrix_name] = os.path.relpath(overview_path, output_dir_base)

    print(f"\nDone! Q/K/V decomposition visualizations saved to: {output_dir_base}/")
    print(f"Key outputs:")
    for matrix_name, overview_path in all_overviews.items():
        print(f"  - {matrix_name} overview grid: {overview_path}")


if __name__ == "__main__":
    main()
