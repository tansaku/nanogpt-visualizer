#!/usr/bin/env python3
"""
Attention Flow Wordmap Visualizer

Visualizes the step-by-step transformation of word representations
through the attention mechanism (Q, K, V, and final output).
"""

import os
import sys
import torch
import numpy as np
import pickle
import re
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from wordcloud import WordCloud
import json
import matplotlib.pyplot as plt
import io
import html
import umap
from sklearn.decomposition import PCA

load_dotenv()

# --- Model and Tokenizer Loading ---


def load_checkpoint(checkpoint_path):
    """Loads the NanoGPT checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint


def load_tokenizer():
    """Loads the word tokenizer."""
    meta_path = os.environ.get("NANOGPT_META_PATH", "./meta_word.pkl")
    if not os.path.exists(meta_path):
        print(f"Tokenizer not found at {meta_path}")
        return None, None
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta.get("stoi"), meta.get("itos")


def tokenize_sentence(sentence, stoi):
    """Tokenizes a sentence into words and IDs."""
    words = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
    words_in_vocab = [w for w in words if w in stoi]
    token_ids = [stoi[w] for w in words_in_vocab]
    return words_in_vocab, token_ids


# --- Representation and Transformation Calculation ---


def gelu(x):
    """GELU activation function."""
    return (
        0.5
        * x
        * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3.0))))
    )


def layernorm(x, gamma, beta, eps=1e-5):
    """Layer normalization."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_normalized = (x - mean) / np.sqrt(var + eps)
    return gamma * x_normalized + beta


def get_token_only_representations(token_ids, wte):
    """Gets token embeddings without positional info."""
    token_reprs = []
    for token_id in token_ids:
        token_reprs.append(wte[token_id].numpy())
    return np.array(token_reprs)


def get_representations(token_ids, wte, wpe):
    """Gets combined token + positional embeddings."""
    combined_reprs = []
    for i, token_id in enumerate(token_ids):
        token_emb = wte[token_id].numpy()
        pos_emb = wpe[i].numpy() if i < wpe.shape[0] else np.zeros_like(token_emb)
        combined_reprs.append(token_emb + pos_emb)
    return np.array(combined_reprs)


def get_block_weights(model_state, model_args, layer_idx=0):
    """Extracts all weights for a given transformer block."""
    n_embd = model_args["n_embd"]

    def get_param(key, is_bias=False):
        param = model_state.get(key)
        if param is None and is_bias:
            print(f"Info: Bias key '{key}' not found, assuming zero bias.")
            # Determine correct shape for the bias vector
            if "attn.c_attn.bias" in key:
                shape = 3 * n_embd
            elif "mlp.c_fc.bias" in key:
                shape = 4 * n_embd
            else:  # All other biases have n_embd size
                shape = n_embd
            return torch.zeros(shape)
        return param

    param_defs = {
        "ln1_g": (f"transformer.h.{layer_idx}.ln_1.weight", False),
        "ln1_b": (f"transformer.h.{layer_idx}.ln_1.bias", True),
        "c_attn_w": (f"transformer.h.{layer_idx}.attn.c_attn.weight", False),
        "c_attn_b": (f"transformer.h.{layer_idx}.attn.c_attn.bias", True),
        "c_proj_w": (f"transformer.h.{layer_idx}.attn.c_proj.weight", False),
        "c_proj_b": (f"transformer.h.{layer_idx}.attn.c_proj.bias", True),
        "ln2_g": (f"transformer.h.{layer_idx}.ln_2.weight", False),
        "ln2_b": (f"transformer.h.{layer_idx}.ln_2.bias", True),
        "mlp_fc_w": (f"transformer.h.{layer_idx}.mlp.c_fc.weight", False),
        "mlp_fc_b": (f"transformer.h.{layer_idx}.mlp.c_fc.bias", True),
        "mlp_proj_w": (f"transformer.h.{layer_idx}.mlp.c_proj.weight", False),
        "mlp_proj_b": (f"transformer.h.{layer_idx}.mlp.c_proj.bias", True),
    }

    weights = {
        name: get_param(key, is_bias) for name, (key, is_bias) in param_defs.items()
    }

    # Check for missing *weights*; biases are now handled.
    missing_weights = [
        param_defs[name][0]
        for name, param in weights.items()
        if param is None and not name.endswith("_b")
    ]

    if missing_weights:
        print(f"Error: Could not find all *weight* tensors for layer {layer_idx}.")
        print("The following weight keys were not found in the checkpoint:")
        for key in missing_weights:
            print(f"  - {key}")
        return None

    # Convert all tensors to numpy, transposing Linear layers' weights for matmul
    for k, v in weights.items():
        if isinstance(v, torch.Tensor):
            if k.endswith("_w"):  # Transpose weight matrices for (in, out) format
                weights[k] = v.T.numpy()
            else:
                weights[k] = v.numpy()

    # Split QKV weights and biases from the combined attention matrix/vector
    W_qkv = weights["c_attn_w"]
    b_qkv = weights["c_attn_b"]
    weights["W_q"] = W_qkv[:, :n_embd]
    weights["W_k"] = W_qkv[:, n_embd : 2 * n_embd]
    weights["W_v"] = W_qkv[:, 2 * n_embd :]
    weights["b_q"], weights["b_k"], weights["b_v"] = np.split(b_qkv, 3)

    return weights


def calculate_transformer_block_flow(x, weights, n_head):
    """Calculates all intermediate representations in a full transformer block."""
    n_embd = x.shape[1]
    d_k = n_embd // n_head

    # --- 1. Attention Sub-layer ---
    # a. LayerNorm
    x_ln1 = layernorm(x, weights["ln1_g"], weights["ln1_b"])

    # b. Q, K, V calculation
    q = x_ln1 @ weights["W_q"] + weights["b_q"]
    k = x_ln1 @ weights["W_k"] + weights["b_k"]
    v = x_ln1 @ weights["W_v"] + weights["b_v"]

    # c. Attention scores and weights (simplified single-head style)
    scores = (q @ k.T) / np.sqrt(d_k)
    e_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = e_scores / e_scores.sum(axis=-1, keepdims=True)

    # d. Weighted sum of values
    z = attn_weights @ v

    # e. Projection
    attn_proj = z @ weights["c_proj_w"] + weights["c_proj_b"]

    # f. First residual connection
    x_after_attn = x + attn_proj

    # --- 2. MLP Sub-layer ---
    # a. LayerNorm
    x_ln2 = layernorm(x_after_attn, weights["ln2_g"], weights["ln2_b"])

    # b. Feed-forward network
    mlp_fc = x_ln2 @ weights["mlp_fc_w"] + weights["mlp_fc_b"]
    mlp_gelu = gelu(mlp_fc)
    mlp_out = mlp_gelu @ weights["mlp_proj_w"] + weights["mlp_proj_b"]

    # c. Second residual connection
    x_final = x_after_attn + mlp_out

    # Return dict of all n_embd-sized vectors for visualization
    return {
        "Input (x)": x,
        "After LN1": x_ln1,
        "Query (q)": q,
        "Key (k)": k,
        "Value (v)": v,
        "Attn Out (z)": z,
        "Attn Proj": attn_proj,
        "After Resid1": x_after_attn,
        "After LN2": x_ln2,
        "MLP Out": mlp_out,
        "Block Output": x_final,
    }, attn_weights


# --- Word Cloud and Grid Generation ---


def load_dimension_wordmaps(model_dir, n_embd):
    """Loads all pre-generated dimension wordmap images."""
    wordmap_dir = os.path.join("visualizations", model_dir, "embedding_wordmaps")
    if not os.path.exists(wordmap_dir):
        print(f"ERROR: Dimension wordmaps not found in {wordmap_dir}")
        print("Please run visualize_tokens.py first.")
        return None, None

    wordmap_images = {}
    wordmap_size = None
    for dim in range(n_embd):
        path = os.path.join(wordmap_dir, f"dimension_{dim}.png")
        if os.path.exists(path):
            img = Image.open(path)
            if wordmap_size is None:
                wordmap_size = img.size
            wordmap_images[dim] = img

    if not wordmap_images:
        print("No wordmap images found.")
        return None, None

    print(f"Loaded {len(wordmap_images)} dimension wordmaps.")
    return wordmap_images, wordmap_size


def create_grid_for_vector(
    vector, n_embd, wordmap_images, wordmap_size, all_values_for_opacity_scaling
):
    """Creates a 6x6 grid visualization for a single n_embd vector."""
    grid_cols = int(np.ceil(np.sqrt(n_embd)))
    grid_rows = int(np.ceil(n_embd / grid_cols))

    # Scale down wordmaps for the grid
    thumb_size = 60
    thumb_w = int(wordmap_size[0] * (thumb_size / wordmap_size[1]))
    thumb_h = thumb_size

    canvas_w = grid_cols * thumb_w
    canvas_h = grid_rows * thumb_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    # Use percentile-based mapping for better contrast, same as in sentence_flow
    all_abs_values = np.abs(all_values_for_opacity_scaling.flatten())
    min_val, max_val = (
        all_values_for_opacity_scaling.min(),
        all_values_for_opacity_scaling.max(),
    )

    for dim in range(n_embd):
        if dim not in wordmap_images:
            continue

        activation = vector[dim]

        # Calculate opacity using percentile-based approach for better contrast
        if max_val > min_val:
            percentile = np.mean(all_abs_values <= abs(activation))
            opacity = np.power(
                percentile, 0.5
            )  # Use square root to boost mid-range values
            opacity = max(0.1, min(1.0, opacity))  # Clamp and set a minimum visibility
        else:
            opacity = 0.5

        # Get the wordmap image and apply transformations
        wordmap_img = wordmap_images[dim].resize(
            (thumb_w, thumb_h), Image.Resampling.LANCZOS
        )
        if wordmap_img.mode != "RGBA":
            wordmap_img = wordmap_img.convert("RGBA")

        # For negative activations, invert the colors for consistency with sentence_flow.py
        if activation < 0:
            img_array = np.array(wordmap_img)
            # Invert RGB channels, but leave alpha untouched
            img_array[:, :, :3] = 255 - img_array[:, :, :3]
            wordmap_img = Image.fromarray(img_array, "RGBA")

        # Apply opacity
        alpha = wordmap_img.split()[-1]
        alpha = alpha.point(lambda p: int(p * opacity))
        wordmap_img.putalpha(alpha)

        # Paste onto canvas
        row, col = dim // grid_cols, dim % grid_cols
        x, y = col * thumb_w, row * thumb_h
        canvas.paste(wordmap_img, (x, y), wordmap_img)

    return canvas


def create_2d_journey_plot(
    base_map_2d,
    vocab_itos,
    probe_word_indices,
    probe_word_vectors,
    probe_word_labels,
    umap_reducer,
    title,
    previous_probe_word_vectors_2d=None,
    final_output_vector=None,
    key_word_vectors_2d=None,
    key_word_labels=None,
):
    """Creates a 2D scatter plot showing the journey of probe words."""
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)

    # 1. Plot the entire vocabulary as a background
    ax.scatter(base_map_2d[:, 0], base_map_2d[:, 1], c="lightgray", alpha=0.1, s=10)

    # NEW: Plot key vocabulary words for context
    if key_word_vectors_2d is not None and key_word_labels:
        ax.scatter(
            key_word_vectors_2d[:, 0],
            key_word_vectors_2d[:, 1],
            c="gray",
            alpha=0.4,
            s=25,
            marker="x",
        )
        for i, label in enumerate(key_word_labels):
            ax.text(
                key_word_vectors_2d[i, 0] + 0.05,
                key_word_vectors_2d[i, 1] + 0.05,
                label,
                fontsize=8,
                color="gray",
                alpha=0.7,
            )

    # 2. Project the current probe word vectors into 2D space
    probe_word_vectors_2d = (
        umap_reducer.transform(probe_word_vectors)
        if probe_word_vectors.ndim > 1
        else umap_reducer.transform([probe_word_vectors])
    )

    # 3. Plot the probe words
    colors = plt.cm.rainbow(np.linspace(0, 1, len(probe_word_labels)))
    for i, label in enumerate(probe_word_labels):
        x, y = probe_word_vectors_2d[i]
        ax.scatter(x, y, color=colors[i], s=150, label=label, edgecolors="black")
        ax.text(x + 0.05, y + 0.05, label, fontsize=9, color=colors[i], weight="bold")

        # 4. If previous positions are provided, draw arrows to show the journey
        if previous_probe_word_vectors_2d is not None:
            prev_x, prev_y = previous_probe_word_vectors_2d[i]
            ax.annotate(
                "",
                xy=(x, y),
                xytext=(prev_x, prev_y),
                arrowprops=dict(
                    arrowstyle="->",
                    color=colors[i],
                    lw=1.5,
                    alpha=0.7,
                ),
            )

    # NEW: Plot the final output vector if provided
    if final_output_vector is not None:
        final_vector_2d = umap_reducer.transform([final_output_vector])[0]
        ax.scatter(
            final_vector_2d[0],
            final_vector_2d[1],
            color="black",
            s=250,
            marker="*",
            label="Model's Final Output",
            zorder=10,
        )
        ax.text(
            final_vector_2d[0] + 0.05,
            final_vector_2d[1] + 0.05,
            "Final Output",
            fontsize=12,
            color="black",
            weight="heavy",
        )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf), probe_word_vectors_2d


# --- Main Execution ---


def get_num_layers(model_state):
    """Determines the number of layers in the model."""
    max_layer = -1
    for key in model_state.keys():
        match = re.match(r"transformer\.h\.(\d+)\.", key)
        if match:
            max_layer = max(max_layer, int(match.group(1)))
    return max_layer + 1


def main():
    """Main function to generate the attention flow visualization."""

    # Config
    checkpoint_path = os.environ.get("MODEL")
    probe_sentence = os.environ.get("PROBE_SENTENCE", "knock knock whos there cat")

    if not checkpoint_path:
        print("Error: MODEL environment variable not set.")
        sys.exit(1)

    # Load model and data
    checkpoint = load_checkpoint(checkpoint_path)
    model_args = checkpoint["model_args"]
    model_state = checkpoint["model"]
    stoi, itos = load_tokenizer()
    wte = model_state["transformer.wte.weight"]
    wpe = model_state["transformer.wpe.weight"]

    if not stoi or not itos:
        sys.exit(1)

    # Prepare inputs first to get the list of words
    words, token_ids = tokenize_sentence(probe_sentence, stoi)

    # --- NEW: Create 2D UMAP projection of the entire vocabulary ---
    print("Creating 2D UMAP projection of the entire vocabulary...")
    full_vocab_vectors = wte.numpy()
    n_embd = model_args["n_embd"]

    # It's often good practice to run PCA before UMAP for high-dimensional data
    if n_embd > 50:
        print(
            f"Embedding dimension ({n_embd}) > 50. Reducing with PCA to 50 dimensions first."
        )
        pca = PCA(n_components=50)
        preprocessed_vectors = pca.fit_transform(full_vocab_vectors)
    else:
        preprocessed_vectors = full_vocab_vectors

    # Now run UMAP on the PCA-reduced data
    umap_reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, random_state=42
    )
    base_map_2d = umap_reducer.fit_transform(preprocessed_vectors)
    print("UMAP projection created.")

    # --- Define key words to always highlight on the plot ---
    key_words_to_highlight = ["knock", "who", "there", "cat", "dog", "man", "woman"]
    # Add words from the probe sentence to this list automatically
    key_words_to_highlight.extend(words)
    key_words_to_highlight = sorted(
        list(set(key_words_to_highlight))
    )  # remove duplicates and sort

    key_word_indices = [stoi.get(w) for w in key_words_to_highlight if w in stoi]
    key_word_vectors_2d = base_map_2d[key_word_indices] if key_word_indices else None

    # --- Start of multi-layer processing ---
    num_layers = get_num_layers(model_state)
    print(f"Model has {num_layers} layers. Visualizing full flow.")

    # NEW: Get both token-only and combined representations
    token_only_reps = get_token_only_representations(token_ids, wte)
    x = get_representations(
        token_ids, wte, wpe
    )  # This is combined, becomes input to Block 0

    all_reps_by_layer = {}
    all_attn_by_layer = {}

    # NEW: Add a special "Input" block for clarity
    all_reps_by_layer["input"] = {
        "Token Embedding": token_only_reps,
        "Combined Embedding (Input to Block 0)": x,
    }

    # Loop through all transformer blocks
    for layer_idx in range(num_layers):
        print(f"\n--- Processing Layer {layer_idx} ---")
        weights = get_block_weights(model_state, model_args, layer_idx)
        if weights is None:
            print(f"Failed to get weights for layer {layer_idx}. Aborting.")
            sys.exit(1)

        representations, attention_weights = calculate_transformer_block_flow(
            x, weights, model_args.get("n_head", 1)
        )
        all_reps_by_layer[f"layer_{layer_idx}"] = representations
        all_attn_by_layer[f"layer_{layer_idx}"] = attention_weights
        x = representations["Block Output"]  # Output of this layer is input to next

    # Process final layers after transformer blocks
    print("\n--- Processing Final Layers ---")
    final_reps = {}
    final_prediction_data = None
    final_ln_g = model_state.get("transformer.ln_f.weight")
    final_ln_b_tensor = model_state.get("transformer.ln_f.bias")
    final_ln_b = (
        final_ln_b_tensor.numpy()
        if final_ln_b_tensor is not None
        else np.zeros(model_args["n_embd"])
    )

    if final_ln_g is not None:
        x_ln_f = layernorm(x, final_ln_g.numpy(), final_ln_b)
        final_reps["After Final Layer Normalisation"] = x_ln_f
        all_reps_by_layer["final"] = final_reps

        # The lm_head weights are the same as the token embedding weights
        lm_head_w = wte.numpy()
        logits = x_ln_f @ lm_head_w.T
        final_reps["Final Linear Layer"] = logits

        # --- NEW: Dot Product Breakdown Visualization ---
        # Get the top predicted token to explain its logit
        last_token_logits = logits[-1]
        top_prediction_idx = np.argmax(last_token_logits)
        top_prediction_word = itos[top_prediction_idx]

        # Get the vector for the last input word after final norm
        final_norm_vector_last_word = x_ln_f[-1]

        # Get the embedding for the predicted word
        predicted_word_embedding = wte[top_prediction_idx].numpy()

        # Calculate element-wise product, showing each dimension's contribution to the logit
        dot_product_breakdown = final_norm_vector_last_word * predicted_word_embedding

        # This breakdown is only for the last word; pad for other words to fit the data structure
        padded_breakdown = np.zeros_like(x_ln_f)
        padded_breakdown[-1] = dot_product_breakdown

        breakdown_key = f"Dot Product Breakdown for {top_prediction_word}"
        final_reps[breakdown_key] = padded_breakdown
        # --- End NEW ---

        # Get probabilities for the token following the last input token
        probs = np.exp(last_token_logits) / np.sum(np.exp(last_token_logits))  # Softmax

        # Get top 10 predictions
        top_k = 10
        top_k_indices = np.argsort(-probs)[:top_k]
        final_prediction_data = {
            "words": [itos[i] for i in top_k_indices],
            "probabilities": probs[top_k_indices].tolist(),
        }
    else:
        print("Warning: Final layer norm or lm_head not found.")
        final_prediction_data = None

    # --- Create Visualizations ---
    model_dir = os.path.basename(os.path.dirname(checkpoint_path))
    output_dir = os.path.join("visualizations", model_dir, "token_space_journey")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutputting to: {output_dir}")

    # Collect ALL activation values for a consistent opacity scale, excluding logits
    all_values_for_scaling = np.concatenate(
        [
            vectors
            for layer_reps in all_reps_by_layer.values()
            for step_name, vectors in layer_reps.items()
            if step_name != "Final Linear Layer"
        ]
    )

    # Generate grid images for each step, each word, each layer
    image_paths = {}

    # Define the order of layers for processing and generating HTML
    layer_keys = ["input"] + [f"layer_{i}" for i in range(num_layers)] + ["final"]

    previous_step_vectors_2d = {}  # To store 2D vectors for drawing arrows

    for layer_key in layer_keys:
        if layer_key not in all_reps_by_layer:
            continue

        representations = all_reps_by_layer[layer_key]
        print(f"Generating images for: {layer_key}")
        for step_name, all_word_vectors in representations.items():

            # Special handling for the final logit layer
            if step_name == "Final Linear Layer":
                # For logits, we can't project them. Let's create a special plot.
                # Here we will plot the top K predicted token embeddings on the UMAP

                last_token_logits = all_word_vectors[-1]
                top_k_indices = np.argsort(-last_token_logits)[:10]
                top_k_words = [itos[i] for i in top_k_indices]
                top_k_vectors_2d = base_map_2d[top_k_indices]

                final_output_vector = all_reps_by_layer["final"][
                    "After Final Layer Normalisation"
                ][-1]

                grid_img, _ = create_logit_plot(
                    base_map_2d,
                    top_k_vectors_2d,
                    top_k_words,
                    "Top 10 Predictions on Vocab Map",
                    key_word_vectors_2d,
                    key_words_to_highlight,
                    final_output_vector,
                    umap_reducer,
                )

            else:
                # Use the new 2D plot generation function
                plot_title = f"{step_name} (Layer: {layer_key})"
                previous_vectors_2d = previous_step_vectors_2d.get(layer_key)

                grid_img, current_vectors_2d = create_2d_journey_plot(
                    base_map_2d=base_map_2d,
                    vocab_itos=itos,
                    probe_word_indices=token_ids,
                    probe_word_vectors=all_word_vectors,
                    probe_word_labels=words,
                    umap_reducer=umap_reducer,
                    title=plot_title,
                    previous_probe_word_vectors_2d=previous_vectors_2d,
                    key_word_vectors_2d=key_word_vectors_2d,
                    key_word_labels=key_words_to_highlight,
                )
                # Store the current vectors for the next step in this layer
                previous_step_vectors_2d[layer_key] = current_vectors_2d

            # Sanitize step_name for filename
            sanitized_step_name = (
                step_name.replace("'", "")
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
            )
            # We create one image per step, not per word
            filename = f"{layer_key}_{sanitized_step_name}.png"
            path = os.path.join(output_dir, filename)
            grid_img.save(path)

            # Update image_paths to have a different structure
            # Key by layer and step, since we have one image per step now
            if layer_key not in image_paths:
                image_paths[layer_key] = {}
            image_paths[layer_key][step_name] = filename

    # Generate HTML page to display everything in a grid
    generate_html_page(
        output_dir,
        model_dir,
        probe_sentence,
        words,
        image_paths,
        all_reps_by_layer,
        all_attn_by_layer,
        final_prediction_data,
        top_prediction_word,
    )

    print("\nDone! Full model flow visualization created.")
    print(f"View the interactive summary at: {output_dir}/index.html")


def create_logit_plot(
    base_map_2d,
    top_k_vectors_2d,
    top_k_labels,
    title,
    key_word_vectors_2d,
    key_word_labels,
    final_output_vector,
    umap_reducer,
):
    """Creates a 2D plot showing the location of top predicted tokens."""
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)

    # 1. Plot the entire vocabulary and key words as a background
    ax.scatter(base_map_2d[:, 0], base_map_2d[:, 1], c="lightgray", alpha=0.1, s=10)
    if key_word_vectors_2d is not None and key_word_labels:
        ax.scatter(
            key_word_vectors_2d[:, 0],
            key_word_vectors_2d[:, 1],
            c="gray",
            alpha=0.4,
            s=25,
            marker="x",
        )
        for i, label in enumerate(key_word_labels):
            ax.text(
                key_word_vectors_2d[i, 0] + 0.05,
                key_word_vectors_2d[i, 1] + 0.05,
                label,
                fontsize=8,
                color="gray",
                alpha=0.7,
            )

    # 2. Plot the Top-K predicted tokens
    colors = plt.cm.viridis(np.linspace(0.8, 0, len(top_k_labels)))
    for i, label in enumerate(top_k_labels):
        x, y = top_k_vectors_2d[i]
        size = 200 - i * 15  # Make top prediction largest
        ax.scatter(
            x, y, color=colors[i], s=size, label=f"#{i+1}: {label}", edgecolors="black"
        )
        ax.text(x + 0.05, y + 0.05, label, fontsize=10, color=colors[i], weight="bold")

    # 3. Plot the final output vector (before projection to logits)
    final_vector_2d = umap_reducer.transform([final_output_vector])[0]
    ax.scatter(
        final_vector_2d[0],
        final_vector_2d[1],
        color="red",
        s=300,
        marker="*",
        label="Model's Final Output",
        zorder=10,
        edgecolors="black",
    )
    ax.text(
        final_vector_2d[0] + 0.05,
        final_vector_2d[1] + 0.05,
        "Final Output",
        fontsize=12,
        color="red",
        weight="heavy",
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf), None


def create_logits_barchart(logit_vector, itos, top_k=5):
    """Creates a bar chart image of the top k logits."""
    top_k_indices = np.argsort(-logit_vector)[:top_k]
    top_k_logits = logit_vector[top_k_indices]
    top_k_words = [itos.get(i, f"unk_{i}") for i in top_k_indices]

    # Create plot
    fig, ax = plt.subplots(figsize=(3, 2), dpi=150)
    bars = ax.barh(np.arange(top_k), top_k_logits, color="skyblue")
    ax.set_yticks(np.arange(top_k))
    ax.set_yticklabels(top_k_words, fontsize=8)
    ax.invert_yaxis()  # Highest on top
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Logit Value", fontsize=8)

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" {width:.2f}",
            va="center",
            ha="left" if width > 0 else "right",
            fontsize=7,
        )

    plt.tight_layout()

    # Adjust x-axis limit to prevent label overlap
    _, xmax = plt.xlim()
    plt.xlim(xmax=xmax * 1.15)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    return Image.open(buf)


def get_code_snippet_dict(top_prediction_word=""):
    """Returns a dictionary of all code snippets."""

    breakdown_key = f"Dot Product Breakdown for {top_prediction_word}"

    code_map = {
        "Input (x)": "x = self.transformer.drop(tok_emb + pos_emb)",
        "After LN1": "x = x + self.attn(self.ln_1(x))  // self.ln_1(x) is applied first",
        "Query (q)": "q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)",
        "Key (k)": "q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)",
        "Value (v)": "q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)",
        "Attn Out (z)": "y = att @ v  // Weighted sum of Value vectors",
        "Attn Proj": "y = self.resid_dropout(self.c_proj(y))",
        "After Resid1": "x = x + self.attn(self.ln_1(x)) // First residual connection",
        "After LN2": "x = x + self.mlp(self.ln_2(x)) // self.ln_2(x) is applied first",
        "MLP Out": "x = self.mlp(x) // Full MLP block: fc -> gelu -> proj -> dropout",
        "Block Output": "x = x + self.mlp(self.ln_2(x)) // Second residual connection",
        "After Final Layer Normalisation": "x = self.transformer.ln_f(x) // Final layer normalization",
        "Final Linear Layer": "logits = self.lm_head(x) // Final projection to vocabulary",
    }
    code_map[
        breakdown_key
    ] = f"""# The logit for '{top_prediction_word}' is the dot product:
# final_norm_vector @ wte['{top_prediction_word}']

# This visualization shows the element-wise product of that operation.
# Each grid square is a dimension, and its brightness shows its contribution
# to the final logit score for '{top_prediction_word}'.

# A bright square means that dimension was highly active in *both* the
# final thought vector and the embedding for '{top_prediction_word}',
# strongly pushing the model to predict '{top_prediction_word}'.

# NOTE: The word-cloud for each dimension is just a label based on
# which words activate it most across the whole vocabulary. It is not
# the content of the dimension itself.
logit_contribution = final_norm_vector * wte['{top_prediction_word}']"""
    return code_map


def generate_html_page(
    output_dir,
    model_name,
    probe_sentence,
    words,
    image_paths,
    all_reps_by_layer,
    all_attn_by_layer,
    final_prediction_data,
    top_prediction_word,
):
    """Generates an HTML page with collapsible sections for each layer."""

    # Create a JSON-serializable version of the data by converting numpy arrays to lists
    serializable_reps = {
        layer_key: {step_name: vectors.tolist() for step_name, vectors in reps.items()}
        for layer_key, reps in all_reps_by_layer.items()
    }
    all_data_json = json.dumps(serializable_reps)
    words_json = json.dumps(words)
    image_paths_json = json.dumps(image_paths)
    code_snippets_dict = get_code_snippet_dict(top_prediction_word)
    code_snippets_json = json.dumps(code_snippets_dict)

    layers_html = ""
    num_layers = sum(1 for key in all_reps_by_layer if key.startswith("layer_"))

    # Define the order and titles for the HTML sections
    html_titles = {
        "input": "Input Embeddings",
        **{f"layer_{i}": f"Transformer Block {i}" for i in range(num_layers)},
        "final": "Final Projection",
    }
    ordered_keys = ["input"] + [f"layer_{i}" for i in range(num_layers)] + ["final"]

    # Generate HTML for each main section (Input, Transformer Blocks, Final)
    for layer_key in ordered_keys:
        if layer_key not in all_reps_by_layer:
            continue

        summary_title = html_titles.get(layer_key, "Details")
        # Keep the first block, input, and final projection sections open by default
        is_open = layer_key in ["input", "final"] or layer_key == "layer_0"
        details_options = "open" if is_open else ""

        representations = all_reps_by_layer[layer_key]
        attention_weights = all_attn_by_layer.get(layer_key)

        # Simplified header, as we now have one image per column
        header_cols = list(representations.keys())

        # Table header
        table_head_html = "".join(f"<th>{html.escape(col)}</th>" for col in header_cols)

        # Table body - now we have only one row per layer section
        row_html = "<tr>"
        for step_name in header_cols:
            # Get the single image for this step
            img_path = image_paths.get(layer_key, {}).get(step_name, "")
            safe_title = html.escape(step_name)

            # The modal click might not be relevant anymore or needs rethinking.
            # For now, let's just display the image.
            row_html += (
                f"<td><img src='{img_path}' title='{safe_title}' loading='lazy'></td>"
            )
        row_html += "</tr>"
        table_body_html = row_html

        layers_html += f"""
        <details {details_options}>
            <summary><h2>{summary_title}</h2></summary>
            <div class="table-container">
                <table>
                    <thead><tr>{table_head_html}</tr></thead>
                    <tbody>{table_body_html}</tbody>
                </table>
            </div>
        </details>
        """

    # Generate HTML for Final Prediction
    if final_prediction_data:
        prediction_html = "<div class='prediction-container'>"

        # New table-based layout for top predictions with embedding visualizations
        if "embedding_img_paths" in final_prediction_data:
            prediction_html += """
            <p style="text-align:left; max-width: 80%; margin: 1em auto; font-style: italic; color: #666;">
                Below are the raw embedding representations for the top 10 potential next tokens.
                You can compare the visualization for the top prediction with the
                'Dot Product Breakdown' visualization in the 'Final Projection' section above.
                The 'Dot Product Breakdown' shows how the model's final internal state
                aligns with a token's embedding to produce a high logit score.
            </p>
            <table class='data-table' style='margin: 1em auto; width: 80%;'>
               <thead><tr><th>Token</th><th>Probability</th><th>Embedding Visualization</th></tr></thead>
               <tbody>
            """
            for i, word in enumerate(final_prediction_data["words"]):
                prob = final_prediction_data["probabilities"][i]
                img_path = final_prediction_data["embedding_img_paths"][i]
                safe_word = html.escape(word)
                prediction_html += f"""
                    <tr>
                        <td class='word-label'>{safe_word}</td>
                        <td>
                            <div class="prediction-prob-container">
                                <div class="bar" style="width: {prob*100*3}px; background-color: rgba(220, 53, 69, {0.2 + prob*0.8});"></div>
                                <span class="bar-value" style="margin-left: 5px;">{(prob*100):.2f}%</span>
                            </div>
                        </td>
                        <td><img src='{img_path}' loading='lazy' style='max-width: 200px; height: auto;' title='Embedding for "{safe_word}"'></td>
                    </tr>
                """
            prediction_html += "</tbody></table>"
        else:
            # Fallback to old bar chart view
            prediction_html += "<div class='prediction-bar-container'>"
            for word, prob in zip(
                final_prediction_data["words"], final_prediction_data["probabilities"]
            ):
                prediction_html += f"""
                    <div class="bar-row">
                        <span class="bar-label">{html.escape(word)}</span>
                        <div class="bar" style="width: {{prob*100*4}}px; background-color: rgba(220, 53, 69, {{0.2 + prob*0.8}});"></div>
                        <span class="bar-value">{{prob*100:0.2f}}%</span>
                    </div>
                """
            prediction_html += "</div>"

        prediction_html += "</div>"

        layers_html += f"""
        <details open>
            <summary><h2>Next Token Prediction</h2></summary>
            {prediction_html}
        </details>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Full Model Flow Visualization</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; margin: 2em; background: #f0f2f5; color: #333; }}
            h1, h2, p {{ text-align: center; }}
            details {{ background: white; border-radius: 8px; margin-bottom: 1em; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
            summary {{ font-size: 1.5em; font-weight: bold; padding: 0.8em; cursor: pointer; }}
            .container {{ max-width: 98%; margin: auto; }}
            .table-container {{ overflow-x: auto; padding: 1em; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 1em; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; vertical-align: middle; min-width: 150px; }}
            th {{ background-color: #f8f9fa; font-size: 1em; }}
            img {{ max-width: 100%; height: auto; border-radius: 4px; cursor: pointer; }}
            .word-label {{ font-weight: bold; font-size: 1.1em; }}
            .pos {{ font-weight: normal; color: #666; font-size: 0.8em; }}
            .attention-cell {{ min-width: 250px; }}
            .attention-bar-container {{ display: flex; flex-direction: column; gap: 4px; }}
            .bar-row {{ display: flex; align-items: center; gap: 5px; font-size: 0.8em; }}
            .bar-label {{ width: 80px; text-align: right; }}
            .bar {{ height: 16px; border-radius: 3px; border: 1px solid #eee; }}
            .bar-value {{ font-family: monospace; }}

            /* Modal Styles */
            .modal {{
                display: none; position: fixed; z-index: 1000; left: 0; top: 0;
                width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.7);
            }}
            .modal-content {{
                background-color: #fefefe; margin: 5% auto; padding: 20px;
                border: 1px solid #888; width: 80%; max-width: 1200px;
                border-radius: 8px;
            }}
            .modal-header {{
                display: flex; justify-content: space-between; align-items: center;
                border-bottom: 1px solid #ddd; padding-bottom: 10px;
            }}
            .modal-header h2 {{ text-align: left; }}
            .close {{ color: #aaa; font-size: 28px; font-weight: bold; cursor: pointer; }}
            .modal-body {{ display: flex; gap: 20px; margin-top: 20px; }}
            .modal-image-container {{ flex: 1; }}
            .modal-data-container {{ flex: 1; }}
            .code-snippet {{
                background: #2d2d2d; color: #dcdcdc; padding: 15px;
                border-radius: 5px; font-family: monospace; white-space: pre;
            }}
            .data-table {{ width: 100%; border-collapse: collapse; font-size: 1em; }}
            .data-table th, .data-table td {{ border: 1px solid #eee; padding: 8px; text-align: left; vertical-align: middle; }}
            .data-table th {{ background-color: #f2f2f2; }}

            .prediction-prob-container {{
                display: flex;
                align-items: center;
                gap: 8px;
            }}

            .prediction-container {{ padding: 2em; }}
            .prediction-bar-container {{ display: flex; flex-direction: column; gap: 6px; max-width: 500px; margin: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Full Model Flow Visualization</h1>
            <p><strong>Model:</strong> {model_name}<br><strong>Probe Sentence:</strong> "{probe_sentence}"</p>
            {layers_html}
        </div>

        <!-- Data Injection using JSON script tags -->
        <script id="all-data" type="application/json">{all_data_json}</script>
        <script id="all-words" type="application/json">{words_json}</script>
        <script id="all-image-paths" type="application/json">{image_paths_json}</script>
        <script id="code-snippets" type="application/json">{code_snippets_json}</script>

        <!-- Modal is currently disabled for this visualization -->
        <!--
        <div id="modal" class="modal">
            ...
        </div>
        -->

        <script>
            // Modal functionality is disabled for this view.
        </script>
    </body>
    </html>
    """

    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated HTML page: {html_path}")


if __name__ == "__main__":
    main()
