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


# --- Main Execution ---


def main():
    """Main function to generate the attention flow visualization."""

    # Config
    checkpoint_path = os.environ.get("MODEL")
    probe_sentence = os.environ.get("PROBE_SENTENCE", "knock knock whos there bob")
    layer_idx = 0

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

    # Prepare inputs
    words, token_ids = tokenize_sentence(probe_sentence, stoi)
    x = get_representations(token_ids, wte, wpe)  # Initial combined representations
    weights = get_block_weights(model_state, model_args, layer_idx)

    if weights is None:
        print("Failed to get transformer block weights.")
        sys.exit(1)

    # Calculate all attention steps
    representations, attention_weights = calculate_transformer_block_flow(
        x, weights, model_args.get("n_head", 1)
    )

    # --- Create Visualizations ---
    model_dir = os.path.basename(os.path.dirname(checkpoint_path))
    output_dir = os.path.join("visualizations", model_dir, "attention_flow_wordmaps")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputting to: {output_dir}")

    # Load base dimension wordmaps
    wordmap_images, wordmap_size = load_dimension_wordmaps(
        model_dir, model_args["n_embd"]
    )
    if wordmap_images is None:
        sys.exit(1)

    # Collect all activation values to create a consistent opacity scale
    all_values_for_scaling = np.concatenate(list(representations.values()))

    # Generate grid images for each step and each word
    image_paths = {}
    for i, word in enumerate(words):
        for step_name, all_word_vectors in representations.items():
            vector = all_word_vectors[i]
            grid_img = create_grid_for_vector(
                vector,
                model_args["n_embd"],
                wordmap_images,
                wordmap_size,
                all_values_for_scaling,
            )
            filename = f"{word}_{i}_{step_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
            path = os.path.join(output_dir, filename)
            grid_img.save(path)
            image_paths[(word, i, step_name)] = filename

    # Generate HTML page to display everything in a grid
    generate_html_page(
        output_dir,
        model_dir,
        probe_sentence,
        words,
        image_paths,
        attention_weights,
        list(representations.keys()),
    )

    print("\nDone! Wordmap grids for each step of the transformer block created.")
    print(f"View the interactive summary at: {output_dir}/index.html")


def generate_html_page(
    output_dir,
    model_name,
    probe_sentence,
    words,
    image_paths,
    attention_weights,
    header_cols,
):
    """Generates an HTML page to display the attention flow grid."""

    # Add the 'Attention' column in the correct place
    if "Attn Out (z)" in header_cols:
        pos = header_cols.index("Attn Out (z)") + 1
        header_cols.insert(pos, "Attention")
    else:
        header_cols.append("Attention")

    # Generate the rows for the main grid
    rows_html = ""
    for i, word in enumerate(words):
        rows_html += "<tr>"
        rows_html += (
            f"<td class='word-label'>'{word}' <span class='pos'>(pos {i})</span></td>"
        )

        # Wordmap and attention columns
        for step_name in header_cols:
            if step_name == "Attention":
                # Attention column
                attention_viz_html = "<div class='attention-bar-container'>"
                for j, target_word in enumerate(words):
                    weight = attention_weights[i, j]
                    attention_viz_html += f"""
                        <div class="bar-row">
                            <span class="bar-label">{target_word}</span>
                            <div class="bar" style="width: {weight*100*2}px; background-color: rgba(75, 192, 192, {weight*5});"></div>
                            <span class="bar-value">{weight:.3f}</span>
                        </div>
                    """
                attention_viz_html += "</div>"
                rows_html += f"<td class='attention-cell'>{attention_viz_html}</td>"
            else:
                img_path = image_paths.get((word, i, step_name), "")
                rows_html += f"<td><img src='{img_path}' title='{step_name}' loading='lazy'></td>"

        rows_html += "</tr>"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Transformer Block Flow Visualization</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; margin: 2em; background: #f0f2f5; color: #333; }}
            .container {{ max-width: 95%; margin: auto; background: white; padding: 2em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
            h1, h2, p {{ text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 2em; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; vertical-align: middle; }}
            th {{ background-color: #f8f9fa; font-size: 1.1em; }}
            img {{ max-width: 100%; height: auto; border-radius: 4px; }}
            .word-label {{ font-weight: bold; font-size: 1.2em; }}
            .pos {{ font-weight: normal; color: #666; font-size: 0.8em; }}
            .attention-cell {{ width: 250px; }}
            .attention-bar-container {{ display: flex; flex-direction: column; gap: 4px; }}
            .bar-row {{ display: flex; align-items: center; gap: 5px; font-size: 0.8em; }}
            .bar-label {{ width: 50px; text-align: right; }}
            .bar {{ height: 12px; border-radius: 3px; border: 1px solid #eee; }}
            .bar-value {{ font-family: monospace; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Transformer Block Flow Visualization</h1>
            <p><strong>Model:</strong> {model_name}<br><strong>Probe Sentence:</strong> "{probe_sentence}"</p>
            <h2>Each row shows a word's journey through the first transformer block.</h2>
            <table>
                <thead>
                    <tr>
                        <th>Word</th>
                        {''.join(f"<th>{col}</th>" for col in header_cols)}
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """

    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated HTML page: {html_path}")


if __name__ == "__main__":
    main()
