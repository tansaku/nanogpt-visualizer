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


def get_representations(token_ids, wte, wpe):
    """Gets combined token + positional embeddings."""
    combined_reprs = []
    for i, token_id in enumerate(token_ids):
        token_emb = wte[token_id].numpy()
        pos_emb = wpe[i].numpy() if i < wpe.shape[0] else np.zeros_like(token_emb)
        combined_reprs.append(token_emb + pos_emb)
    return np.array(combined_reprs)


def get_qkv_matrices(model_state, model_args, layer_idx=0):
    """Extracts Q, K, V weight matrices from the model state."""
    n_embd = model_args["n_embd"]
    key = f"transformer.h.{layer_idx}.attn.c_attn.weight"
    if key not in model_state:
        return None, None, None

    c_attn_weight = model_state[key]
    if c_attn_weight.shape[0] == 3 * n_embd:
        c_attn_weight = c_attn_weight.T

    W_q = c_attn_weight[:, :n_embd].numpy()
    W_k = c_attn_weight[:, n_embd : 2 * n_embd].numpy()
    W_v = c_attn_weight[:, 2 * n_embd :].numpy()
    return W_q, W_k, W_v


def calculate_attention_flow(x, W_q, W_k, W_v, n_head=1, d_k=None):
    """Calculates all intermediate representations in the attention flow."""
    if d_k is None:
        d_k = x.shape[1] // n_head

    # 1. Transform to Q, K, V spaces
    q = x @ W_q
    k = x @ W_k
    v = x @ W_v

    # 2. Calculate attention scores and weights
    scores = (q @ k.T) / np.sqrt(d_k)
    e_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = e_scores / e_scores.sum(axis=-1, keepdims=True)

    # 3. Calculate final output vectors
    z = weights @ v

    return q, k, v, weights, z


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

    max_abs_val = np.max(np.abs(all_values_for_opacity_scaling))

    for dim in range(n_embd):
        if dim not in wordmap_images:
            continue

        activation = vector[dim]

        # Calculate opacity based on this activation relative to the max absolute activation
        opacity = 0.5  # default
        if max_abs_val > 0:
            norm_opacity = abs(activation) / max_abs_val
            opacity = (
                0.1 + 0.9 * norm_opacity
            )  # Scale to be more visible, from 10% to 100%

        opacity = max(0.05, min(1.0, opacity))  # Clamp

        # Get the wordmap image and apply transformations
        wordmap_img = wordmap_images[dim].resize(
            (thumb_w, thumb_h), Image.Resampling.LANCZOS
        )
        if wordmap_img.mode != "RGBA":
            wordmap_img = wordmap_img.convert("RGBA")

        # Use red for negative activations
        if activation < 0:
            r, g, b, a = wordmap_img.split()
            # Create a red overlay
            red_overlay = Image.new("RGB", wordmap_img.size, (204, 0, 0))
            wordmap_img = Image.blend(
                wordmap_img.convert("RGB"), red_overlay, alpha=0.5
            )
            wordmap_img.putalpha(a)

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
    W_q, W_k, W_v = get_qkv_matrices(model_state, model_args, layer_idx)

    if W_q is None:
        print("Failed to get QKV matrices.")
        sys.exit(1)

    # Calculate all attention steps
    q, k, v, weights, z = calculate_attention_flow(
        x,
        W_q,
        W_k,
        W_v,
        model_args.get("n_head", 1),
        model_args.get("n_embd") // model_args.get("n_head", 1),
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
    all_values_for_scaling = np.concatenate([x, q, k, v, z])

    # Generate grid images for each step and each word
    image_paths = {}
    for i, word in enumerate(words):
        steps = {
            "Input (x)": x[i],
            "Query (q)": q[i],
            "Key (k)": k[i],
            "Value (v)": v[i],
            "Output (z)": z[i],
        }
        for step_name, vector in steps.items():
            grid_img = create_grid_for_vector(
                vector,
                model_args["n_embd"],
                wordmap_images,
                wordmap_size,
                all_values_for_scaling,
            )
            filename = f"{word}_{i}_{step_name.replace(' ', '_')}.png"
            path = os.path.join(output_dir, filename)
            grid_img.save(path)
            image_paths[(word, i, step_name)] = filename

    # Generate HTML page to display everything in a grid
    generate_html_page(
        output_dir, model_dir, probe_sentence, words, image_paths, weights
    )

    print("\nDone! Wordmap grids for each step of the attention flow created.")
    print(f"View the interactive summary at: {output_dir}/index.html")


def generate_html_page(
    output_dir, model_name, probe_sentence, words, image_paths, attention_weights
):
    """Generates an HTML page to display the attention flow grid."""

    header_cols = [
        "Input (x)",
        "Query (q)",
        "Key (k)",
        "Value (v)",
        "Attention",
        "Output (z)",
    ]

    # Generate the rows for the main grid
    rows_html = ""
    for i, word in enumerate(words):
        rows_html += "<tr>"
        rows_html += (
            f"<td class='word-label'>'{word}' <span class='pos'>(pos {i})</span></td>"
        )

        # Wordmap columns
        for step_name in ["Input (x)", "Query (q)", "Key (k)", "Value (v)"]:
            img_path = image_paths.get((word, i, step_name), "")
            rows_html += f"<td><img src='{img_path}' loading='lazy'></td>"

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

        # Output wordmap column
        img_path = image_paths.get((word, i, "Output (z)"), "")
        rows_html += f"<td><img src='{img_path}' loading='lazy'></td>"
        rows_html += "</tr>"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Attention Flow - {model_name}</title>
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
            <h1>Attention Flow Wordmap Visualization</h1>
            <p><strong>Model:</strong> {model_name}<br><strong>Probe Sentence:</strong> "{probe_sentence}"</p>
            <h2>Each row shows a word's journey through the first attention layer.</h2>
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
