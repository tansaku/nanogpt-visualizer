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
    probe_sentence = os.environ.get("PROBE_SENTENCE", "knock knock whos there bob")

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

    # --- Start of multi-layer processing ---
    num_layers = get_num_layers(model_state)
    print(f"Model has {num_layers} layers. Visualizing full flow.")

    # Prepare inputs
    words, token_ids = tokenize_sentence(probe_sentence, stoi)
    x = get_representations(token_ids, wte, wpe)  # Initial combined representations

    all_reps_by_layer = {}
    all_attn_by_layer = {}

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
    final_ln_g = model_state.get("transformer.ln_f.weight")
    final_ln_b = model_state.get("transformer.ln_f.bias")

    if final_ln_g is not None and final_ln_b is not None:
        x_ln_f = layernorm(x, final_ln_g.numpy(), final_ln_b.numpy())
        final_reps["After Final LN"] = x_ln_f

        # Project logits back into embedding space for visualization
        lm_head_w = model_state.get("lm_head.weight")
        if lm_head_w is not None:
            logits = x_ln_f @ lm_head_w.T.numpy()
            final_reps["Logits (Projected)"] = logits @ wte.numpy()
        all_reps_by_layer["final"] = final_reps
    else:
        print("Warning: Final layer norm or lm_head not found.")

    # --- Create Visualizations ---
    model_dir = os.path.basename(os.path.dirname(checkpoint_path))
    output_dir = os.path.join("visualizations", model_dir, "full_model_flow")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutputting to: {output_dir}")

    # Load base dimension wordmaps
    wordmap_images, wordmap_size = load_dimension_wordmaps(
        model_dir, model_args["n_embd"]
    )
    if wordmap_images is None:
        sys.exit(1)

    # Collect ALL activation values for a consistent opacity scale
    all_values_for_scaling = np.concatenate(
        [v for layer_reps in all_reps_by_layer.values() for v in layer_reps.values()]
    )

    # Generate grid images for each step, each word, each layer
    image_paths = {}
    for layer_key, representations in all_reps_by_layer.items():
        print(f"Generating images for: {layer_key}")
        for step_name, all_word_vectors in representations.items():
            for i, word in enumerate(words):
                vector = all_word_vectors[i]
                grid_img = create_grid_for_vector(
                    vector,
                    model_args["n_embd"],
                    wordmap_images,
                    wordmap_size,
                    all_values_for_scaling,
                )
                filename = f"{layer_key}_{word}_{i}_{step_name.replace(' ', '_').replace('(', '').replace(')', '')}.png"
                path = os.path.join(output_dir, filename)
                grid_img.save(path)
                # Use a string key for JSON compatibility
                key = f"{layer_key}||{word}||{i}||{step_name}"
                image_paths[key] = filename

    # Generate HTML page to display everything in a grid
    generate_html_page(
        output_dir,
        model_dir,
        probe_sentence,
        words,
        image_paths,
        all_reps_by_layer,
        all_attn_by_layer,
    )

    print("\nDone! Full model flow visualization created.")
    print(f"View the interactive summary at: {output_dir}/index.html")


def get_code_snippet(step_name):
    """Returns a hard-coded explanation for each calculation step."""

    code_map = {
        "Input (x)": "x = token_embedding + positional_embedding",
        "After LN1": "x_ln1 = layernorm(x, ln1_gamma, ln1_beta)",
        "Query (q)": "q = x_ln1 @ W_q + b_q",
        "Key (k)": "k = x_ln1 @ W_k + b_k",
        "Value (v)": "v = x_ln1 @ W_v + b_v",
        "Attn Out (z)": "z = softmax((q @ k.T) / sqrt(d_k)) @ v",
        "Attn Proj": "attn_proj = z @ c_proj_w + c_proj_b",
        "After Resid1": "x = x + attn_proj",
        "After LN2": "x_ln2 = layernorm(x, ln2_gamma, ln2_beta)",
        "MLP Out": "mlp_out = gelu(x_ln2 @ mlp_fc_w + mlp_fc_b) @ mlp_proj_w + mlp_proj_b",
        "Block Output": "x = x_after_attn + mlp_out",
        "After Final LN": "x_final = layernorm(x, ln_f_gamma, ln_f_beta)",
        "Logits (Projected)": "logits = x_final @ lm_head.T",
    }
    return code_map.get(step_name, "No code snippet available.")


def generate_html_page(
    output_dir,
    model_name,
    probe_sentence,
    words,
    image_paths,
    all_reps_by_layer,
    all_attn_by_layer,
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

    layers_html = ""
    num_layers = sum(1 for key in all_reps_by_layer if key.startswith("layer_"))

    # Generate HTML for each Transformer Block
    for layer_idx in range(num_layers):
        layer_key = f"layer_{layer_idx}"
        representations = all_reps_by_layer[layer_key]
        attention_weights = all_attn_by_layer[layer_key]

        header_cols = list(representations.keys())
        if "Attn Out (z)" in header_cols:
            pos = header_cols.index("Attn Out (z)") + 1
            header_cols.insert(pos, "Attention")

        # Table header
        table_head_html = "<th>Word</th>" + "".join(
            f"<th>{col}</th>" for col in header_cols
        )

        # Table body
        table_body_html = ""
        for i, word in enumerate(words):
            row_html = f"<tr><td class='word-label'>'{word}'<br><span class='pos'>(pos {i})</span></td>"
            for step_name in header_cols:
                if step_name == "Attention":
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
                    row_html += f"<td class='attention-cell'>{attention_viz_html}</td>"
                else:
                    img_path = image_paths.get(
                        f"{layer_key}||{word}||{i}||{step_name}", ""
                    )
                    row_html += f"<td><img src='{img_path}' title='{step_name}' loading='lazy' onclick='openModal(\"{layer_key}\", {i}, \"{step_name}\")'></td>"
            row_html += "</tr>"
            table_body_html += row_html

        layers_html += f"""
        <details {'open' if layer_idx == 0 else ''}>
            <summary><h2>Transformer Block {layer_idx}</h2></summary>
            <div class="table-container">
                <table>
                    <thead><tr>{table_head_html}</tr></thead>
                    <tbody>{table_body_html}</tbody>
                </table>
            </div>
        </details>
        """

    # Generate HTML for Final Layers
    if "final" in all_reps_by_layer:
        final_reps = all_reps_by_layer["final"]
        header_cols = list(final_reps.keys())
        table_head_html = "<th>Word</th>" + "".join(
            f"<th>{col}</th>" for col in header_cols
        )
        table_body_html = ""
        for i, word in enumerate(words):
            row_html = f"<tr><td class='word-label'>'{word}'<br><span class='pos'>(pos {i})</span></td>"
            for step_name in header_cols:
                img_path = image_paths.get(f"final||{word}||{i}||{step_name}", "")
                row_html += f"<td><img src='{img_path}' title='{step_name}' loading='lazy' onclick='openModal(\"final\", {i}, \"{step_name}\")'></td>"
            row_html += "</tr>"
            table_body_html += row_html

        layers_html += f"""
        <details open>
            <summary><h2>Final Projection</h2></summary>
            <div class="table-container">
                <table>
                    <thead><tr>{table_head_html}</tr></thead>
                    <tbody>{table_body_html}</tbody>
                </table>
            </div>
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
            .bar-label {{ width: 50px; text-align: right; }}
            .bar {{ height: 12px; border-radius: 3px; border: 1px solid #eee; }}
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
            .data-table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
            .data-table th, .data-table td {{ border: 1px solid #eee; padding: 6px; text-align: left; }}
            .data-table th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Full Model Flow Visualization</h1>
            <p><strong>Model:</strong> {model_name}<br><strong>Probe Sentence:</strong> "{probe_sentence}"</p>
            {layers_html}
        </div>

        <!-- Modal HTML Structure -->
        <div id="modal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h2 id="modal-title"></h2>
                    <span class="close" onclick="closeModal()">&times;</span>
                </div>
                <div class="modal-body">
                    <div class="modal-image-container">
                        <img id="modal-img" src="" style="width:100%">
                    </div>
                    <div class="modal-data-container">
                        <h3>Code:</h3>
                        <div id="modal-code" class="code-snippet"></div>
                        <h3>Top 10 Activating Dimensions:</h3>
                        <table id="modal-data-table" class="data-table">
                           <thead><tr><th>Dim</th><th>Value</th></tr></thead>
                           <tbody></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const allData = {all_data_json};
            const allWords = {words_json};
            const allImagePaths = {image_paths_json};
            const codeSnippets = {{
                "Input (x)": "x = token_embedding + positional_embedding",
                "After LN1": "x_ln1 = layernorm(x, ln1_gamma, ln1_beta)",
                "Query (q)": "q = x_ln1 @ W_q + b_q",
                "Key (k)": "k = x_ln1 @ W_k + b_k",
                "Value (v)": "v = x_ln1 @ W_v + b_v",
                "Attn Out (z)": "z = softmax((q @ k.T) / sqrt(d_k)) @ v",
                "Attn Proj": "attn_proj = z @ c_proj_w + c_proj_b",
                "After Resid1": "x = x + attn_proj",
                "After LN2": "x_ln2 = layernorm(x, ln2_gamma, ln2_beta)",
                "MLP Out": "mlp_out = gelu(x_ln2 @ mlp_fc_w + mlp_fc_b) @ mlp_proj_w + mlp_proj_b",
                "Block Output": "x = x_after_attn + mlp_out",
                "After Final LN": "x_final = layernorm(x, ln_f_gamma, ln_f_beta)",
                "Logits (Projected)": "logits = x_final @ lm_head.T"
            }};

            function openModal(layerKey, wordIndex, stepName) {{
                const word = allWords[wordIndex];
                const vector = allData[layerKey][stepName][wordIndex];

                // Construct the string key to look up the image path
                const key = `${{layerKey}}||${{word}}||${{wordIndex}}||${{stepName}}`;
                const imgPath = allImagePaths[key];

                // Populate Modal
                document.getElementById('modal-title').innerText = `${{stepName}} for '${{word}}' (Layer ${{layerKey.split('_')[1] || 'Final'}})`;
                document.getElementById('modal-img').src = imgPath;
                document.getElementById('modal-code').innerText = codeSnippets[stepName];

                // Populate data table
                const tableBody = document.querySelector("#modal-data-table tbody");
                tableBody.innerHTML = "";
                const sortedDims = vector.map((val, i) => ([i, val]))
                                       .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

                for(let i=0; i<10; i++) {{
                    const [dim, val] = sortedDims[i];
                    const row = tableBody.insertRow();
                    row.innerHTML = `<td>${{dim}}</td><td>${{val.toFixed(6)}}</td>`;
                }}

                document.getElementById('modal').style.display = 'block';
            }}

            function closeModal() {{
                document.getElementById('modal').style.display = 'none';
            }}

            // Close modal if user clicks outside of it
            window.onclick = function(event) {{
                const modal = document.getElementById('modal');
                if (event.target == modal) {{
                    modal.style.display = "none";
                }}
            }}
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
