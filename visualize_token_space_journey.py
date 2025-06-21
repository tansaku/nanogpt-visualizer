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
import json
import plotly.graph_objects as go
import plotly.colors
import html
import umap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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


# --- Interactive Plot Generation ---


def create_interactive_plot(
    base_map_2d,
    vocab_itos,
    umap_reducer,
    title,
    probe_word_vectors=None,
    probe_word_labels=None,
    original_probe_word_vectors_2d=None,
    previous_probe_word_vectors_2d=None,
    final_output_vector=None,
    top_k_vectors_2d=None,
    top_k_labels=None,
    dot_product_vector=None,
    dot_product_target_word_vec_2d=None,
    dot_product_target_word_label=None,
    key_word_vectors_2d=None,
    key_word_labels=None,
):
    """
    Creates a single, versatile interactive 2D scatter plot using Plotly.

    This function can handle:
    - Standard token journey steps.
    - Final logit predictions plot.
    - Dot product breakdown plot.
    """
    fig = go.Figure()

    # 1. Plot the entire vocabulary as a background
    fig.add_trace(
        go.Scatter(
            x=base_map_2d[:, 0],
            y=base_map_2d[:, 1],
            mode="markers",
            marker=dict(color="lightgray", size=3, opacity=0.3),
            hoverinfo="text",
            text=[f"Vocab: {word}" for word in vocab_itos],
            name="Vocabulary",
        )
    )

    # 2. Plot key vocabulary words for context
    if key_word_vectors_2d is not None and key_word_labels:
        fig.add_trace(
            go.Scatter(
                x=key_word_vectors_2d[:, 0],
                y=key_word_vectors_2d[:, 1],
                mode="text",
                text=key_word_labels,
                textposition="top center",
                textfont=dict(size=10, color="#555"),
                hoverinfo="text",
                hovertext=key_word_labels,
                name="Keywords",
            )
        )

    # 3. Project the current probe word vectors (if this is a journey step)
    probe_word_vectors_2d = None
    if probe_word_vectors is not None:
        probe_word_vectors_2d = (
            umap_reducer.transform(probe_word_vectors)
            if probe_word_vectors.ndim > 1
            else umap_reducer.transform([probe_word_vectors])
        )

        # 4. Draw arrows and 'ghost' dots from previous positions
        if previous_probe_word_vectors_2d is not None:
            colors = plotly.colors.qualitative.Plotly
            for i in range(len(probe_word_vectors_2d)):
                color = colors[i % len(colors)]
                # Ghost dot
                fig.add_trace(
                    go.Scatter(
                        x=[previous_probe_word_vectors_2d[i, 0]],
                        y=[previous_probe_word_vectors_2d[i, 1]],
                        mode="markers",
                        marker=dict(
                            color=color,
                            symbol="circle-open",
                            size=10,
                            opacity=0.6,
                            line=dict(width=1.5),
                        ),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )
                # Arrow
                fig.add_annotation(
                    x=probe_word_vectors_2d[i, 0],
                    y=probe_word_vectors_2d[i, 1],
                    ax=previous_probe_word_vectors_2d[i, 0],
                    ay=previous_probe_word_vectors_2d[i, 1],
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=1,
                    arrowcolor=color,
                    opacity=0.7,
                )

        # 5. Plot the probe words at their current and original positions
        colors = plotly.colors.qualitative.Plotly
        for i, label in enumerate(probe_word_labels):
            x, y = probe_word_vectors_2d[i]
            color = colors[i % len(colors)]

            # Plot original position as an 'x'
            if original_probe_word_vectors_2d is not None:
                orig_x, orig_y = original_probe_word_vectors_2d[i]
                fig.add_trace(
                    go.Scatter(
                        x=[orig_x],
                        y=[orig_y],
                        mode="markers",
                        marker=dict(symbol="x", color=color, size=8, opacity=0.8),
                        hoverinfo="text",
                        hovertext=f"{label} (Original Embedding)",
                        name=f"{label} (Original)",
                        showlegend=False,
                    )
                )

            # Plot current position as a solid dot
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers+text",
                    marker=dict(
                        color=color, size=12, line=dict(width=2, color="DarkSlateGrey")
                    ),
                    text=label,
                    textposition="top right",
                    textfont=dict(color=color, size=12),
                    hoverinfo="text",
                    hovertext=f"Probe: {label}",
                    name=label,
                    showlegend=True,
                )
            )

    # 6. Plot the Top-K predicted tokens (for the logit plot)
    if top_k_vectors_2d is not None:
        colors = plotly.colors.sequential.Viridis
        for i, label in enumerate(top_k_labels):
            x, y = top_k_vectors_2d[i]
            size = 20 - i * 1.5  # Make top prediction largest
            color = colors[int(i * (len(colors) / len(top_k_labels)))]
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    marker=dict(
                        color=color,
                        size=size,
                        line=dict(width=1, color="black"),
                    ),
                    hoverinfo="text",
                    hovertext=f"#{i+1}: {label}",
                    name=f"#{i+1}: {label}",
                    showlegend=True,
                )
            )

    # 7. Plot the final output vector
    final_vector_2d = None
    if final_output_vector is not None:
        final_vector_2d = umap_reducer.transform([final_output_vector])[0]
        fig.add_trace(
            go.Scatter(
                x=[final_vector_2d[0]],
                y=[final_vector_2d[1]],
                mode="markers+text",
                marker=dict(
                    symbol="star",
                    color="red",
                    size=15,
                    line=dict(width=1, color="black"),
                ),
                text=["Final Output"],
                textposition="top right",
                name="Model's Final Output",
                showlegend=True,
            )
        )

    # 8. Handle Dot Product visualization
    if dot_product_vector is not None:
        # We need the final_vector_2d for the arrow start point
        if final_output_vector is not None:
            final_vector_2d = umap_reducer.transform([final_output_vector])[0]

        if final_vector_2d is not None:
            # This vector is not in the original space, it's a contribution vector.
            # We can't directly plot it. Instead, we draw a line from the final
            # output vector to the embedding of the predicted word.
            fig.add_annotation(
                x=dot_product_target_word_vec_2d[0],  # End at the target word
                y=dot_product_target_word_vec_2d[1],
                ax=final_vector_2d[0],  # Start at the final output vector
                ay=final_vector_2d[1],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=2,
                arrowwidth=2,
                arrowcolor="red",
                opacity=0.8,
            )
        # Highlight the target word's embedding
        fig.add_trace(
            go.Scatter(
                x=[dot_product_target_word_vec_2d[0]],
                y=[dot_product_target_word_vec_2d[1]],
                mode="markers",
                marker=dict(
                    symbol="diamond",
                    color="red",
                    size=16,
                    line=dict(width=2, color="black"),
                ),
                hoverinfo="text",
                hovertext=f"Predicted: {dot_product_target_word_label}",
                name=f"Predicted: {dot_product_target_word_label}",
                showlegend=True,
            )
        )

    # Final layout adjustments
    # Set fixed range to show the full vocabulary map initially
    x_range = [base_map_2d[:, 0].min() - 1, base_map_2d[:, 0].max() + 1]
    y_range = [base_map_2d[:, 1].min() - 1, base_map_2d[:, 1].max() + 1]

    fig.update_layout(
        title=dict(text=title, x=0.5),
        showlegend=True,
        width=600,  # Give plots a fixed width
        height=700,
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        margin=dict(l=40, r=40, b=40, t=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
    )
    fig.update_xaxes(gridcolor="lightgrey", zerolinecolor="grey")
    fig.update_yaxes(gridcolor="lightgrey", zerolinecolor="grey")

    plot_html = fig.to_html(full_html=False, include_plotlyjs=False)
    return plot_html, probe_word_vectors_2d


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
    # Start with the probe words
    key_words_to_highlight = list(set(words))

    # NEW: Find diverse words using k-means to get a better map of the space
    print("Finding diverse keywords using K-Means clustering...")
    n_clusters = 15  # How many representative words to find
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(
        base_map_2d
    )

    # Find the word closest to each cluster center
    for i in range(n_clusters):
        center = kmeans.cluster_centers_[i]
        # Calculate Euclidean distance from the center to all points
        distances = np.linalg.norm(base_map_2d - center, axis=1)
        closest_word_index = np.argmin(distances)
        key_words_to_highlight.append(itos[closest_word_index])

    key_words_to_highlight = sorted(
        list(set(key_words_to_highlight))
    )  # remove duplicates and sort
    print(f"Keywords for highlighting: {key_words_to_highlight}")

    key_word_indices = [stoi.get(w) for w in key_words_to_highlight if w in stoi]
    key_word_vectors_2d = base_map_2d[key_word_indices] if key_word_indices else None

    # --- Start of multi-layer processing ---
    num_layers = get_num_layers(model_state)
    print(f"Model has {num_layers} layers. Visualizing full flow.")

    # NEW: Get both token-only and combined representations
    token_only_reps = get_token_only_representations(token_ids, wte)
    original_probe_word_vectors_2d = umap_reducer.transform(
        token_only_reps
    )  # Get original 2D positions
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

    # Generate interactive plots for each step, each word, each layer
    plot_htmls = {}

    # Define the order of layers for processing and generating HTML
    layer_keys = ["input"] + [f"layer_{i}" for i in range(num_layers)] + ["final"]

    previous_step_vectors_2d = {}  # To store 2D vectors for drawing arrows

    for layer_key in layer_keys:
        if layer_key not in all_reps_by_layer:
            continue

        representations = all_reps_by_layer[layer_key]
        print(f"Generating plots for: {layer_key}")
        for step_name, all_word_vectors in representations.items():
            plot_title = f"{step_name} (Layer: {layer_key})"
            plot_html, current_vectors_2d = None, None

            # Special handling for different plot types
            if step_name == "Final Linear Layer":
                # Plot the top K predicted token embeddings on the UMAP
                last_token_logits = all_word_vectors[-1]
                top_k_indices = np.argsort(-last_token_logits)[:10]
                top_k_words = [itos[i] for i in top_k_indices]
                top_k_vectors_2d = base_map_2d[top_k_indices]
                final_output_vector = all_reps_by_layer["final"][
                    "After Final Layer Normalisation"
                ][-1]

                plot_html, _ = create_interactive_plot(
                    base_map_2d=base_map_2d,
                    vocab_itos=itos,
                    umap_reducer=umap_reducer,
                    title="Top 10 Predictions on Vocab Map",
                    top_k_vectors_2d=top_k_vectors_2d,
                    top_k_labels=top_k_words,
                    final_output_vector=final_output_vector,
                    key_word_vectors_2d=key_word_vectors_2d,
                    key_word_labels=key_words_to_highlight,
                )

            elif step_name.startswith("Dot Product Breakdown"):
                final_output_vector = all_reps_by_layer["final"][
                    "After Final Layer Normalisation"
                ][-1]
                predicted_word_idx = stoi[top_prediction_word]
                predicted_word_vec_2d = base_map_2d[predicted_word_idx]

                plot_html, _ = create_interactive_plot(
                    base_map_2d=base_map_2d,
                    vocab_itos=itos,
                    umap_reducer=umap_reducer,
                    title=plot_title,
                    dot_product_vector=all_word_vectors[-1],
                    dot_product_target_word_vec_2d=predicted_word_vec_2d,
                    dot_product_target_word_label=top_prediction_word,
                    final_output_vector=final_output_vector,
                    probe_word_vectors=all_reps_by_layer["final"][
                        "After Final Layer Normalisation"
                    ],
                    probe_word_labels=words,
                    original_probe_word_vectors_2d=original_probe_word_vectors_2d,
                    key_word_vectors_2d=key_word_vectors_2d,
                    key_word_labels=key_words_to_highlight,
                )

            else:
                # Standard journey plot
                previous_vectors_2d = previous_step_vectors_2d.get(layer_key)
                plot_html, current_vectors_2d = create_interactive_plot(
                    base_map_2d=base_map_2d,
                    vocab_itos=itos,
                    probe_word_vectors=all_word_vectors,
                    probe_word_labels=words,
                    umap_reducer=umap_reducer,
                    title=plot_title,
                    original_probe_word_vectors_2d=original_probe_word_vectors_2d,
                    previous_probe_word_vectors_2d=previous_vectors_2d,
                    key_word_vectors_2d=key_word_vectors_2d,
                    key_word_labels=key_words_to_highlight,
                )
                if current_vectors_2d is not None:
                    previous_step_vectors_2d[layer_key] = current_vectors_2d

            # Store the generated HTML for the plot
            if plot_html:
                if layer_key not in plot_htmls:
                    plot_htmls[layer_key] = {}
                plot_htmls[layer_key][step_name] = plot_html

    # Generate HTML page to display everything
    generate_html_page(
        output_dir,
        model_dir,
        probe_sentence,
        words,
        plot_htmls,
        all_reps_by_layer,
        all_attn_by_layer,
        final_prediction_data,
        top_prediction_word,
    )

    print("\nDone! Full model flow visualization created.")
    print(f"View the interactive summary at: {os.path.join(output_dir, 'index.html')}")


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
    plot_htmls,
    all_reps_by_layer,
    all_attn_by_layer,
    final_prediction_data,
    top_prediction_word,
):
    """Generates an HTML page with collapsible sections for each layer."""

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
        is_open = layer_key in ["input", "final"] or layer_key == "layer_0"
        details_options = "open" if is_open else ""

        representations = all_reps_by_layer[layer_key]

        header_cols = list(representations.keys())
        table_head_html = "".join(f"<th>{html.escape(col)}</th>" for col in header_cols)

        row_html = "<tr>"
        for step_name in header_cols:
            plot_div = plot_htmls.get(layer_key, {}).get(step_name, "")
            row_html += f"<td>{plot_div}</td>"
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
    prediction_html = ""
    if final_prediction_data:
        prediction_html += "<div class='prediction-container'>"
        prediction_html += "<div class='prediction-bar-container'>"
        for word, prob in zip(
            final_prediction_data["words"], final_prediction_data["probabilities"]
        ):
            prediction_html += f"""
                <div class="bar-row">
                    <span class="bar-label">{html.escape(word)}</span>
                    <div class="bar" style="width: {prob*100*4}px; background-color: rgba(220, 53, 69, {0.2 + prob*0.8});"></div>
                    <span class="bar-value">{(prob*100):.2f}%</span>
                </div>
            """
        prediction_html += "</div></div>"

        layers_html += f"""
        <details open>
            <summary><h2>Next Token Prediction</h2></summary>
            <div style="padding: 1em; text-align: left; max-width: 80%; margin: 1em auto; background: #fff8e1; border-left: 5px solid #ffc107; border-radius: 4px;">
                <strong>How to Interpret the Final Plots:</strong>
                <ul style="margin-top: 0.5em;">
                    <li>The <strong>'Top 10 Predictions'</strong> plot shows the location of the best candidate words in the vocabulary space, relative to the model's final output state (the red star).</li>
                    <li>The <strong>'Dot Product Breakdown'</strong> plot shows the final step. The red arrow from the 'Final Output' to the predicted word ('{top_prediction_word}') illustrates the relationship that matters: the dot product.</li>
                    <li><strong>Important:</strong> Proximity in this 2D UMAP projection does not guarantee a high dot product score in the original high-dimensional space. The UMAP plot is for visualizing the general structure, not for definitively judging which token will be chosen. The highest logit (from the dot product) wins.</li>
                </ul>
            </div>
            {prediction_html}
        </details>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Full Model Flow Visualization</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; margin: 2em; background: #f0f2f5; color: #333; }}
            h1, h2, p {{ text-align: center; }}
            details {{ background: white; border-radius: 8px; margin-bottom: 1em; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
            summary {{ font-size: 1.5em; font-weight: bold; padding: 0.8em; cursor: pointer; }}
            .container {{ max-width: 98%; margin: auto; }}
            .table-container {{ overflow-x: auto; padding: 1em; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 1em; }}
            th, td {{ border: 1px solid #ddd; padding: 0; text-align: center; vertical-align: middle; min-width: 620px; }}
            th {{ padding: 8px; background-color: #f8f9fa; font-size: 1em; }}
            .word-label {{ font-weight: bold; font-size: 1.1em; }}
            .attention-bar-container {{ display: flex; flex-direction: column; gap: 4px; }}
            .bar-row {{ display: flex; align-items: center; gap: 5px; font-size: 0.9em; }}
            .bar-label {{ width: 100px; text-align: right; }}
            .bar {{ height: 18px; border-radius: 3px; border: 1px solid #eee; }}
            .bar-value {{ font-family: monospace; }}
            .prediction-container {{ padding: 2em; }}
            .prediction-bar-container {{ display: flex; flex-direction: column; gap: 6px; max-width: 500px; margin: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Interactive Token Journey Visualization</h1>
            <p><strong>Model:</strong> {model_name}<br><strong>Probe Sentence:</strong> "{probe_sentence}"</p>
            <p style="text-align:center; max-width: 80%; margin: 1em auto; font-style: italic; color: #666;">
                Each plot below is interactive. You can pan, zoom, and hover over points to see the corresponding words. The initial view shows the entire vocabulary space. For each probe word, the solid dot is its current position, and the 'x' is its original embedding position.
            </p>
            {layers_html}
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
