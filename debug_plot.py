#!/usr/bin/env python3
"""
Debug script for creating a single interactive 2D plot of word embeddings.
"""

import os
import sys
import torch
import numpy as np
import pickle
import re
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.colors
import umap
from sklearn.cluster import KMeans

load_dotenv()

# --- Copied from visualize_token_space_journey.py ---


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


def get_representations(token_ids, wte, wpe):
    """Gets combined token + positional embeddings."""
    combined_reprs = []
    for i, token_id in enumerate(token_ids):
        token_emb = wte[token_id].numpy()
        pos_emb = wpe[i].numpy() if i < wpe.shape[0] else np.zeros_like(token_emb)
        combined_reprs.append(token_emb + pos_emb)
    return np.array(combined_reprs)


# --- Plotting Function ---


def create_debug_plot(
    base_map_2d,
    vocab_itos,
    probe_vectors_2d,
    probe_labels,
    key_word_vectors_2d,
    key_word_labels,
    title,
):
    """Creates an interactive 2D scatter plot with a vocabulary cloud."""
    fig = go.Figure()

    # 1. Plot the entire vocabulary using Scattergl for performance
    fig.add_trace(
        go.Scattergl(
            x=base_map_2d[:, 0],
            y=base_map_2d[:, 1],
            mode="markers",
            marker=dict(color="#e0e0e0", size=5, opacity=0.6),
            text=[vocab_itos.get(i, "") for i in range(len(base_map_2d))],
            hoverinfo="text",
            name="Vocabulary",
        )
    )

    # 2. Plot key context words
    fig.add_trace(
        go.Scatter(
            x=key_word_vectors_2d[:, 0],
            y=key_word_vectors_2d[:, 1],
            mode="text",
            text=key_word_labels,
            textfont=dict(size=10, color="#666666"),
            hoverinfo="none",
            name="Context Words",
        )
    )

    # 3. Plot the probe words
    colors = plotly.colors.qualitative.Plotly
    for i, label in enumerate(probe_labels):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=[probe_vectors_2d[i, 0]],
                y=[probe_vectors_2d[i, 1]],
                mode="markers+text",
                marker=dict(
                    color=color, size=12, line=dict(width=2, color="DarkSlateGrey")
                ),
                text=label,
                textposition="top center",
                name=label,
            )
        )

    # Final layout updates
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="white",
    )
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor="LightGray", zerolinecolor="Gray"
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor="LightGray", zerolinecolor="Gray"
    )

    return fig


# --- Main Execution ---


def main():
    """Main execution function."""
    # --- Config ---
    checkpoint_path = os.environ.get("MODEL")
    probe_sentence = os.environ.get("PROBE_SENTENCE", "knock knock whos there")

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(
            f"Error: Checkpoint file not found. 'MODEL' env var was '{checkpoint_path}'"
        )
        print(
            "Please ensure the MODEL variable in your .env file points to a valid checkpoint."
        )
        sys.exit(1)

    # --- Load Model and Data ---
    checkpoint = load_checkpoint(checkpoint_path)
    model_args = checkpoint["model_args"]
    stoi, itos = load_tokenizer()

    if not stoi or not itos:
        sys.exit(1)

    model_state = checkpoint["model"]
    wte = model_state["transformer.wte.weight"]
    wpe = model_state["transformer.wpe.weight"]

    # --- Prepare Embeddings ---
    words, token_ids = tokenize_sentence(probe_sentence, stoi)
    probe_vectors = get_representations(token_ids, wte, wpe)

    # --- UMAP Projection ---
    print("Creating 2D UMAP projection of the entire vocabulary...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    base_map_2d = umap_reducer.fit_transform(wte.numpy())
    print("UMAP projection created.")

    probe_vectors_2d = umap_reducer.transform(probe_vectors)

    # --- Select Keywords for Highlighting ---
    print("Finding diverse keywords using K-Means clustering...")
    n_keywords = 20
    kmeans = KMeans(n_clusters=n_keywords, random_state=42, n_init=10)
    kmeans.fit(base_map_2d)

    key_word_indices = []
    for i in range(n_keywords):
        cluster_points = np.where(kmeans.labels_ == i)[0]
        if len(cluster_points) > 0:
            # Find the point closest to the cluster center
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(base_map_2d[cluster_points] - center, axis=1)
            closest_point_idx = cluster_points[np.argmin(distances)]
            key_word_indices.append(closest_point_idx)

    key_word_labels = [itos.get(i, "?") for i in key_word_indices]
    key_word_vectors = wte[key_word_indices].numpy()
    key_word_vectors_2d = umap_reducer.transform(key_word_vectors)

    # --- Generate and Save Plot ---
    print("Generating plot...")
    fig = create_debug_plot(
        base_map_2d=base_map_2d,
        vocab_itos=itos,
        probe_vectors_2d=probe_vectors_2d,
        probe_labels=words,
        key_word_vectors_2d=key_word_vectors_2d,
        key_word_labels=key_word_labels,
        title="Initial Token Embeddings in 2D Vocabulary Space",
    )

    output_path = "debug_plot.html"
    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
