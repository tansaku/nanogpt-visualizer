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
                for i in range(min(25, len(vocab))):
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

    # Clean up training text before tokenizing
    training_text = training_text.replace("<|endoftext|>", " ")

    # Tokenize using same logic as WordTokenizer
    import re

    def tokenize(text):
        return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

    training_tokens = set(tokenize(training_text))
    training_tokens.add("<|endoftext|>")  # Add manually if needed

    print(f"✅ Found {len(training_tokens)} unique training tokens")

    tokens_in_vocab = set(vocab.values())
    tokens_not_in_training = tokens_in_vocab - training_tokens

    if tokens_not_in_training:
        if len(tokens_not_in_training) > 0:
            print("\n🚨 WARNING: Some vocab tokens do not appear in training data")
            for i, token in enumerate(sorted(tokens_not_in_training)):
                print(f"  {i}: {token}")
                if i == 9:
                    print(f"  ... and {len(tokens_not_in_training) - 10} more (if any)")
                    break

    return training_tokens

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
    # Store actual values for data display
    dimension_data = {"positive": [], "negative": []}

    for idx, score in zip(pos_indices, pos_scores):
        if score > 0:
            token_str = str(vocab.get(idx, f"token_{idx}"))
            word_frequencies[token_str] = float(score)
            word_colors[token_str] = "#000000"
            dimension_data["positive"].append(
                {"word": token_str, "value": float(score), "index": int(idx)}
            )

    for idx, score in zip(neg_indices, neg_scores):
        if dim_values[idx] < 0:
            token_str = str(vocab.get(idx, f"token_{idx}"))
            if token_str not in word_frequencies:
                word_frequencies[token_str] = float(score)
                word_colors[token_str] = "#CC0000"
                dimension_data["negative"].append(
                    {
                        "word": token_str,
                        "value": float(dim_values[idx]),
                        "index": int(idx),
                    }
                )

    if not word_frequencies:
        print(f"No valid tokens for dimension {dimension}")
        return None

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

    return dimension_data


def main():
    # Get paths from environment variables
    checkpoint_path = os.environ.get("MODEL")
    training_data_path = os.environ.get("TRAINING_DATA")

    if not checkpoint_path:
        print("Error: MODEL environment variable not set")
        print("Please set MODEL to the path of your checkpoint file")
        sys.exit(1)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Extract model name from checkpoint path
    # e.g., /path/to/knock_6_1_96_words/ckpt.pt -> knock_6_1_96_words
    model_dir = os.path.basename(os.path.dirname(checkpoint_path))

    embeddings, vocab_size, model_args = load_checkpoint(checkpoint_path)
    nanogpt_path = os.environ.get("NANOGPT_PATH")
    vocab = create_vocabulary_mapping(vocab_size, nanogpt_path)

    training_tokens = None
    if training_data_path:
        training_tokens = analyze_training_data(training_data_path, vocab)

    # Create model-specific output directory
    output_dir = os.path.join("visualizations", model_dir, "embedding_wordmaps")
    os.makedirs(output_dir, exist_ok=True)

    n_embd = embeddings.shape[1]
    # Generate all dimensions instead of just first 5
    dimensions_to_visualize = n_embd

    print(f"Creating visualizations for all {dimensions_to_visualize} dimensions...")
    print(f"Output directory: {output_dir}")

    # Collect dimension data for HTML
    all_dimension_data = {}

    for dim in range(dimensions_to_visualize):
        dimension_data = create_word_cloud(embeddings, vocab, dim, output_dir)
        if dimension_data:
            all_dimension_data[dim] = dimension_data
        if (dim + 1) % 10 == 0:
            print(f"  Completed {dim + 1}/{dimensions_to_visualize} dimensions")

    # Generate HTML index page
    generate_html_index(
        output_dir,
        model_dir,
        model_args,
        vocab_size,
        dimensions_to_visualize,
        all_dimension_data,
    )

    print(f"\nDone! Visualizations saved to: {output_dir}/")
    print(f"Created {dimensions_to_visualize} word cloud images")
    print(f"View at: {output_dir}/index.html")


def generate_html_index(
    output_dir, model_name, model_args, vocab_size, num_dimensions, dimension_data
):
    """Generate an HTML index page for viewing all embedding dimensions."""

    # Convert dimension data to JSON for embedding in HTML
    import json

    dimension_data_json = json.dumps(dimension_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embedding Visualizations - {model_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1800px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        .header p {{
            margin: 5px 0;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(6, 1fr);
            gap: 20px;
            margin-top: 20px;
        }}
        .dimension-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
            cursor: pointer;
        }}
        .dimension-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .dimension-card img {{
            width: 100%;
            height: 150px;
            object-fit: cover;
        }}
        .dimension-card .info {{
            padding: 15px;
            text-align: center;
        }}
        .dimension-card .info h3 {{
            margin: 0 0 5px 0;
            color: #333;
        }}
        .modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            justify-content: center;
            align-items: center;
            padding: 20px;
            box-sizing: border-box;
        }}
        .modal-content {{
            background: white;
            border-radius: 8px;
            max-width: 90%;
            max-height: 90%;
            overflow: auto;
            position: relative;
        }}
        .modal .close {{
            position: absolute;
            top: 10px;
            right: 20px;
            color: #666;
            font-size: 30px;
            cursor: pointer;
            z-index: 1001;
        }}
        .modal-header {{
            padding: 20px;
            border-bottom: 1px solid #eee;
            background: #f8f9fa;
        }}
        .modal-body {{
            padding: 20px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .modal-image {{
            flex: 1;
            min-width: 300px;
        }}
        .modal-image img {{
            width: 100%;
            border-radius: 4px;
        }}
        .modal-data {{
            flex: 1;
            min-width: 300px;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        .data-table th {{
            background: #667eea;
            color: white;
            padding: 10px;
            text-align: left;
        }}
        .data-table td {{
            padding: 8px 10px;
            border-bottom: 1px solid #eee;
        }}
        .data-table .positive {{
            color: #000;
        }}
        .data-table .negative {{
            color: #CC0000;
        }}
        .data-section {{
            margin-bottom: 20px;
        }}
        .data-section h4 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #495057;
        }}
        .stat-card .value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .button-group {{
            margin: 10px 0;
            text-align: center;
        }}
        .button-group button {{
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 0 5px;
            border-radius: 4px;
            cursor: pointer;
        }}
        .button-group button:hover {{
            background: #5a67d8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Embedding Visualizations</h1>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Vocabulary Size:</strong> {vocab_size:,} tokens</p>
            <p><strong>Embedding Dimensions:</strong> {num_dimensions}</p>
        </div>

        <div class="content">
            <div class="stats">
                <div class="stat-card">
                    <h3>Vocabulary Size</h3>
                    <div class="value">{vocab_size:,}</div>
                </div>
                <div class="stat-card">
                    <h3>Embedding Dimensions</h3>
                    <div class="value">{num_dimensions}</div>
                </div>
                <div class="stat-card">
                    <h3>Total Parameters</h3>
                    <div class="value">{vocab_size * num_dimensions:,}</div>
                </div>
            </div>

            <h2>Embedding Dimensions</h2>
            <p>Click on any dimension to view the word cloud and exact embedding values</p>

            <div class="grid">"""

    # Add dimension cards
    for dim in range(num_dimensions):
        html_content += f"""
                <div class="dimension-card" onclick="openModal({dim})">
                    <img src="dimension_{dim}.png" alt="Dimension {dim}">
                    <div class="info">
                        <h3>Dimension {dim}</h3>
                    </div>
                </div>"""

    html_content += f"""
            </div>
        </div>
    </div>

    <div class="modal" id="modal" onclick="closeModal(event)">
        <div class="modal-content" onclick="event.stopPropagation()">
            <span class="close" onclick="closeModal()">&times;</span>
            <div class="modal-header">
                <h2 id="modal-title">Dimension X</h2>
                <div class="button-group">
                    <button onclick="showSection('image')">Word Cloud</button>
                    <button onclick="showSection('data')">Exact Values</button>
                    <button onclick="showSection('both')">Both</button>
                </div>
            </div>
            <div class="modal-body">
                <div class="modal-image" id="modal-image-section">
                    <img id="modal-img" src="" alt="">
                </div>
                <div class="modal-data" id="modal-data-section">
                    <div class="data-section">
                        <h4>Highest Positive Values</h4>
                        <table class="data-table" id="positive-table">
                            <thead>
                                <tr><th>Word</th><th>Value</th><th>Token Index</th></tr>
                            </thead>
                            <tbody id="positive-tbody"></tbody>
                        </table>
                    </div>
                    <div class="data-section">
                        <h4>Most Negative Values</h4>
                        <table class="data-table" id="negative-table">
                            <thead>
                                <tr><th>Word</th><th>Value</th><th>Token Index</th></tr>
                            </thead>
                            <tbody id="negative-tbody"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const dimensionData = {dimension_data_json};
        let currentView = 'both';

        function openModal(dimension) {{
            const modal = document.getElementById('modal');
            const modalImg = document.getElementById('modal-img');
            const modalTitle = document.getElementById('modal-title');

            modalImg.src = `dimension_${{dimension}}.png`;
            modalTitle.textContent = `Dimension ${{dimension}}`;

            loadDimensionData(dimension);
            showSection(currentView);
            modal.style.display = 'flex';
        }}

        function loadDimensionData(dimension) {{
            const data = dimensionData[dimension];
            if (!data) {{
                console.log('No data for dimension', dimension);
                return;
            }}

            const positiveTbody = document.getElementById('positive-tbody');
            const negativeTbody = document.getElementById('negative-tbody');

            positiveTbody.innerHTML = '';
            negativeTbody.innerHTML = '';

            if (data.positive) {{
                data.positive.forEach(item => {{
                    const row = positiveTbody.insertRow();
                    row.innerHTML = `<td class="positive">${{item.word}}</td><td class="positive">${{item.value.toFixed(4)}}</td><td class="positive">${{item.index}}</td>`;
                }});
            }}

            if (data.negative) {{
                data.negative.forEach(item => {{
                    const row = negativeTbody.insertRow();
                    row.innerHTML = `<td class="negative">${{item.word}}</td><td class="negative">${{item.value.toFixed(4)}}</td><td class="negative">${{item.index}}</td>`;
                }});
            }}
        }}

        function showSection(section) {{
            currentView = section;
            const imageSection = document.getElementById('modal-image-section');
            const dataSection = document.getElementById('modal-data-section');

            if (section === 'image') {{
                imageSection.style.display = 'block';
                dataSection.style.display = 'none';
            }} else if (section === 'data') {{
                imageSection.style.display = 'none';
                dataSection.style.display = 'block';
            }} else {{ // both
                imageSection.style.display = 'block';
                dataSection.style.display = 'block';
            }}
        }}

        function closeModal(event) {{
            if (event && event.target !== event.currentTarget) return;
            document.getElementById('modal').style.display = 'none';
        }}

        // Close modal on escape key
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') {{
                closeModal();
            }}
        }});
    </script>
</body>
</html>"""

    # Write HTML file
    html_path = os.path.join(output_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated HTML index: {html_path}")


if __name__ == "__main__":
    main()
