# NanoGPT Transformer Visualizer

This repository contains a suite of Python scripts to generate in-depth visualizations of the internal workings of a NanoGPT model. It is designed to trace the flow of information step-by-step through the transformer architecture, from the initial embeddings to the final prediction.

The primary output is a detailed HTML page (`index.html`) that shows the state of the model's representations at every stage of processing for a given input sentence.

![Visualization Screenshot](https://user-images.githubusercontent.com/123/456-example.png) <!-- Placeholder image -->

## Features

- **Full Model Flow:** Visualize the transformation of token representations through every layer of the transformer.
- **Input Analysis:** See the difference between a token's raw embedding and its position-aware combined embedding.
- **Attention Deep Dive:** Inspect attention patterns, Q, K, and V vectors, and their compositions.
- **Final Prediction Breakdown:** Analyze the model's final output, including the dot product breakdown that determines the next token prediction.
- **Interactive HTML Report:** An easy-to-navigate HTML page with collapsible sections, tooltips, and pop-up modals with more details.
- **GitHub Pages Deployment:** Automatically deploy your generated visualizations to a live website.

## How to Use

### 1. Prerequisites

- Python 3.9+
- [Pipenv](https://pipenv.pypa.io/en/latest/) for environment management.

### 2. Setup

First, clone this repository:
```bash
git clone https://github.com/tansaku/nanogpt-visualizer.git
cd nanogpt-visualizer
```

Install the required dependencies using Pipenv:
```bash
pipenv install
pipenv shell
```

### 3. Model and Data Setup

This tool expects your NanoGPT model files to be organized in a specific way within the `visualizations/` directory.

1.  Create a directory for your model, for example: `visualizations/my_model_name/`.
2.  Place your trained model checkpoint file and tokenizer metadata file in this directory:
    - `visualizations/my_model_name/ckpt.pt`
    - `visualizations/my_model_name/meta.pkl`

Your directory structure should look like this:
```
visualizations/
└── my_model_name/
    ├── ckpt.pt
    └── meta.pkl
```

### 4. Generate the Visualization

The main script to run is `visualize_attention_flow_wordmaps.py`. It requires three environment variables to be set.

```bash
# Set environment variables
export MODEL="visualizations/my_model_name/ckpt.pt"
export NANOGPT_META_PATH="visualizations/my_model_name/meta.pkl"
export PROBE_SENTENCE="the quick brown fox"

# Run the script
python visualize_attention_flow_wordmaps.py
```

-   `MODEL`: The path to your model checkpoint file.
-   `NANOGPT_META_PATH`: The path to your tokenizer metadata file.
-   `PROBE_SENTENCE`: The input sentence you want to visualize.

The script will generate a comprehensive set of images and an `index.html` file inside a new directory: `visualizations/my_model_name/full_model_flow/`.

### 5. View and Deploy

You can open `visualizations/my_model_name/full_model_flow/index.html` in your local browser to view the interactive report.

This repository is configured with a GitHub Action to automatically deploy the contents of the `visualizations/knock_6_1_36_words/full_model_flow` directory to GitHub Pages. To use this for your own model:

1.  Update the path in `.github/workflows/deploy.yml` to point to your model's output directory.
2.  Commit and push all the generated files from your model's `full_model_flow` directory.
3.  The visualization will be available at `https://<your-username>.github.io/nanogpt-visualizer/`.

## Included Visualization Scripts

This repository contains several scripts for different kinds of analysis:

-   `visualize_attention_flow_wordmaps.py`: The main script for the full, end-to-end model visualization.
-   `visualize_tokens.py`: Generates word clouds for each dimension of the token embedding space.
-   `visualize_positions.py`: Generates word clouds for each dimension of the positional embedding space.
-   `visualize_qkv_decomposition.py`: Decomposes Q, K, and V transformations.
-   ... and others for more specific deep dives.

Each script can be explored for more targeted analysis of the model's components.
