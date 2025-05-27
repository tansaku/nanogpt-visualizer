# NanoGPT Token Visualizer (Minimal)

A minimal tool to visualize token embeddings from NanoGPT checkpoint files.

## What it does

1. Loads a NanoGPT `.pt` checkpoint file
2. Extracts token embeddings
3. Creates word cloud visualizations for embedding dimensions

## Installation

```bash
pipenv install
pipenv shell
```

## Usage

```bash
python visualize_tokens.py /path/to/checkpoint.pt
```

## Output

Creates word cloud images showing the most influential tokens for each embedding dimension.
