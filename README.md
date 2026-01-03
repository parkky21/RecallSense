# Image Retrieval System

An image retrieval system that uses BLIP for image captioning and Qwen3-Embedding-0.6B for semantic search.

## Setup

1. Install dependencies using UV:
```bash
uv sync
```

2. Make sure you have a CUDA-compatible GPU for optimal performance.

## Usage

### 1. Index Images

Run the indexing script to process all images in a folder:

```bash
uv run index_images.py
```

The script will:
- Ask for a folder path containing images
- Generate captions for all images using BLIP
- Create embeddings using Qwen3-Embedding-0.6B
- Store embeddings and metadata in `embeddings_data/` folder

### 2. Search Images

Run the search script to find similar images:

```bash
uv run search_images.py
```

The script will:
- Ask for a search query
- Find the top 3 most similar images based on semantic similarity
- Display the images with their captions and similarity scores

## Files

- `index_images.py`: Indexes images by generating captions and embeddings
- `search_images.py`: Searches through indexed images and displays results
- `embeddings_data/`: Directory where embeddings and metadata are stored
  - `embeddings_qwen.npy`: NumPy array of Qwen embeddings
  - `embeddings_gte.npy`: NumPy array of GTE-multilingual embeddings
  - `metadata.json`: JSON file with image paths, captions, and timing information

## Requirements

- Python >= 3.12
- CUDA-compatible GPU (recommended)
- Transformers >= 4.51.0
- Sentence-transformers >= 2.7.0

