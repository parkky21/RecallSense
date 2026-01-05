# Image Retrieval System

A modular image retrieval system that uses BLIP for image captioning and configurable embedding models for semantic search.

## Features

- **Modular Architecture**: Easily enable/disable embedding models via configuration
- **Multiple Embedding Models**: Support for Qwen, GTE-multilingual, and EmbeddingGemma
- **Flexible Configuration**: Choose which models to use without code changes
- **Incremental Indexing**: Only processes new images, skips already indexed ones
- **On-the-fly Search**: Search additional folders without pre-indexing

## Setup

1. Install dependencies using UV:
```bash
uv sync
```

2. Configure which models to use by editing `config.py`:
```python
EMBEDDING_MODELS = {
    "qwen": False,      # Qwen/Qwen3-Embedding-0.6B
    "gte": True,        # Alibaba-NLP/gte-multilingual-base (DEFAULT)
    "gemma": False,     # google/embeddinggemma-300m
}
```

3. Make sure you have a CUDA-compatible GPU for optimal performance (optional, works on CPU too).

## Usage

### 1. Index Images

Run the indexing script to process all images in a folder:

```bash
uv run index_images.py
```

The script will:
- Show which models are enabled
- Ask for a folder path containing images
- Generate captions for all images using BLIP
- Create embeddings using all enabled models
- Store embeddings and metadata in `embeddings_data/` folder

### 2. Search Images

Run the search script to find similar images:

```bash
uv run search_images.py
```

The script will:
- Show which models are enabled
- Optionally index images from an additional folder
- Ask for a search query
- Find the top 3 most similar images using all enabled models
- Display the images with their captions and similarity scores

## Project Structure

```
ImageRetrival/
├── config.py              # Configuration file - enable/disable models here
├── embedding_models.py    # Modular embedding model classes
├── index_images.py       # Image indexing script
├── search_images.py      # Image search script
├── embeddings_data/      # Stored embeddings and metadata
│   ├── embeddings_*.npy  # NumPy arrays of embeddings (one per model)
│   └── metadata.json     # Image paths, captions, and timing info
└── README.md            # This file
```

## Configuration

### Enabling/Disabling Models

Edit `config.py` to enable or disable embedding models:

```python
EMBEDDING_MODELS = {
    "qwen": False,   # Set to True to enable
    "gte": True,     # Currently enabled (default)
    "gemma": False,  # Set to True to enable
}
```

### Available Models

1. **Qwen** (`qwen`): Qwen/Qwen3-Embedding-0.6B
   - Good accuracy, moderate speed
   - Supports query-specific encoding

2. **GTE-multilingual** (`gte`): Alibaba-NLP/gte-multilingual-base
   - Fastest performance
   - Multilingual support
   - Currently enabled by default

3. **EmbeddingGemma** (`gemma`): google/embeddinggemma-300m
   - Moderate speed
   - Built-in similarity computation
   - Separate query/document encoding

## Files

- `config.py`: Configuration file to enable/disable embedding models
- `embedding_models.py`: Modular embedding model implementations
- `index_images.py`: Indexes images by generating captions and embeddings
- `search_images.py`: Searches through indexed images and displays results
- `embeddings_data/`: Directory where embeddings and metadata are stored
  - `embeddings_*.npy`: NumPy arrays of embeddings (one file per enabled model)
  - `metadata.json`: JSON file with image paths, captions, timing, and enabled models

## Requirements

- Python >= 3.12
- CUDA-compatible GPU (recommended, but works on CPU)
- Transformers >= 4.51.0
- Sentence-transformers >= 3.0.0
- torch >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- pillow >= 10.0.0

## Adding New Models

To add a new embedding model:

1. Add the model configuration to `config.py`:
```python
EMBEDDING_MODELS = {
    "new_model": False,  # Add your new model
}

MODEL_CONFIGS = {
    "new_model": {
        "name": "model-name/path",
        # ... other config
    }
}
```

2. Create a model class in `embedding_models.py`:
```python
class NewModelEmbedding(BaseEmbeddingModel):
    def __init__(self):
        super().__init__("model-name/path")
    # Implement required methods
```

3. Register it in the factory function in `embedding_models.py`:
```python
model_classes = {
    "new_model": NewModelEmbedding,
    # ...
}
```

## Notes

- Only enabled models will be loaded and used
- Embeddings are stored separately for each model
- The system automatically handles missing embeddings (regenerates if needed)
- All enabled models are used simultaneously for comparison

