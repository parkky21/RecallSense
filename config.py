"""
Configuration file for image retrieval system.
Enable/disable embedding models here.
"""

# Embedding Models Configuration
# Set to True to enable a model, False to disable
EMBEDDING_MODELS = {
    "qwen": False,      # Qwen/Qwen3-Embedding-0.6B
    "gte": True,        # Alibaba-NLP/gte-multilingual-base (DEFAULT)
    "gemma": True,     # google/embeddinggemma-300m
}

# Model configurations
MODEL_CONFIGS = {
    "qwen": {
        "name": "Qwen/Qwen3-Embedding-0.6B",
        "model_class": "QwenEmbeddingModel",
        "encode_method": "encode",
        "encode_kwargs": {},
        "query_encode_method": "encode",
        "query_encode_kwargs": {"prompt_name": "query"},
        "similarity_method": "custom",  # Uses cosine similarity
    },
    "gte": {
        "name": "Alibaba-NLP/gte-multilingual-base",
        "model_class": "GTEEmbeddingModel",
        "encode_method": "encode",
        "encode_kwargs": {"normalize_embeddings": True},
        "query_encode_method": "encode",
        "query_encode_kwargs": {"normalize_embeddings": True},
        "similarity_method": "custom",  # Uses cosine similarity
        "trust_remote_code": True,
    },
    "gemma": {
        "name": "google/embeddinggemma-300m",
        "model_class": "GemmaEmbeddingModel",
        "encode_method": "encode_document",
        "encode_kwargs": {},
        "query_encode_method": "encode_query",
        "query_encode_kwargs": {},
        "similarity_method": "model",  # Uses model's similarity method
    },
}

# Get list of enabled models
def get_enabled_models():
    """Return list of enabled model keys."""
    return [key for key, enabled in EMBEDDING_MODELS.items() if enabled]

# Get model config
def get_model_config(model_key):
    """Get configuration for a specific model."""
    return MODEL_CONFIGS.get(model_key, {})

