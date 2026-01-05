"""
Embedding Models Module
Contains classes for different embedding models.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional


class BaseEmbeddingModel:
    """Base class for embedding models."""
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize the embedding model."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, **kwargs)
    
    def encode_documents(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode documents/captions into embeddings."""
        raise NotImplementedError
    
    def encode_query(self, query: str, **kwargs) -> np.ndarray:
        """Encode a query into an embedding."""
        raise NotImplementedError
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          document_embeddings: np.ndarray) -> np.ndarray:
        """Compute similarity between query and documents."""
        raise NotImplementedError


class QwenEmbeddingModel(BaseEmbeddingModel):
    """Qwen embedding model implementation."""
    
    def __init__(self):
        super().__init__("Qwen/Qwen3-Embedding-0.6B")
    
    def encode_documents(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode documents using Qwen model."""
        return self.model.encode(texts, **kwargs)
    
    def encode_query(self, query: str, **kwargs) -> np.ndarray:
        """Encode query using Qwen model with query prompt."""
        return self.model.encode(query, prompt_name="query", **kwargs)
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          document_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity."""
        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = document_embeddings / (np.linalg.norm(document_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        return similarities


class GTEEmbeddingModel(BaseEmbeddingModel):
    """GTE-multilingual embedding model implementation."""
    
    def __init__(self):
        super().__init__("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
    
    def encode_documents(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode documents using GTE model."""
        default_kwargs = {"normalize_embeddings": True}
        default_kwargs.update(kwargs)
        return self.model.encode(texts, **default_kwargs)
    
    def encode_query(self, query: str, **kwargs) -> np.ndarray:
        """Encode query using GTE model."""
        default_kwargs = {"normalize_embeddings": True}
        default_kwargs.update(kwargs)
        # GTE doesn't have separate query encoding, use regular encode
        result = self.model.encode([query], **default_kwargs)
        return result[0] if len(result) > 0 else result
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          document_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity."""
        # Normalize embeddings
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = document_embeddings / (np.linalg.norm(document_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        similarities = np.dot(doc_norms, query_norm)
        return similarities


class GemmaEmbeddingModel(BaseEmbeddingModel):
    """EmbeddingGemma model implementation."""
    
    def __init__(self):
        # Determine device and dtype
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use bfloat16 on CUDA if available, otherwise float32
        if device == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32
        
        # Initialize with device and dtype
        self.model_name = "google/embeddinggemma-300m"
        self.model = SentenceTransformer(
            self.model_name,
            device=device,
            model_kwargs={
                "dtype": dtype
            }
        )
    
    def encode_documents(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode documents using EmbeddingGemma model."""
        return self.model.encode_document(texts, **kwargs)
    
    def encode_query(self, query: str, **kwargs) -> np.ndarray:
        """Encode query using EmbeddingGemma model."""
        return self.model.encode_query(query, **kwargs)
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          document_embeddings: np.ndarray) -> np.ndarray:
        """Compute similarity using model's built-in method."""
        # EmbeddingGemma has its own similarity method
        similarities_tensor = self.model.similarity(query_embedding, document_embeddings)
        # Convert to numpy array
        if hasattr(similarities_tensor, 'cpu'):
            similarities = similarities_tensor.cpu().numpy()[0]
        else:
            similarities = np.array(similarities_tensor)[0]
        return similarities


# Model factory
def create_embedding_model(model_key: str) -> BaseEmbeddingModel:
    """Factory function to create embedding model instances."""
    model_classes = {
        "qwen": QwenEmbeddingModel,
        "gte": GTEEmbeddingModel,
        "gemma": GemmaEmbeddingModel,
    }
    
    if model_key not in model_classes:
        raise ValueError(f"Unknown model key: {model_key}")
    
    return model_classes[model_key]()

