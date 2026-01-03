"""
Image Search Script
This script takes a query, searches through stored embeddings,
and displays the top 3 most similar images.
Can also search through additional folders with on-the-fly indexing.
"""

import json
import numpy as np
import torch
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_image_files(folder_path):
    """Get all image files from the folder."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    # Use a set to avoid duplicates (Windows filesystem is case-insensitive)
    image_files_set = set()
    for ext in image_extensions:
        # Search for both lowercase and uppercase extensions
        image_files_set.update(folder.glob(f'*{ext}'))
        image_files_set.update(folder.glob(f'*{ext.upper()}'))
    
    # Convert to sorted list
    return sorted(image_files_set)


def normalize_path(path):
    """Normalize path for comparison (resolve to absolute path)."""
    return str(Path(path).resolve())


def load_embeddings_data():
    """Load stored embeddings and metadata."""
    output_dir = Path("embeddings_data")
    
    qwen_embeddings_file = output_dir / "embeddings_qwen.npy"
    gte_embeddings_file = output_dir / "embeddings_gte.npy"
    gemma_embeddings_file = output_dir / "embeddings_gemma.npy"
    metadata_file = output_dir / "metadata.json"
    
    if (qwen_embeddings_file.exists() and gte_embeddings_file.exists() and 
        gemma_embeddings_file.exists() and metadata_file.exists()):
        try:
            qwen_embeddings = np.load(qwen_embeddings_file)
            gte_embeddings = np.load(gte_embeddings_file)
            gemma_embeddings = np.load(gemma_embeddings_file)
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            return (qwen_embeddings, gte_embeddings, gemma_embeddings), metadata["image_paths"], metadata["captions"]
        except Exception as e:
            print(f"Warning: Could not load existing data: {e}")
            return None, [], []
    return None, [], []


def caption_image(image_path, processor, model, device):
    """Generate caption for an image using BLIP."""
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def index_folder_images(folder_path, processor, caption_model, qwen_model, gte_model, gemma_model, device, existing_paths_set):
    """Index images from a folder on-the-fly."""
    print(f"\nIndexing images from: {folder_path}")
    
    # Get image files
    image_files = get_image_files(folder_path)
    
    if not image_files:
        print("No image files found in the specified folder.")
        return None, None, None, [], []
    
    # Filter out already indexed images
    new_image_files = []
    for img_path in image_files:
        normalized = normalize_path(img_path)
        if normalized not in existing_paths_set:
            new_image_files.append(img_path)
    
    if not new_image_files:
        print("All images in this folder are already indexed.")
        return None, None, None, [], []
    
    print(f"Found {len(new_image_files)} new image(s) to index...")
    
    # Process images
    captions = []
    image_paths = []
    
    for idx, image_path in enumerate(new_image_files, 1):
        print(f"  [{idx}/{len(new_image_files)}] Processing: {image_path.name}")
        caption = caption_image(image_path, processor, caption_model, device)
        if caption:
            captions.append(caption)
            image_paths.append(str(image_path))
            print(f"    Caption: {caption}")
    
    if not captions:
        return None, None, None, [], []
    
    # Generate embeddings with all three models
    print(f"Generating embeddings for {len(captions)} image(s)...")
    start_time = time.time()
    qwen_embeddings = qwen_model.encode(captions)
    qwen_time = time.time() - start_time
    print(f"  Qwen: {qwen_time:.2f} seconds")
    
    start_time = time.time()
    gte_embeddings = gte_model.encode(captions, normalize_embeddings=True)
    gte_time = time.time() - start_time
    print(f"  GTE: {gte_time:.2f} seconds")
    
    start_time = time.time()
    gemma_embeddings = gemma_model.encode_document(captions)
    gemma_time = time.time() - start_time
    print(f"  EmbeddingGemma: {gemma_time:.2f} seconds")
    
    return qwen_embeddings, gte_embeddings, gemma_embeddings, image_paths, captions


def compute_similarity(query_embedding, document_embeddings):
    """Compute cosine similarity between query and document embeddings."""
    # Normalize embeddings
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    doc_norms = document_embeddings / (np.linalg.norm(document_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity
    similarities = np.dot(doc_norms, query_norm)
    return similarities


def display_images(image_paths, captions, similarities, top_k=3, model_name=""):
    """Display the top K images with their captions and similarity scores."""
    fig, axes = plt.subplots(1, top_k, figsize=(15, 5))
    
    if top_k == 1:
        axes = [axes]
    
    title_prefix = f"[{model_name}] " if model_name else ""
    
    for idx, (img_path, caption, sim) in enumerate(zip(image_paths, captions, similarities)):
        try:
            img = mpimg.imread(img_path)
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(f"{title_prefix}Similarity: {sim:.4f}\n{caption}", 
                              fontsize=10, wrap=True)
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Error loading image:\n{img_path}\n{e}", 
                          ha='center', va='center', fontsize=8)
            axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def display_all_models(qwen_paths, qwen_captions, qwen_similarities, 
                       gte_paths, gte_captions, gte_similarities,
                       gemma_paths, gemma_captions, gemma_similarities, top_k=3):
    """Display images from all three models side-by-side in a 3-row layout."""
    fig, axes = plt.subplots(3, top_k, figsize=(15, 15))
    
    if top_k == 1:
        axes = axes.reshape(3, 1)
    
    # Display Qwen results (top row)
    for idx, (img_path, caption, sim) in enumerate(zip(qwen_paths, qwen_captions, qwen_similarities)):
        try:
            img = mpimg.imread(img_path)
            axes[0, idx].imshow(img)
            axes[0, idx].axis('off')
            axes[0, idx].set_title(f"[Qwen] Similarity: {sim:.4f}\n{caption}", 
                                  fontsize=8, wrap=True)
        except Exception as e:
            axes[0, idx].text(0.5, 0.5, f"Error loading image:\n{img_path}\n{e}", 
                            ha='center', va='center', fontsize=7)
            axes[0, idx].axis('off')
    
    # Display GTE results (middle row)
    for idx, (img_path, caption, sim) in enumerate(zip(gte_paths, gte_captions, gte_similarities)):
        try:
            img = mpimg.imread(img_path)
            axes[1, idx].imshow(img)
            axes[1, idx].axis('off')
            axes[1, idx].set_title(f"[GTE] Similarity: {sim:.4f}\n{caption}", 
                                  fontsize=8, wrap=True)
        except Exception as e:
            axes[1, idx].text(0.5, 0.5, f"Error loading image:\n{img_path}\n{e}", 
                            ha='center', va='center', fontsize=7)
            axes[1, idx].axis('off')
    
    # Display EmbeddingGemma results (bottom row)
    for idx, (img_path, caption, sim) in enumerate(zip(gemma_paths, gemma_captions, gemma_similarities)):
        try:
            img = mpimg.imread(img_path)
            axes[2, idx].imshow(img)
            axes[2, idx].axis('off')
            axes[2, idx].set_title(f"[EmbeddingGemma] Similarity: {sim:.4f}\n{caption}", 
                                  fontsize=8, wrap=True)
        except Exception as e:
            axes[2, idx].text(0.5, 0.5, f"Error loading image:\n{img_path}\n{e}", 
                            ha='center', va='center', fontsize=7)
            axes[2, idx].axis('off')
    
    # Add row labels
    fig.text(0.02, 0.83, 'Qwen Model', rotation=90, fontsize=11, fontweight='bold', va='center')
    fig.text(0.02, 0.50, 'GTE Model', rotation=90, fontsize=11, fontweight='bold', va='center')
    fig.text(0.02, 0.17, 'EmbeddingGemma', rotation=90, fontsize=11, fontweight='bold', va='center')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05)
    plt.show()


def main():
    # Load existing indexed data
    print("Loading existing embeddings and metadata...")
    document_embeddings, image_paths, captions = load_embeddings_data()
    
    if document_embeddings is not None:
        print(f"Loaded {len(image_paths)} indexed image(s).")
        existing_paths_set = {normalize_path(p) for p in image_paths}
    else:
        print("No existing indexed images found.")
        document_embeddings = None
        image_paths = []
        captions = []
        existing_paths_set = set()
    
    # Ask if user wants to search additional folders
    additional_folder = input("\nEnter additional folder path to search (or press Enter to skip): ").strip()
    
    # Remove quotes if present
    if additional_folder:
        if additional_folder.startswith('"') and additional_folder.endswith('"'):
            additional_folder = additional_folder[1:-1]
        if additional_folder.startswith("'") and additional_folder.endswith("'"):
            additional_folder = additional_folder[1:-1]
    
    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    processor = None
    caption_model = None
    
    # If additional folder is provided, index it
    if additional_folder:
        print(f"\nSetting up models for indexing...")
        if device == "cuda":
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU (this will be slower)")
        
        print("Loading BLIP captioning model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", 
            torch_dtype=dtype
        ).to(device)
    
    # Load embedding models
    print("Loading Qwen embedding model...")
    qwen_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    
    print("Loading GTE-multilingual embedding model...")
    gte_model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
    
    print("Loading EmbeddingGemma model...")
    gemma_model = SentenceTransformer("google/embeddinggemma-300m")
    
    # Index additional folder if provided
    if additional_folder and processor and caption_model:
        new_qwen_embeddings, new_gte_embeddings, new_gemma_embeddings, new_paths, new_captions = index_folder_images(
            additional_folder, processor, caption_model, qwen_model, gte_model, gemma_model, device, existing_paths_set
        )
        
        if new_qwen_embeddings is not None and len(new_paths) > 0:
            # Merge with existing data
            if document_embeddings is not None:
                existing_qwen, existing_gte, existing_gemma = document_embeddings
                all_qwen_embeddings = np.vstack([existing_qwen, new_qwen_embeddings])
                all_gte_embeddings = np.vstack([existing_gte, new_gte_embeddings])
                all_gemma_embeddings = np.vstack([existing_gemma, new_gemma_embeddings])
                all_paths = image_paths + new_paths
                all_captions = captions + new_captions
            else:
                all_qwen_embeddings = new_qwen_embeddings
                all_gte_embeddings = new_gte_embeddings
                all_gemma_embeddings = new_gemma_embeddings
                all_paths = new_paths
                all_captions = new_captions
            
            document_embeddings = (all_qwen_embeddings, all_gte_embeddings, all_gemma_embeddings)
            image_paths = all_paths
            captions = all_captions
            print(f"\nTotal images available for search: {len(image_paths)}")
    
    if document_embeddings is None or len(image_paths) == 0:
        print("\nNo images available for search. Please index some images first.")
        return
    
    qwen_embeddings, gte_embeddings, gemma_embeddings = document_embeddings
    
    # Get query from user
    query = input("\nEnter your search query: ").strip()
    
    if not query:
        print("Query cannot be empty.")
        return
    
    # Search with all three models
    print(f"\nSearching with all three models for query: '{query}'...")
    print("-" * 60)
    
    # Qwen search
    print("\n[Qwen Model]")
    start_time = time.time()
    qwen_query_embedding = qwen_model.encode(query, prompt_name="query")
    qwen_similarities = compute_similarity(qwen_query_embedding, qwen_embeddings)
    qwen_search_time = time.time() - start_time
    
    # GTE search
    print("[GTE-multilingual Model]")
    start_time = time.time()
    gte_query_embedding = gte_model.encode([query], normalize_embeddings=True)
    gte_similarities = compute_similarity(gte_query_embedding[0], gte_embeddings)
    gte_search_time = time.time() - start_time
    
    # EmbeddingGemma search
    print("[EmbeddingGemma Model]")
    start_time = time.time()
    gemma_query_embedding = gemma_model.encode_query(query)
    # Use model's similarity method for EmbeddingGemma
    gemma_similarities_tensor = gemma_model.similarity(gemma_query_embedding, gemma_embeddings)
    gemma_similarities = gemma_similarities_tensor.cpu().numpy()[0] if hasattr(gemma_similarities_tensor, 'cpu') else np.array(gemma_similarities_tensor)[0]
    gemma_search_time = time.time() - start_time
    
    print(f"\nSearch timing:")
    print(f"  Qwen: {qwen_search_time:.4f} seconds")
    print(f"  GTE: {gte_search_time:.4f} seconds")
    print(f"  EmbeddingGemma: {gemma_search_time:.4f} seconds")
    
    # Get top 3 results for each model
    top_k = 3
    qwen_top_indices = np.argsort(qwen_similarities)[::-1][:top_k]
    gte_top_indices = np.argsort(gte_similarities)[::-1][:top_k]
    gemma_top_indices = np.argsort(gemma_similarities)[::-1][:top_k]
    
    # Display Qwen results
    print(f"\n{'='*60}")
    print(f"Top {top_k} results - Qwen Model:")
    print("-" * 60)
    
    qwen_top_paths = []
    qwen_top_captions = []
    qwen_top_similarities = []
    
    for rank, idx in enumerate(qwen_top_indices, 1):
        similarity_score = qwen_similarities[idx]
        caption = captions[idx]
        img_path = image_paths[idx]
        
        print(f"\n{rank}. Similarity: {similarity_score:.4f}")
        print(f"   Image: {img_path}")
        print(f"   Caption: {caption}")
        
        qwen_top_paths.append(img_path)
        qwen_top_captions.append(caption)
        qwen_top_similarities.append(similarity_score)
    
    # Display GTE results
    print(f"\n{'='*60}")
    print(f"Top {top_k} results - GTE-multilingual Model:")
    print("-" * 60)
    
    gte_top_paths = []
    gte_top_captions = []
    gte_top_similarities = []
    
    for rank, idx in enumerate(gte_top_indices, 1):
        similarity_score = gte_similarities[idx]
        caption = captions[idx]
        img_path = image_paths[idx]
        
        print(f"\n{rank}. Similarity: {similarity_score:.4f}")
        print(f"   Image: {img_path}")
        print(f"   Caption: {caption}")
        
        gte_top_paths.append(img_path)
        gte_top_captions.append(caption)
        gte_top_similarities.append(similarity_score)
    
    # Display EmbeddingGemma results
    print(f"\n{'='*60}")
    print(f"Top {top_k} results - EmbeddingGemma Model:")
    print("-" * 60)
    
    gemma_top_paths = []
    gemma_top_captions = []
    gemma_top_similarities = []
    
    for rank, idx in enumerate(gemma_top_indices, 1):
        similarity_score = gemma_similarities[idx]
        caption = captions[idx]
        img_path = image_paths[idx]
        
        print(f"\n{rank}. Similarity: {similarity_score:.4f}")
        print(f"   Image: {img_path}")
        print(f"   Caption: {caption}")
        
        gemma_top_paths.append(img_path)
        gemma_top_captions.append(caption)
        gemma_top_similarities.append(similarity_score)
    
    # Display images from all three models side-by-side
    print("\nDisplaying images from all three models (Qwen on top, GTE in middle, EmbeddingGemma on bottom)...")
    try:
        display_all_models(
            qwen_top_paths, qwen_top_captions, qwen_top_similarities,
            gte_top_paths, gte_top_captions, gte_top_similarities,
            gemma_top_paths, gemma_top_captions, gemma_top_similarities,
            top_k=top_k
        )
    except Exception as e:
        print(f"Error displaying images: {e}")
        print("You can manually view the images at the paths listed above.")


if __name__ == "__main__":
    main()

