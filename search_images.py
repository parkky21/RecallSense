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
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from config import get_enabled_models, get_model_config
from embedding_models import create_embedding_model


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


def load_embeddings_data(enabled_models):
    """Load stored embeddings and metadata."""
    output_dir = Path("embeddings_data")
    metadata_file = output_dir / "metadata.json"
    
    if not metadata_file.exists():
        return None, [], []
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")
        return None, [], []
    
    image_paths = metadata.get("image_paths", [])
    captions = metadata.get("captions", [])
    
    # Load embeddings for enabled models
    embeddings_dict = {}
    all_exist = True
    
    for model_key in enabled_models:
        embeddings_file = output_dir / f"embeddings_{model_key}.npy"
        if embeddings_file.exists():
            try:
                embeddings_dict[model_key] = np.load(embeddings_file)
            except Exception as e:
                print(f"Warning: Could not load {model_key} embeddings: {e}")
                all_exist = False
        else:
            all_exist = False
    
    if all_exist and len(image_paths) > 0:
        return embeddings_dict, image_paths, captions
    
    if embeddings_dict:
        return embeddings_dict, image_paths, captions
    
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


def index_folder_images(folder_path, processor, caption_model, embedding_models, device, existing_paths_set, enabled_models):
    """Index images from a folder on-the-fly."""
    print(f"\nIndexing images from: {folder_path}")
    
    # Get image files
    image_files = get_image_files(folder_path)
    
    if not image_files:
        print("No image files found in the specified folder.")
        return {}, [], []
    
    # Filter out already indexed images
    new_image_files = []
    for img_path in image_files:
        normalized = normalize_path(img_path)
        if normalized not in existing_paths_set:
            new_image_files.append(img_path)
    
    if not new_image_files:
        print("All images in this folder are already indexed.")
        return {}, [], []
    
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
        return {}, [], []
    
    # Generate embeddings with enabled models
    print(f"Generating embeddings for {len(captions)} image(s)...")
    embeddings_dict = {}
    
    for model_key in enabled_models:
        if model_key in embedding_models:
            print(f"  {model_key}...")
            start_time = time.time()
            embeddings_dict[model_key] = embedding_models[model_key].encode_documents(captions)
            elapsed_time = time.time() - start_time
            print(f"    {model_key}: {elapsed_time:.2f} seconds")
    
    return embeddings_dict, image_paths, captions


def display_all_models(results_dict, top_k=3):
    """Display images from all enabled models in a grid layout."""
    num_models = len(results_dict)
    if num_models == 0:
        print("No results to display.")
        return
    
    fig, axes = plt.subplots(num_models, top_k, figsize=(15, 5 * num_models))
    
    if num_models == 1:
        axes = axes.reshape(1, -1)
    if top_k == 1:
        axes = axes.reshape(-1, 1)
    
    for row_idx, (model_key, (paths, captions, similarities)) in enumerate(results_dict.items()):
        for col_idx, (img_path, caption, sim) in enumerate(zip(paths, captions, similarities)):
            try:
                img = mpimg.imread(img_path)
                axes[row_idx, col_idx].imshow(img)
                axes[row_idx, col_idx].axis('off')
                axes[row_idx, col_idx].set_title(f"[{model_key.upper()}] Similarity: {sim:.4f}\n{caption}", 
                                                fontsize=8, wrap=True)
            except Exception as e:
                axes[row_idx, col_idx].text(0.5, 0.5, f"Error loading image:\n{img_path}\n{e}", 
                                          ha='center', va='center', fontsize=7)
                axes[row_idx, col_idx].axis('off')
        
        # Add row label
        fig.text(0.02, 0.95 - (row_idx * 0.9 / num_models), f'{model_key.upper()} Model', 
                rotation=90, fontsize=11, fontweight='bold', va='center')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05)
    plt.show()


def main():
    # Get enabled models from config
    enabled_models = get_enabled_models()
    
    if not enabled_models:
        print("Error: No embedding models are enabled in config.py")
        print("Please enable at least one model in config.py")
        return
    
    print(f"Enabled models: {', '.join(enabled_models)}")
    
    # Load existing indexed data
    print("\nLoading existing embeddings and metadata...")
    embeddings_dict, image_paths, captions = load_embeddings_data(enabled_models)
    
    if embeddings_dict is not None:
        print(f"Loaded {len(image_paths)} indexed image(s).")
        existing_paths_set = {normalize_path(p) for p in image_paths}
    else:
        print("No existing indexed images found.")
        embeddings_dict = {}
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
    print("\nLoading embedding models...")
    embedding_models = {}
    for model_key in enabled_models:
        model_config = get_model_config(model_key)
        print(f"  Loading {model_config['name']}...")
        embedding_models[model_key] = create_embedding_model(model_key)
    
    # Index additional folder if provided
    if additional_folder and processor and caption_model:
        new_embeddings_dict, new_paths, new_captions = index_folder_images(
            additional_folder, processor, caption_model, embedding_models, device, existing_paths_set, enabled_models
        )
        
        if new_embeddings_dict and len(new_paths) > 0:
            # Merge with existing data
            if embeddings_dict:
                for model_key in enabled_models:
                    if model_key in embeddings_dict and model_key in new_embeddings_dict:
                        embeddings_dict[model_key] = np.vstack([embeddings_dict[model_key], new_embeddings_dict[model_key]])
                    elif model_key in new_embeddings_dict:
                        embeddings_dict[model_key] = new_embeddings_dict[model_key]
                image_paths = image_paths + new_paths
                captions = captions + new_captions
            else:
                embeddings_dict = new_embeddings_dict
                image_paths = new_paths
                captions = new_captions
            print(f"\nTotal images available for search: {len(image_paths)}")
    
    if not embeddings_dict or len(image_paths) == 0:
        print("\nNo images available for search. Please index some images first.")
        return
    
    # Get query from user
    query = input("\nEnter your search query: ").strip()
    
    if not query:
        print("Query cannot be empty.")
        return
    
    # Search with all enabled models
    print(f"\nSearching with all enabled models for query: '{query}'...")
    print("-" * 60)
    
    search_results = {}
    search_times = {}
    
    for model_key in enabled_models:
        if model_key not in embeddings_dict:
            print(f"Warning: {model_key} embeddings not found. Skipping.")
            continue
        
        print(f"\n[{model_key.upper()} Model]")
        start_time = time.time()
        
        # Encode query
        query_embedding = embedding_models[model_key].encode_query(query)
        
        # Compute similarities
        similarities = embedding_models[model_key].compute_similarity(
            query_embedding, embeddings_dict[model_key]
        )
        
        search_time = time.time() - start_time
        search_times[model_key] = search_time
        
        # Get top 3 results
        top_k = 3
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Store results
        top_paths = []
        top_captions = []
        top_similarities = []
        
        for rank, idx in enumerate(top_indices, 1):
            similarity_score = similarities[idx]
            caption = captions[idx]
            img_path = image_paths[idx]
            
            print(f"\n{rank}. Similarity: {similarity_score:.4f}")
            print(f"   Image: {img_path}")
            print(f"   Caption: {caption}")
            
            top_paths.append(img_path)
            top_captions.append(caption)
            top_similarities.append(similarity_score)
        
        search_results[model_key] = (top_paths, top_captions, top_similarities)
    
    # Display timing
    print(f"\nSearch timing:")
    for model_key, search_time in search_times.items():
        print(f"  {model_key}: {search_time:.4f} seconds")
    
    # Display images
    print("\nDisplaying images from all enabled models...")
    try:
        display_all_models(search_results, top_k=3)
    except Exception as e:
        print(f"Error displaying images: {e}")
        print("You can manually view the images at the paths listed above.")


if __name__ == "__main__":
    main()
