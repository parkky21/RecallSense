"""
Image Indexing Script
This script processes all images in a folder, generates captions using BLIP,
and embeds them using configurable embedding models.
"""

import torch
import json
import time
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np

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


def load_existing_data(enabled_models):
    """Load existing embeddings and metadata if they exist."""
    output_dir = Path("embeddings_data")
    metadata_file = output_dir / "metadata.json"
    
    if not metadata_file.exists():
        return None, [], [], {}
    
    # Load metadata
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load metadata: {e}")
        return None, [], [], {}
    
    existing_paths = metadata.get("image_paths", [])
    existing_captions = metadata.get("captions", [])
    existing_times = metadata.get("embedding_times", {})
    
    # Load embeddings for enabled models only
    existing_embeddings = {}
    all_exist = True
    
    for model_key in enabled_models:
        embeddings_file = output_dir / f"embeddings_{model_key}.npy"
        if embeddings_file.exists():
            try:
                existing_embeddings[model_key] = np.load(embeddings_file)
            except Exception as e:
                print(f"Warning: Could not load {model_key} embeddings: {e}")
                all_exist = False
        else:
            all_exist = False
    
    if all_exist and len(existing_paths) > 0:
        return existing_embeddings, existing_paths, existing_captions, existing_times
    
    # Partial data exists
    if existing_embeddings:
        return existing_embeddings, existing_paths, existing_captions, existing_times
    
    return None, [], [], {}


def normalize_path(path):
    """Normalize path for comparison (resolve to absolute path)."""
    return str(Path(path).resolve())


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


def main():
    # Get enabled models from config
    enabled_models = get_enabled_models()
    
    if not enabled_models:
        print("Error: No embedding models are enabled in config.py")
        print("Please enable at least one model in config.py")
        return
    
    print(f"Enabled models: {', '.join(enabled_models)}")
    
    # Configuration
    folder_path = input("\nEnter the folder path containing images: ").strip()
    
    # Remove quotes if present
    if folder_path.startswith('"') and folder_path.endswith('"'):
        folder_path = folder_path[1:-1]
    if folder_path.startswith("'") and folder_path.endswith("'"):
        folder_path = folder_path[1:-1]
    
    print(f"\nProcessing images from: {folder_path}")
    
    # Get all image files
    image_files = get_image_files(folder_path)
    
    if not image_files:
        print("No image files found in the specified folder.")
        return
    
    print(f"Found {len(image_files)} image(s) in folder.\n")
    
    # Load existing data
    existing_embeddings_dict, existing_paths, existing_captions, existing_times = load_existing_data(enabled_models)
    
    # Check which models need regeneration
    models_to_regenerate = []
    if existing_embeddings_dict is not None and len(existing_paths) > 0:
        print(f"Found {len(existing_paths)} already indexed image(s).")
        existing_paths_set = {normalize_path(p) for p in existing_paths}
        
        # Check which models are missing
        for model_key in enabled_models:
            if model_key not in existing_embeddings_dict:
                models_to_regenerate.append(model_key)
                print(f"Warning: {model_key} embeddings are missing. Will regenerate.")
    else:
        existing_paths_set = set()
        existing_embeddings_dict = {}
        existing_paths = []
        existing_captions = []
        existing_times = {}
        models_to_regenerate = enabled_models.copy()
    
    # Filter out already processed images
    new_image_files = []
    for img_path in image_files:
        normalized = normalize_path(img_path)
        if normalized not in existing_paths_set:
            new_image_files.append(img_path)
    
    if not new_image_files and not models_to_regenerate:
        print("\nAll images are already indexed with all enabled models. No new images to process.")
        return
    
    if models_to_regenerate and not new_image_files:
        print(f"\nWill regenerate embeddings for {len(existing_paths)} existing image(s) using existing captions.")
    elif new_image_files:
        print(f"Found {len(new_image_files)} new image(s) to process.\n")
    
    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    if device == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU (this will be slower)")
    
    # Load BLIP captioning model (only if we have new images)
    processor = None
    caption_model = None
    if new_image_files:
        print("\nLoading BLIP captioning model...")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)
        caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base", 
            dtype=dtype
        ).to(device)
    
    # Load embedding models
    print("\nLoading embedding models...")
    embedding_models = {}
    for model_key in enabled_models:
        model_config = get_model_config(model_key)
        print(f"  Loading {model_config['name']}...")
        embedding_models[model_key] = create_embedding_model(model_key)
    
    # Process new images
    new_captions = []
    new_image_paths = []
    
    if new_image_files:
        for idx, image_path in enumerate(new_image_files, 1):
            print(f"[{idx}/{len(new_image_files)}] Processing: {image_path.name}")
            
            # Generate caption
            caption = caption_image(image_path, processor, caption_model, device)
            
            if caption:
                new_captions.append(caption)
                new_image_paths.append(str(image_path))
                print(f"  Caption: {caption}")
            else:
                print(f"  Failed to generate caption for {image_path.name}")
    
    # Generate embeddings for existing images (if models need regeneration)
    existing_embeddings_new = {}
    existing_times_new = {}
    
    if models_to_regenerate and existing_captions:
        print(f"\nRegenerating embeddings for {len(existing_captions)} existing image(s)...")
        for model_key in models_to_regenerate:
            if model_key in embedding_models:
                print(f"  Generating {model_key} embeddings...")
                start_time = time.time()
                existing_embeddings_new[model_key] = embedding_models[model_key].encode_documents(existing_captions)
                elapsed_time = time.time() - start_time
                existing_times_new[model_key] = elapsed_time
                print(f"    {model_key}: {elapsed_time:.2f} seconds")
    
    # Generate embeddings for new images
    new_embeddings = {}
    new_times = {}
    
    if new_captions:
        print(f"\nGenerating embeddings for {len(new_captions)} new caption(s)...")
        for model_key in enabled_models:
            if model_key in embedding_models:
                print(f"  Generating {model_key} embeddings...")
                start_time = time.time()
                new_embeddings[model_key] = embedding_models[model_key].encode_documents(new_captions)
                elapsed_time = time.time() - start_time
                new_times[model_key] = elapsed_time
                print(f"    {model_key}: {elapsed_time:.2f} seconds")
    
    if not new_captions and not models_to_regenerate:
        print("\nNo new captions were generated.")
        if existing_embeddings_dict:
            print("Keeping existing indexed data.")
        return
    
    # Merge embeddings
    all_embeddings = {}
    for model_key in enabled_models:
        if model_key in existing_embeddings_dict:
            existing_emb = existing_embeddings_dict[model_key]
            if model_key in new_embeddings:
                all_embeddings[model_key] = np.vstack([existing_emb, new_embeddings[model_key]])
            else:
                all_embeddings[model_key] = existing_emb
        elif model_key in existing_embeddings_new:
            # Regenerated embeddings for existing images
            if model_key in new_embeddings:
                all_embeddings[model_key] = np.vstack([existing_embeddings_new[model_key], new_embeddings[model_key]])
            else:
                all_embeddings[model_key] = existing_embeddings_new[model_key]
        elif model_key in new_embeddings:
            all_embeddings[model_key] = new_embeddings[model_key]
    
    # Update paths and captions
    if new_image_paths:
        all_paths = existing_paths + new_image_paths
        all_captions = existing_captions + new_captions
    else:
        all_paths = existing_paths
        all_captions = existing_captions
    
    # Update timing information
    embedding_times = existing_times.copy()
    for model_key in enabled_models:
        if model_key in new_times:
            embedding_times[model_key] = embedding_times.get(model_key, 0) + new_times[model_key]
            if new_captions:
                embedding_times[f"per_image_{model_key}"] = new_times[model_key] / len(new_captions)
        if model_key in existing_times_new:
            embedding_times[model_key] = embedding_times.get(model_key, 0) + existing_times_new[model_key]
            if existing_captions:
                embedding_times[f"per_image_{model_key}"] = existing_times_new[model_key] / len(existing_captions)
    
    # Store results
    output_dir = Path("embeddings_data")
    output_dir.mkdir(exist_ok=True)
    
    # Save embeddings
    for model_key in enabled_models:
        if model_key in all_embeddings:
            embeddings_file = output_dir / f"embeddings_{model_key}.npy"
            np.save(embeddings_file, all_embeddings[model_key])
            print(f"Saved {model_key} embeddings to: {embeddings_file}")
    
    # Save metadata
    metadata = {
        "image_paths": all_paths,
        "captions": all_captions,
        "embedding_times": embedding_times,
        "enabled_models": enabled_models
    }
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to: {metadata_file}")
    
    # Summary
    print(f"\n✓ Successfully indexed {len(new_image_paths)} new image(s)!")
    print(f"  Total indexed images: {len(all_paths)}")
    if new_times:
        times_str = ", ".join([f"{k}: {v:.2f}s" for k, v in new_times.items()])
        print(f"  Embedding times - {times_str}")
        per_image_str = ", ".join([f"{k}: {embedding_times.get(f'per_image_{k}', 0):.4f}s" 
                                   for k in enabled_models if f'per_image_{k}' in embedding_times])
        if per_image_str:
            print(f"  Per image - {per_image_str}")


if __name__ == "__main__":
    main()
