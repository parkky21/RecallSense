"""
Image Indexing Script
This script processes all images in a folder, generates captions using BLIP,
embeds them using Qwen3-Embedding-0.6B, and stores the embeddings locally.
"""

import torch
import json
import time
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer
import numpy as np


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


def load_existing_data():
    """Load existing embeddings and metadata if they exist."""
    output_dir = Path("embeddings_data")
    qwen_embeddings_file = output_dir / "embeddings_qwen.npy"
    gte_embeddings_file = output_dir / "embeddings_gte.npy"
    gemma_embeddings_file = output_dir / "embeddings_gemma.npy"
    old_embeddings_file = output_dir / "embeddings.npy"  # Old format
    metadata_file = output_dir / "metadata.json"
    
    # Check for new format (all three models)
    if (qwen_embeddings_file.exists() and gte_embeddings_file.exists() and 
        gemma_embeddings_file.exists() and metadata_file.exists()):
        try:
            qwen_embeddings = np.load(qwen_embeddings_file)
            gte_embeddings = np.load(gte_embeddings_file)
            gemma_embeddings = np.load(gemma_embeddings_file)
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            return (qwen_embeddings, gte_embeddings, gemma_embeddings), metadata["image_paths"], metadata["captions"], metadata.get("embedding_times", {})
        except Exception as e:
            print(f"Warning: Could not load existing data: {e}")
            return None, [], [], {}
    
    # Check for old format (two models - Qwen and GTE)
    if qwen_embeddings_file.exists() and gte_embeddings_file.exists() and metadata_file.exists():
        print("Warning: Found embeddings with only Qwen and GTE models.")
        print("Please re-index to include EmbeddingGemma model.")
        try:
            qwen_embeddings = np.load(qwen_embeddings_file)
            gte_embeddings = np.load(gte_embeddings_file)
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            # Return with None for gemma to trigger re-indexing
            return (qwen_embeddings, gte_embeddings, None), metadata["image_paths"], metadata["captions"], metadata.get("embedding_times", {})
        except Exception as e:
            print(f"Warning: Could not load existing data: {e}")
            return None, [], [], {}
    
    # Check for old format (single embeddings file)
    if old_embeddings_file.exists() and metadata_file.exists():
        print("Warning: Found old format embeddings. Please re-index to use all models.")
        print("The old embeddings will be ignored. Run index_images.py to re-index your images.")
        return None, [], [], {}
    
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
    # Configuration
    folder_path = input("Enter the folder path containing images: ").strip()
    
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
    existing_data = load_existing_data()
    existing_embeddings_tuple, existing_paths, existing_captions, existing_times = existing_data
    
    # Check if EmbeddingGemma embeddings are missing
    gemma_missing = False
    if existing_embeddings_tuple is not None and len(existing_paths) > 0:
        print(f"Found {len(existing_paths)} already indexed image(s).")
        # Create a set of normalized existing paths for quick lookup
        existing_paths_set = {normalize_path(p) for p in existing_paths}
        if len(existing_embeddings_tuple) == 3:
            existing_qwen_embeddings, existing_gte_embeddings, existing_gemma_embeddings = existing_embeddings_tuple
            if existing_gemma_embeddings is None:
                gemma_missing = True
                print("Warning: EmbeddingGemma embeddings are missing. Will regenerate for all images.")
        else:
            # Old format with only 2 models
            existing_qwen_embeddings, existing_gte_embeddings = existing_embeddings_tuple
            existing_gemma_embeddings = None
            gemma_missing = True
            print("Warning: EmbeddingGemma embeddings are missing. Will regenerate for all images.")
    else:
        existing_paths_set = set()
        existing_embeddings_tuple = None
        existing_qwen_embeddings = None
        existing_gte_embeddings = None
        existing_gemma_embeddings = None
        existing_paths = []
        existing_captions = []
        existing_times = {}
    
    # Filter out already processed images (unless EmbeddingGemma is missing)
    new_image_files = []
    if gemma_missing:
        # If EmbeddingGemma is missing, we need to regenerate embeddings for all existing images
        # But we can reuse existing captions, so we don't need to re-process images
        print("EmbeddingGemma embeddings missing. Will regenerate using existing captions.")
        # Check if all images in folder are already indexed
        all_indexed = True
        for img_path in image_files:
            normalized = normalize_path(img_path)
            if normalized not in existing_paths_set:
                all_indexed = False
                new_image_files.append(img_path)
        
        if all_indexed:
            print("All images are already indexed. Will regenerate EmbeddingGemma embeddings only.")
        else:
            print(f"Found {len(new_image_files)} new image(s) to process, plus regenerating EmbeddingGemma for existing images.")
    else:
        for img_path in image_files:
            normalized = normalize_path(img_path)
            if normalized not in existing_paths_set:
                new_image_files.append(img_path)
    
    if not new_image_files and not gemma_missing:
        print("\nAll images are already indexed. No new images to process.")
        return
    
    if gemma_missing:
        print(f"\nWill process {len(new_image_files)} new image(s) and regenerate EmbeddingGemma embeddings for {len(existing_paths)} existing image(s).")
    else:
        print(f"Found {len(new_image_files)} new image(s) to process.\n")
    
    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    if device == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available, using CPU (this will be slower)")
    
    # Load models (only if we have new images to process)
    print("Loading BLIP captioning model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base", 
        torch_dtype=dtype
    ).to(device)
    
    print("Loading Qwen embedding model...")
    qwen_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    
    print("Loading GTE-multilingual embedding model...")
    gte_model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
    
    print("Loading EmbeddingGemma model...")
    gemma_model = SentenceTransformer("google/embeddinggemma-300m")
    
    # Process new images
    new_captions = []
    new_image_paths = []
    
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
    
    # Handle EmbeddingGemma regeneration for existing images
    gemma_existing_time = 0
    if gemma_missing and existing_captions:
        print(f"\nRegenerating EmbeddingGemma embeddings for {len(existing_captions)} existing image(s)...")
        start_time = time.time()
        existing_gemma_embeddings = gemma_model.encode_document(existing_captions)
        gemma_existing_time = time.time() - start_time
        print(f"  EmbeddingGemma (existing): {gemma_existing_time:.2f} seconds")
    
    # Generate embeddings for new images with all three models
    new_qwen_embeddings = None
    new_gte_embeddings = None
    new_gemma_embeddings = None
    qwen_time = 0
    gte_time = 0
    gemma_time = 0
    
    if new_captions:
        print(f"\nGenerating embeddings for {len(new_captions)} new caption(s)...")
        
        # Qwen embeddings
        print("  Generating Qwen embeddings...")
        start_time = time.time()
        new_qwen_embeddings = qwen_model.encode(new_captions)
        qwen_time = time.time() - start_time
        print(f"    Qwen: {qwen_time:.2f} seconds")
        
        # GTE embeddings
        print("  Generating GTE-multilingual embeddings...")
        start_time = time.time()
        new_gte_embeddings = gte_model.encode(new_captions, normalize_embeddings=True)
        gte_time = time.time() - start_time
        print(f"    GTE: {gte_time:.2f} seconds")
        
        # EmbeddingGemma embeddings
        print("  Generating EmbeddingGemma embeddings...")
        start_time = time.time()
        new_gemma_embeddings = gemma_model.encode_document(new_captions)
        gemma_time = time.time() - start_time
        print(f"    EmbeddingGemma: {gemma_time:.2f} seconds")
    elif not gemma_missing:
        print("\nNo new captions were generated.")
        if existing_embeddings_tuple is not None:
            print("Keeping existing indexed data.")
        return
    
    # Store timing information
    embedding_times = existing_times.copy()
    if new_captions:
        embedding_times["qwen"] = embedding_times.get("qwen", 0) + qwen_time
        embedding_times["gte"] = embedding_times.get("gte", 0) + gte_time
        embedding_times["gemma"] = embedding_times.get("gemma", 0) + gemma_time
        embedding_times["per_image_qwen"] = qwen_time / len(new_captions) if len(new_captions) > 0 else 0
        embedding_times["per_image_gte"] = gte_time / len(new_captions) if len(new_captions) > 0 else 0
        embedding_times["per_image_gemma"] = gemma_time / len(new_captions) if len(new_captions) > 0 else 0
    
    if gemma_missing and existing_captions:
        embedding_times["gemma"] = embedding_times.get("gemma", 0) + gemma_existing_time
        embedding_times["per_image_gemma"] = gemma_existing_time / len(existing_captions) if len(existing_captions) > 0 else 0
    
    # Merge with existing data
    if existing_qwen_embeddings is not None:
        if new_captions:
            all_qwen_embeddings = np.vstack([existing_qwen_embeddings, new_qwen_embeddings])
            all_gte_embeddings = np.vstack([existing_gte_embeddings, new_gte_embeddings])
            all_paths = existing_paths + new_image_paths
            all_captions = existing_captions + new_captions
        else:
            # Only regenerating EmbeddingGemma
            all_qwen_embeddings = existing_qwen_embeddings
            all_gte_embeddings = existing_gte_embeddings
            all_paths = existing_paths
            all_captions = existing_captions
        
        # Handle EmbeddingGemma embeddings
        if gemma_missing:
            # Use regenerated EmbeddingGemma embeddings
            all_gemma_embeddings = existing_gemma_embeddings
            if new_captions:
                # Merge with new EmbeddingGemma embeddings if any
                all_gemma_embeddings = np.vstack([all_gemma_embeddings, new_gemma_embeddings])
        elif existing_gemma_embeddings is not None:
            if new_captions:
                all_gemma_embeddings = np.vstack([existing_gemma_embeddings, new_gemma_embeddings])
            else:
                all_gemma_embeddings = existing_gemma_embeddings
        else:
            all_gemma_embeddings = new_gemma_embeddings
        
        if new_captions:
            print(f"Merged with {len(existing_paths)} existing image(s).")
        else:
            print(f"Regenerated EmbeddingGemma embeddings for {len(existing_paths)} existing image(s).")
    else:
        all_qwen_embeddings = new_qwen_embeddings
        all_gte_embeddings = new_gte_embeddings
        all_gemma_embeddings = new_gemma_embeddings
        all_paths = new_image_paths
        all_captions = new_captions
    
    # Store results
    output_dir = Path("embeddings_data")
    output_dir.mkdir(exist_ok=True)
    
    # Save embeddings as numpy arrays
    qwen_embeddings_file = output_dir / "embeddings_qwen.npy"
    gte_embeddings_file = output_dir / "embeddings_gte.npy"
    gemma_embeddings_file = output_dir / "embeddings_gemma.npy"
    np.save(qwen_embeddings_file, all_qwen_embeddings)
    np.save(gte_embeddings_file, all_gte_embeddings)
    np.save(gemma_embeddings_file, all_gemma_embeddings)
    print(f"Saved Qwen embeddings to: {qwen_embeddings_file}")
    print(f"Saved GTE embeddings to: {gte_embeddings_file}")
    print(f"Saved EmbeddingGemma embeddings to: {gemma_embeddings_file}")
    
    # Save metadata (image paths, captions, and timing)
    metadata = {
        "image_paths": all_paths,
        "captions": all_captions,
        "embedding_times": embedding_times
    }
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata to: {metadata_file}")
    
    print(f"\n✓ Successfully indexed {len(new_image_paths)} new image(s)!")
    print(f"  Total indexed images: {len(all_paths)}")
    print(f"  Embedding times - Qwen: {qwen_time:.2f}s, GTE: {gte_time:.2f}s, EmbeddingGemma: {gemma_time:.2f}s")
    print(f"  Per image - Qwen: {embedding_times['per_image_qwen']:.4f}s, GTE: {embedding_times['per_image_gte']:.4f}s, EmbeddingGemma: {embedding_times['per_image_gemma']:.4f}s")


if __name__ == "__main__":
    main()

