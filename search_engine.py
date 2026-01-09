import json
import numpy as np
import torch
import time
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import shutil

from config import get_enabled_models, get_model_config, get_caption_model_name
from embedding_models import create_embedding_model


class SearchEngine:
    def __init__(self):
        self.enabled_models = get_enabled_models()
        self.output_dir = Path("embeddings_data")
        self.embeddings_dict = {}
        self.image_paths = []
        self.captions = []
        self.embedding_models = {}
        
        # Captioning models (lazy loaded)
        self.processor = None
        self.caption_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    def load_data(self):
        """Load stored embeddings and metadata."""
        metadata_file = self.output_dir / "metadata.json"
        
        if not metadata_file.exists():
            print("No metadata file found.")
            return False
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load metadata: {e}")
            return False
        
        self.image_paths = metadata.get("image_paths", [])
        self.captions = metadata.get("captions", [])
        
        # Load embeddings for enabled models
        self.embeddings_dict = {}
        all_exist = True
        
        for model_key in self.enabled_models:
            embeddings_file = self.output_dir / f"embeddings_{model_key}.npy"
            if embeddings_file.exists():
                try:
                    self.embeddings_dict[model_key] = np.load(embeddings_file)
                except Exception as e:
                    print(f"Warning: Could not load {model_key} embeddings: {e}")
                    all_exist = False
            else:
                all_exist = False
        
        # Load embedding models for search
        for model_key in self.enabled_models:
             if model_key not in self.embedding_models:
                print(f"Loading {model_key} model...")
                self.embedding_models[model_key] = create_embedding_model(model_key)

        return True

    def _load_caption_model(self):
        """Lazy load the captioning model."""
        if self.processor is None:
            model_name = get_caption_model_name()
            print(f"Loading BLIP captioning model ({model_name})...")
            self.processor = BlipProcessor.from_pretrained(model_name, use_fast=False)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(
                model_name, 
                dtype=self.dtype
            ).to(self.device)

    def _get_image_files(self, folder_path):
        """Get all image files from the folder."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
        folder = Path(folder_path)
        
        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
        
        image_files_set = set()
        for ext in image_extensions:
            image_files_set.update(folder.glob(f'*{ext}'))
            image_files_set.update(folder.glob(f'*{ext.upper()}'))
        
        return sorted(image_files_set)

    def _normalize_path(self, path):
        return str(Path(path).resolve())

    def index_folder_generator(self, folder_path):
        """Index images from a folder, yielding progress updates."""
        try:
            self._load_caption_model()
            
            # Ensure models are loaded
            for model_key in self.enabled_models:
                if model_key not in self.embedding_models:
                    self.embedding_models[model_key] = create_embedding_model(model_key)

            yield {"status": "info", "message": f"Scanning folder: {folder_path}"}
            
            existing_paths_set = {self._normalize_path(p) for p in self.image_paths}
            image_files = self._get_image_files(folder_path)
            
            new_image_files = []
            for img_path in image_files:
                normalized = self._normalize_path(img_path)
                if normalized not in existing_paths_set:
                    new_image_files.append(img_path)
            
            if not new_image_files:
                yield {"status": "complete", "message": "All images are already indexed.", "total": 0, "current": 0}
                return
            
            total_images = len(new_image_files)
            yield {"status": "start", "total": total_images, "message": f"Found {total_images} new images to index."}
            
            new_captions = []
            new_paths = []
            
            for idx, image_path in enumerate(new_image_files, 1):
                try:
                    # Yield progress before processing
                    yield {"status": "progress", "current": idx, "total": total_images, "file": image_path.name}
                    
                    raw_image = Image.open(image_path).convert('RGB')
                    inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
                    out = self.caption_model.generate(
                        **inputs,
                        max_new_tokens=50,
                        repetition_penalty=1.1
                    )
                    caption = self.processor.decode(out[0], skip_special_tokens=True)
                    
                    new_captions.append(caption)
                    new_paths.append(str(image_path))
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    # Could yield an error event here if desired

            if not new_captions:
                 yield {"status": "error", "message": "Failed to caption images."}
                 return

            # Generate embeddings
            yield {"status": "embedding", "message": "Generating embeddings..."}
            
            new_embeddings_dict = {}
            for model_key in self.enabled_models:
                if model_key in self.embedding_models:
                    new_embeddings_dict[model_key] = self.embedding_models[model_key].encode_documents(new_captions)

            # Merge and Validate
            self.image_paths.extend(new_paths)
            self.captions.extend(new_captions)
            
            for model_key in self.enabled_models:
                if model_key in new_embeddings_dict:
                    if model_key in self.embeddings_dict:
                        self.embeddings_dict[model_key] = np.vstack([self.embeddings_dict[model_key], new_embeddings_dict[model_key]])
                    else:
                        self.embeddings_dict[model_key] = new_embeddings_dict[model_key]

            self._save_data()
            yield {"status": "complete", "indexed_count": len(new_paths), "message": "Indexing complete!"}

        except Exception as e:
            yield {"status": "error", "message": str(e)}

    # Legacy method wrapper if needed, or just remove original index_folder
    def index_folder(self, folder_path):
        """Legacy wrapper for backward compatibility if needed."""
        # This is blocking, but we can iterate the generator
        last_status = {}
        for status in self.index_folder_generator(folder_path):
            last_status = status
            if status.get("status") == "progress":
                print(f"Processing {status['current']}/{status['total']}: {status['file']}")
        return last_status


    def _save_data(self):
        """Save embeddings and metadata to disk."""
        self.output_dir.mkdir(exist_ok=True)
        
        metadata = {
            "image_paths": self.image_paths,
            "captions": self.captions
        }
        
        with open(self.output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
        for model_key, embeddings in self.embeddings_dict.items():
            np.save(self.output_dir / f"embeddings_{model_key}.npy", embeddings)

    def search(self, query, top_k=20):
        """Search for images."""
        if not self.embeddings_dict or not self.image_paths:
            return {"error": "No indexed images."}
        
        # Ensure models are loaded
        for model_key in self.enabled_models:
            if model_key not in self.embedding_models:
                self.embedding_models[model_key] = create_embedding_model(model_key)

        results = {}
        
        for model_key in self.enabled_models:
            if model_key not in self.embeddings_dict:
                continue
                
            query_embedding = self.embedding_models[model_key].encode_query(query)
            similarities = self.embedding_models[model_key].compute_similarity(
                query_embedding, self.embeddings_dict[model_key]
            )
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            model_results = []
            for idx in top_indices:
                model_results.append({
                    "path": self.image_paths[idx],
                    "caption": self.captions[idx],
                    "similarity": float(similarities[idx])
                })
            
            results[model_key] = model_results
            
        return results
