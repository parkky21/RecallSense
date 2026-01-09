#!/usr/bin/env python3
"""
Script to download BLIP captioning models.
Pre-downloads the model to avoid delays during first use.
"""

from transformers import BlipProcessor, BlipForConditionalGeneration
from config import get_caption_model_name, CAPTION_MODELS
import sys

def download_blip_model(model_name: str):
    """Download and cache a BLIP model."""
    print(f"Downloading BLIP model: {model_name}")
    print("This may take a few minutes depending on your internet connection...")
    print("-" * 60)
    
    try:
        # Download processor
        print(f"📥 Downloading processor for {model_name}...")
        processor = BlipProcessor.from_pretrained(model_name, use_fast=False)
        print("✅ Processor downloaded successfully")
        
        # Download model
        print(f"📥 Downloading model {model_name}...")
        caption_model = BlipForConditionalGeneration.from_pretrained(model_name)
        print("✅ Model downloaded successfully")
        
        print("-" * 60)
        print(f"✅ Successfully downloaded and cached: {model_name}")
        print(f"   Model location: ~/.cache/huggingface/hub/")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space")
        print("3. Check Hugging Face authentication if required")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_key = sys.argv[1].lower()
        if model_key in CAPTION_MODELS:
            model_name = CAPTION_MODELS[model_key]
        else:
            print(f"❌ Unknown model key: {model_key}")
            print(f"Available models: {list(CAPTION_MODELS.keys())}")
            sys.exit(1)
    else:
        # Use the configured model
        model_name = get_caption_model_name()
        print(f"Using configured model: {model_name}")
    
    success = download_blip_model(model_name)
    sys.exit(0 if success else 1)
