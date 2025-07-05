#!/usr/bin/env python3
"""
Filter Flickr30k Dataset for Space Images
Loads the dataset and filters for space-related images with human-written captions
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
from io import BytesIO
import time
from datasets import load_dataset
import re

class Flickr30kSpaceFilter:
    """Filter Flickr30k dataset for space-related images"""
    
    def __init__(self, output_dir: str = "images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Space keywords for filtering
        self.space_keywords = [
            "space", "galaxy", "planet", "moon", "nasa", "orbit",
            "nebula", "telescope", "milky way", "cosmos", "universe", "solar system",
            "mars", "jupiter", "saturn", "venus", "mercury", "uranus", "neptune",
            "star", "supernova", "black hole", "hubble", "iss", "spacex", "meteor", "comet", "asteroid", 
            "lunar", "eclipse", "deep space", "white hole", "pulsar", "quasar", "light-year",
            "spaceship", "observatory", "gravity", "cosmic", "exoplanet", "extraterrestrial",
            "sky", "skywatch", "satellite", "astronaut", "spacecraft", "rocket", "launch",
            "constellation", "aurora", "meteorite", "solar", "stellar", "interstellar",
            "astronomical", "celestial", "orbital", "space station", "rover", "probe"
        ]
        
        # Create keyword patterns for better matching
        self.space_patterns = [re.compile(rf'\b{keyword}\b', re.IGNORECASE) for keyword in self.space_keywords]
    
    def is_space_related(self, caption: str) -> bool:
        """Check if caption contains space-related keywords"""
        caption_lower = caption.lower()
        
        # Check for exact keyword matches
        for pattern in self.space_patterns:
            if pattern.search(caption):
                return True
        
        # Additional checks for compound terms
        if any(term in caption_lower for term in ["milky way", "black hole", "deep space", "solar system"]):
            return True
        
        return False
    
    def download_image(self, image, filename: str) -> bool:
        """Save PIL Image to output directory"""
        try:
            # Save image
            filepath = self.output_dir / filename
            image.save(filepath)
            return True
            
        except Exception as e:
            print(f"❌ Failed to save {filename}: {e}")
            return False
    
    def filter_and_download(self, max_images: int = 100) -> Dict:
        """Filter Flickr30k dataset and download space-related images with proper splits"""
        print(f"🔍 Loading Flickr30k dataset...")
        
        try:
            dataset = load_dataset("nlphuji/flickr30k", split="test")
            print(f"✅ Loaded {len(dataset)} images from Flickr30k")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return {}
        
        print(f"🔍 Filtering for space-related images...")
        print(f"🔍 Keywords being searched: {', '.join(self.space_keywords[:10])}...")
        print()
        
        # First pass: find space-related images (with limit)
        print(f"🔍 Finding space-related images (max: {max_images})...")
        space_candidates = []
        
        for i, item in enumerate(dataset):
            # Stop if we have enough images
            if len(space_candidates) >= max_images:
                break
                
            # Check all captions for space keywords
            captions = item['caption']  # This is a list of captions
            is_space = False
            best_caption = ""
            matched_keyword = ""
            
            for caption in captions:
                if self.is_space_related(caption):
                    is_space = True
                    best_caption = caption
                    # Find which keyword matched
                    for keyword in self.space_keywords:
                        if keyword.lower() in caption.lower():
                            matched_keyword = keyword
                            break
                    break
            
            if is_space:
                space_candidates.append({
                    'item': item,
                    'caption': best_caption,
                    'matched_keyword': matched_keyword,
                    'index': i
                })
            
            # Progress update
            if (i + 1) % 1000 == 0:
                print(f"  📊 Processed {i+1} images, found {len(space_candidates)} space candidates")
        
        print(f"✅ Found {len(space_candidates)} space-related images")
        
        # Create train/val/test splits (80/10/10)
        total_images = len(space_candidates)
        train_size = int(0.8 * total_images)
        val_size = int(0.1 * total_images)
        test_size = total_images - train_size - val_size
        
        train_candidates = space_candidates[:train_size]
        val_candidates = space_candidates[train_size:train_size + val_size]
        test_candidates = space_candidates[train_size + val_size:]
        
        print(f"📊 Creating splits:")
        print(f"   Train: {len(train_candidates)} images (80%)")
        print(f"   Validation: {len(val_candidates)} images (10%)")
        print(f"   Test: {len(test_candidates)} images (10%)")
        
        # Download images for each split
        splits = {
            'train': train_candidates,
            'validation': val_candidates,
            'test': test_candidates
        }
        
        space_images = []
        downloaded_count = 0
        preview_count = 0
        
        for split_name, candidates in splits.items():
            print(f"\n📥 Downloading {split_name} split ({len(candidates)} images)...")
            
            for i, candidate in enumerate(candidates):
                # Show preview for first few matches
                if preview_count < 10:
                    print(f"🎯 Found space image #{preview_count + 1} ({split_name}):")
                    print(f"   Keyword matched: '{candidate['matched_keyword']}'")
                    print(f"   Caption: '{candidate['caption']}'")
                    print(f"   Image: {candidate['item']['image']}")
                    print()
                    preview_count += 1
                
                # Download the image
                image = candidate['item']['image']
                filename = f"flickr30k_space_{split_name}_{i+1:04d}.jpg"
                
                print(f"  📥 Downloading {split_name} {i+1}/{len(candidates)}: {filename}")
                print(f"    Caption: {candidate['caption'][:80]}...")
                
                if self.download_image(image, filename):
                    space_images.append({
                        'filename': filename,
                        'caption': candidate['caption'],
                        'split': split_name,
                        'original_index': candidate['index'],
                        'image': str(image),
                        'matched_keyword': candidate['matched_keyword']
                    })
                    downloaded_count += 1
                    time.sleep(0.5)  # Be nice to servers
                else:
                    print(f"    ❌ Download failed")
        
        print(f"✅ Downloaded {len(space_images)} space-related images across all splits")
        
        print(f"✅ Downloaded {len(space_images)} space-related images")
        
        # Save metadata
        metadata = {
            'total_processed': len(dataset),
            'space_images_found': len(space_images),
            'space_keywords_used': self.space_keywords,
            'splits': {
                'train': len([img for img in space_images if img['split'] == 'train']),
                'validation': len([img for img in space_images if img['split'] == 'validation']),
                'test': len([img for img in space_images if img['split'] == 'test'])
            },
            'images': space_images
        }
        
        metadata_file = self.output_dir / "flickr30k_space_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create labels file for training (only train split)
        train_labels = {}
        for img_data in space_images:
            if img_data['split'] == 'train':
                train_labels[img_data['filename']] = img_data['caption']
        
        labels_file = self.output_dir / "rich_labels.json"
        with open(labels_file, 'w') as f:
            json.dump(train_labels, f, indent=2)
        
        # Create separate label files for each split
        for split_name in ['train', 'validation', 'test']:
            split_labels = {}
            for img_data in space_images:
                if img_data['split'] == split_name:
                    split_labels[img_data['filename']] = img_data['caption']
            
            split_labels_file = self.output_dir / f"labels_{split_name}.json"
            with open(split_labels_file, 'w') as f:
                json.dump(split_labels, f, indent=2)
        
        print(f"📁 Metadata saved to: {metadata_file}")
        print(f"📁 Labels saved to: {labels_file}")
        
        return metadata
    
    def show_samples(self, num_samples: int = 5):
        """Show sample images and captions"""
        print(f"\n📋 Sample Space Images ({num_samples} of {len(list(self.output_dir.glob('*.jpg')))}):")
        print("=" * 80)
        
        # Load metadata
        metadata_file = self.output_dir / "flickr30k_space_metadata.json"
        if not metadata_file.exists():
            print("❌ No metadata file found. Run filtering first.")
            return
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Show samples
        for i, img_data in enumerate(metadata['images'][:num_samples], 1):
            print(f"{i:2d}. {img_data['filename']}")
            print(f"    Caption: {img_data['caption']}")
            print()
    
    def get_statistics(self) -> Dict:
        """Get statistics about the filtered dataset"""
        print(f"\n📊 Dataset Statistics:")
        
        # Count images
        image_files = list(self.output_dir.glob("*.jpg")) + list(self.output_dir.glob("*.png"))
        print(f"   Total images: {len(image_files)}")
        
        # Load metadata
        metadata_file = self.output_dir / "flickr30k_space_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            print(f"   Images processed: {metadata['total_processed']}")
            print(f"   Space images found: {metadata['space_images_found']}")
            print(f"   Filter rate: {metadata['space_images_found']/metadata['total_processed']*100:.2f}%")
        
        # Check labels file
        labels_file = self.output_dir / "rich_labels.json"
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                labels = json.load(f)
            print(f"   Captions created: {len(labels)}")
        
        return {
            'total_images': len(image_files),
            'has_metadata': metadata_file.exists(),
            'has_labels': labels_file.exists()
        }

def main():
    """Main filtering workflow"""
    print("🌟 Flickr30k Space Image Filter")
    print("=" * 50)
    
    # Initialize filter
    filter_tool = Flickr30kSpaceFilter("images")
    
    # Get user input
    max_images = input("How many space images to download? (default: 100): ").strip()
    max_images = int(max_images) if max_images.isdigit() else 100
    
    print(f"\n🚀 Starting filtering process...")
    print(f"   Max images: {max_images}")
    print(f"   Splits: Train (80%) / Validation (10%) / Test (10%)")
    print(f"   Keywords: {len(filter_tool.space_keywords)} space-related terms")
    
    # Filter and download
    metadata = filter_tool.filter_and_download(max_images)
    
    if metadata:
        # Show samples
        filter_tool.show_samples(5)
        
        # Show statistics
        filter_tool.get_statistics()
        
        print(f"\n✅ Filtering complete!")
        print(f"\n📚 Next steps:")
        print(f"   1. Review the sample captions above")
        print(f"   2. Check images/ folder for downloaded images")
        print(f"   3. Run training: python space_clip_trainer.py")
        print(f"   4. Compare models: python compare_models.py")
    else:
        print("❌ Filtering failed. Check the error messages above.")

if __name__ == "__main__":
    main() 