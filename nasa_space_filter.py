#!/usr/bin/env python3
"""
NASA Space Image Filter
Downloads high-quality space images from NASA's APOD (Astronomy Picture of the Day) dataset
"""

import json
import time
import requests
import os
from pathlib import Path
from typing import Dict, List
from PIL import Image
import re

# Try to load dotenv, fallback to manual loading if not available
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(filename):
        """Simple fallback for loading .env files"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value

class NASASpaceFilter:
    """Filter and download NASA space images with proper astronomical captions"""
    
    def __init__(self, output_dir: str = "images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load environment variables
        load_dotenv('config.env')
        
        # NASA APOD dataset has high-quality space images with detailed captions
        print("🚀 Initializing NASA Space Image Filter")
        print("   Using NASA APOD API for high-quality astronomical images")
        print("   Images come with detailed astronomical descriptions")
    
    def download_image(self, image_url: str, filename: str) -> bool:
        """Download image from NASA URL"""
        try:
            response = requests.get(image_url, timeout=15)
            response.raise_for_status()
            
            # Save image
            filepath = self.output_dir / filename
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            # Verify image can be opened
            with Image.open(filepath) as img:
                img.verify()
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {image_url}: {e}")
            return False
    
    def filter_and_download(self, max_images: int = 100) -> Dict:
        """Download NASA APOD images using NASA's API"""
        print(f"🔍 Fetching NASA APOD images...")
        
        # NASA APOD API endpoint
        api_url = "https://api.nasa.gov/planetary/apod"
        
        # Get API key from environment variables
        api_key = os.getenv('NASA_API_KEY', 'DEMO_KEY')
        start_date = os.getenv('APOD_START_DATE', '2023-01-01')
        end_date = os.getenv('APOD_END_DATE', '2024-12-31')
        
        if api_key == 'DEMO_KEY':
            print(f"   Using NASA APOD API (demo key - limited to 1000 requests/day)")
            print(f"   Get your free API key at: https://api.nasa.gov/")
            print(f"   Then update config.env with your API key")
        else:
            print(f"   Using NASA APOD API with your API key")
        print()
        
        # Fetch recent APOD images
        space_candidates = []
        
        try:
            # Fetch images in batches
            params = {
                'api_key': api_key,
                'start_date': start_date,
                'end_date': end_date,
                'count': min(max_images * 2, 100)  # Get more than needed to filter
            }
            
            print(f"🔍 Fetching NASA APOD images from {start_date} to {end_date}...")
            response = requests.get(api_url, params=params, timeout=30)
            response.raise_for_status()
            
            apod_data = response.json()
            print(f"✅ Fetched {len(apod_data)} NASA APOD entries")
            
            # Filter for images with valid URLs
            for i, item in enumerate(apod_data):
                if len(space_candidates) >= max_images:
                    break
                
                # Check if it's an image (not video)
                if item.get('media_type') == 'image' and item.get('hdurl'):
                    title = item.get('title', '')
                    explanation = item.get('explanation', '')
                    date = item.get('date', '')
                    url = item.get('hdurl')
                    
                    # Combine title and explanation for rich caption
                    caption = f"{title}. {explanation}"
                    
                    space_candidates.append({
                        'item': item,
                        'caption': caption,
                        'title': title,
                        'explanation': explanation,
                        'url': url,
                        'date': date,
                        'index': i
                    })
                
                # Progress update
                if (i + 1) % 20 == 0:
                    print(f"  📊 Processed {i+1} entries, found {len(space_candidates)} valid images")
            
            print(f"✅ Found {len(space_candidates)} NASA space images")
            
        except Exception as e:
            print(f"❌ Failed to fetch NASA APOD data: {e}")
            print(f"   This might be due to API rate limits or network issues")
            return {}
        
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
                    print(f"🎯 Found NASA space image #{preview_count + 1} ({split_name}):")
                    print(f"   Title: '{candidate['title']}'")
                    print(f"   Date: {candidate['date']}")
                    print(f"   URL: {candidate['url'][:60]}...")
                    print()
                    preview_count += 1
                
                # Download the image
                image_url = candidate['url']
                filename = f"nasa_apod_{split_name}_{i+1:04d}.jpg"
                
                print(f"  📥 Downloading {split_name} {i+1}/{len(candidates)}: {filename}")
                print(f"    Title: {candidate['title'][:80]}...")
                
                if self.download_image(image_url, filename):
                    space_images.append({
                        'filename': filename,
                        'caption': candidate['caption'],
                        'title': candidate['title'],
                        'explanation': candidate['explanation'],
                        'split': split_name,
                        'date': candidate['date'],
                        'original_index': candidate['index'],
                        'url': image_url
                    })
                    downloaded_count += 1
                    time.sleep(1)  # Be nice to NASA servers
                else:
                    print(f"    ❌ Download failed")
        
        print(f"✅ Downloaded {len(space_images)} NASA space images across all splits")
        
        # Save metadata
        metadata = {
            'dataset_source': 'NASA APOD API',
            'api_key_used': 'DEMO_KEY',
            'date_range': f"{start_date} to {end_date}",
            'total_processed': len(apod_data) if 'apod_data' in locals() else 0,
            'space_images_found': len(space_images),
            'splits': {
                'train': len([img for img in space_images if img['split'] == 'train']),
                'validation': len([img for img in space_images if img['split'] == 'validation']),
                'test': len([img for img in space_images if img['split'] == 'test'])
            },
            'images': space_images
        }
        
        metadata_file = self.output_dir / "nasa_apod_metadata.json"
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
        print(f"\n📋 Sample NASA Space Images ({num_samples} of {len(list(self.output_dir.glob('*.jpg')))}):")
        print("=" * 80)
        
        # Load metadata
        metadata_file = self.output_dir / "nasa_apod_metadata.json"
        if not metadata_file.exists():
            print("❌ No metadata file found. Run filtering first.")
            return
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Show samples
        for i, img_data in enumerate(metadata['images'][:num_samples], 1):
            print(f"{i:2d}. {img_data['filename']}")
            print(f"    Title: {img_data['title']}")
            print(f"    Date: {img_data['date']}")
            print(f"    Caption: {img_data['caption'][:100]}...")
            print()

def main():
    """Main function"""
    print("🚀 NASA Space Image Filter")
    print("=" * 50)
    print("Downloads high-quality space images from NASA APOD dataset")
    print("Images come with detailed astronomical descriptions")
    print()
    
    filter_tool = NASASpaceFilter("images")
    
    # Get user input
    max_images = input("How many NASA space images to download? (default: 100): ").strip()
    max_images = int(max_images) if max_images.isdigit() else 100
    
    print(f"\n🚀 Starting NASA filtering process...")
    print(f"   Max images: {max_images}")
    print(f"   Splits: Train (80%) / Validation (10%) / Test (10%)")
    print(f"   Source: NASA APOD dataset")
    
    # Filter and download
    metadata = filter_tool.filter_and_download(max_images)
    
    if metadata:
        print(f"\n✅ Successfully downloaded {metadata['space_images_found']} NASA space images!")
        print(f"📊 Split distribution:")
        for split, count in metadata['splits'].items():
            print(f"   {split.capitalize()}: {count} images")
        
        # Show samples
        filter_tool.show_samples(3)
    else:
        print("❌ Failed to download images")

if __name__ == "__main__":
    main() 