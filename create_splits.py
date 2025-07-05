#!/usr/bin/env python3
"""
Create Train/Validation/Test Splits
Splits accumulated NASA images into proper ML splits
"""

import json
import random
from pathlib import Path
from typing import Dict, List

def create_splits(data_dir: str = "images", train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1):
    """Create train/validation/test splits from accumulated images"""
    
    data_dir = Path(data_dir)
    
    # Load all labels
    labels_file = data_dir / "rich_labels.json"
    if not labels_file.exists():
        print("❌ No labels file found. Run nasa_space_filter.py first.")
        return
    
    with open(labels_file, 'r') as f:
        all_labels = json.load(f)
    
    print(f"📊 Found {len(all_labels)} total images")
    
    # Get all image filenames
    image_files = list(all_labels.keys())
    
    # Shuffle for random split
    random.seed(42)  # For reproducible splits
    random.shuffle(image_files)
    
    # Calculate split sizes
    total_images = len(image_files)
    train_size = int(train_ratio * total_images)
    val_size = int(val_ratio * total_images)
    test_size = total_images - train_size - val_size
    
    # Create splits
    train_files = image_files[:train_size]
    val_files = image_files[train_size:train_size + val_size]
    test_files = image_files[train_size + val_size:]
    
    print(f"📊 Creating splits:")
    print(f"   Train: {len(train_files)} images ({train_ratio*100:.0f}%)")
    print(f"   Validation: {len(val_files)} images ({val_ratio*100:.0f}%)")
    print(f"   Test: {len(test_files)} images ({test_ratio*100:.0f}%)")
    
    # Create split label files
    splits = {
        'train': train_files,
        'validation': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        split_labels = {filename: all_labels[filename] for filename in files}
        
        split_labels_file = data_dir / f"labels_{split_name}.json"
        with open(split_labels_file, 'w') as f:
            json.dump(split_labels, f, indent=2)
        
        print(f"   ✅ Saved {split_name} labels: {len(split_labels)} images")
    
    # Save split metadata
    split_metadata = {
        'total_images': total_images,
        'split_ratios': {
            'train': train_ratio,
            'validation': val_ratio,
            'test': test_ratio
        },
        'split_sizes': {
            'train': len(train_files),
            'validation': len(val_files),
            'test': len(test_files)
        },
        'split_files': {
            'train': train_files,
            'validation': val_files,
            'test': test_files
        }
    }
    
    split_metadata_file = data_dir / "split_metadata.json"
    with open(split_metadata_file, 'w') as f:
        json.dump(split_metadata, f, indent=2)
    
    print(f"📁 Split metadata saved to: {split_metadata_file}")
    
    # Show sample images from each split
    print(f"\n📋 Sample images from each split:")
    for split_name, files in splits.items():
        print(f"\n{split_name.capitalize()} split (first 3):")
        for i, filename in enumerate(files[:3]):
            caption = all_labels[filename][:80] + "..." if len(all_labels[filename]) > 80 else all_labels[filename]
            print(f"   {i+1}. {filename}: {caption}")

def main():
    """Main function"""
    print("🔀 Create Train/Validation/Test Splits")
    print("=" * 50)
    print("Splits accumulated NASA images into proper ML splits")
    print()
    
    # Get user input for split ratios
    print("Enter split ratios (must sum to 1.0):")
    train_ratio = float(input("Train ratio (default: 0.8): ") or "0.8")
    val_ratio = float(input("Validation ratio (default: 0.1): ") or "0.1")
    test_ratio = float(input("Test ratio (default: 0.1): ") or "0.1")
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        print(f"❌ Ratios sum to {total_ratio}, not 1.0. Please adjust.")
        return
    
    print(f"\n🚀 Creating splits with ratios: {train_ratio:.1f}/{val_ratio:.1f}/{test_ratio:.1f}")
    
    # Create splits
    create_splits(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    
    print(f"\n✅ Splits created successfully!")
    print(f"💡 You can now run the training script with the split labels.")

if __name__ == "__main__":
    main() 