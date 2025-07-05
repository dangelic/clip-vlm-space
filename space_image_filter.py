#!/usr/bin/env python3
"""
Utility script to filter space images from a dataset using CLIP
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
from space_clip_classifier import SpaceCLIPClassifier
import argparse

class SpaceImageFilter:
    """
    Filter space images from a dataset using CLIP classification
    """
    
    def __init__(self, threshold: float = 0.1):
        """
        Initialize the space image filter
        
        Args:
            threshold: Minimum confidence score to consider an image as space-related
        """
        self.classifier = SpaceCLIPClassifier()
        self.threshold = threshold
        
        # Keywords that indicate space-related content
        self.space_keywords = [
            "galaxy", "nebula", "star", "planet", "moon", "asteroid", "comet",
            "black hole", "supernova", "cosmic", "solar", "constellation",
            "meteor", "satellite", "space", "telescope", "astronaut",
            "spacecraft", "mars", "hubble", "james webb", "iss", "aurora",
            "milky way", "andromeda", "orion"
        ]
    
    def is_space_image(self, image_path: str) -> Tuple[bool, float, str]:
        """
        Determine if an image is space-related
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (is_space, confidence, top_category)
        """
        try:
            predictions = self.classifier.classify_image(image_path, top_k=1)
            if predictions:
                top_category, confidence = predictions[0]
                
                # Check if the top category is space-related
                is_space = any(keyword in top_category.lower() for keyword in self.space_keywords)
                
                return is_space and confidence >= self.threshold, confidence, top_category
            else:
                return False, 0.0, "unknown"
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False, 0.0, "error"
    
    def filter_directory(self, input_dir: str, output_dir: str | None = None) -> Dict:
        """
        Filter space images from a directory
        
        Args:
            input_dir: Directory containing images to filter
            output_dir: Directory to copy space images to (optional)
            
        Returns:
            Dictionary with 'space_images' and 'non_space_images' lists
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images in {input_dir}")
        
        space_images = []
        non_space_images = []
        
        for i, image_file in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {image_file.name}")
            
            is_space, confidence, category = self.is_space_image(str(image_file))
            
            if is_space:
                space_images.append({
                    'path': str(image_file),
                    'confidence': confidence,
                    'category': category
                })
                print(f"  ✅ Space image: {category} ({confidence:.3f})")
            else:
                non_space_images.append(str(image_file))
                print(f"  ❌ Non-space image: {category} ({confidence:.3f})")
        
        results = {
            'space_images': space_images,
            'non_space_images': non_space_images,
            'total_images': len(image_files),
            'space_count': len(space_images),
            'non_space_count': len(non_space_images)
        }
        
        # Save results
        if output_dir is not None:
            self._save_results(results, output_dir)
        
        return results
    
    def _save_results(self, results: Dict, output_dir: str):
        """Save filtering results to a JSON file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / "filtering_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
    
    def create_space_dataset(self, input_dir: str, output_dir: str):
        """
        Create a space-only dataset by copying filtered images
        
        Args:
            input_dir: Directory containing images to filter
            output_dir: Directory to save space images
        """
        import shutil
        
        results = self.filter_directory(input_dir)
        space_images = results['space_images']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying {len(space_images)} space images to {output_dir}")
        
        for i, img_info in enumerate(space_images, 1):
            src_path = Path(img_info['path'])
            dst_path = output_path / f"space_{i:04d}_{img_info['category']}_{img_info['confidence']:.3f}{src_path.suffix}"
            
            try:
                shutil.copy2(src_path, dst_path)
                print(f"  Copied: {src_path.name} -> {dst_path.name}")
            except Exception as e:
                print(f"  Error copying {src_path.name}: {e}")
        
        print(f"\n✅ Space dataset created with {len(space_images)} images")

def main():
    parser = argparse.ArgumentParser(description="Filter space images from a dataset using CLIP")
    parser.add_argument("input_dir", help="Directory containing images to filter")
    parser.add_argument("--output-dir", help="Directory to save results")
    parser.add_argument("--threshold", type=float, default=0.1, 
                       help="Minimum confidence threshold (default: 0.1)")
    parser.add_argument("--create-dataset", action="store_true",
                       help="Create a space-only dataset by copying filtered images")
    
    args = parser.parse_args()
    
    # Initialize filter
    filter_tool = SpaceImageFilter(threshold=args.threshold)
    
    print(f"🚀 Space Image Filter")
    print(f"Input directory: {args.input_dir}")
    print(f"Confidence threshold: {args.threshold}")
    print("=" * 50)
    
    if args.create_dataset:
        if not args.output_dir:
            args.output_dir = "space_dataset"
        filter_tool.create_space_dataset(args.input_dir, args.output_dir)
    else:
        results = filter_tool.filter_directory(args.input_dir, args.output_dir)
        
        print(f"\n📊 Filtering Results:")
        print(f"Total images: {results['total_images']}")
        print(f"Space images: {results['space_count']}")
        print(f"Non-space images: {results['non_space_count']}")
        print(f"Space percentage: {results['space_count']/results['total_images']*100:.1f}%")

if __name__ == "__main__":
    main() 