#!/usr/bin/env python3
"""
Demo script for the CLIP Space Image Classifier
"""

from space_clip_classifier import SpaceCLIPClassifier
import os
import torch

def demo_basic_classification():
    """Demonstrate basic image classification"""
    print("🚀 Initializing CLIP Space Classifier...")
    classifier = SpaceCLIPClassifier()
    
    # Example space images from Unsplash with captions
    test_images = [
        {
            "url": "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=800",
            "caption": "A field of bright stars scattered across the night sky"
        },
        {
            "url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",
            "caption": "A view of deep space with distant galaxies and nebulae"
        },
        {
            "url": "https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?w=800",
            "caption": "A colorful nebula with swirling gas clouds"
        },
        {
            "url": "https://images.unsplash.com/photo-1516339901601-2e1b62dc0c45?w=800",
            "caption": "A planet with visible surface features and atmosphere"
        },
        {
            "url": "https://images.unsplash.com/photo-1614728894747-a83421e2b9c9?w=800",
            "caption": "A black hole concept with glowing accretion disk"
        },
        {
            "url": "https://images.unsplash.com/photo-1543722530-d2c3201371e7?w=800",
            "caption": "A spiral galaxy with bright arms and a central bulge"
        },
    ]
    
    print("\n🔍 Classifying Space Images...")
    print("=" * 50)
    
    for i, img in enumerate(test_images, 1):
        print(f"\n📸 Image {i}:")
        try:
            # Use the caption as text input if available
            if "caption" in img:
                inputs = classifier.processor(
                    text=[img["caption"]],
                    images=classifier.load_image(img["url"]),
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                inputs = {k: v.to(classifier.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = classifier.model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = torch.nn.functional.softmax(logits_per_image, dim=1)
                top_probs, top_indices = torch.topk(probs, 3, dim=1)
                print("   Using caption as text input:")
                for j in range(3):
                    print(f"   {j+1}. {img['caption']}: {top_probs[0][j].item():.3f}")
                # Show visualization for each image
                print(f"\n📊 Generating visualization for Image {i}...")
                classifier.visualize_classification(img["url"], top_k=5)
            else:
                predictions = classifier.classify_image(img["url"], top_k=3)
                for j, (category, confidence) in enumerate(predictions, 1):
                    print(f"   {j}. {category.capitalize()}: {confidence:.3f}")
                print(f"\n📊 Generating visualization for Image {i}...")
                classifier.visualize_classification(img["url"], top_k=5)
        except Exception as e:
            print(f"   ❌ Error: {e}")
    print("\n" + "=" * 50)
    print("✅ Classification complete!")

def demo_custom_image():
    """Demonstrate classification with a custom image"""
    print("\n🎯 Custom Image Classification Demo")
    print("=" * 50)
    
    classifier = SpaceCLIPClassifier()
    
    # You can replace this with your own space image
    custom_image = "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?w=800"
    
    print(f"Classifying: {custom_image}")
    
    try:
        predictions = classifier.classify_image(custom_image, top_k=5)
        
        print("\nTop 5 Predictions:")
        for i, (category, confidence) in enumerate(predictions, 1):
            print(f"   {i}. {category.capitalize()}: {confidence:.3f}")
        
        # Note: Visualization is now handled in demo_basic_classification
        
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_space_categories():
    """Show all available space categories"""
    print("\n📋 Available Space Categories")
    print("=" * 50)
    
    classifier = SpaceCLIPClassifier()
    
    categories = classifier.space_categories
    print(f"Total categories: {len(categories)}")
    print("\nCategories:")
    
    for i, category in enumerate(categories, 1):
        print(f"   {i:2d}. {category}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    print("🌟 CLIP Space Image Classifier Demo")
    print("=" * 50)
    
    # Show available categories
    demo_space_categories()
    
    # Basic classification demo
    demo_basic_classification()
    
    # Custom image demo with visualization
    demo_custom_image()
    
    print("\n🎉 Demo completed! Check out the visualizations above.")
    print("\n💡 To use with your own images:")
    print("   1. Replace the image URLs in demo.py")
    print("   2. Or use local file paths")
    print("   3. Run: python demo.py") 