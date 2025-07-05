#!/usr/bin/env python3
"""
Demo script for the CLIP Space Image Classifier
"""

from space_clip_classifier import SpaceCLIPClassifier
import os

def demo_basic_classification():
    """Demonstrate basic image classification"""
    print("🚀 Initializing CLIP Space Classifier...")
    classifier = SpaceCLIPClassifier()
    
    # Example space images from Unsplash
    test_images = [
        "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?w=800",  # Galaxy
        "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=800",  # Stars
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",  # Space
    ]
    
    print("\n🔍 Classifying Space Images...")
    print("=" * 50)
    
    for i, image_url in enumerate(test_images, 1):
        print(f"\n📸 Image {i}:")
        try:
            predictions = classifier.classify_image(image_url, top_k=3)
            
            for j, (category, confidence) in enumerate(predictions, 1):
                print(f"   {j}. {category.capitalize()}: {confidence:.3f}")
                
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
        
        # Show visualization
        print("\n📊 Generating visualization...")
        classifier.visualize_classification(custom_image, top_k=5)
        
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
    
    print("\n🎉 Demo completed! Check out the visualization above.")
    print("\n💡 To use with your own images:")
    print("   1. Replace the image URLs in demo.py")
    print("   2. Or use local file paths")
    print("   3. Run: python demo.py") 