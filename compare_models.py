#!/usr/bin/env python3
"""
Model Comparison Demo - Compare pretrained vs fine-tuned CLIP models
"""

from space_clip_classifier import SpaceCLIPClassifier
from space_clip_trainer import ModelComparison
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple

def compare_single_image(image_path: str, fine_tuned_path: str = None, caption: str = ""):
    """Compare models on a single image"""
    print(f"🔍 Comparing models on: {image_path}")
    if caption:
        print(f"📝 Caption: {caption}")
    print("=" * 60)
    
    # Initialize classifiers
    pretrained = SpaceCLIPClassifier()
    
    if fine_tuned_path:
        fine_tuned = SpaceCLIPClassifier(fine_tuned_path=fine_tuned_path)
        print("✅ Loaded both pretrained and fine-tuned models")
    else:
        fine_tuned = None
        print("⚠️  No fine-tuned model provided - showing only pretrained results")
    
    # Get predictions using categories (default method)
    pretrained_preds = pretrained.classify_image(image_path, top_k=5)
    
    print("\n📊 Pretrained CLIP Results (Categories):")
    for i, (category, confidence) in enumerate(pretrained_preds, 1):
        print(f"   {i}. {category.capitalize()}: {confidence:.3f}")
    
    if fine_tuned:
        fine_tuned_preds = fine_tuned.classify_image(image_path, top_k=5)
        
        print("\n📊 Fine-tuned CLIP Results (Categories):")
        for i, (category, confidence) in enumerate(fine_tuned_preds, 1):
            print(f"   {i}. {category.capitalize()}: {confidence:.3f}")
        
        # Compare top predictions
        print("\n🔄 Category Comparison:")
        pretrained_top = pretrained_preds[0]
        fine_tuned_top = fine_tuned_preds[0]
        
        print(f"   Pretrained top: {pretrained_top[0]} ({pretrained_top[1]:.3f})")
        print(f"   Fine-tuned top:  {fine_tuned_top[0]} ({fine_tuned_top[1]:.3f})")
        
        if pretrained_top[0] == fine_tuned_top[0]:
            print("   ✅ Same top prediction!")
        else:
            print("   ❌ Different top predictions")
    
    # If caption is provided, also compare using caption
    if caption:
        print(f"\n📝 Caption-based Comparison:")
        print(f"Caption: '{caption}'")
        
        # Pretrained with caption
        pretrained_caption_score = pretrained.classify_with_caption(image_path, caption)
        print(f"   Pretrained caption match: {pretrained_caption_score:.3f}")
        
        if fine_tuned:
            fine_tuned_caption_score = fine_tuned.classify_with_caption(image_path, caption)
            print(f"   Fine-tuned caption match: {fine_tuned_caption_score:.3f}")
            
            improvement = fine_tuned_caption_score - pretrained_caption_score
            print(f"   Improvement: {improvement:+.3f}")
            if improvement > 0:
                print("   🎉 Fine-tuned model better matches the caption!")
            else:
                print("   ⚠️  Pretrained model better matches the caption")
        
        # Show visualizations
        comparison = ModelComparison(fine_tuned_model_path=fine_tuned_path)
        comparison.visualize_comparison(image_path, top_k=5)
    else:
        # Just show category-based visualization
        if fine_tuned:
            comparison = ModelComparison(fine_tuned_model_path=fine_tuned_path)
            comparison.visualize_comparison(image_path, top_k=5)
        else:
            pretrained.visualize_classification(image_path, top_k=5)

def compare_multiple_images(image_paths: List[str], fine_tuned_path: str = None):
    """Compare models on multiple images and show detailed results for each"""
    print("🔍 Comparing models on multiple images...")
    print("=" * 60)
    
    pretrained = SpaceCLIPClassifier()
    fine_tuned = None
    if fine_tuned_path:
        fine_tuned = SpaceCLIPClassifier(fine_tuned_path=fine_tuned_path)
    
    # Test images with captions (same as main function)
    test_images = [
        {
            "url": "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?w=800",
            "caption": "A stunning spiral galaxy with bright stars and cosmic dust clouds"
        },
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
        }
    ]
    
    results = {
        'pretrained': [],
        'fine_tuned': [],
        'agreement': 0,
        'total': len(test_images)
    }
    
    for i, img in enumerate(test_images, 1):
        print(f"\n📸 Image {i}/{len(test_images)}: {img['url']}")
        print(f"📝 Caption: {img['caption']}")
        
        # Category-based predictions
        pretrained_preds = pretrained.classify_image(img['url'], top_k=1)
        results['pretrained'].append(pretrained_preds[0])
        
        print(f"   Pretrained (categories): {pretrained_preds[0][0]} ({pretrained_preds[0][1]:.3f})")
        
        # Caption-based predictions
        pretrained_caption_score = pretrained.classify_with_caption(img['url'], img['caption'])
        print(f"   Pretrained (caption): {pretrained_caption_score:.3f}")
        
        if fine_tuned:
            fine_tuned_preds = fine_tuned.classify_image(img['url'], top_k=1)
            results['fine_tuned'].append(fine_tuned_preds[0])
            
            fine_tuned_caption_score = fine_tuned.classify_with_caption(img['url'], img['caption'])
            
            print(f"   Fine-tuned (categories): {fine_tuned_preds[0][0]} ({fine_tuned_preds[0][1]:.3f})")
            print(f"   Fine-tuned (caption): {fine_tuned_caption_score:.3f}")
            
            # Check if top predictions agree
            if pretrained_preds[0][0] == fine_tuned_preds[0][0]:
                results['agreement'] += 1
                print(f"   ✅ Category agreement: {pretrained_preds[0][0]}")
            else:
                print(f"   ❌ Category disagreement: {pretrained_preds[0][0]} vs {fine_tuned_preds[0][0]}")
            
            # Compare caption scores
            caption_improvement = fine_tuned_caption_score - pretrained_caption_score
            print(f"   📝 Caption improvement: {caption_improvement:+.3f}")
        else:
            print(f"   ⚠️  No fine-tuned model for comparison")
    
    # Show aggregate statistics
    print(f"\n📈 Aggregate Statistics:")
    print(f"   Total images: {results['total']}")
    if fine_tuned:
        agreement_rate = results['agreement'] / results['total']
        print(f"   Category agreement rate: {agreement_rate:.1%}")
        print(f"   Disagreement rate: {1-agreement_rate:.1%}")

def create_comparison_visualization(image_paths: List[str], fine_tuned_path: str = None):
    """Create a comprehensive comparison visualization"""
    if not fine_tuned_path:
        print("⚠️  No fine-tuned model provided for comparison")
        return
    
    pretrained = SpaceCLIPClassifier()
    fine_tuned = SpaceCLIPClassifier(fine_tuned_path=fine_tuned_path)
    
    # Get predictions for all images
    pretrained_preds = []
    fine_tuned_preds = []
    
    for image_path in image_paths:
        pretrained_preds.append(pretrained.classify_image(image_path, top_k=1)[0])
        fine_tuned_preds.append(fine_tuned.classify_image(image_path, top_k=1)[0])
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Confidence comparison
    pretrained_conf = [pred[1] for pred in pretrained_preds]
    fine_tuned_conf = [pred[1] for pred in fine_tuned_preds]
    
    x = np.arange(len(image_paths))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, pretrained_conf, width, label='Pretrained', color='lightblue')
    bars2 = ax1.bar(x + width/2, fine_tuned_conf, width, label='Fine-tuned', color='lightgreen')
    
    ax1.set_xlabel('Image Index')
    ax1.set_ylabel('Confidence Score')
    ax1.set_title('Confidence Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Img {i+1}' for i in range(len(image_paths))])
    ax1.legend()
    
    # Agreement analysis
    agreements = [1 if p[0] == f[0] else 0 for p, f in zip(pretrained_preds, fine_tuned_preds)]
    agreement_rate = sum(agreements) / len(agreements)
    
    ax2.bar(['Agreement', 'Disagreement'], 
            [agreement_rate, 1-agreement_rate], 
            color=['green', 'red'])
    ax2.set_ylabel('Proportion')
    ax2.set_title(f'Prediction Agreement ({agreement_rate:.1%})')
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main comparison demo"""
    print("🌟 CLIP Model Comparison Demo")
    print("=" * 60)
    
    # Test images with captions (same as demo)
    test_images = [
        {
            "url": "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?w=800",
            "caption": "A stunning spiral galaxy with bright stars and cosmic dust clouds"
        },
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
        }
    ]
    
    # Fine-tuned model path (uncomment when you have a trained model)
    fine_tuned_path = "fine_tuned_clip"  # Use your trained model
    
    print("1. Single image comparison")
    compare_single_image(test_images[0]["url"], fine_tuned_path, test_images[0]["caption"])
    
    print("\n" + "=" * 60)
    print("2. Multiple image comparison")
    compare_multiple_images([img["url"] for img in test_images], fine_tuned_path)
    
    if fine_tuned_path:
        print("\n" + "=" * 60)
        print("3. Comprehensive visualization")
        create_comparison_visualization([img["url"] for img in test_images], fine_tuned_path)
    
    print("\n💡 To use with your fine-tuned model:")
    print("   1. Train a model using space_clip_trainer.py")
    print("   2. Set fine_tuned_path = 'your_model_directory'")
    print("   3. Run this script again")

if __name__ == "__main__":
    main() 