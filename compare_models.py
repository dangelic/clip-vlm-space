#!/usr/bin/env python3
"""
Model Comparison Demo - Compare pretrained vs fine-tuned CLIP models
"""

from space_clip_classifier import SpaceCLIPClassifier
from space_clip_trainer import ModelComparison
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple

def compare_single_image(image_path: str, fine_tuned_path: str = None):
    """Compare models on a single image"""
    print(f"🔍 Comparing models on: {image_path}")
    print("=" * 60)
    
    # Initialize classifiers
    pretrained = SpaceCLIPClassifier()
    
    if fine_tuned_path:
        fine_tuned = SpaceCLIPClassifier(fine_tuned_path=fine_tuned_path)
        print("✅ Loaded both pretrained and fine-tuned models")
    else:
        fine_tuned = None
        print("⚠️  No fine-tuned model provided - showing only pretrained results")
    
    # Get predictions
    pretrained_preds = pretrained.classify_image(image_path, top_k=5)
    
    print("\n📊 Pretrained CLIP Results:")
    for i, (category, confidence) in enumerate(pretrained_preds, 1):
        print(f"   {i}. {category.capitalize()}: {confidence:.3f}")
    
    if fine_tuned:
        fine_tuned_preds = fine_tuned.classify_image(image_path, top_k=5)
        
        print("\n📊 Fine-tuned CLIP Results:")
        for i, (category, confidence) in enumerate(fine_tuned_preds, 1):
            print(f"   {i}. {category.capitalize()}: {confidence:.3f}")
        
        # Compare top predictions
        print("\n🔄 Comparison:")
        pretrained_top = pretrained_preds[0]
        fine_tuned_top = fine_tuned_preds[0]
        
        print(f"   Pretrained top: {pretrained_top[0]} ({pretrained_top[1]:.3f})")
        print(f"   Fine-tuned top:  {fine_tuned_top[0]} ({fine_tuned_top[1]:.3f})")
        
        if pretrained_top[0] == fine_tuned_top[0]:
            print("   ✅ Same top prediction!")
        else:
            print("   ❌ Different top predictions")
        
        # Show visualizations
        comparison = ModelComparison(fine_tuned_model_path=fine_tuned_path)
        comparison.visualize_comparison(image_path, top_k=5)
    else:
        # Just show pretrained visualization
        pretrained.visualize_classification(image_path, top_k=5)

def compare_multiple_images(image_paths: List[str], fine_tuned_path: str = None):
    """Compare models on multiple images and show aggregate statistics"""
    print("🔍 Comparing models on multiple images...")
    print("=" * 60)
    
    pretrained = SpaceCLIPClassifier()
    fine_tuned = None
    if fine_tuned_path:
        fine_tuned = SpaceCLIPClassifier(fine_tuned_path=fine_tuned_path)
    
    results = {
        'pretrained': [],
        'fine_tuned': [],
        'agreement': 0,
        'total': len(image_paths)
    }
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n📸 Image {i}/{len(image_paths)}: {image_path}")
        
        pretrained_preds = pretrained.classify_image(image_path, top_k=1)
        results['pretrained'].append(pretrained_preds[0])
        
        if fine_tuned:
            fine_tuned_preds = fine_tuned.classify_image(image_path, top_k=1)
            results['fine_tuned'].append(fine_tuned_preds[0])
            
            # Check if top predictions agree
            if pretrained_preds[0][0] == fine_tuned_preds[0][0]:
                results['agreement'] += 1
                print(f"   ✅ Agreement: {pretrained_preds[0][0]}")
            else:
                print(f"   ❌ Disagreement: {pretrained_preds[0][0]} vs {fine_tuned_preds[0][0]}")
        else:
            print(f"   Pretrained: {pretrained_preds[0][0]} ({pretrained_preds[0][1]:.3f})")
    
    # Show aggregate statistics
    print(f"\n📈 Aggregate Statistics:")
    print(f"   Total images: {results['total']}")
    
    if fine_tuned:
        agreement_rate = results['agreement'] / results['total']
        print(f"   Agreement rate: {agreement_rate:.1%} ({results['agreement']}/{results['total']})")
        
        # Average confidence comparison
        pretrained_avg_conf = np.mean([pred[1] for pred in results['pretrained']])
        fine_tuned_avg_conf = np.mean([pred[1] for pred in results['fine_tuned']])
        
        print(f"   Average confidence - Pretrained: {pretrained_avg_conf:.3f}")
        print(f"   Average confidence - Fine-tuned:  {fine_tuned_avg_conf:.3f}")
        
        # Show confidence improvement
        if fine_tuned_avg_conf > pretrained_avg_conf:
            improvement = fine_tuned_avg_conf - pretrained_avg_conf
            print(f"   🎉 Fine-tuned model shows {improvement:.3f} higher average confidence")
        else:
            difference = pretrained_avg_conf - fine_tuned_avg_conf
            print(f"   📉 Fine-tuned model shows {difference:.3f} lower average confidence")

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
    
    # Test images
    test_images = [
        "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?w=800",  # Galaxy
        "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=800",  # Stars
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",  # Space
        "https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?w=800",  # Nebula
        "https://images.unsplash.com/photo-1516339901601-2e1b62dc0c45?w=800",  # Planet
    ]
    
    # Fine-tuned model path (uncomment when you have a trained model)
    fine_tuned_path = None  # "fine_tuned_clip"
    
    print("1. Single image comparison")
    compare_single_image(test_images[0], fine_tuned_path)
    
    print("\n" + "=" * 60)
    print("2. Multiple image comparison")
    compare_multiple_images(test_images, fine_tuned_path)
    
    if fine_tuned_path:
        print("\n" + "=" * 60)
        print("3. Comprehensive visualization")
        create_comparison_visualization(test_images, fine_tuned_path)
    
    print("\n💡 To use with your fine-tuned model:")
    print("   1. Train a model using space_clip_trainer.py")
    print("   2. Set fine_tuned_path = 'your_model_directory'")
    print("   3. Run this script again")

if __name__ == "__main__":
    main() 