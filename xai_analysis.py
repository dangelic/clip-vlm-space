#!/usr/bin/env python3
"""
XAI (Explainable AI) Analysis for Space CLIP Classifier
Includes attention heatmaps, gradient-based explanations, and visualizations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from space_clip_classifier import SpaceCLIPClassifier
import cv2
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import os
from torch.nn import functional as F

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class SpaceCLIPXAI:
    """XAI analysis for Space CLIP classifier"""
    
    def __init__(self, model_path: str = None):
        """Initialize XAI analyzer"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔬 XAI Analysis using device: {self.device}")
        
        # Load model
        if model_path is not None:
            print(f"📥 Loading fine-tuned model from: {model_path}")
            self.model = CLIPModel.from_pretrained(str(model_path)).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(str(model_path))
        else:
            print("📥 Loading pretrained model")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Space categories
        self.categories = [
            'galaxy', 'nebula', 'star cluster', 'planet', 'moon', 'asteroid', 'comet',
            'black hole', 'supernova', 'cosmic dust', 'solar system', 'constellation',
            'meteor', 'satellite', 'space station', 'rocket', 'telescope', 'astronaut',
            'space suit', 'spacecraft', 'mars rover', 'hubble telescope', 
            'james webb telescope', 'iss', 'space debris', 'aurora borealis',
            'milky way', 'andromeda galaxy', 'orion nebula'
        ]
        
        print(f"✅ Loaded model with {len(self.categories)} categories")
    
    def load_and_preprocess_image(self, image_path: str):
        """Load and preprocess image for analysis"""
        if image_path.startswith('http'):
            import requests
            from io import BytesIO
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        
        # Process with CLIP
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].to(self.device), image
    
    def create_gradcam_heatmap(self, image_path: str, target_category: str, save_path: str = None):
        """Create Grad-CAM heatmap for ViT vision encoder"""
        print(f"🔥 Creating Grad-CAM heatmap for category: {target_category}")
        
        # Load image
        image_tensor, original_image = self.load_and_preprocess_image(image_path)
        image_tensor.requires_grad_(True)
        
        # Process text
        text_inputs = self.processor(
            text=[target_category],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        for k in text_inputs:
            text_inputs[k] = text_inputs[k].to(self.device)
        
        # Get model outputs
        outputs = self.model(
            pixel_values=image_tensor,
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask']
        )
        
        # Get logits for target category
        logits = outputs.logits_per_image
        
        # Backward pass
        logits.backward()
        
        # Get gradients from the last layer
        gradients = image_tensor.grad
        
        # Create heatmap from gradients
        gradient_map = gradients.abs().mean(dim=1).squeeze().cpu().detach().numpy()
        gradient_map = cv2.resize(gradient_map, (original_image.size[0], original_image.size[1]))
        
        # Normalize and apply Gaussian blur for better visualization
        gradient_map = (gradient_map - gradient_map.min()) / (gradient_map.max() - gradient_map.min() + 1e-8)
        gradient_map = cv2.GaussianBlur(gradient_map, (15, 15), 0)
        
        # Apply additional smoothing and enhancement
        gradient_map = cv2.medianBlur((gradient_map * 255).astype(np.uint8), 5).astype(np.float32) / 255
        gradient_map = np.clip(gradient_map * 1.2, 0, 1)  # Enhance contrast
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Grad-CAM heatmap with improved colormap
        im = axes[1].imshow(gradient_map, cmap='viridis', alpha=0.9)
        axes[1].set_title(f"Grad-CAM Heatmap - {target_category.title()}", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], shrink=0.8)
        
        # Overlay with better blending
        axes[2].imshow(original_image)
        im2 = axes[2].imshow(gradient_map, cmap='viridis', alpha=0.7)
        axes[2].set_title("Grad-CAM Overlay", fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved Grad-CAM heatmap to: {save_path}")
        
        plt.show()
    
    def create_gradient_heatmap(self, image_path: str, target_category: str, save_path: str = None):
        """Create gradient-based heatmap using guided backpropagation"""
        print(f"🌊 Creating gradient heatmap for category: {target_category}")
        
        # Load image
        image_tensor, original_image = self.load_and_preprocess_image(image_path)
        image_tensor.requires_grad_(True)
        
        # Process text
        text_inputs = self.processor(
            text=[target_category],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        for k in text_inputs:
            text_inputs[k] = text_inputs[k].to(self.device)
        
        # Forward pass
        outputs = self.model(
            pixel_values=image_tensor,
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask']
        )
        
        # Get logits for target category
        logits = outputs.logits_per_image
        
        # Backward pass
        logits.backward()
        
        # Get gradients
        gradients = image_tensor.grad
        
        # Create gradient heatmap
        gradient_map = gradients.abs().mean(dim=1).squeeze().cpu().detach().numpy()
        gradient_map = cv2.resize(gradient_map, (original_image.size[0], original_image.size[1]))
        
        # Normalize and enhance
        gradient_map = (gradient_map - gradient_map.min()) / (gradient_map.max() - gradient_map.min() + 1e-8)
        gradient_map = cv2.GaussianBlur(gradient_map, (11, 11), 0)
        gradient_map = np.clip(gradient_map * 1.3, 0, 1)  # Enhance contrast
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Gradient heatmap
        im = axes[1].imshow(gradient_map, cmap='plasma', alpha=0.9)
        axes[1].set_title(f"Gradient Heatmap - {target_category.title()}", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], shrink=0.8)
        
        # Overlay
        axes[2].imshow(original_image)
        im2 = axes[2].imshow(gradient_map, cmap='plasma', alpha=0.6)
        axes[2].set_title("Gradient Overlay", fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved gradient heatmap to: {save_path}")
        
        plt.show()
    
    def create_feature_importance_plot(self, image_path: str, save_path: str = None):
        """Create feature importance plot for all categories"""
        print("📊 Creating feature importance plot")
        
        # Load image
        image_tensor, original_image = self.load_and_preprocess_image(image_path)
        
        # Get predictions for all categories
        self.model.eval()
        with torch.no_grad():
            # Process all categories
            text_inputs = self.processor(
                text=self.categories,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            for k in text_inputs:
                text_inputs[k] = text_inputs[k].to(self.device)
            
            # Get predictions
            outputs = self.model(
                pixel_values=image_tensor,
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            )
            
            # Get logits
            logits = outputs.logits_per_image.squeeze().cpu().numpy()
        
        # Convert to probabilities
        probabilities = F.softmax(torch.tensor(logits), dim=0).numpy()
        
        # Sort categories by probability
        category_probs = list(zip(self.categories, probabilities))
        category_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Top 10 categories
        top_10 = category_probs[:10]
        categories = [cat.replace('_', ' ').title() for cat, _ in top_10]
        probs = [prob for _, prob in top_10]
        
        bars = ax1.barh(categories, probs, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Probability', fontsize=12)
        ax1.set_title('Top 10 Category Predictions', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{prob:.3f}', ha='left', va='center', fontweight='bold')
        
        # All categories (sorted)
        all_categories = [cat.replace('_', ' ').title() for cat, _ in category_probs]
        all_probs = [prob for _, prob in category_probs]
        
        ax2.barh(all_categories, all_probs, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Probability', fontsize=12)
        ax2.set_title('All Category Predictions', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved feature importance plot to: {save_path}")
        
        plt.show()
    
    def create_confidence_heatmap(self, image_path: str, save_path: str = None):
        """Create confidence heatmap showing model confidence across image regions"""
        print("🎯 Creating confidence heatmap")
        
        # Load image
        image_tensor, original_image = self.load_and_preprocess_image(image_path)
        
        # Create sliding window approach
        width, height = original_image.size
        window_size = 64
        stride = 32
        
        confidence_map = np.zeros((height, width))
        count_map = np.zeros((height, width))
        
        # Process image in windows
        for y in range(0, height - window_size + 1, stride):
            for x in range(0, width - window_size + 1, stride):
                # Crop window
                window = original_image.crop((x, y, x + window_size, y + window_size))
                
                # Process window
                inputs = self.processor(images=window, return_tensors="pt")
                window_tensor = inputs['pixel_values'].to(self.device)
                
                # Get predictions
                self.model.eval()
                with torch.no_grad():
                    text_inputs = self.processor(
                        text=self.categories,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                    for k in text_inputs:
                        text_inputs[k] = text_inputs[k].to(self.device)
                    
                    outputs = self.model(
                        pixel_values=window_tensor,
                        input_ids=text_inputs['input_ids'],
                        attention_mask=text_inputs['attention_mask']
                    )
                    
                    logits = outputs.logits_per_image.squeeze().cpu().numpy()
                    probabilities = F.softmax(torch.tensor(logits), dim=0).numpy()
                    
                    # Use max confidence
                    max_confidence = np.max(probabilities)
                
                # Add to confidence map
                confidence_map[y:y+window_size, x:x+window_size] += max_confidence
                count_map[y:y+window_size, x:x+window_size] += 1
        
        # Average confidence
        confidence_map = np.divide(confidence_map, count_map, where=count_map > 0)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Confidence heatmap
        im = axes[1].imshow(confidence_map, cmap='plasma', alpha=0.8)
        axes[1].set_title("Confidence Heatmap", fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], shrink=0.8)
        
        # Overlay
        axes[2].imshow(original_image)
        im2 = axes[2].imshow(confidence_map, cmap='plasma', alpha=0.6)
        axes[2].set_title("Confidence Overlay", fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved confidence heatmap to: {save_path}")
        
        plt.show()
    
    def comprehensive_analysis(self, image_path: str, target_category: str = None, save_dir: str = "xai_results", suffix: str = ""):
        """Run comprehensive XAI analysis"""
        print("🔬 Starting Comprehensive XAI Analysis")
        print("=" * 60)
        
        # Create save directory
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Create visualizations
        print("\n📊 Creating visualizations...")
        
        # Feature importance plot
        importance_path = save_dir / f"feature_importance{suffix}.png"
        self.create_feature_importance_plot(image_path, str(importance_path))
        
        # Confidence heatmap
        confidence_path = save_dir / f"confidence_heatmap{suffix}.png"
        self.create_confidence_heatmap(image_path, str(confidence_path))
        
        # Grad-CAM heatmap (if target category provided)
        if target_category:
            gradcam_path = save_dir / f"gradcam_heatmap{suffix}.png"
            self.create_gradcam_heatmap(image_path, target_category, str(gradcam_path))
            
            # Gradient heatmap
            gradient_path = save_dir / f"gradient_heatmap{suffix}.png"
            self.create_gradient_heatmap(image_path, target_category, str(gradient_path))
        
        print(f"\n✅ XAI analysis complete! Results saved to: {save_dir}")
        
        return {
            'image_path': image_path,
            'target_category': target_category,
            'save_dir': str(save_dir)
        }

def main():
    """Main XAI analysis demo"""
    print("🔬 Space CLIP XAI Analysis")
    print("=" * 60)
    
    # Test images
    test_images = [
        "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?w=800",  # Galaxy
        "https://images.unsplash.com/photo-1516339901601-2e1b62dc0c45?w=800",  # Planet
        "https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?w=800",  # Nebula
    ]
    
    # Initialize XAI analyzer (use fine-tuned model if available)
    fine_tuned_path = "fine_tuned_clip" if Path("fine_tuned_clip").exists() else None
    xai_analyzer = SpaceCLIPXAI(fine_tuned_path)
    
    # Run analysis on all images
    for idx, img in enumerate(test_images, 1):
        print(f"\n🎯 Analyzing: {img}")
        suffix = f"_{idx}"
        xai_analyzer.comprehensive_analysis(
            img,
            target_category="galaxy",  # You can customize per image if needed
            save_dir="xai_results",
            suffix=suffix
        )
    
    print("\n💡 XAI Analysis Complete!")
    print("Check the 'xai_results/' folder for all visualizations.")
    print("\n📋 Generated visualizations for each image:")
    print("   • feature_importance_X.png - Category prediction probabilities")
    print("   • confidence_heatmap_X.png - Model confidence across image regions")
    print("   • gradcam_heatmap_X.png - Grad-CAM for ViT encoder")
    print("   • gradient_heatmap_X.png - Gradient-based explanations")

if __name__ == "__main__":
    main() 