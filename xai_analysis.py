#!/usr/bin/env python3
"""
XAI (Explainable AI) Analysis for Space CLIP Classifier
Includes attention heatmaps, gradient-based explanations, and visualizations
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO
import cv2
import os
from transformers import CLIPProcessor, CLIPModel
from space_clip_classifier import SpaceCLIPClassifier
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class SpaceCLIPXAI:
    """XAI analysis for Space CLIP classifier"""
    
    def __init__(self, model_path: str = "fine_tuned_clip"):
        """Initialize XAI analyzer with fine-tuned CLIP model"""
        print("🔬 Space CLIP XAI Analysis")
        print("=" * 60)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔬 XAI Analysis using device: {self.device}")
        
        # Load model
        print(f"📥 Loading fine-tuned model from: {model_path}")
        self.model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Get categories from model config
        self.categories = self.model.config.text_config.vocab_size
        print(f"✅ Loaded model with {self.categories} categories")
        
        # Create results directory
        os.makedirs("xai_results", exist_ok=True)
        
        # Test images with captions
        self.test_images = [
            {
                "url": "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?w=800",
                "caption": "A stunning spiral galaxy with bright stars and cosmic dust clouds"
            },
            {
                "url": "https://images.unsplash.com/photo-1516339901601-2e1b62dc0c45?w=800", 
                "caption": "Deep space nebula with colorful gas clouds and distant stars"
            },
            {
                "url": "https://images.unsplash.com/photo-1534796636912-3b95b3ab5986?w=800",
                "caption": "Astronaut floating in space with Earth visible in the background"
            }
        ]

    def load_and_preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, Image.Image]:
        """Load and preprocess image for analysis"""
        if image_path.startswith('http'):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        
        # Preprocess for model
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].to(self.device), image

    def get_model_predictions(self, image_tensor: torch.Tensor, text_inputs: Dict) -> Dict:
        """Get model predictions and confidence scores"""
        with torch.no_grad():
            outputs = self.model(
                pixel_values=image_tensor,
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            )
            
            logits = outputs.logits_per_image
            probs = F.softmax(logits, dim=-1)
            
            return {
                'logits': logits.cpu().numpy(),
                'probabilities': probs.cpu().numpy(),
                'outputs': outputs
            }

    def create_feature_importance_plot(self, predictions: Dict, save_path: str = None):
        """Create feature importance plot with beeswarm"""
        print("📊 Creating feature importance plot")
        
        probs = predictions['probabilities'].flatten()
        
        # Get top categories (assuming we have category names)
        top_indices = np.argsort(probs)[::-1][:10]
        top_probs = probs[top_indices]
        top_categories = [f"Category {i}" for i in top_indices]
        
        # Create beeswarm plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Bar plot
        bars = ax1.bar(range(len(top_categories)), top_probs, color='skyblue', alpha=0.7)
        ax1.set_title("Top Category Predictions", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Categories")
        ax1.set_ylabel("Probability")
        ax1.set_xticks(range(len(top_categories)))
        ax1.set_xticklabels(top_categories, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, prob in zip(bars, top_probs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Beeswarm plot
        # Create data points for beeswarm
        y_positions = np.random.normal(0, 0.1, len(probs))
        ax2.scatter(probs, y_positions, alpha=0.6, s=50, c=probs, cmap='viridis')
        ax2.set_title("Probability Distribution (Beeswarm)", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Probability")
        ax2.set_ylabel("Distribution")
        ax2.set_ylim(-0.5, 0.5)
        ax2.axvline(x=np.mean(probs), color='red', linestyle='--', alpha=0.7, label=f'Mean: {np.mean(probs):.3f}')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved feature importance plot to: {save_path}")
        
        plt.show()

    def create_confidence_heatmap(self, image_tensor: torch.Tensor, save_path: str = None):
        """Create confidence heatmap across image regions"""
        print("🎯 Creating confidence heatmap")
        
        # Get attention weights from the model
        with torch.no_grad():
            outputs = self.model.vision_model(image_tensor, output_attentions=True)
            attentions = outputs.attentions[-1]  # Last layer attention
            
            # Average attention across heads
            attention_map = attentions.mean(dim=1).squeeze().cpu().numpy()
            attention_map = attention_map.mean(axis=0)  # Average across layers
            
            # Handle different attention map sizes
            total_patches = attention_map.shape[0]
            
            # Try to find the closest square root
            size = int(np.sqrt(total_patches))
            if size * size != total_patches:
                # If not a perfect square, use the closest approximation
                size = int(np.sqrt(total_patches))
                # Pad or truncate to make it square
                if size * size < total_patches:
                    size += 1
                
                # Pad with zeros if needed
                target_size = size * size
                if total_patches < target_size:
                    padding = target_size - total_patches
                    attention_map = np.pad(attention_map, (0, padding), mode='constant')
                else:
                    attention_map = attention_map[:target_size]
            
            attention_map = attention_map.reshape(size, size)
            
            # Resize to image dimensions
            attention_map = cv2.resize(attention_map, (224, 224))
            
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original image
            original_image = self.tensor_to_pil(image_tensor)
            axes[0].imshow(original_image)
            axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Confidence heatmap
            im = axes[1].imshow(attention_map, cmap='hot', alpha=0.8)
            axes[1].set_title("Model Attention Heatmap", fontsize=14, fontweight='bold')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], shrink=0.8)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"💾 Saved confidence heatmap to: {save_path}")
            
            plt.show()

    def tensor_to_pil(self, tensor):
        """Convert tensor back to PIL image for visualization"""
        # Remove batch dimension if present
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        # Denormalize and convert to PIL
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        tensor = (tensor * 255).byte()
        return Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy())

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
        """Create gradient-based heatmap"""
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
        axes[0].axis('off')
        axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
        
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

    def analyze_description_contribution(self, image_path: str, description: str, save_path: str = None):
        """Analyze how descriptions contribute to model predictions"""
        print(f"📝 Analyzing description contribution: {description[:50]}...")
        
        # Load image
        image_tensor, original_image = self.load_and_preprocess_image(image_path)
        
        # Process description
        text_inputs = self.processor(
            text=[description],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        for k in text_inputs:
            text_inputs[k] = text_inputs[k].to(self.device)
        
        # Get predictions with description
        predictions_with_desc = self.get_model_predictions(image_tensor, text_inputs)
        
        # Get predictions without description (using generic text)
        generic_text_inputs = self.processor(
            text=["space image"],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        for k in generic_text_inputs:
            generic_text_inputs[k] = generic_text_inputs[k].to(self.device)
        
        predictions_without_desc = self.get_model_predictions(image_tensor, generic_text_inputs)
        
        # Compare predictions
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # With description
        probs_with = predictions_with_desc['probabilities'].flatten()
        top_indices_with = np.argsort(probs_with)[::-1][:10]
        top_probs_with = probs_with[top_indices_with]
        top_categories_with = [f"Cat {i}" for i in top_indices_with]
        
        axes[0, 0].bar(range(len(top_categories_with)), top_probs_with, color='green', alpha=0.7)
        axes[0, 0].set_title("Predictions WITH Description", fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel("Probability")
        axes[0, 0].set_xticks(range(len(top_categories_with)))
        axes[0, 0].set_xticklabels(top_categories_with, rotation=45, ha='right')
        
        # Without description
        probs_without = predictions_without_desc['probabilities'].flatten()
        top_indices_without = np.argsort(probs_without)[::-1][:10]
        top_probs_without = probs_without[top_indices_without]
        top_categories_without = [f"Cat {i}" for i in top_indices_without]
        
        axes[0, 1].bar(range(len(top_categories_without)), top_probs_without, color='red', alpha=0.7)
        axes[0, 1].set_title("Predictions WITHOUT Description", fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel("Probability")
        axes[0, 1].set_xticks(range(len(top_categories_without)))
        axes[0, 1].set_xticklabels(top_categories_without, rotation=45, ha='right')
        
        # Difference plot
        # Align predictions by category index
        all_categories = list(set(top_indices_with) | set(top_indices_without))
        diff_probs = []
        for cat in all_categories:
            prob_with = probs_with[cat] if cat < len(probs_with) else 0
            prob_without = probs_without[cat] if cat < len(probs_without) else 0
            diff_probs.append(prob_with - prob_without)
        
        colors = ['green' if diff > 0 else 'red' for diff in diff_probs]
        axes[1, 0].bar(range(len(all_categories)), diff_probs, color=colors, alpha=0.7)
        axes[1, 0].set_title("Description Impact (With - Without)", fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel("Probability Difference")
        axes[1, 0].set_xticks(range(len(all_categories)))
        axes[1, 0].set_xticklabels([f"Cat {i}" for i in all_categories], rotation=45, ha='right')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Beeswarm of differences
        y_positions = np.random.normal(0, 0.1, len(diff_probs))
        scatter = axes[1, 1].scatter(diff_probs, y_positions, c=diff_probs, cmap='RdYlGn', alpha=0.7, s=100)
        axes[1, 1].set_title("Description Impact Distribution", fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel("Probability Difference")
        axes[1, 1].set_ylabel("Distribution")
        axes[1, 1].set_ylim(-0.5, 0.5)
        axes[1, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=axes[1, 1], shrink=0.8)
        
        plt.suptitle(f"Description Analysis: {description[:60]}...", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved description analysis to: {save_path}")
        
        plt.show()

    def word_attribution_beeswarm(self, image_path: str, caption: str, save_path: str = None):
        """Create a beeswarm plot showing the impact of each word in the caption on the model's output."""
        print(f"🐝 Creating word-level attribution beeswarm for caption: {caption}")
        image_tensor, _ = self.load_and_preprocess_image(image_path)
        words = caption.split()
        baseline_text_inputs = self.processor(
            text=[caption],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        for k in baseline_text_inputs:
            baseline_text_inputs[k] = baseline_text_inputs[k].to(self.device)
        baseline_pred = self.get_model_predictions(image_tensor, baseline_text_inputs)
        baseline_probs = baseline_pred['probabilities'].flatten()
        top_class = int(np.argmax(baseline_probs))
        top_prob = baseline_probs[top_class]
        # Attribution: mask each word, measure drop in top class probability
        impacts = []
        for i, word in enumerate(words):
            masked_words = words.copy()
            masked_words[i] = "[MASK]"
            masked_caption = " ".join(masked_words)
            text_inputs = self.processor(
                text=[masked_caption],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            for k in text_inputs:
                text_inputs[k] = text_inputs[k].to(self.device)
            pred = self.get_model_predictions(image_tensor, text_inputs)
            probs = pred['probabilities'].flatten()
            impact = top_prob - probs[top_class]
            impacts.append(impact)
        # Beeswarm plot
        fig, ax = plt.subplots(figsize=(12, 4))
        y_positions = np.random.normal(0, 0.03, len(words))
        scatter = ax.scatter(impacts, y_positions, c=impacts, cmap='coolwarm', s=120, edgecolor='k')
        for i, (impact, y) in enumerate(zip(impacts, y_positions)):
            ax.text(impact, y+0.04, words[i], ha='center', va='bottom', fontsize=10, rotation=30)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Decrease in Top Class Probability when Word is Masked')
        ax.set_yticks([])
        ax.set_title('Word-level Attribution Beeswarm for Caption', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, shrink=0.8, label='Impact')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved word attribution beeswarm to: {save_path}")
        plt.show()

    def comprehensive_analysis(self, image_path: str, caption: str, suffix: str = ""):
        """Run comprehensive XAI analysis on a single image"""
        print(f"🔬 Starting Comprehensive XAI Analysis")
        print("=" * 60)
        print("\n📊 Creating visualizations...")
        image_tensor, _ = self.load_and_preprocess_image(image_path)
        text_inputs = self.processor(
            text=[caption],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        for k in text_inputs:
            text_inputs[k] = text_inputs[k].to(self.device)
        predictions = self.get_model_predictions(image_tensor, text_inputs)
        self.create_feature_importance_plot(
            predictions, 
            f"xai_results/feature_importance{suffix}.png"
        )
        self.create_confidence_heatmap(
            image_tensor, 
            f"xai_results/confidence_heatmap{suffix}.png"
        )
        target_category = "galaxy"  # Default category
        self.create_gradcam_heatmap(
            image_path, 
            target_category, 
            f"xai_results/gradcam_heatmap{suffix}.png"
        )
        self.create_gradient_heatmap(
            image_path, 
            target_category, 
            f"xai_results/gradient_heatmap{suffix}.png"
        )
        # Word-level attribution beeswarm
        self.word_attribution_beeswarm(
            image_path,
            caption,
            f"xai_results/word_attribution_beeswarm{suffix}.png"
        )
        print(f"\n✅ XAI analysis complete! Results saved to: xai_results")

def main():
    """Main function to run XAI analysis"""
    xai_analyzer = SpaceCLIPXAI()
    
    # Only analyze the first image
    test_image = xai_analyzer.test_images[0]
    
    print(f"\n🎯 Analyzing: {test_image['url']}")
    print(f"📝 Description: {test_image['caption']}")
    
    xai_analyzer.comprehensive_analysis(
        test_image['url'],
        test_image['caption'],
        suffix="_1"
    )
    
    print(f"\n💡 XAI Analysis Complete!")
    print("Check the 'xai_results/' folder for all visualizations.")
    
    print(f"\n📋 Generated visualizations:")
    print("   • feature_importance_1.png - Category prediction probabilities with beeswarm")
    print("   • confidence_heatmap_1.png - Model confidence across image regions")
    print("   • gradcam_heatmap_1.png - Grad-CAM for ViT encoder")
    print("   • gradient_heatmap_1.png - Gradient-based explanations")
    print("   • word_attribution_beeswarm_1.png - Word-level attribution beeswarm")

if __name__ == "__main__":
    main() 