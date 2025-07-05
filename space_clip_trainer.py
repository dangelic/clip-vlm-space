#!/usr/bin/env python3
"""
Space CLIP Trainer - Fine-tune CLIP model on space images
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, get_linear_schedule_with_warmup
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
from typing import List, Tuple, Dict
from space_clip_classifier import SpaceCLIPClassifier
import requests
from io import BytesIO

class SpaceImageDataset(Dataset):
    """Dataset for space images with text labels"""
    
    def __init__(self, data_dir: str, processor, transform=None, labels=None):
        """
        Args:
            data_dir: Directory containing space images
            processor: CLIP processor for text/image preprocessing
            transform: Optional image transformations
            labels: Optional pre-loaded labels dictionary
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.transform = transform
        
        # Find all image files
        self.image_files = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for ext in image_extensions:
            self.image_files.extend(self.data_dir.glob(f"*{ext}"))
            self.image_files.extend(self.data_dir.glob(f"*{ext.upper()}"))
        
        # Use provided labels or load from file
        if labels is not None:
            self.labels = labels
            self.labels_file = None  # No file to save to when using provided labels
        else:
            # Load labels from JSON file if it exists
            self.labels_file = self.data_dir / "labels.json"
            if self.labels_file.exists():
                with open(self.labels_file, 'r') as f:
                    self.labels = json.load(f)
            else:
                # Create default labels based on filenames
                self.labels = self._create_default_labels()
                self._save_labels()
    
    def _create_default_labels(self) -> Dict[str, str]:
        """Create default labels based on filenames"""
        labels = {}
        for img_file in self.image_files:
            filename = img_file.stem.lower()
            # Rich, descriptive labeling for better multimodality
            if 'galaxy' in filename:
                if 'spiral' in filename:
                    labels[str(img_file)] = "a spiral galaxy with distinct spiral arms and bright central bulge"
                elif 'elliptical' in filename:
                    labels[str(img_file)] = "an elliptical galaxy with smooth, featureless appearance"
                else:
                    labels[str(img_file)] = "a galaxy with bright central region and surrounding stars"
            elif 'nebula' in filename:
                if 'emission' in filename:
                    labels[str(img_file)] = "a colorful emission nebula with bright pink and blue gas clouds"
                elif 'planetary' in filename:
                    labels[str(img_file)] = "a planetary nebula with glowing gas shells around a central star"
                else:
                    labels[str(img_file)] = "a nebula with colorful gas and dust clouds"
            elif 'planet' in filename:
                if 'gas' in filename or 'jupiter' in filename:
                    labels[str(img_file)] = "a gas giant planet with colorful atmospheric bands and storms"
                elif 'rocky' in filename or 'mars' in filename:
                    labels[str(img_file)] = "a rocky planet with cratered surface and reddish terrain"
                else:
                    labels[str(img_file)] = "a planet with visible surface features and atmosphere"
            elif 'star' in filename:
                if 'cluster' in filename:
                    labels[str(img_file)] = "a star cluster with hundreds of bright stars grouped together"
                else:
                    labels[str(img_file)] = "bright stars scattered across a dark space background"
            elif 'black' in filename and 'hole' in filename:
                labels[str(img_file)] = "a black hole with bright accretion disk and powerful jets of plasma"
            elif 'supernova' in filename:
                labels[str(img_file)] = "a supernova explosion with bright expanding gas shells"
            elif 'aurora' in filename:
                labels[str(img_file)] = "aurora borealis with colorful light curtains in the night sky"
            elif 'telescope' in filename or 'hubble' in filename:
                labels[str(img_file)] = "space telescope view of distant astronomical objects"
            else:
                labels[str(img_file)] = "a beautiful space scene with stars, galaxies, and cosmic phenomena"
        return labels
    
    def _save_labels(self):
        """Save labels to JSON file"""
        if self.labels_file is not None:
            with open(self.labels_file, 'w') as f:
                json.dump(self.labels, f, indent=2)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label = self.labels.get(str(img_path), "space")
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Process with CLIP processor
        inputs = self.processor(
            text=[label],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'label': label,
            'image_path': str(img_path)
        }

class SpaceCLIPTrainer:
    """Trainer for fine-tuning CLIP on space images"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load pretrained model
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Training parameters
        self.learning_rate = 5e-5
        self.batch_size = 8
        self.num_epochs = 3
        self.warmup_steps = 100
        
        print(f"Initialized trainer with {model_name}")
    
    def prepare_data(self, data_dir: str) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare training, validation, and test data"""
        data_dir = Path(data_dir)
        
        # Load labels for each split
        train_labels_file = data_dir / "labels_train.json"
        val_labels_file = data_dir / "labels_validation.json"
        test_labels_file = data_dir / "labels_test.json"
        
        if not all([train_labels_file.exists(), val_labels_file.exists(), test_labels_file.exists()]):
            print(f"❌ Split label files not found. Please run filter_flickr30k.py first.")
            return None, None, None
        
        with open(train_labels_file, 'r') as f:
            train_labels = json.load(f)
        with open(val_labels_file, 'r') as f:
            val_labels = json.load(f)
        with open(test_labels_file, 'r') as f:
            test_labels = json.load(f)
        
        # Create datasets for each split
        train_dataset = SpaceImageDataset(data_dir, self.processor, labels=train_labels)
        val_dataset = SpaceImageDataset(data_dir, self.processor, labels=val_labels)
        test_dataset = SpaceImageDataset(data_dir, self.processor, labels=test_labels)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2
        )
        
        print(f"Prepared data:")
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Validation: {len(val_dataset)} samples")
        print(f"   Test: {len(test_dataset)} samples")
        return train_loader, val_loader, test_loader
    
    def train(self, data_dir: str, output_dir: str = "fine_tuned_clip"):
        """Train the model"""
        print("🚀 Starting CLIP fine-tuning...")
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(data_dir)
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        training_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                pixel_values = batch['pixel_values'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values
                )
                
                # CLIP contrastive loss
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                
                # Create labels (diagonal matrix for positive pairs)
                batch_size = logits_per_image.size(0)
                labels = torch.arange(batch_size).to(self.device)
                
                # Calculate loss
                loss = (nn.CrossEntropyLoss()(logits_per_image, labels) + 
                       nn.CrossEntropyLoss()(logits_per_text, labels)) / 2
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    pixel_values = batch['pixel_values'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values
                    )
                    
                    logits_per_image = outputs.logits_per_image
                    logits_per_text = outputs.logits_per_text
                    
                    batch_size = logits_per_image.size(0)
                    labels = torch.arange(batch_size).to(self.device)
                    
                    loss = (nn.CrossEntropyLoss()(logits_per_image, labels) + 
                           nn.CrossEntropyLoss()(logits_per_text, labels)) / 2
                    
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save_model(output_dir)
                print(f"  ✅ Saved best model (val_loss: {best_val_loss:.4f})")
            
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
        
        # Save training history
        with open(f"{output_dir}/training_history.json", 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"\n🎉 Training completed! Best validation loss: {best_val_loss:.4f}")
        return training_history
    
    def save_model(self, output_dir: str):
        """Save the fine-tuned model"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_path)
        self.processor.save_pretrained(output_path)
        print(f"Model saved to {output_path}")
    
    def load_model(self, model_path: str):
        """Load a fine-tuned model"""
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        print(f"Loaded fine-tuned model from {model_path}")

class ModelComparison:
    """Compare pretrained vs fine-tuned CLIP models"""
    
    def __init__(self, pretrained_model_path: str = None, fine_tuned_model_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load models
        self.pretrained_classifier = SpaceCLIPClassifier()
        if fine_tuned_model_path:
            self.fine_tuned_classifier = SpaceCLIPClassifier(fine_tuned_model_path)
        else:
            self.fine_tuned_classifier = None
    
    def compare_predictions(self, image_path: str, top_k: int = 5) -> Dict:
        """Compare predictions between models"""
        results = {
            'image_path': image_path,
            'pretrained': self.pretrained_classifier.classify_image(image_path, top_k),
        }
        
        if self.fine_tuned_classifier:
            results['fine_tuned'] = self.fine_tuned_classifier.classify_image(image_path, top_k)
        
        return results
    
    def visualize_comparison(self, image_path: str, top_k: int = 5):
        """Visualize comparison between models"""
        results = self.compare_predictions(image_path, top_k)
        
        # Load image
        image = self.pretrained_classifier.load_image(image_path)
        
        if self.fine_tuned_classifier:
            # Create side-by-side comparison
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Show image
            axes[0].imshow(image)
            axes[0].set_title("Space Image")
            axes[0].axis('off')
            
            # Pretrained results
            pretrained_cats = [pred[0] for pred in results['pretrained']]
            pretrained_conf = [pred[1] for pred in results['pretrained']]
            
            bars1 = axes[1].barh(range(len(pretrained_cats)), pretrained_conf, color='lightblue')
            axes[1].set_yticks(range(len(pretrained_cats)))
            axes[1].set_yticklabels(pretrained_cats)
            axes[1].set_xlabel('Confidence Score')
            axes[1].set_title('Pretrained CLIP')
            
            # Fine-tuned results
            fine_tuned_cats = [pred[0] for pred in results['fine_tuned']]
            fine_tuned_conf = [pred[1] for pred in results['fine_tuned']]
            
            bars2 = axes[2].barh(range(len(fine_tuned_cats)), fine_tuned_conf, color='lightgreen')
            axes[2].set_yticks(range(len(fine_tuned_cats)))
            axes[2].set_yticklabels(fine_tuned_cats)
            axes[2].set_xlabel('Confidence Score')
            axes[2].set_title('Fine-tuned CLIP')
            
            plt.tight_layout()
            plt.show()
        else:
            # Just show pretrained results
            self.pretrained_classifier.visualize_classification(image_path, top_k)

def main():
    """Example usage"""
    print("🌟 Space CLIP Trainer & Comparison Tool")
    print("=" * 50)
    
    # Initialize trainer
    trainer = SpaceCLIPTrainer()
    
    # Example: Train on a dataset (uncomment when you have data)
    # data_dir = "path/to/your/space/images"
    # if os.path.exists(data_dir):
    #     trainer.train(data_dir, "fine_tuned_clip")
    
    # Example: Compare models
    comparison = ModelComparison()
    
    # Test images
    test_images = [
        "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?w=800",
        "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=800",
    ]
    
    print("\n🔍 Comparing models on test images...")
    for i, image_url in enumerate(test_images, 1):
        print(f"\nImage {i}:")
        results = comparison.compare_predictions(image_url, top_k=3)
        
        print("Pretrained CLIP:")
        for j, (category, confidence) in enumerate(results['pretrained'], 1):
            print(f"  {j}. {category}: {confidence:.3f}")
        
        if 'fine_tuned' in results:
            print("Fine-tuned CLIP:")
            for j, (category, confidence) in enumerate(results['fine_tuned'], 1):
                print(f"  {j}. {category}: {confidence:.3f}")
    
    # Show comparison visualization
    comparison.visualize_comparison(test_images[0], top_k=5)

if __name__ == "__main__":
    main() 