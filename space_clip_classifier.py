import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO
from typing import List, Tuple

class SpaceCLIPClassifier:
    """
    A CLIP-based classifier for space images using pretrained models.
    No training required - uses OpenAI's pretrained CLIP model.
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the CLIP classifier with a pretrained model.
        
        Args:
            model_name: HuggingFace model name for CLIP
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load pretrained CLIP model and processor
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Space-related text categories for classification
        self.space_categories = [
            "galaxy", "nebula", "star cluster", "planet", "moon", "asteroid",
            "comet", "black hole", "supernova", "cosmic dust", "solar system",
            "constellation", "meteor", "satellite", "space station", "rocket",
            "telescope", "astronaut", "space suit", "spacecraft", "mars rover",
            "hubble telescope", "james webb telescope", "iss", "space debris",
            "aurora borealis", "milky way", "andromeda galaxy", "orion nebula"
        ]
        
        print(f"Loaded CLIP model with {len(self.space_categories)} space categories")
    
    def load_image(self, image_path: str) -> Image.Image:
        """
        Load an image from file path or URL.
        
        Args:
            image_path: Path to image file or URL
            
        Returns:
            PIL Image object
        """
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)
        
        return image.convert('RGB')
    
    def classify_image(self, image_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Classify a space image using CLIP.
        
        Args:
            image_path: Path to image file or URL
            top_k: Number of top predictions to return
            
        Returns:
            List of (category, confidence) tuples
        """
        # Load and preprocess image
        image = self.load_image(image_path)
        
        # Prepare inputs for CLIP
        inputs = self.processor(
            text=self.space_categories,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Get CLIP predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = F.softmax(logits_per_image, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k, dim=1)
        
        results = []
        for i in range(top_k):
            category = self.space_categories[top_indices[0][i].item()]
            confidence = top_probs[0][i].item()
            results.append((category, confidence))
        
        return results
    
    def visualize_classification(self, image_path: str, top_k: int = 5):
        """
        Visualize the classification results for an image.
        
        Args:
            image_path: Path to image file or URL
            top_k: Number of top predictions to show
        """
        # Get predictions
        predictions = self.classify_image(image_path, top_k)
        
        # Load image
        image = self.load_image(image_path)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Show image
        ax1.imshow(image)
        ax1.set_title("Space Image")
        ax1.axis('off')
        
        # Show predictions
        categories = [pred[0] for pred in predictions]
        confidences = [pred[1] for pred in predictions]
        
        bars = ax2.barh(range(len(categories)), confidences, color='skyblue')
        ax2.set_yticks(range(len(categories)))
        ax2.set_yticklabels(categories)
        ax2.set_xlabel('Confidence Score')
        ax2.set_title('CLIP Classification Results')
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Example usage of the SpaceCLIPClassifier.
    """
    # Initialize classifier
    classifier = SpaceCLIPClassifier()
    
    # Example space images (you can replace these with your own)
    example_images = [
        "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?w=800",  # Galaxy
        "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?w=800",  # Stars
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800",  # Space
    ]
    
    print("=== Space Image Classification with CLIP ===\n")
    
    # Classify each image
    for i, image_path in enumerate(example_images):
        print(f"Image {i+1}:")
        predictions = classifier.classify_image(image_path, top_k=3)
        
        for j, (category, confidence) in enumerate(predictions):
            print(f"  {j+1}. {category}: {confidence:.3f}")
        
        print()
    
    # Visualize classification for first image
    print("\n=== Generating Visualization ===")
    classifier.visualize_classification(example_images[0], top_k=5)

if __name__ == "__main__":
    main() 