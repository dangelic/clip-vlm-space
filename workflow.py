#!/usr/bin/env python3
"""
Complete Workflow for Space CLIP Training
Shows all steps and allows running them separately
"""

import os
import sys
from pathlib import Path

def show_workflow():
    """Show the complete workflow"""
    print("🌟 Complete Space CLIP Training Workflow")
    print("=" * 60)
    print()
    print("📋 Workflow Steps:")
    print("   1. 📥 Prepare Data     - Download images & create captions")
    print("   2. 🎯 Train Model      - Fine-tune CLIP on space images")
    print("   3. 🔍 Compare Models   - Pretrained vs fine-tuned")
    print("   4. 📊 Evaluate Results - Analyze performance")
    print("   5. 🧠 XAI Analysis     - Visualize model explanations")
    print()
    print("💡 Each step can be run independently")
    print()

def step1_prepare_data():
    """Step 1: Prepare data"""
    print("📥 Step 1: Data Preparation")
    print("-" * 40)
    print("This will:")
    print("  • Load NASA APOD dataset (Astronomy Picture of the Day)")
    print("  • Download high-quality space images with detailed astronomical captions")
    print("  • Images are numbered sequentially and can be run multiple times")
    print("  • Organize data in 'images/' folder")
    print()
    
    if input("Run NASA APOD data preparation? (y/n): ").lower() == 'y':
        print("\n🚀 Running NASA APOD data preparation...")
        os.system("python nasa_space_filter.py")
    else:
        print("⏭️  Skipping data preparation")

def step1b_create_splits():
    """Step 1b: Create train/validation/test splits"""
    print("\n🔀 Step 1b: Create Splits")
    print("-" * 40)
    print("This will:")
    print("  • Load all accumulated NASA images")
    print("  • Create train/validation/test splits (80/10/10)")
    print("  • Generate split label files for training")
    print()
    
    if input("Create train/validation/test splits? (y/n): ").lower() == 'y':
        print("\n🚀 Creating splits...")
        os.system("python create_splits.py")
    else:
        print("⏭️  Skipping split creation")

def step2_train_model():
    """Step 2: Train model"""
    print("\n🎯 Step 2: Model Training")
    print("-" * 40)
    print("This will:")
    print("  • Load pretrained CLIP model")
    print("  • Fine-tune on your space images")
    print("  • Save fine-tuned model to 'fine_tuned_clip/'")
    print("  • Show training progress and metrics")
    print()
    
    # Check if data exists
    images_dir = Path("images")
    if not images_dir.exists():
        print("❌ No 'images/' folder found. Run Step 1 first.")
        return
    
    if input("Run model training? (y/n): ").lower() == 'y':
        print("\n🚀 Running model training...")
        # Import and run training
        try:
            from space_clip_trainer import SpaceCLIPTrainer
            trainer = SpaceCLIPTrainer()
            trainer.train("images", "fine_tuned_clip")
        except Exception as e:
            print(f"❌ Training failed: {e}")
    else:
        print("⏭️  Skipping model training")

def step3_compare_models():
    """Step 3: Compare models"""
    print("\n🔍 Step 3: Model Comparison")
    print("-" * 40)
    print("This will:")
    print("  • Load both pretrained and fine-tuned models")
    print("  • Show all test images with their captions")
    print("  • Display model comparisons for each image")
    print("  • Show side-by-side visualizations")
    print()
    
    if input("Run model comparison demo? (y/n): ").lower() == 'y':
        print("\n🚀 Running model comparison demo...")
        os.system("python demo.py")
    else:
        print("⏭️  Skipping model comparison")

def step4_evaluate_results():
    """Step 4: Evaluate results"""
    print("\n📊 Step 4: Results Evaluation")
    print("-" * 40)
    print("This will:")
    print("  • Analyze training metrics")
    print("  • Show improvement statistics")
    print("  • Generate performance reports")
    print("  • Suggest next steps")
    print()
    
    if input("Run results evaluation? (y/n): ").lower() == 'y':
        print("\n🚀 Running results evaluation...")
        evaluate_results()
    else:
        print("⏭️  Skipping results evaluation")

def step5_xai_analysis():
    """Step 5: XAI analysis"""
    print("\n🧠 Step 5: XAI Analysis")
    print("-" * 40)
    print("This will:")
    print("  • Generate feature importance plots for all test images")
    print("  • Generate confidence heatmaps for all test images")
    print("  • Generate Grad-CAM and gradient-based heatmaps for all test images")
    print("  • Save all visualizations to the 'xai_results/' folder")
    print()
    if input("Run XAI analysis? (y/n): ").lower() == 'y':
        print("\n🚀 Running XAI analysis...")
        os.system("python xai_analysis.py")
    else:
        print("⏭️  Skipping XAI analysis")

def evaluate_results():
    """Evaluate training results"""
    import json
    from pathlib import Path
    
    print("📈 Evaluating Results...")
    
    # Check training history
    history_file = Path("fine_tuned_clip/training_history.json")
    if history_file.exists():
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        print(f"\n📊 Training Metrics:")
        print(f"   Final training loss: {history['train_loss'][-1]:.4f}")
        print(f"   Final validation loss: {history['val_loss'][-1]:.4f}")
        print(f"   Best validation loss: {min(history['val_loss']):.4f}")
        
        # Calculate improvement
        initial_loss = history['val_loss'][0]
        final_loss = history['val_loss'][-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        
        print(f"   Improvement: {improvement:.1f}%")
        
        if improvement > 0:
            print("   🎉 Model improved during training!")
        else:
            print("   ⚠️  Model didn't improve - consider more data or different parameters")
    
    # Check data statistics
    images_dir = Path("images")
    if images_dir.exists():
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        print(f"\n📁 Data Statistics:")
        print(f"   Total training images: {len(image_files)}")
        
        captions_file = images_dir / "rich_labels.json"
        if captions_file.exists():
            with open(captions_file, 'r') as f:
                captions = json.load(f)
            print(f"   Captions created: {len(captions)}")
    
    print(f"\n💡 Next Steps:")
    print(f"   • Test on new images: python demo.py")
    print(f"   • Use fine-tuned model: SpaceCLIPClassifier('fine_tuned_clip')")
    print(f"   • Collect more data for better results")

def run_all_steps():
    """Run all steps in sequence"""
    print("🚀 Running Complete Workflow")
    print("=" * 60)
    step1_prepare_data()
    step1b_create_splits()
    step2_train_model()
    step3_compare_models()
    step4_evaluate_results()
    step5_xai_analysis()
    print("\n🎉 Workflow Complete!")
    print("Check the results in the generated folders and files.")

def main():
    """Main workflow interface"""
    show_workflow()
    print("Choose an option:")
    print("   a. Run all steps")
    print("   b. Step 1: Prepare Data")
    print("   c. Step 1b: Create Splits")
    print("   d. Step 2: Train Model")
    print("   e. Step 3: Compare Models")
    print("   f. Step 4: Evaluate Results")
    print("   g. Step 5: XAI Analysis")
    print("   q. Exit")
    print()
    choice = input("Enter your choice (a-g, q): ").strip().lower()
    if choice == 'a':
        run_all_steps()
    elif choice == 'b':
        step1_prepare_data()
    elif choice == 'c':
        step1b_create_splits()
    elif choice == 'd':
        step2_train_model()
    elif choice == 'e':
        step3_compare_models()
    elif choice == 'f':
        step4_evaluate_results()
    elif choice == 'g':
        step5_xai_analysis()
    elif choice == 'q':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please run again.")

if __name__ == "__main__":
    main() 