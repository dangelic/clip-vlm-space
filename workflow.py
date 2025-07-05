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
    print("  • Create proper ML splits: 80% train, 10% validation, 10% test")
    print("  • Organize data in 'images/' folder")
    print()
    
    if input("Run NASA APOD data preparation? (y/n): ").lower() == 'y':
        print("\n🚀 Running NASA APOD data preparation...")
        os.system("python nasa_space_filter.py")
    else:
        print("⏭️  Skipping data preparation")

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
    print("  • Compare predictions on test images")
    print("  • Show side-by-side visualizations")
    print("  • Calculate agreement rates and confidence")
    print()
    
    # Check if fine-tuned model exists
    fine_tuned_dir = Path("fine_tuned_clip")
    if not fine_tuned_dir.exists():
        print("❌ No 'fine_tuned_clip/' folder found. Run Step 2 first.")
        return
    
    if input("Run model comparison? (y/n): ").lower() == 'y':
        print("\n🚀 Running model comparison...")
        os.system("python compare_models.py")
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
    step2_train_model()
    step3_compare_models()
    step4_evaluate_results()
    
    print("\n🎉 Workflow Complete!")
    print("Check the results in the generated folders and files.")

def main():
    """Main workflow interface"""
    show_workflow()
    
    print("Choose an option:")
    print("   1. Run all steps")
    print("   2. Step 1: Prepare Data")
    print("   3. Step 2: Train Model")
    print("   4. Step 3: Compare Models")
    print("   5. Step 4: Evaluate Results")
    print("   6. Exit")
    print()
    
    choice = input("Enter your choice (1-6): ").strip()
    
    if choice == '1':
        run_all_steps()
    elif choice == '2':
        step1_prepare_data()
    elif choice == '3':
        step2_train_model()
    elif choice == '4':
        step3_compare_models()
    elif choice == '5':
        step4_evaluate_results()
    elif choice == '6':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Please run again.")

if __name__ == "__main__":
    main() 