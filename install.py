#!/usr/bin/env python3
"""
Installation script for CLIP Space Image Classifier
"""

import subprocess
import sys
import os
from pathlib import Path

def in_venv():
    return (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        os.environ.get('VIRTUAL_ENV') is not None
    )

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def create_venv(venv_path="venv"):
    print(f"\n🔒 Creating virtual environment at '{venv_path}'...")
    subprocess.check_call([sys.executable, "-m", "venv", venv_path])
    print(f"✅ Virtual environment created at '{venv_path}'!")
    print(f"👉 Activate it with: source {venv_path}/bin/activate\nThen re-run this script.")
    sys.exit(0)

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def test_installation():
    """Test if the installation works"""
    print("\n🧪 Testing installation...")
    try:
        import torch
        import transformers
        import PIL
        import matplotlib
        import numpy
        print("✅ All required packages imported successfully!")
        print("🔄 Testing CLIP model loading...")
        from space_clip_classifier import SpaceCLIPClassifier
        classifier = SpaceCLIPClassifier()
        print("✅ CLIP model loaded successfully!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

def create_example_script():
    example_script = '''#!/usr/bin/env python3
"""
Example usage of CLIP Space Image Classifier
"""
from space_clip_classifier import SpaceCLIPClassifier

def main():
    print("🚀 Initializing CLIP Space Classifier...")
    classifier = SpaceCLIPClassifier()
    image_url = "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?w=800"
    print(f"🔍 Classifying image: {image_url}")
    predictions = classifier.classify_image(image_url, top_k=5)
    print("\\n📊 Classification Results:")
    for i, (category, confidence) in enumerate(predictions, 1):
        print(f"   {i}. {category.capitalize()}: {confidence:.3f}")
    print("\\n📈 Generating visualization...")
    classifier.visualize_classification(image_url, top_k=5)
    print("\\n✅ Example completed!")

if __name__ == "__main__":
    main()
'''
    with open("example_usage.py", "w") as f:
        f.write(example_script)
    print("✅ Created example_usage.py")

def main():
    print("🌟 CLIP Space Image Classifier - Installation")
    print("=" * 50)
    if not check_python_version():
        sys.exit(1)
    if not in_venv():
        print("\n⚠️  You are NOT in a Python virtual environment!")
        print("This is required for safe installation (see PEP 668).\n")
        venv_path = "venv"
        if not Path(venv_path).exists():
            resp = input(f"Would you like to create a virtual environment at '{venv_path}' now? [Y/n]: ").strip().lower()
            if resp in ("", "y", "yes"):
                create_venv(venv_path)
            else:
                print(f"❌ Please create and activate a venv, then re-run this script.")
                print(f"   python3 -m venv {venv_path}\n   source {venv_path}/bin/activate")
                sys.exit(1)
        else:
            print(f"❌ Please activate your venv with: source {venv_path}/bin/activate\nThen re-run this script.")
            sys.exit(1)
    if not install_requirements():
        sys.exit(1)
    if not test_installation():
        print("\n❌ Installation test failed. Please check the error messages above.")
        sys.exit(1)
    create_example_script()
    print("\n🎉 Installation completed successfully!")
    print("\n📚 Next steps:")
    print("   1. Run the demo: python demo.py")
    print("   2. Try the example: python example_usage.py")
    print("   3. Use with your own images: python space_clip_classifier.py")
    print("   4. Filter a dataset: python space_image_filter.py /path/to/images")
    print("\n💡 Tips:")
    print("   - The first run will download the CLIP model (~500MB)")
    print("   - Use GPU if available for faster processing")
    print("   - Check README.md for detailed usage instructions")

if __name__ == "__main__":
    main() 