

import os
import subprocess
import sys
from pathlib import Path

def print_header():
    print("="*60)
    print("🕳️  Pothole Detection Setup")
    print("="*60)

def create_structure():
    """Create project directory structure"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        'data/images/train',
        'data/images/val',
        'data/labels/train',
        'data/labels/val',
        'models',
        'results'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {dir_path}")
    
    # Create placeholder files
    placeholders = {
        'data/images/train/README.txt': 'Place training images here',
        'data/images/val/README.txt': 'Place validation images here',
        'data/labels/train/README.txt': 'Place training labels here (.txt files in YOLO format)',
        'data/labels/val/README.txt': 'Place validation labels here',
        'models/README.txt': 'Trained models will be saved here',
        'results/README.txt': 'Detection results will be saved here'
    }
    
    for file_path, content in placeholders.items():
        with open(file_path, 'w') as f:
            f.write(content)
    
    return True

def install_requirements():
    """Install required packages"""
    print("\n🔧 Installing requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Could not install requirements automatically")
        print("Please run: pip install -r requirements.txt")
        return False

def check_gpu():
    """Check GPU availability"""
    print("\n🎮 Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name}")
            return True
        else:
            print("ℹ️  No GPU detected. Training will use CPU (slower)")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False

def show_instructions():
    """Show usage instructions"""
    print("\n📚 INSTRUCTIONS:")
    print("-"*40)
    
    print("\n1. 📊 Prepare your dataset:")
    print("   - Images in: data/images/train/ and data/images/val/")
    print("   - Labels in: data/labels/train/ and data/labels/val/")
    print("   - Labels must be in YOLO format: class x_center y_center width height")
    
    print("\n2. 🚀 Train the model:")
    print("   python train.py --epochs 100 --batch 16")
    
    print("\n3. 🔍 Detect potholes:")
    print("   python detect.py --source test.jpg")
    print("   python detect.py --source test.mp4")
    print("   python detect.py --source 0  (for webcam)")
    
    print("\n4. 🎯 To achieve 89% accuracy:")
    print("   - Use 1000+ images")
    print("   - Train for 100+ epochs")
    print("   - Use YOLOv8m or larger")
    print("   - Enable augmentations")
    
    print("\n5. 📈 Monitor training:")
    print("   - Check runs/detect/pothole_v8m/ for results")
    print("   - View training plots and metrics")

def main():
    print_header()
    
    # Create structure
    create_structure()
    
    # Install requirements
    install_req = input("\nInstall requirements now? (y/n): ")
    if install_req.lower() == 'y':
        install_requirements()
    
    # Check GPU
    check_gpu()
    
    # Show instructions
    show_instructions()
    
    print("\n" + "="*60)
    print("✅ Setup Complete!")
    print("="*60)
    
    print("\n🎯 Next steps:")
    print("1. Add your pothole images and labels")
    print("2. Run: python train.py")
    print("3. Run: python detect.py --source your_image.jpg")

if __name__ == "__main__":
    main()