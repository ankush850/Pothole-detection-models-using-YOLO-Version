from ultralytics import YOLO
import torch
import os
import argparse

def train_pothole_detector(
    model_size='m',
    epochs=100,
    batch=16,
    imgsz=640,
    data_config='config.yaml',
    device=None
):
    """  Train a YOLOv8 model for pothole detection with optimized parameters."""
    print("="*60)
    print("🕳️  Pothole Detection Training")
    print("="*60)
    
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"📱 Device: {device}")
    print(f"📐 Model: YOLOv8{model_size}")
    print(f"📈 Epochs: {epochs}")
    print(f"📦 Batch: {batch}")
    print(f"🖼️  Image Size: {imgsz}")
    
    # Load model
    print("\n📦 Loading model...")
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Training parameters optimized for 89% accuracy
    training_args = {
        'data': data_config,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'patience': 50,  # Early stopping
        'device': device,
        'workers': 8,
        'optimizer': 'AdamW',  # Better optimizer
        'lr0': 0.001,  # Learning rate
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # Data augmentations (crucial for 89% accuracy)
        'augment': True,
        'degrees': 15.0,  # Increased rotation
        'translate': 0.2,  # Increased translation
        'scale': 0.5,
        'shear': 5.0,  # Increased shear
        'perspective': 0.0005,
        'flipud': 0.0,
        'fliplr': 0.5,  # Horizontal flip
        'mosaic': 1.0,
        'mixup': 0.3,  # Increased mixup
        'copy_paste': 0.0,
        'erasing': 0.4,  # Random erasing
        
        # Advanced settings
        'close_mosaic': 15,
        'amp': True,  # Mixed precision
        'fraction': 1.0,
        'profile': False,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,  # Cosine learning rate scheduler
        'label_smoothing': 0.0,
        'nbs': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        
        # Save settings
        'save': True,
        'save_period': 10,
        'cache': True if device != 'cpu' else False,
        'resume': False,
        'plots': True,
        'exist_ok': True,
        'name': f'pothole_v8{model_size}',
    }
    
    print("\n🚀 Starting training...")
    print("This may take a while. Grab a coffee! ☕")
    
    # Train the model
    try:
        results = model.train(**training_args)
        
        print("\n✅ Training completed!")
        
        # Validate the model
        print("\n📊 Validating model...")
        metrics = model.val(
            data=data_config,
            imgsz=imgsz,
            batch=batch*2,
            conf=0.25,
            iou=0.45,
            device=device,
            plots=True
        )
        
        # Print results
        print("\n" + "="*60)
        print("📈 TRAINING RESULTS")
        print("="*60)
        print(f"mAP@0.5: {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.p:.4f}")
        print(f"Recall: {metrics.box.r:.4f}")
        
        # Check if 89% accuracy achieved
        if metrics.box.map50 >= 0.89:
            print(f"\n🎉 SUCCESS! Achieved {metrics.box.map50:.2%} accuracy (Target: 89%)")
        else:
            print(f"\n⚠️  Accuracy: {metrics.box.map50:.2%} (Target: 89%)")
            print("Try increasing epochs or using more data")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Train pothole detection model')
    parser.add_argument('--model', type=str, default='m',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu/mps) or None for auto')
    
    args = parser.parse_args()
    
    # Check if config exists
    if not os.path.exists('config.yaml'):
        print("❌ config.yaml not found!")
        print("Make sure you have the config.yaml file")
        return
    
    # Check if data exists
    if not os.path.exists('data/images/train'):
        print("❌ Training data not found!")
        print("Please place your data in:")
        print("  data/images/train/")
        print("  data/labels/train/")
        print("  data/images/val/")
        print("  data/labels/val/")
        return
    
    # Start training
    train_pothole_detector(
        model_size=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )

if __name__ == "__main__":
    main()