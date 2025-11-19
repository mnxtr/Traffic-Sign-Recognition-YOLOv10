#!/usr/bin/env python3
"""
YOLOv10 Training Script for BRSSD Dataset (Improved Version)
Bangladeshi Road Sign Symbol Detection
Includes GPU auto-detection, better error handling, and progress tracking
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml
import argparse
import torch

def check_gpu():
    """Check GPU availability and return device info"""
    print("\n" + "="*60)
    print("System Configuration")
    print("="*60)
    
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✓ GPU Available: {gpu_name}")
        print(f"  GPU Memory: {gpu_memory:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
        device = '0'  # Use first GPU
    else:
        print("⚠️  No GPU detected - training will use CPU")
        print("  Note: CPU training will be significantly slower")
        device = 'cpu'
    
    print("="*60 + "\n")
    return device

def verify_dataset(data_yaml_path):
    """Verify dataset structure and configuration"""
    print("Verifying dataset configuration...")
    
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {data_yaml_path}")
    
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_path = Path(config['path'])
    
    # Check directories
    required_dirs = ['train/images', 'train/labels', 'valid/images', 'valid/labels']
    for dir_path in required_dirs:
        full_path = dataset_path / dir_path
        if not full_path.exists():
            raise FileNotFoundError(f"Required directory not found: {full_path}")
    
    # Count images
    train_images = list((dataset_path / 'train/images').glob('*.*'))
    val_images = list((dataset_path / 'valid/images').glob('*.*'))
    test_images = list((dataset_path / 'test/images').glob('*.*')) if (dataset_path / 'test/images').exists() else []
    
    print(f"✓ Dataset verified:")
    print(f"  Training images: {len(train_images)}")
    print(f"  Validation images: {len(val_images)}")
    print(f"  Test images: {len(test_images)}")
    print(f"  Number of classes: {config['nc']}")
    print(f"  Dataset path: {dataset_path.absolute()}")
    
    return config

def train_yolov10(model_size='n', epochs=100, batch=16, imgsz=640, data_yaml='brssd_data.yaml', device='auto'):
    """Train YOLOv10 on BRSSD dataset"""
    
    # Auto-detect GPU if device is 'auto'
    if device == 'auto':
        device = check_gpu()
    
    # Verify dataset
    config = verify_dataset(data_yaml)
    
    # Model selection
    model_variants = {
        'n': 'yolov10n.pt',  # Nano - fastest
        's': 'yolov10s.pt',  # Small
        'm': 'yolov10m.pt',  # Medium
        'l': 'yolov10l.pt',  # Large
        'x': 'yolov10x.pt'   # Extra large - most accurate
    }
    
    model_path = model_variants.get(model_size, 'yolov10n.pt')
    
    print(f"\n{'='*60}")
    print(f"Training YOLOv10-{model_size.upper()} on BRSSD Dataset")
    print(f"{'='*60}\n")
    
    print("Configuration:")
    print(f"  Model: {model_path}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch}")
    print(f"  Image size: {imgsz}")
    print(f"  Dataset: {data_yaml}")
    print(f"  Device: {device}")
    
    # Initialize model
    print("\nInitializing model...")
    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise
    
    # Adjust batch size for CPU
    if device == 'cpu' and batch > 8:
        print(f"\n⚠️  Reducing batch size from {batch} to 8 for CPU training")
        batch = 8
    
    # Training parameters
    training_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'name': f'YOLOv10{model_size}_BRSSD',
        'patience': 50,
        'save': True,
        'device': device,
        'workers': 8 if device != 'cpu' else 4,
        'project': 'runs/brssd',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'val': True,
        'plots': True,
        'save_period': -1,  # Save checkpoint every N epochs (-1 to disable)
        
        # Augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        
        # Learning rate
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
    }
    
    # Start training
    print("\nStarting training...\n")
    print("=" * 60)
    
    try:
        results = model.train(**training_args)
        
        print(f"\n{'='*60}")
        print("Training completed successfully!")
        print(f"{'='*60}\n")
        
        # Validation
        print("Running final validation...")
        metrics = model.val()
        
        print("\nValidation Metrics:")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"  Precision: {metrics.box.mp:.4f}")
        print(f"  Recall: {metrics.box.mr:.4f}")
        
        # Save best model info
        results_dir = f"runs/brssd/YOLOv10{model_size}_BRSSD"
        print(f"\n{'='*60}")
        print("Model Saved:")
        print(f"  Best weights: {results_dir}/weights/best.pt")
        print(f"  Last weights: {results_dir}/weights/last.pt")
        print(f"  Results: {results_dir}/")
        print(f"{'='*60}\n")
        
        return model, results, metrics
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Partial results may be saved in runs/brssd/")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv10 on BRSSD Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (nano model, 100 epochs)
  python train_brssd_improved.py
  
  # Train small model for 50 epochs with batch size 32
  python train_brssd_improved.py --model s --epochs 50 --batch 32
  
  # Train on CPU explicitly
  python train_brssd_improved.py --device cpu
  
  # Quick test run (1 epoch)
  python train_brssd_improved.py --epochs 1 --batch 4
        """
    )
    
    parser.add_argument('--model', choices=['n', 's', 'm', 'l', 'x'], default='n',
                       help='Model size: n(nano), s(small), m(medium), l(large), x(xlarge)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--data', type=str, default='brssd_data.yaml', help='Dataset YAML file')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use: auto, cpu, 0, 1, etc.')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("BRSSD YOLOv10 Training Script")
    print("="*60)
    
    try:
        model, results, metrics = train_yolov10(
            model_size=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            data_yaml=args.data,
            device=args.device
        )
        print("\n✓ Training pipeline completed successfully!")
        return 0
    except FileNotFoundError as e:
        print(f"\n✗ File not found: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure the BRSSD dataset is downloaded")
        print("  2. Check that brssd_data.yaml exists and points to the correct paths")
        print("  3. Run: python download_brssd.py --source all")
        return 1
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        print("\nPlease check the error message above and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
