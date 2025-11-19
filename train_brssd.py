#!/usr/bin/env python3
"""
YOLOv10 Training Script for BRSSD Dataset
Bangladeshi Road Sign Symbol Detection
"""

import os
from pathlib import Path
from ultralytics import YOLO
import yaml
import argparse

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
    
    print(f"✓ Dataset verified:")
    print(f"  Training images: {len(train_images)}")
    print(f"  Validation images: {len(val_images)}")
    print(f"  Number of classes: {config['nc']}")
    
    return config

def train_yolov10(model_size='n', epochs=100, batch=16, imgsz=640, data_yaml='brssd_data.yaml'):
    """Train YOLOv10 on BRSSD dataset"""
    
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
    
    # Initialize model
    print("\nInitializing model...")
    model = YOLO(model_path)
    
    # Training parameters
    training_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch,
        'name': f'YOLOv10{model_size}_BRSSD',
        'patience': 50,
        'save': True,
        'device': 'auto',  # Auto-detect GPU, use CPU if not available
        'workers': 8,
        'project': 'runs/brssd',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'val': True,
        'plots': True,
        
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
        
        # Learning rate
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        
        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
    }
    
    # Start training
    print("\nStarting training...\n")
    results = model.train(**training_args)
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"{'='*60}\n")
    
    # Validation
    print("Running validation...")
    metrics = model.val()
    
    print("\nValidation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    
    # Save best model info
    print(f"\nBest model saved at: runs/brssd/YOLOv10{model_size}_BRSSD/weights/best.pt")
    
    return model, results, metrics

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv10 on BRSSD Dataset')
    parser.add_argument('--model', choices=['n', 's', 'm', 'l', 'x'], default='n',
                       help='Model size: n(nano), s(small), m(medium), l(large), x(xlarge)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--data', type=str, default='brssd_data.yaml', help='Dataset YAML file')
    
    args = parser.parse_args()
    
    try:
        model, results, metrics = train_yolov10(
            model_size=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            data_yaml=args.data
        )
        print("\n✓ Training completed successfully!")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
