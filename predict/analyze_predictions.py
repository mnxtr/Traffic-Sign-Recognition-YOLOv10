#!/usr/bin/env python3
"""
Analyze YOLOv10 Traffic Sign Predictions
Shows statistics about predicted traffic signs
"""

import os
from collections import Counter
import json

def analyze_predictions():
    """Analyze prediction results in the predict folder"""
    
    labels_dir = "labels"
    images_dir = "."
    
    # Count images and labels
    images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    print("=" * 60)
    print("TRAFFIC SIGN PREDICTION ANALYSIS")
    print("=" * 60)
    print(f"\n📊 Dataset Overview:")
    print(f"  • Total images: {len(images)}")
    print(f"  • Images with labels: {len(labels)}")
    print(f"  • Images without labels: {len(images) - len(labels)}")
    print(f"  • Coverage: {len(labels)/len(images)*100:.1f}%")
    
    # Analyze class distribution
    class_counts = Counter()
    total_detections = 0
    multi_object_images = 0
    
    for label_file in labels:
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                multi_object_images += 1
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    total_detections += 1
    
    print(f"\n🎯 Detection Statistics:")
    print(f"  • Total detections: {total_detections}")
    print(f"  • Average detections per image: {total_detections/len(labels):.2f}")
    print(f"  • Images with multiple signs: {multi_object_images}")
    print(f"  • Unique traffic sign classes: {len(class_counts)}")
    
    print(f"\n🚦 Traffic Sign Class Distribution:")
    print(f"  {'Class ID':<12} {'Count':<8} {'Percentage':<12} {'Bar'}")
    print(f"  {'-'*12} {'-'*8} {'-'*12} {'-'*30}")
    
    for class_id, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_detections) * 100
        bar = '█' * int(percentage / 2)
        print(f"  Class {class_id:<6} {count:<8} {percentage:>6.2f}%      {bar}")
    
    # Sample detections
    print(f"\n📝 Sample Predictions:")
    sample_count = 0
    for label_file in sorted(labels)[:3]:
        if sample_count >= 3:
            break
        label_path = os.path.join(labels_dir, label_file)
        image_name = label_file.replace('.txt', '.jpg')
        print(f"\n  Image: {image_name}")
        with open(label_path, 'r') as f:
            for i, line in enumerate(f.readlines(), 1):
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id, x, y, w, h = parts[0], parts[1], parts[2], parts[3], parts[4]
                    print(f"    Detection {i}: Class {class_id} at ({x}, {y}) size ({w}, {h})")
        sample_count += 1
    
    # Class mapping suggestion
    print(f"\n💡 Class ID Mapping (examples based on common traffic signs):")
    class_mapping = {
        22: "Speed limit or regulatory sign",
        23: "Warning sign",
        26: "Priority/give way sign",
        28: "Information sign",
        29: "Direction sign",
        32: "Parking or stop sign",
        35: "Pedestrian crossing sign",
        36: "General warning/caution sign"
    }
    
    for class_id, description in class_mapping.items():
        if class_id in class_counts:
            print(f"  Class {class_id}: {description}")
    
    print(f"\n{'='*60}")
    print(f"Note: Exact sign meanings require the original class.yaml file")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    analyze_predictions()
