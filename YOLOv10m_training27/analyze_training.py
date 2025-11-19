#!/usr/bin/env python3
"""
Analyze YOLOv10m Training Results - Training27
Extracts insights from confusion matrix and performance curves
"""

from PIL import Image
import os

def analyze_training_results():
    """Analyze training result visualizations"""
    
    print("=" * 70)
    print("YOLOv10m TRAINING RESULTS ANALYSIS - Training27")
    print("=" * 70)
    print(f"\nTraining Date: January 26, 2025")
    print(f"Model: YOLOv10m (Medium variant)")
    print(f"Task: Traffic Sign Detection")
    
    print("\n" + "=" * 70)
    print("📁 AVAILABLE METRICS FILES")
    print("=" * 70)
    
    files = {
        'Performance Curves': [
            ('F1_curve.png', 'F1 Score vs Confidence threshold'),
            ('P_curve.png', 'Precision vs Confidence threshold'),
            ('R_curve.png', 'Recall vs Confidence threshold'),
            ('PR_curve.png', 'Precision-Recall curve'),
        ],
        'Confusion Matrices': [
            ('confusion_matrix.png', 'Raw confusion matrix (count-based)'),
            ('confusion_matrix_normalized.png', 'Normalized confusion matrix (percentage)'),
        ],
        'Validation Examples': [
            ('val_batch0_labels.jpg', 'Ground truth labels (mosaic view)'),
            ('val_batch0_pred.jpg', 'Model predictions (mosaic view)'),
        ]
    }
    
    for category, file_list in files.items():
        print(f"\n{category}:")
        for filename, description in file_list:
            if os.path.exists(filename):
                img = Image.open(filename)
                size_kb = os.path.getsize(filename) / 1024
                print(f"  ✓ {filename}")
                print(f"    - {description}")
                print(f"    - Dimensions: {img.size[0]}x{img.size[1]} pixels")
                print(f"    - Size: {size_kb:.0f} KB")
            else:
                print(f"  ✗ {filename} - NOT FOUND")
    
    print("\n" + "=" * 70)
    print("🎯 UNDERSTANDING THE METRICS")
    print("=" * 70)
    
    print("\n1. CONFUSION MATRIX")
    print("   • Shows true vs predicted classes for each traffic sign")
    print("   • Diagonal = correct predictions")
    print("   • Off-diagonal = misclassifications")
    print("   • Normalized version shows percentages (easier to interpret)")
    
    print("\n2. PRECISION-RECALL (PR) CURVE")
    print("   • Precision = TP / (TP + FP) - how many predictions are correct")
    print("   • Recall = TP / (TP + FN) - how many true signs are found")
    print("   • Area under curve (AUC) indicates overall performance")
    print("   • Higher is better for both metrics")
    
    print("\n3. F1 SCORE CURVE")
    print("   • F1 = 2 × (Precision × Recall) / (Precision + Recall)")
    print("   • Harmonic mean of precision and recall")
    print("   • Shows optimal confidence threshold")
    print("   • Peak F1 = best balance between precision and recall")
    
    print("\n4. PRECISION & RECALL CURVES")
    print("   • Show how metrics change with confidence threshold")
    print("   • P_curve: Higher threshold = better precision, fewer detections")
    print("   • R_curve: Lower threshold = better recall, more detections")
    print("   • Choose threshold based on use case:")
    print("     - Safety-critical: Optimize for recall (don't miss signs)")
    print("     - Low false positives: Optimize for precision")
    
    print("\n5. VALIDATION BATCH COMPARISON")
    print("   • val_batch0_labels.jpg: What signs should be detected")
    print("   • val_batch0_pred.jpg: What the model actually detected")
    print("   • Visual comparison shows model strengths/weaknesses")
    print("   • Look for: missed signs, false positives, bbox accuracy")
    
    print("\n" + "=" * 70)
    print("🔍 HOW TO INTERPRET RESULTS")
    print("=" * 70)
    
    print("\n✅ GOOD SIGNS:")
    print("   • Strong diagonal in confusion matrix")
    print("   • PR curve hugging top-right corner (high precision & recall)")
    print("   • F1 score peak > 0.8")
    print("   • val_batch predictions match labels closely")
    
    print("\n⚠️  WARNING SIGNS:")
    print("   • Off-diagonal values in confusion matrix (class confusion)")
    print("   • PR curve sagging in middle (poor precision-recall balance)")
    print("   • Low F1 peak < 0.6")
    print("   • Many missed or false detections in validation")
    
    print("\n" + "=" * 70)
    print("📊 TYPICAL YOLO PERFORMANCE RANGES")
    print("=" * 70)
    
    performance_ranges = {
        'Excellent': ('>90%', 'Production ready, state-of-the-art'),
        'Very Good': ('80-90%', 'Suitable for most applications'),
        'Good': ('70-80%', 'Usable with some limitations'),
        'Fair': ('60-70%', 'Needs improvement, specific use cases'),
        'Poor': ('<60%', 'Requires retraining or more data'),
    }
    
    print("\nmAP@0.5 (mean Average Precision at IoU=0.5):")
    for level, (range_val, desc) in performance_ranges.items():
        print(f"   • {level:12} {range_val:8} - {desc}")
    
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70)
    
    print("\n1. VIEW CONFUSION MATRIX FIRST")
    print("   Command: display confusion_matrix_normalized.png")
    print("   Look for:")
    print("   • Which classes are confused with each other")
    print("   • Classes with low accuracy")
    
    print("\n2. CHECK PR CURVE")
    print("   Command: display PR_curve.png")
    print("   Look for:")
    print("   • Overall mAP value")
    print("   • Per-class performance")
    
    print("\n3. ANALYZE F1 CURVE")
    print("   Command: display F1_curve.png")
    print("   Look for:")
    print("   • Optimal confidence threshold (peak F1)")
    print("   • All classes vs per-class F1 scores")
    
    print("\n4. COMPARE VALIDATION PREDICTIONS")
    print("   Command: display val_batch0_labels.jpg val_batch0_pred.jpg")
    print("   Look for:")
    print("   • Bounding box accuracy")
    print("   • Missed detections (in labels but not predictions)")
    print("   • False positives (in predictions but not labels)")
    
    print("\n" + "=" * 70)
    print("🚀 NEXT STEPS")
    print("=" * 70)
    
    print("\n1. Visual Inspection (using display or image viewer):")
    print("   cd /home/mnxtr/Traffic-Sign-Recognition-YOLOv10/YOLOv10m_training27")
    print("   display confusion_matrix_normalized.png")
    print("   display PR_curve.png")
    print("   display F1_curve.png")
    
    print("\n2. Compare with other training runs:")
    print("   unzip YOLOv10m_training22.zip")
    print("   unzip YOLOv10m_training23.zip")
    print("   # Compare metrics to see improvement over time")
    
    print("\n3. Use model for inference:")
    print("   # If model weights (.pt file) are available")
    print("   # Run predictions on new images")
    
    print("\n4. Fine-tune if needed:")
    print("   # Based on confusion matrix, identify weak classes")
    print("   # Collect more data for those specific sign types")
    print("   # Adjust class weights or augmentation")
    
    print("\n" + "=" * 70)
    print("Note: This analysis is based on visualization files only.")
    print("For detailed metrics (mAP, loss curves, etc.), check:")
    print("  • results.csv (if available)")
    print("  • Training logs")
    print("  • Model weights metadata")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    analyze_training_results()
