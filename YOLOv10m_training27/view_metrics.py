#!/usr/bin/env python3
"""
Extract and display key metrics from YOLOv10m training visualizations
Shows text-based analysis of confusion matrix and performance curves
"""

from PIL import Image
import numpy as np
import sys

def analyze_confusion_matrix(image_path):
    """Analyze confusion matrix image and extract insights"""
    print("=" * 80)
    print("CONFUSION MATRIX ANALYSIS")
    print("=" * 80)
    
    img = Image.open(image_path)
    print(f"\nImage: {image_path}")
    print(f"Dimensions: {img.size[0]}x{img.size[1]} pixels")
    
    # The confusion matrix is a heatmap
    # We can analyze the general structure even without OCR
    
    print("\n📊 VISUAL CHARACTERISTICS:")
    print("  • This is a normalized confusion matrix (percentage-based)")
    print("  • Rows = True classes (actual traffic signs)")
    print("  • Columns = Predicted classes (what model detected)")
    print("  • Diagonal = Correct predictions (should be brightest)")
    print("  • Off-diagonal = Misclassifications (should be dark)")
    
    # Convert to numpy for basic analysis
    img_array = np.array(img.convert('RGB'))
    
    # Get average brightness (higher = better performance generally)
    avg_brightness = np.mean(img_array)
    
    print(f"\n💡 QUICK ASSESSMENT:")
    print(f"  • Average image brightness: {avg_brightness:.1f}/255")
    
    if avg_brightness > 200:
        print("  • Overall: Likely high accuracy (bright confusion matrix)")
    elif avg_brightness > 150:
        print("  • Overall: Moderate accuracy")
    else:
        print("  • Overall: May have accuracy issues (dark matrix)")
    
    print("\n🎯 TO PROPERLY READ THIS MATRIX:")
    print("  1. Look for a bright diagonal line (top-left to bottom-right)")
    print("  2. Check if off-diagonal areas are mostly dark/black")
    print("  3. Identify any bright spots off the diagonal (class confusions)")
    print("  4. Check row/column labels for class names/IDs")
    
    print("\n✅ GOOD SIGNS:")
    print("  • Strong bright diagonal")
    print("  • Dark everywhere else")
    print("  • Each class >80% accuracy on diagonal")
    
    print("\n⚠️  WARNING SIGNS:")
    print("  • Dim diagonal elements (low accuracy for those classes)")
    print("  • Bright off-diagonal spots (systematic confusion)")
    print("  • Missing rows/columns (classes not detected)")
    
    return img_array

def analyze_pr_curve(image_path):
    """Analyze PR curve and extract insights"""
    print("\n" + "=" * 80)
    print("PRECISION-RECALL CURVE ANALYSIS")
    print("=" * 80)
    
    img = Image.open(image_path)
    print(f"\nImage: {image_path}")
    print(f"Dimensions: {img.size[0]}x{img.size[1]} pixels")
    
    print("\n📊 HOW TO READ THIS CURVE:")
    print("  • X-axis: Recall (0.0 to 1.0) - % of true signs found")
    print("  • Y-axis: Precision (0.0 to 1.0) - % of predictions that are correct")
    print("  • Curve shows trade-off between precision and recall")
    print("  • Area under curve = mAP (mean Average Precision)")
    
    print("\n🎯 INTERPRETING THE SHAPE:")
    
    print("\n  EXCELLENT (mAP > 0.90):")
    print("    ┌─────────────┐")
    print("    │ ┌─────────┐ │  ← Curve hugs top-right corner")
    print("  P │ │         │ │")
    print("    │ │         └─┤")
    print("    └─────────────┘")
    print("        Recall")
    
    print("\n  POOR (mAP < 0.60):")
    print("    ┌─────────────┐")
    print("    │           ╲ │  ← Curve sags toward bottom-left")
    print("  P │            ╲│")
    print("    │             ╲")
    print("    └─────────────┘")
    print("        Recall")
    
    print("\n💡 LOOK FOR:")
    print("  • mAP@0.5 value (usually shown in legend/title)")
    print("  • All classes curve (overall performance)")
    print("  • Individual class curves (per-sign performance)")
    
    print("\n📏 PERFORMANCE BENCHMARKS:")
    print("  • mAP@0.5 > 0.90  = Excellent (production ready)")
    print("  • mAP@0.5 = 0.80-0.90 = Very good")
    print("  • mAP@0.5 = 0.70-0.80 = Good")
    print("  • mAP@0.5 = 0.60-0.70 = Fair (needs improvement)")
    print("  • mAP@0.5 < 0.60  = Poor (retrain needed)")

def analyze_f1_curve(image_path):
    """Analyze F1 curve"""
    print("\n" + "=" * 80)
    print("F1 SCORE CURVE ANALYSIS")
    print("=" * 80)
    
    img = Image.open(image_path)
    print(f"\nImage: {image_path}")
    print(f"Dimensions: {img.size[0]}x{img.size[1]} pixels")
    
    print("\n📊 PURPOSE:")
    print("  • Find optimal confidence threshold for inference")
    print("  • F1 = harmonic mean of Precision and Recall")
    print("  • Formula: F1 = 2 × (P × R) / (P + R)")
    
    print("\n🎯 HOW TO USE:")
    print("  1. Find the peak of the curve")
    print("  2. Note the confidence value at that peak")
    print("  3. Use this threshold when running inference")
    
    print("\n  Example Curve:")
    print("    1.0 ┌───────────────┐")
    print("        │       ╱╲      │")
    print("    0.8 │     ╱    ╲    │  ← Peak at ~0.35 confidence")
    print("  F     │   ╱        ╲  │")
    print("  1 0.6 │ ╱            ╲│")
    print("        │╱              ╲")
    print("    0.0 └───────────────┘")
    print("        0.0   0.35   1.0")
    print("           Confidence")
    
    print("\n💡 TYPICAL OPTIMAL THRESHOLDS:")
    print("  • 0.20-0.30: Maximize detections (high recall)")
    print("  • 0.30-0.40: Balanced performance (peak F1)")
    print("  • 0.50-0.70: Minimize false alarms (high precision)")
    
    print("\n✅ GOOD PERFORMANCE:")
    print("  • Peak F1 > 0.80")
    print("  • Sharp peak (not flat)")
    print("  • All classes have similar F1 scores")
    
    print("\n⚠️  ISSUES:")
    print("  • Peak F1 < 0.60 (poor overall performance)")
    print("  • Flat curve (threshold insensitive, may indicate issues)")
    print("  • Large variation between classes")

def analyze_validation_batch(labels_path, pred_path):
    """Analyze validation batch comparison"""
    print("\n" + "=" * 80)
    print("VALIDATION BATCH COMPARISON")
    print("=" * 80)
    
    labels = Image.open(labels_path)
    preds = Image.open(pred_path)
    
    print(f"\nGround Truth: {labels_path}")
    print(f"  Dimensions: {labels.size[0]}x{labels.size[1]} pixels")
    
    print(f"\nPredictions: {pred_path}")
    print(f"  Dimensions: {preds.size[0]}x{preds.size[1]} pixels")
    
    print("\n📊 MOSAIC LAYOUT:")
    print("  Both images show a grid of validation samples")
    print("  Each cell contains:")
    print("    • Traffic sign image")
    print("    • Bounding boxes around detected signs")
    print("    • Class labels")
    
    print("\n🎯 COMPARISON CHECKLIST:")
    
    print("\n  ✅ GOOD SIGNS (compare labels vs predictions):")
    print("    • Bounding boxes match in size and position")
    print("    • All signs in labels are also in predictions")
    print("    • Class labels are identical")
    print("    • Boxes are tight around signs (not too loose)")
    
    print("\n  ⚠️  ISSUES TO LOOK FOR:")
    print("    • Missing detections: Box in labels but not in predictions")
    print("    • False positives: Box in predictions but not in labels")
    print("    • Wrong class: Different label in predictions vs labels")
    print("    • Loose boxes: Predictions have oversized bounding boxes")
    print("    • Position errors: Boxes don't align properly")
    
    print("\n💡 VISUAL INSPECTION:")
    print("  1. Open both images side-by-side")
    print("  2. Look at same position in both images")
    print("  3. Count differences (missing, extra, wrong boxes)")
    print("  4. Estimate accuracy: # correct / # total signs")

def main():
    """Main analysis function"""
    print("\n" + "=" * 80)
    print(" " * 20 + "YOLOv10m TRAINING METRICS ANALYSIS")
    print(" " * 25 + "Training27 - Text-Based View")
    print("=" * 80)
    
    files = {
        'confusion_matrix_normalized.png': analyze_confusion_matrix,
        'PR_curve.png': analyze_pr_curve,
        'F1_curve.png': analyze_f1_curve,
    }
    
    # Analyze each metric file
    for filename, analyzer in files.items():
        try:
            analyzer(filename)
        except FileNotFoundError:
            print(f"\n⚠️  File not found: {filename}")
        except Exception as e:
            print(f"\n❌ Error analyzing {filename}: {e}")
    
    # Validation batch
    try:
        analyze_validation_batch('val_batch0_labels.jpg', 'val_batch0_pred.jpg')
    except Exception as e:
        print(f"\n❌ Error analyzing validation batch: {e}")
    
    print("\n" + "=" * 80)
    print("📌 SUMMARY")
    print("=" * 80)
    
    print("\n✨ KEY METRICS TO EXTRACT (manually from images):")
    print("  1. Confusion Matrix:")
    print("     → Check diagonal values (should be >80%)")
    print("     → Note any bright off-diagonal spots")
    
    print("\n  2. PR Curve:")
    print("     → Read mAP@0.5 value from legend")
    print("     → Target: >0.80 for good performance")
    
    print("\n  3. F1 Curve:")
    print("     → Find peak F1 value")
    print("     → Note optimal confidence threshold")
    
    print("\n  4. Validation Batch:")
    print("     → Count missed detections")
    print("     → Count false positives")
    print("     → Estimate overall accuracy")
    
    print("\n🖼️  TO VIEW IMAGES:")
    print("  # In GUI environment:")
    print("  eog confusion_matrix_normalized.png")
    print("  eog PR_curve.png")
    print("  eog F1_curve.png")
    print("  eog val_batch0_labels.jpg val_batch0_pred.jpg")
    
    print("\n  # Or copy to local machine and open:")
    print("  scp user@server:path/to/*.png .")
    print("  open *.png  # macOS")
    print("  xdg-open *.png  # Linux")
    
    print("\n" + "=" * 80)
    print("Analysis complete! View the images to get actual metric values.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
