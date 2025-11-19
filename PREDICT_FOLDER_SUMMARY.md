# Traffic Sign Recognition - Predict Folder Analysis

**Generated:** 2025-11-18  
**Location:** `/home/mnxtr/Traffic-Sign-Recognition-YOLOv10/predict/`

---

## 📊 Overview

The `predict/` folder contains **YOLOv10 model prediction results** on traffic sign images.

### Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Images** | 170 images |
| **Images with Predictions** | 56 images (32.9%) |
| **Images without Predictions** | 114 images (67.1%) |
| **Image Format** | JPEG, 640x640 pixels, RGB |
| **Total Size** | 27 MB |

### Detection Statistics

| Metric | Value |
|--------|-------|
| **Total Detections** | 67 traffic signs |
| **Average per Image** | 1.20 signs/image |
| **Multi-sign Images** | 10 images |
| **Unique Classes** | 8 traffic sign types |

---

## 🚦 Traffic Sign Class Distribution

The model detected **8 different traffic sign classes** across 56 images:

| Class ID | Count | Percentage | Description (Estimated) |
|----------|-------|------------|-------------------------|
| **36** | 24 | 35.82% | ⚠️ General warning/caution sign |
| **26** | 17 | 25.37% | 🔺 Priority/give way sign |
| **23** | 11 | 16.42% | ⚠️ Warning sign |
| **28** | 8 | 11.94% | ℹ️ Information sign |
| **32** | 3 | 4.48% | 🅿️ Parking or stop sign |
| **35** | 2 | 2.99% | 🚶 Pedestrian crossing sign |
| **29** | 1 | 1.49% | ➡️ Direction sign |
| **22** | 1 | 1.49% | 🔢 Speed limit/regulatory sign |

**Top 3 Classes:** Class 36 (36%), Class 26 (25%), Class 23 (16%)

---

## 📁 Folder Structure

```
predict/
├── *.jpg              # 170 traffic sign images (640x640)
└── labels/
    └── *.txt          # 56 YOLO format annotation files
```

### Label Format (YOLO)
Each `.txt` file contains:
```
<class_id> <x_center> <y_center> <width> <height>
```
- Values are **normalized** (0.0 to 1.0)
- Multiple lines = multiple detections in one image

---

## 📝 Sample Predictions

### Example 1: Single Detection
**Image:** `0216206a-c7-1_jpg.rf.9826673d394117790937e834c9cc5947_aug1.jpg`
```
Class 32 at (0.334, 0.771) size (0.669, 0.458)
→ Stop/Parking sign in center-bottom area
```

### Example 2: Multiple Detections
**Image:** `07b48bfc-e32-4_jpg.rf.8fa48bd62b62bc110ba1f3f601a978e9_aug0.jpg`
```
1. Class 35 at (0.974, 0.763) size (0.052, 0.204)
   → Pedestrian crossing sign (small, bottom-right)
   
2. Class 26 at (0.382, 0.280) size (0.757, 0.559)
   → Priority sign (large, center-left)
```

---

## 📈 Model Performance Insights

Based on the detection distribution:

✅ **Strengths:**
- Good detection of warning signs (Class 36, 23) - 52% of all detections
- Reliable priority sign detection (Class 26) - 25% coverage
- Handles multi-sign scenarios (10 images with 2+ signs)

⚠️ **Observations:**
- 67% of images have no predictions (possible negatives or detection threshold)
- Some classes rarely detected (22, 29, 35) - may indicate:
  - Class imbalance in training data
  - Difficult sign types
  - Lower confidence thresholds

---

## 🎯 Training Results (YOLOv10m_training27)

The latest model checkpoint includes:

### Performance Metrics
- **Confusion Matrix** (normalized and standard)
- **Precision-Recall Curve**
- **F1 Score Curve**
- **Precision Curve**
- **Recall Curve**

### Validation Examples
- `val_batch0_labels.jpg` - Ground truth annotations
- `val_batch0_pred.jpg` - Model predictions

**Metrics Files:** Located in `YOLOv10m_training27/` folder

---

## 🔍 Key Findings

1. **Dataset Coverage**: Only 33% of images have predictions
   - Possible true negatives (no signs)
   - Or detection confidence threshold filtering

2. **Class Imbalance**: 
   - Top 3 classes account for 78% of all detections
   - Rare classes (22, 29) need more training data

3. **Detection Quality**:
   - Average 1.2 signs per image (reasonable for traffic scenarios)
   - 10 images with multiple signs show model handles complex scenes

4. **Image Quality**:
   - Standardized 640x640 resolution
   - Augmented dataset (many `_aug0`, `_aug1` suffixes)

---

## 💡 Recommendations

### For Better Predictions
1. **Lower Confidence Threshold**: May increase detection rate beyond 33%
2. **Class Balancing**: Add more samples for rare classes (22, 29, 35)
3. **Post-processing**: Review unlabeled images for false negatives

### For Production Use
1. Use ensemble with multiple checkpoints (training22-27)
2. Implement confidence scoring per class
3. Add temporal smoothing for video/sequential frames

---

## 📚 Related Files

- **Notebook:** `Traffic_Sign_Recognition_YOLOv10.ipynb` (training code)
- **Preprocessing:** `prétraitement_des_données_version1.py`
- **Dataset Info:** `README.roboflow.txt` (1,359 images, CC BY 4.0)
- **Model Checkpoints:** `YOLOv10m_training{22,23,25,27}.zip`

---

## 🔧 Usage

### Run Analysis Script
```bash
cd /home/mnxtr/Traffic-Sign-Recognition-YOLOv10/predict
python3 analyze_predictions.py
```

### Visualize Results
```bash
cd /home/mnxtr/Traffic-Sign-Recognition-YOLOv10/YOLOv10m_training27
# View confusion matrix, PR curves, validation examples
display confusion_matrix.png
display val_batch0_pred.jpg
```

---

**Note:** Exact class meanings require the original `data.yaml` or `classes.txt` file from the Roboflow dataset.
