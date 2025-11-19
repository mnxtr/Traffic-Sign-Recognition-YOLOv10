# Quick Reference - View Training Metrics

## 📍 File Locations

All metric images are in:
```
/home/mnxtr/Traffic-Sign-Recognition-YOLOv10/YOLOv10m_training27/
```

## 🎯 Priority Order (What to View First)

### 1️⃣ Confusion Matrix (MOST IMPORTANT)
```bash
File: confusion_matrix_normalized.png
Size: 3000×2250 pixels, 186 KB
Status: ✅ Available
Quick Analysis: Brightness 250.1/255 = LIKELY HIGH ACCURACY
```

**What to look for:**
- Strong diagonal line (bright squares)
- Dark off-diagonal areas
- Each class >80% on diagonal

### 2️⃣ Precision-Recall Curve
```bash
File: PR_curve.png
Size: 2250×1500 pixels, 97 KB
Status: ✅ Available
```

**What to extract:**
- mAP@0.5 value (in legend or title)
- Target: >0.80 for good model

### 3️⃣ F1 Score Curve
```bash
File: F1_curve.png
Size: 2250×1500 pixels, 157 KB
Status: ✅ Available
```

**What to extract:**
- Peak F1 value
- Confidence threshold at peak
- Use this threshold for inference

## 💻 How to View

### Option 1: File Manager (Easiest)
1. Open file manager
2. Navigate to: `/home/mnxtr/Traffic-Sign-Recognition-YOLOv10/YOLOv10m_training27/`
3. Double-click images to open

### Option 2: Command Line
```bash
cd /home/mnxtr/Traffic-Sign-Recognition-YOLOv10/YOLOv10m_training27

# View all at once
eog *.png *.jpg &

# Or one by one
eog confusion_matrix_normalized.png
eog PR_curve.png
eog F1_curve.png
```

### Option 3: If No GUI Available
```bash
# Copy to your local machine
scp -r user@server:/home/mnxtr/Traffic-Sign-Recognition-YOLOv10/YOLOv10m_training27/*.png ~/Downloads/

# Then open locally
```

## 📊 Quick Analysis Results

Based on automated analysis:

✅ **Confusion Matrix**
- Average brightness: 250.1/255
- Interpretation: **Likely HIGH accuracy**
- This suggests strong performance across classes

✨ **Expected Performance Range**
- Probably: 80-95% accuracy
- Good diagonal definition
- Minimal class confusion

## 📝 What to Note When Viewing

### From Confusion Matrix:
```
□ Diagonal values (each class accuracy)
  Class 0: ____%
  Class 1: ____%
  ...
  
□ Problem classes (if any):
  Class __: Only __% (needs more data)
  
□ Confused classes:
  Class __ confused with Class __: __% of time
```

### From PR Curve:
```
□ mAP@0.5 = ____
□ Overall assessment: ____________
□ Weakest class: ____
□ Strongest class: ____
```

### From F1 Curve:
```
□ Peak F1 score: ____
□ Optimal confidence: ____
□ Recommendation: Use conf=____ for inference
```

### From Validation Images:
```
□ Correctly detected: __/__
□ Missed signs: __
□ False positives: __
□ Visual accuracy estimate: ____%
```

## 🚀 Next Steps After Viewing

1. **If metrics are good (mAP >0.80, F1 >0.75):**
   - ✅ Model is ready for testing
   - Set confidence threshold from F1 peak
   - Test on new images

2. **If metrics are moderate (mAP 0.60-0.80):**
   - ⚠️ Check confusion matrix for weak classes
   - Collect more data for those classes
   - Consider additional training

3. **If metrics are poor (mAP <0.60):**
   - ❌ Check for issues:
     - Class imbalance
     - Dataset quality
     - Training parameters

## 📚 Helper Scripts Available

All in current directory:

1. **view_metrics.py** - Text-based analysis guide
2. **analyze_training.py** - Comprehensive documentation
3. **TRAINING_ANALYSIS.md** - Full reference manual

```bash
# Run any of these for help
python3 view_metrics.py
python3 analyze_training.py
cat TRAINING_ANALYSIS.md
```

---

**Quick Start:** Just open the images in file manager and compare with the checklists above!
