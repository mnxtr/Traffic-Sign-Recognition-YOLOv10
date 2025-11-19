# BRSSD YOLOv10 Notebook Fixes Guide

## Issues Fixed

This guide addresses the following issues in your Jupyter notebooks:

### 1. **Pip Install Warning** (`!pip` vs `%pip`)

**Issue:** Using `!pip install` in Jupyter notebooks can install packages in the wrong Python environment.

**Fix:** Replace all instances of `!pip install` with `%pip install`

**Files Affected:**
- `BRSSD_YOLOv10_Training_GPU.ipynb` (line 51)
- `BRSSD_YOLOv10_Training.ipynb` (line 27)
- `Traffic_Sign_Recognition_YOLOv10.ipynb` (line 2155)

**How to Fix:**

1. Open each notebook in Jupyter
2. Find the cell with `!pip install`
3. Replace with:
   ```python
   %pip install ultralytics roboflow -q
   ```

**Why this matters:**
- `!pip` runs in a shell subprocess and may install to a different Python environment
- `%pip` ensures packages are installed in the same environment as the notebook kernel
- This prevents "ModuleNotFoundError" even after installation

---

### 2. **Mixed Detection/Segmentation Dataset Warning**

**Issue:** Warning message during training:
```
WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 2, 
len(boxes) = 6319. To resolve this only boxes will be used and all segments 
will be removed.
```

**What it means:**
- Your dataset has a few segmentation annotations mixed with detection boxes
- YOLOv10 is configured for detection, not segmentation
- The model will automatically ignore the 2 segmentation annotations

**Impact:**
- ⚠️ **LOW SEVERITY** - Training will continue normally
- Only 2 images out of 6189 have segmentation data
- These will be treated as detection-only images

**Fix (Optional):**
To clean the dataset:
```bash
# This removes the segmentation annotations, keeping only bounding boxes
python3 -c "
from pathlib import Path
import re

label_dir = Path('BRSSD/train/labels')
for label_file in label_dir.glob('*.txt'):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    cleaned_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:  # Only keep standard YOLO format (class x y w h)
            cleaned_lines.append(line)
    
    with open(label_file, 'w') as f:
        f.writelines(cleaned_lines)

print('✓ Dataset cleaned')
"
```

**Recommendation:** 
- No action needed - the warning is informational
- Training will work correctly as-is

---

### 3. **GPU Auto-Detection**

**Issue:** The original `train_brssd.py` hardcoded `device='cpu'`

**Fix:** Updated to `device='auto'` in the improved script

**Files Fixed:**
- ✓ `train_brssd.py` - Updated
- ✓ `train_brssd_improved.py` - New improved version

**What changed:**
```python
# Before:
'device': 'cpu',  # Use CPU since no GPU is available

# After:
'device': 'auto',  # Auto-detect GPU, use CPU if not available
```

---

## Recommended Usage

### For Local Training:

Use the improved script:
```bash
# Quick test (1 epoch)
python3 train_brssd_improved.py --epochs 1 --batch 4

# Full training with default settings
python3 train_brssd_improved.py

# Custom configuration
python3 train_brssd_improved.py --model s --epochs 50 --batch 32
```

### For Google Colab:

1. Upload `BRSSD_YOLOv10_Training_GPU.ipynb` to Colab
2. **Important:** In the first code cell, change:
   ```python
   !pip install ultralytics roboflow -q
   ```
   to:
   ```python
   %pip install ultralytics roboflow -q
   ```
3. Enable GPU: Runtime → Change runtime type → GPU (T4)
4. Run all cells

---

## Files Summary

| File | Status | Purpose |
|------|--------|---------|
| `train_brssd.py` | ✓ Fixed | Original script (device updated to 'auto') |
| `train_brssd_improved.py` | ✓ New | Enhanced version with better error handling |
| `BRSSD_YOLOv10_Training_GPU.ipynb` | ⚠️ Needs manual fix | Google Colab notebook (change !pip to %pip) |
| `BRSSD_YOLOv10_Training.ipynb` | ⚠️ Needs manual fix | Alternative notebook (change !pip to %pip) |
| `brssd_data.yaml` | ✓ OK | Dataset configuration |

---

## Testing the Fixes

Run a quick test to verify everything works:

```bash
# Test with 1 epoch (will complete in a few minutes)
python3 train_brssd_improved.py --epochs 1 --batch 4 --imgsz 640

# Check if GPU is detected
python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

Expected output:
```
============================================================
System Configuration
============================================================
✓ GPU Available: NVIDIA GeForce RTX 3060
  GPU Memory: 12.00 GB
  CUDA Version: 11.8
============================================================
```

Or if no GPU:
```
============================================================
System Configuration
============================================================
⚠️  No GPU detected - training will use CPU
  Note: CPU training will be significantly slower
============================================================
```

---

## Next Steps

1. **Test the improved script:**
   ```bash
   python3 train_brssd_improved.py --epochs 1 --batch 4
   ```

2. **Fix the notebooks** (if you plan to use Google Colab):
   - Open in Jupyter/Colab
   - Replace `!pip install` with `%pip install`
   - Save

3. **Start full training:**
   ```bash
   # Recommended for first run: small model, fewer epochs
   python3 train_brssd_improved.py --model s --epochs 50 --batch 16
   
   # For best results (will take longer):
   python3 train_brssd_improved.py --model m --epochs 100 --batch 16
   ```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'ultralytics'"
**Solution:**
```bash
pip3 install ultralytics
```

### Issue: "FileNotFoundError: Configuration file not found"
**Solution:**
```bash
# Make sure you're in the project directory
cd /home/mnxtr/Traffic-Sign-Recognition-YOLOv10

# Check if brssd_data.yaml exists
ls -la brssd_data.yaml

# If BRSSD folder is missing, download the dataset
python3 download_brssd.py --source all
```

### Issue: Training is very slow
**Possible causes:**
- Running on CPU instead of GPU
- Batch size too large for available memory
- Too many workers

**Solutions:**
```bash
# Check GPU
python3 -c "import torch; print(torch.cuda.is_available())"

# Reduce batch size
python3 train_brssd_improved.py --batch 8

# Use smaller model
python3 train_brssd_improved.py --model n
```

---

## Performance Expectations

| Model | Parameters | Speed (GPU) | Speed (CPU) | mAP50 (expected) |
|-------|------------|-------------|-------------|------------------|
| YOLOv10n | 2.7M | ~20 min/epoch | ~3 hrs/epoch | ~0.70-0.75 |
| YOLOv10s | 7.2M | ~30 min/epoch | ~5 hrs/epoch | ~0.75-0.80 |
| YOLOv10m | 15.4M | ~45 min/epoch | ~8 hrs/epoch | ~0.80-0.85 |
| YOLOv10l | 24.4M | ~60 min/epoch | ~12 hrs/epoch | ~0.85-0.88 |

*Times are approximate and depend on your hardware*

---

## Questions?

If you encounter any other issues, check:
1. The error message in the terminal
2. The log files in `runs/brssd/YOLOv10n_BRSSD/`
3. GPU availability: `nvidia-smi` (if you have NVIDIA GPU)

Good luck with your training! 🚦
