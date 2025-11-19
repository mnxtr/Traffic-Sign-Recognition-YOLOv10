# BRSSD Integration for YOLOv10 Traffic Sign Recognition

## Overview
This directory contains scripts and configurations for training YOLOv10 models on the **BRSSD (Bangladeshi Road Sign Symbol Dataset)** for traffic sign detection and recognition.

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Run the automated setup script
bash setup_brssd.sh
```

### 2. Download BRSSD Dataset

**Option A: Automatic Download (Try all sources)**
```bash
python3 download_brssd.py --source all
```

**Option B: Download from Roboflow**
```bash
python3 download_brssd.py --source roboflow --roboflow-api-key YOUR_API_KEY
```

**Option C: Download from Kaggle**
```bash
python3 download_brssd.py --source kaggle
```

**Option D: Manual Download**
1. Visit one of these sources:
   - Roboflow Universe: https://universe.roboflow.com (search "BRSSD")
   - Kaggle: https://www.kaggle.com/datasets (search "Bangladesh traffic signs")
   - GitHub: Search for BRSSD repositories

2. Download in YOLOv8/YOLOv10 format
3. Extract to `./BRSSD/` directory with this structure:
   ```
   BRSSD/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── val/
   │   ├── images/
   │   └── labels/
   └── test/ (optional)
       ├── images/
       └── labels/
   ```

### 3. Train Model

**Option A: Python Script (Recommended)**
```bash
# Train with YOLOv10-nano (fastest)
python3 train_brssd.py --model n --epochs 100 --batch 16

# Train with YOLOv10-medium (balanced)
python3 train_brssd.py --model m --epochs 150 --batch 8

# Train with YOLOv10-large (most accurate)
python3 train_brssd.py --model l --epochs 200 --batch 4
```

**Option B: Jupyter Notebook**
```bash
jupyter notebook BRSSD_YOLOv10_Training.ipynb
```

**Option C: Google Colab**
1. Upload `BRSSD_YOLOv10_Training.ipynb` to Google Colab
2. Connect to GPU runtime
3. Run all cells

## 📁 Files and Scripts

| File | Description |
|------|-------------|
| `setup_brssd.sh` | Automated setup script - creates directories, installs dependencies |
| `download_brssd.py` | Multi-source dataset downloader (Roboflow, Kaggle, GitHub) |
| `brssd_data.yaml` | Dataset configuration for YOLOv10 training |
| `train_brssd.py` | Training script with optimal hyperparameters |
| `BRSSD_YOLOv10_Training.ipynb` | Complete Jupyter notebook for training and evaluation |
| `BRSSD/` | Dataset directory (created after download) |

## 🎯 Training Options

### Model Sizes
- **nano (n)**: Fastest, smallest - good for real-time applications
- **small (s)**: Fast with better accuracy
- **medium (m)**: Balanced speed and accuracy
- **large (l)**: High accuracy, slower
- **xlarge (x)**: Highest accuracy, slowest

### Training Parameters
```bash
python3 train_brssd.py \
  --model m \           # Model size
  --epochs 100 \        # Training epochs
  --batch 16 \          # Batch size
  --imgsz 640 \         # Image size
  --data brssd_data.yaml  # Dataset config
```

## 📊 Expected Results

After training completes, you'll find:
- **Best model**: `runs/brssd/YOLOv10{size}_BRSSD/weights/best.pt`
- **Last model**: `runs/brssd/YOLOv10{size}_BRSSD/weights/last.pt`
- **Training plots**: Confusion matrix, F1 curve, PR curve, etc.
- **Validation metrics**: mAP, precision, recall

## 🔧 Configuration

### Update Class Names
Edit `brssd_data.yaml` to match your BRSSD version:

```yaml
nc: 6  # Number of classes

names:
  0: Regulatory
  1: Warning
  2: Mandatory
  3: Prohibitory
  4: Informatory
  5: Guide
```

### Adjust Hyperparameters
Edit training parameters in `train_brssd.py` or pass via command line.

## 🧪 Testing Predictions

After training, test your model:

```python
from ultralytics import YOLO

# Load best model
model = YOLO('runs/brssd/YOLOv10n_BRSSD/weights/best.pt')

# Predict on image
results = model.predict('path/to/image.jpg', conf=0.25)

# Display results
results[0].show()
```

## 📈 Performance Tips

1. **GPU Usage**: Use `device=0` for GPU training (much faster)
2. **Batch Size**: Adjust based on GPU memory (16 for good GPUs, 4-8 for smaller)
3. **Image Size**: 640 is standard, increase to 1024 for better accuracy (slower)
4. **Epochs**: Start with 100, increase if model is still improving
5. **Data Augmentation**: Already configured in the scripts

## 🐛 Troubleshooting

**Dataset not found:**
- Run `download_brssd.py` or manually download BRSSD
- Verify directory structure matches expected format

**Out of memory:**
- Reduce batch size: `--batch 4`
- Use smaller model: `--model n`
- Reduce image size: `--imgsz 416`

**Low accuracy:**
- Increase epochs: `--epochs 200`
- Use larger model: `--model m` or `--model l`
- Verify dataset quality and annotations
- Check class balance

## 📝 Citation

If you use BRSSD dataset, please cite the original paper:
```
@article{brssd,
  title={BRSSD: Bangladeshi Road Sign Symbol Dataset},
  author={Authors},
  journal={Journal},
  year={Year}
}
```

## 🤝 Contributing

Found issues or improvements? Please contribute:
1. Update scripts as needed
2. Test thoroughly
3. Document changes

## 📞 Support

For issues:
1. Check the troubleshooting section
2. Review YOLOv10 documentation: https://docs.ultralytics.com
3. Check BRSSD dataset documentation

## ✅ Checklist

- [ ] Environment setup complete (`bash setup_brssd.sh`)
- [ ] BRSSD dataset downloaded
- [ ] Dataset structure verified
- [ ] Class names updated in `brssd_data.yaml`
- [ ] Training completed successfully
- [ ] Model validated on test set
- [ ] Predictions tested

---

**Happy Training! 🚦🇧🇩**
