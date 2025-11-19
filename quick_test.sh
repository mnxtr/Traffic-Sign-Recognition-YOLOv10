#!/bin/bash
# Quick Test Script for BRSSD YOLOv10 Training
# This runs a 1-epoch test to verify everything works

echo "========================================"
echo "BRSSD YOLOv10 Quick Test"
echo "========================================"
echo ""

# Check if ultralytics is installed
echo "1. Checking dependencies..."
python3 -c "import ultralytics; print('✓ Ultralytics installed')" || {
    echo "✗ Ultralytics not found. Installing..."
    pip3 install ultralytics
}

# Check GPU
echo ""
echo "2. Checking GPU availability..."
python3 -c "import torch; gpu = torch.cuda.is_available(); print(f'✓ GPU Available: {gpu}'); print(f'Device: {torch.cuda.get_device_name(0) if gpu else \"CPU\"}')"

# Check dataset
echo ""
echo "3. Checking dataset..."
if [ -d "BRSSD" ]; then
    echo "✓ BRSSD dataset found"
    echo "  Training images: $(find BRSSD/train/images -type f | wc -l)"
    echo "  Validation images: $(find BRSSD/valid/images -type f | wc -l)"
else
    echo "✗ BRSSD dataset not found!"
    echo "Please run: python3 download_brssd.py --source all"
    exit 1
fi

# Run quick test
echo ""
echo "4. Running 1-epoch test..."
echo "   (This will take a few minutes)"
echo ""
python3 train_brssd_improved.py --epochs 1 --batch 4 --imgsz 640

echo ""
echo "========================================"
echo "Test completed!"
echo "========================================"
echo ""
echo "If successful, start full training with:"
echo "  python3 train_brssd_improved.py --model s --epochs 50 --batch 16"
echo ""
