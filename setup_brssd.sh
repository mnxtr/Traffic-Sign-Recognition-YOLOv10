#!/bin/bash
# BRSSD Setup Script for YOLOv10 Traffic Sign Recognition

echo "=========================================="
echo "BRSSD Dataset Setup for YOLOv10"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Create directory structure
echo -e "\n${YELLOW}[1/5] Creating directory structure...${NC}"
mkdir -p BRSSD/{train,val,test}/{images,labels}
echo -e "${GREEN}✓ Directories created${NC}"

# Install dependencies
echo -e "\n${YELLOW}[2/5] Installing Python dependencies...${NC}"
pip install -q ultralytics roboflow kaggle opencv-python pillow matplotlib seaborn pyyaml

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
fi

# Make download script executable
echo -e "\n${YELLOW}[3/5] Setting up download script...${NC}"
chmod +x download_brssd.py
echo -e "${GREEN}✓ Download script ready${NC}"

# Attempt to download dataset
echo -e "\n${YELLOW}[4/5] Attempting to download BRSSD dataset...${NC}"
python3 download_brssd.py --source all

# Check if dataset was downloaded
if [ -d "BRSSD/train" ] && [ "$(ls -A BRSSD/train/images 2>/dev/null)" ]; then
    echo -e "${GREEN}✓ BRSSD dataset downloaded successfully${NC}"
    
    # Count images
    TRAIN_IMGS=$(find BRSSD/train/images -type f | wc -l)
    VAL_IMGS=$(find BRSSD/val/images -type f | wc -l)
    TEST_IMGS=$(find BRSSD/test/images -type f 2>/dev/null | wc -l)
    
    echo -e "\n${GREEN}Dataset Statistics:${NC}"
    echo "  Training images: $TRAIN_IMGS"
    echo "  Validation images: $VAL_IMGS"
    echo "  Test images: $TEST_IMGS"
else
    echo -e "${YELLOW}⚠ Automatic download failed or incomplete${NC}"
    echo -e "${YELLOW}Please download BRSSD manually and place in ./BRSSD/${NC}"
fi

# Verify configuration
echo -e "\n${YELLOW}[5/5] Verifying configuration...${NC}"
if [ -f "brssd_data.yaml" ]; then
    echo -e "${GREEN}✓ Configuration file ready: brssd_data.yaml${NC}"
else
    echo -e "${RED}✗ Configuration file missing${NC}"
fi

# Summary
echo -e "\n=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "==========================================\n"

echo "Next steps:"
echo "  1. If dataset not downloaded, run: python3 download_brssd.py"
echo "  2. Or manually download BRSSD to ./BRSSD/"
echo "  3. Update brssd_data.yaml with correct class names"
echo "  4. Run training: python3 train_brssd.py"
echo "  5. Or use Jupyter notebook: jupyter notebook BRSSD_YOLOv10_Training.ipynb"

echo -e "\n=========================================="
