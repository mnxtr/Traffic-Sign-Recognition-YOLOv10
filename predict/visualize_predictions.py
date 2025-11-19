#!/usr/bin/env python3
"""
Visualize YOLOv10 Traffic Sign Predictions
Draws bounding boxes on images with class labels
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Class mapping (estimated from analysis)
CLASS_NAMES = {
    22: "Speed Limit",
    23: "Warning",
    26: "Priority/Yield",
    28: "Information",
    29: "Direction",
    32: "Stop/Parking",
    35: "Pedestrian",
    36: "Caution"
}

# Colors for different classes (RGB format for PIL)
CLASS_COLORS = {
    22: (255, 0, 0),      # Red
    23: (255, 165, 0),    # Orange
    26: (255, 255, 0),    # Yellow
    28: (0, 0, 255),      # Blue
    29: (255, 0, 255),    # Magenta
    32: (128, 0, 0),      # Dark Red
    35: (0, 255, 255),    # Cyan
    36: (0, 255, 0)       # Green
}

def draw_boxes_on_image(image_path, label_path, output_path=None):
    """Draw bounding boxes on a single image"""
    
    # Read image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error: Could not read {image_path}: {e}")
        return None
    
    w, h = img.size
    draw = ImageDraw.Draw(img)
    
    # Try to load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Read labels if they exist
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert YOLO format to pixel coordinates
                    x1 = int((x_center - width/2) * w)
                    y1 = int((y_center - height/2) * h)
                    x2 = int((x_center + width/2) * w)
                    y2 = int((y_center + height/2) * h)
                    
                    # Get class name and color
                    class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
                    color = CLASS_COLORS.get(class_id, (255, 255, 255))
                    
                    # Draw bounding box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Draw label background and text
                    label_text = f"{class_name}"
                    bbox = draw.textbbox((x1, y1), label_text, font=font)
                    draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=color)
                    draw.text((x1, y1-20), label_text, fill=(255, 255, 255), font=font)
    
    # Save or return image
    if output_path:
        img.save(output_path)
        print(f"Saved: {output_path}")
    
    return img

def visualize_grid(num_images=9, save_output=True):
    """Create a grid visualization of predictions"""
    
    labels_dir = "labels"
    images_dir = "."
    
    # Get images with labels
    labeled_images = []
    for label_file in sorted(os.listdir(labels_dir)):
        if label_file.endswith('.txt'):
            image_name = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(images_dir, image_name)
            if os.path.exists(image_path):
                labeled_images.append((image_path, os.path.join(labels_dir, label_file)))
    
    # Create grid
    rows = int(num_images ** 0.5)
    cols = (num_images + rows - 1) // rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    for idx, ax in enumerate(axes):
        if idx < len(labeled_images) and idx < num_images:
            img_path, label_path = labeled_images[idx]
            
            # Draw boxes
            img = draw_boxes_on_image(img_path, label_path)
            
            if img is not None:
                ax.imshow(img)
                ax.set_title(os.path.basename(img_path), fontsize=8)
                ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_output:
        output_file = "predictions_grid.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Saved grid visualization: {output_file}")
    
    plt.close()

def visualize_by_class(class_id, max_images=6):
    """Visualize all predictions for a specific class"""
    
    labels_dir = "labels"
    images_dir = "."
    
    # Find images containing this class
    class_images = []
    for label_file in sorted(os.listdir(labels_dir)):
        if label_file.endswith('.txt'):
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) == class_id:
                        image_name = label_file.replace('.txt', '.jpg')
                        image_path = os.path.join(images_dir, image_name)
                        if os.path.exists(image_path):
                            class_images.append((image_path, label_path))
                        break
    
    if not class_images:
        print(f"No images found for class {class_id}")
        return
    
    # Create grid
    num_images = min(len(class_images), max_images)
    rows = 2
    cols = (num_images + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten() if num_images > 1 else [axes]
    
    class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
    fig.suptitle(f"Traffic Sign Class: {class_name} (ID: {class_id})", fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes):
        if idx < len(class_images):
            img_path, label_path = class_images[idx]
            img = draw_boxes_on_image(img_path, label_path)
            
            if img is not None:
                ax.imshow(img)
                ax.set_title(os.path.basename(img_path), fontsize=8)
                ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    output_file = f"class_{class_id}_{class_name.replace('/', '_')}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ Saved class visualization: {output_file}")
    plt.close()

def create_individual_visualizations(output_dir="visualized"):
    """Create individual visualizations for all labeled images"""
    
    os.makedirs(output_dir, exist_ok=True)
    labels_dir = "labels"
    images_dir = "."
    
    count = 0
    for label_file in sorted(os.listdir(labels_dir)):
        if label_file.endswith('.txt'):
            image_name = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(images_dir, image_name)
            label_path = os.path.join(labels_dir, label_file)
            output_path = os.path.join(output_dir, image_name)
            
            if os.path.exists(image_path):
                draw_boxes_on_image(image_path, label_path, output_path)
                count += 1
    
    print(f"\n✅ Created {count} individual visualizations in '{output_dir}/' folder")

def main():
    """Main visualization function"""
    
    print("=" * 60)
    print("TRAFFIC SIGN PREDICTION VISUALIZER")
    print("=" * 60)
    
    print("\n1️⃣  Creating grid visualization (9 images)...")
    visualize_grid(num_images=9)
    
    print("\n2️⃣  Creating visualizations by class...")
    # Visualize top 3 classes
    for class_id in [36, 26, 23]:
        class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
        print(f"  - Class {class_id} ({class_name})")
        visualize_by_class(class_id, max_images=6)
    
    print("\n3️⃣  Creating individual visualizations...")
    create_individual_visualizations()
    
    print("\n" + "=" * 60)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  • predictions_grid.png - Grid of 9 labeled images")
    print("  • class_36_Caution.png - Top class examples")
    print("  • class_26_Priority_Yield.png - Second class examples")
    print("  • class_23_Warning.png - Third class examples")
    print("  • visualized/ - Individual annotated images (56 files)")
    print()

if __name__ == "__main__":
    main()
