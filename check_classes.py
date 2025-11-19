import os
from pathlib import Path

def get_max_class_id(label_dir):
    max_id = -1
    label_files = list(Path(label_dir).glob('*.txt'))
    print(f"Found {len(label_files)} label files in {label_dir}")
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                try:
                    class_id = int(line.split()[0])
                    if class_id > max_id:
                        max_id = class_id
                except (ValueError, IndexError):
                    pass
    return max_id

train_labels = "./BRSSD/train/labels"
if os.path.exists(train_labels):
    max_id = get_max_class_id(train_labels)
    print(f"Max class ID found: {max_id}")
    print(f"Implied number of classes: {max_id + 1}")
else:
    print(f"Directory not found: {train_labels}")
