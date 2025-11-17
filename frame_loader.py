# preprocessing/frame_loader.py

import os
from PIL import Image

def list_frames(frame_root, class_names):
    """
    Return list of (frame_path, label_index) for all frames under frame_root,
    assuming class subfolders.
    """
    items = []
    for idx, cls in enumerate(class_names):
        class_dir = os.path.join(frame_root, cls)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".png")):
                items.append((os.path.join(class_dir, fname), idx))
    return items
