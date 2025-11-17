# inference/infer_v3.py

import torch
from torch.utils.data import DataLoader
from preprocessing.dataset import FrameDataset
from preprocessing.frame_loader import list_frames
from models.inception_v3 import get_inception_v3
import config

def infer(checkpoint_path, frame_root, output_predictions_path):
    model = get_inception_v3(config.NUM_CLASSES).to(config.DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    items = list_frames(frame_root, config.CLASS_NAMES)
    ds = FrameDataset(items)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # Save predictions and labels
    import numpy as np
    np.savez(output_predictions_path, preds=all_preds, labels=all_labels)
    print(f"Saved predictions to {output_predictions_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--frames", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()
    infer(args.ckpt, args.frames, args.out)
