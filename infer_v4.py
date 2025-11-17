# inference/infer_v4.py

import torch
from torch.utils.data import DataLoader
from preprocessing.dataset import FrameDataset
from preprocessing.frame_loader import list_frames
from models.inception_v4 import get_inception_v4
import config
import numpy as np
from evaluation.evaluate import evaluate

def run_inference(frame_root, checkpoint_path, batch_size=32):
    device = config.DEVICE
    model = get_inception_v4(num_classes=config.NUM_CLASSES).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = FrameDataset(list_frames(frame_root, config.CLASS_NAMES))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    evaluate(all_preds, all_labels)

    np.savez("predictions_v4.npz", preds=all_preds, labels=all_labels)
    print("Predictions saved to predictions_v4.npz")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_root", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    args = parser.parse_args()

    run_inference(args.frame_root, args.checkpoint, args.batch_size)
