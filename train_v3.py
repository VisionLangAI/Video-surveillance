# training/train_v3.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing.dataset import FrameDataset
from preprocessing.frame_loader import list_frames
from models.inception_v3 import get_inception_v3
from training.utils import save_checkpoint
import config

def load_split(split_file, frame_root):
    with open(split_file, 'r') as f:
        lines = f.read().splitlines()
    items = []
    for line in lines:
        # assume each line has: video_id or path, class_name
        # you need to adapt depending on how you encode splits
        video_rel, cls = line.split()
        class_idx = config.CLASS_NAMES.index(cls)
        # gather frames in frame_root/<cls>/<video_rel> etc.
        # for simplicity, assume all frames of that class are under class folder
    # Let's simply call list_frames
    items = list_frames(frame_root, config.CLASS_NAMES)
    return items

def train():
    # prepare data
    train_items = load_split(config.TRAIN_SPLIT, config.ALLFRAME_DIR)
    val_items = load_split(config.VAL_SPLIT, config.ALLFRAME_DIR)

    train_ds = FrameDataset(train_items)
    val_ds = FrameDataset(val_items)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    # model
    model = get_inception_v3(config.NUM_CLASSES).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float('inf')
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(config.DEVICE)
                labels = labels.to(config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch}: Train loss {train_loss:.4f}, acc {train_acc:.4f} | Val loss {val_loss:.4f}, acc {val_acc:.4f}")

        # checkpoint
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"inception_v3_epoch{epoch}.pth")
        save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save best
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }, os.path.join(config.CHECKPOINT_DIR, "best_v3.pth"))

if __name__ == "__main__":
    train()
