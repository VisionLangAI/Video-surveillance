# training/train_v4.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing.dataset import FrameDataset
from preprocessing.frame_loader import list_frames
from models.inception_v4 import get_inception_v4
from evaluation.evaluate import evaluate
import config
import numpy as np

def prepare_dataset(frame_root):
    items = list_frames(frame_root, config.CLASS_NAMES)
    return FrameDataset(items)

def train_model(model, train_loader, val_loader, device, num_epochs, lr, checkpoint_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        train_acc = train_correct / train_total
        train_loss /= train_total
        
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        
        val_acc = val_correct / val_total
        val_loss /= val_total
        
        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch+1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_acc': val_acc
            }, checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1}")

    print("Training completed.")
    print("Evaluation on validation set:")
    evaluate(np.array(all_preds), np.array(all_labels))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_root", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--checkpoint", type=str, default=os.path.join(config.CHECKPOINT_DIR, "best_v4.pth"))
    args = parser.parse_args()

    device = config.DEVICE
    model = get_inception_v4(num_classes=config.NUM_CLASSES).to(device)

    dataset = prepare_dataset(args.frame_root)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_siz]()_
