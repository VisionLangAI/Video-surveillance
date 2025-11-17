# preprocessing/dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import config

class FrameDataset(Dataset):
    def __init__(self, frame_label_list, transform=None):
        """
        :param frame_label_list: list of (frame_path, label)
        """
        self.items = frame_label_list
        self.transform = transform or transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label
