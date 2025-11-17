# models/base_model.py

import torch
import torch.nn as nn

class ClassifierBase(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        # assuming backbone outputs feature vector, then add a classification head
        self.fc = nn.Linear(backbone.fc.in_features, num_classes)
        # remove original fc
        backbone.fc = nn.Identity()

    def forward(self, x):
        feats = self.backbone(x)
        out = self.fc(feats)
        return out
