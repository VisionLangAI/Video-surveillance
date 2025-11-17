# models/inception_v3.py

import torch
import torchvision.models as models
from .base_model import BaseClassifier

class InceptionV3Classifier(BaseClassifier):
    """
    Inception V3 Classifier with optional pretrained weights.
    """
    def __init__(self, num_classes, pretrained=True, aux_logits=False):
        # Load torchvision Inception V3
        backbone = models.inception_v3(pretrained=pretrained, aux_logits=aux_logits)
        # Disable auxiliary classifier during training if not needed
        if not aux_logits:
            backbone.aux_logits = False
        
        super(InceptionV3Classifier, self).__init__(backbone, num_classes)
        self.input_size = (299, 299)  # standard InceptionV3 input

    def forward(self, x):
        # Inception v3 requires training mode to handle aux_logits
        return super().forward(x)

def get_inception_v3(num_classes=4, pretrained=True):
    """
    Returns InceptionV3Classifier instance.
    """
    model = InceptionV3Classifier(num_classes=num_classes, pretrained=pretrained)
    return model
