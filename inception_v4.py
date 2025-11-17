# models/inception_v4.py

import torch
import torch.nn as nn
import pretrainedmodels  # third-party pretrained models
from .base_model import BaseClassifier

class InceptionV4Classifier(BaseClassifier):
    """
    Inception V4 classifier using pretrained models.pytorch.
    """
    def __init__(self, num_classes, pretrained='imagenet'):
        """
        :param num_classes: number of output classes
        :param pretrained: 'imagenet' or None
        """
        backbone = pretrainedmodels.__dict__['inceptionv4'](pretrained=pretrained)
        # backbone.last_linear is the original classifier
        super(InceptionV4Classifier, self).__init__(backbone, num_classes)
        self.input_size = (299, 299)

def get_inception_v4(num_classes=4, pretrained='imagenet'):
    model = InceptionV4Classifier(num_classes=num_classes, pretrained=pretrained)
    return model
