# config.py

import os

# Paths
RAW_VIDEO_DIR = "data/raw"
FRAMES_DIR = "data/frames"
KEYFRAME_DIR = os.path.join(FRAMES_DIR, "keyframes")
ALLFRAME_DIR = os.path.join(FRAMES_DIR, "all_frames")

SPLITS_DIR = "data/splits"
TRAIN_SPLIT = os.path.join(SPLITS_DIR, "train.txt")
VAL_SPLIT = os.path.join(SPLITS_DIR, "val.txt")
TEST_SPLIT = os.path.join(SPLITS_DIR, "test.txt")

# Classes to use
CLASS_NAMES = ["Arrest", "Assault", "Abuse", "Arson"]
NUM_CLASSES = len(CLASS_NAMES)

# Training hyperparams
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

# Image size for Inception
IMAGE_SIZE = (299, 299)  # Inception V3/V4 typical input size

# Device
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# Checkpointing
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
