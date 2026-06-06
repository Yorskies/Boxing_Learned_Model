import os

# Base paths
INPUT_DIR = r"f:\Skripsi\Wahyu\Dataset"
OUTPUT_DIR = r"f:\Skripsi\Wahyu\Dataset_Numpy_Normalized"

# Dataset parameters
CLASSES = ['Hook', 'Jab', 'Straight', 'Uppercut']

# Model hyperparameters for padding/truncating
MAX_TIMESTEPS = 30
# 33 landmarks * 4 variables (x, y, z, visibility) = 132
FEATURE_DIM = 132

# Keypoints swap mapping for horizontal flip (Southpaw vs Orthodox)
# Based on MediaPipe Pose topology
# Even indices generally correspond to right side, odd to left side in MP Pose (e.g. 11 left shoulder, 12 right shoulder)
# (Left, Right) pairs
POSE_SWAP_PAIRS = [
    (1, 2), (2, 3), (3, 4), (7, 8), # Face
    (9, 10), # Mouth
    (11, 12), # Shoulders
    (13, 14), # Elbows
    (15, 16), # Wrists
    (17, 18), # Pinky
    (19, 20), # Index
    (21, 22), # Thumb
    (23, 24), # Hip
    (25, 26), # Knee
    (27, 28), # Ankle
    (29, 30), # Heel
    (31, 32)  # Foot Index
]
