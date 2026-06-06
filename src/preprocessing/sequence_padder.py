import numpy as np
import config

def pad_or_truncate_sequence(sequence):
    """
    Standardizes the temporal length of the keypoints sequence to exactly MAX_TIMESTEPS.
    If frames < MAX_TIMESTEPS, apply post-padding with zeros.
    If frames > MAX_TIMESTEPS, apply post-truncating.
    """
    max_len = config.MAX_TIMESTEPS
    feature_dim = config.FEATURE_DIM
    
    num_frames = sequence.shape[0]
    
    if num_frames == max_len:
        return sequence
    elif num_frames > max_len:
        # Post-truncate: keep first max_len frames
        return sequence[:max_len, :]
    else:
        # Post-pad: add zeros to the end
        pad_len = max_len - num_frames
        padding = np.zeros((pad_len, feature_dim), dtype=np.float32)
        padded_sequence = np.vstack((sequence, padding))
        return padded_sequence
