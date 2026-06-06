import numpy as np
import config

def horizontal_flip_keypoints(sequence):
    """
    Horizontally flips the keypoints in the sequence to simulate Southpaw vs Orthodox stance.
    - Flips X coordinates: x_new = 1.0 - x_old
    - Swaps left and right joint indices based on MediaPipe topology.
    """
    # Create a copy so we don't modify the original sequence
    flipped_seq = np.copy(sequence)
    
    num_frames = flipped_seq.shape[0]
    num_landmarks = config.FEATURE_DIM // 4 # Should be 33
    
    # 1. Flip X coordinates
    for i in range(num_landmarks):
        x_idx = i * 4
        # Since padded frames have 0.0 visibility, we only flip valid landmarks
        for t in range(num_frames):
            if flipped_seq[t, x_idx + 3] > 0.0: # visibility > 0 means it's a valid extracted point
                flipped_seq[t, x_idx] = 1.0 - flipped_seq[t, x_idx]
                
    # 2. Swap Left and Right joints
    for left_idx, right_idx in config.POSE_SWAP_PAIRS:
        l_start = left_idx * 4
        l_end = l_start + 4
        r_start = right_idx * 4
        r_end = r_start + 4
        
        # Swap the data for all frames at once using numpy array assignment
        temp = np.copy(flipped_seq[:, l_start:l_end])
        flipped_seq[:, l_start:l_end] = flipped_seq[:, r_start:r_end]
        flipped_seq[:, r_start:r_end] = temp
        
    return flipped_seq

def temporal_random_crop(sequence, crop_len=None):
    """
    Randomly crops a continuous chunk of length `crop_len` from the sequence.
    If sequence length is less than or equal to `crop_len`, returns the sequence as is.
    """
    if crop_len is None:
        crop_len = config.MAX_TIMESTEPS
        
    num_frames = sequence.shape[0]
    if num_frames > crop_len:
        start_idx = np.random.randint(0, num_frames - crop_len + 1)
        return sequence[start_idx : start_idx + crop_len, :]
    return sequence

def speed_interpolation(sequence, speed_factor=None):
    """
    Interpolates the temporal axis of the sequence to simulate speed up or slow down.
    A factor > 1.0 means slow down (more frames).
    A factor < 1.0 means speed up (fewer frames).
    """
    if speed_factor is None:
        # Random speed factor between 0.8 (20% faster) and 1.2 (20% slower)
        # Avoid exactly 1.0 to ensure variation
        factors = [np.random.uniform(0.75, 0.9), np.random.uniform(1.1, 1.25)]
        speed_factor = np.random.choice(factors)
        
    num_frames = sequence.shape[0]
    feature_dim = sequence.shape[1]
    
    new_num_frames = int(num_frames * speed_factor)
    
    if new_num_frames < 2 or num_frames < 2:
        return sequence
        
    old_indices = np.arange(num_frames)
    new_indices = np.linspace(0, num_frames - 1, new_num_frames)
    
    new_sequence = np.zeros((new_num_frames, feature_dim), dtype=np.float32)
    for i in range(feature_dim):
        new_sequence[:, i] = np.interp(new_indices, old_indices, sequence[:, i])
        
    return new_sequence
