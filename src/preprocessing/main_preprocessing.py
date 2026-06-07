import os
import gc
import numpy as np
from tqdm import tqdm

import config
from video_processor import get_video_frames
from mediapipe_extractor import extract_keypoints, compute_engineered_features
from sequence_padder import pad_or_truncate_sequence
from augmentator import horizontal_flip_keypoints, temporal_random_crop, speed_interpolation

def main():
    print("Starting MediaPipe Preprocessing Pipeline...")
    print(f"  RAW_FEATURE_DIM  = {config.RAW_FEATURE_DIM}")
    print(f"  ENGINEERED_FEAT  = {config.ENGINEERED_FEATURE_DIM}")
    print(f"  FEATURE_DIM      = {config.FEATURE_DIM}")
    
    # Iterate through classes
    for cls in config.CLASSES:
        input_cls_path = os.path.join(config.INPUT_DIR, cls)
        
        if not os.path.exists(input_cls_path):
            print(f"Warning: Input class folder {input_cls_path} not found. Skipping.")
            continue
            
        # Output paths
        output_cls_path = os.path.join(config.OUTPUT_DIR, "Original", cls)
        output_aug_path = os.path.join(config.OUTPUT_DIR, "Augmented", cls)
        
        os.makedirs(output_cls_path, exist_ok=True)
        os.makedirs(output_aug_path, exist_ok=True)
        
        # Get video files
        video_files = [f for f in os.listdir(input_cls_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
        print(f"\nProcessing class: {cls} ({len(video_files)} videos)")
        
        for video_file in tqdm(video_files, desc=f"Extracting {cls}"):
            video_path = os.path.join(input_cls_path, video_file)
            base_name = os.path.splitext(video_file)[0]
            
            out_file_orig = os.path.join(output_cls_path, f"{base_name}.npy")
            out_file_aug_flip = os.path.join(output_aug_path, f"{base_name}.npy")
            out_file_aug_crop = os.path.join(output_aug_path, f"{base_name}_crop.npy")
            out_file_aug_speed = os.path.join(output_aug_path, f"{base_name}_speed.npy")
            
            # Skip if all already exist to support resuming
            if all(os.path.exists(f) for f in [out_file_orig, out_file_aug_flip, out_file_aug_crop, out_file_aug_speed]):
                continue
                
            try:
                # 1. Load video frames using memory-efficient generator
                frame_gen = get_video_frames(video_path)
                
                # 2. Extract raw keypoints using MediaPipe (N, 132)
                raw_sequence = extract_keypoints(frame_gen)
                
                # ── Augmentations on RAW 132-dim coordinates ──────────────
                
                # 3a. Temporal augmentation: random crop (before padding)
                crop_raw = temporal_random_crop(raw_sequence)
                
                # 3b. Temporal augmentation: speed interpolation (before padding)
                speed_raw = speed_interpolation(raw_sequence)
                
                # 3c. Spatial augmentation: horizontal flip on raw coordinates
                flip_raw = horizontal_flip_keypoints(raw_sequence)
                
                # ── Compute engineered features (132 → 144) ──────────────
                # Features are computed AFTER augmentations so that angles,
                # velocities, and stance indicators are derived from the
                # correctly transformed coordinates.
                
                orig_feat  = compute_engineered_features(raw_sequence)   # (N, 144)
                flip_feat  = compute_engineered_features(flip_raw)       # (N, 144)
                crop_feat  = compute_engineered_features(crop_raw)       # (M, 144)
                speed_feat = compute_engineered_features(speed_raw)      # (K, 144)
                
                # ── Pad / truncate to standard temporal length ────────────
                standardized_orig  = pad_or_truncate_sequence(orig_feat)
                standardized_flip  = pad_or_truncate_sequence(flip_feat)
                standardized_crop  = pad_or_truncate_sequence(crop_feat)
                standardized_speed = pad_or_truncate_sequence(speed_feat)
                
                # 5. Save directly to .npy files
                np.save(out_file_orig,      standardized_orig)
                np.save(out_file_aug_flip,  standardized_flip)
                np.save(out_file_aug_crop,  standardized_crop)
                np.save(out_file_aug_speed, standardized_speed)
                
            except Exception as e:
                print(f"Error processing video {video_file}: {str(e)}")
            
            # Strict Memory Management
            gc.collect()

    print("\nPreprocessing Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
