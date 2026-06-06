import os
import gc
import numpy as np
from tqdm import tqdm

import config
from video_processor import get_video_frames
from mediapipe_extractor import extract_keypoints
from sequence_padder import pad_or_truncate_sequence
from augmentator import horizontal_flip_keypoints, temporal_random_crop, speed_interpolation

def main():
    print("Starting MediaPipe Preprocessing Pipeline...")
    
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
                
                # 2. Extract keypoints using MediaPipe
                sequence = extract_keypoints(frame_gen)
                
                # --- Temporal Augmentations (Before Padding) ---
                crop_seq = temporal_random_crop(sequence)
                crop_seq_standardized = pad_or_truncate_sequence(crop_seq)
                
                speed_seq = speed_interpolation(sequence)
                speed_seq_standardized = pad_or_truncate_sequence(speed_seq)
                
                # 3. Standardize temporal length (Pad/Truncate) for Original
                standardized_seq = pad_or_truncate_sequence(sequence)
                
                # 4. Create Augmented version (Horizontal Flip) from standardized original
                augmented_seq = horizontal_flip_keypoints(standardized_seq)
                
                # 5. Save directly to .npy files to keep memory usage minimal
                np.save(out_file_orig, standardized_seq)
                np.save(out_file_aug_flip, augmented_seq)
                np.save(out_file_aug_crop, crop_seq_standardized)
                np.save(out_file_aug_speed, speed_seq_standardized)
                
            except Exception as e:
                print(f"Error processing video {video_file}: {str(e)}")
            
            # Strict Memory Management
            gc.collect()

    print("\nPreprocessing Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
