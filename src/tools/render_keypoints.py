import os
import cv2
import numpy as np
import argparse

# Standard MediaPipe Pose Connections
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20),
    (16, 22), (18, 20), (11, 23), (12, 24),
    (23, 24), (23, 25), (24, 26), (25, 27),
    (26, 28), (27, 29), (28, 30), (29, 31),
    (30, 32), (27, 31), (28, 32)
]

def render_keypoints(npy_path, frames_dir, output_dir):
    """
    Load .npy array and overlay the 33 keypoints onto corresponding frames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load sequence: shape (num_frames, 132)
    sequence = np.load(npy_path)
    
    # Get all frames sorted
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
    
    if len(frame_files) == 0:
        print(f"Error: Tidak ada gambar di {frames_dir}")
        return
        
    # We render up to the minimum of available frames or available keypoint timesteps
    num_frames_to_render = min(len(frame_files), sequence.shape[0])
    
    for i in range(num_frames_to_render):
        frame_path = os.path.join(frames_dir, frame_files[i])
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
            
        h, w, _ = frame.shape
        
        # Get keypoints for current frame: shape (132,)
        keypoints = sequence[i]
        
        # Parse keypoints into a list of (x_px, y_px)
        # Sequence is encoded as: [x0, y0, z0, v0, x1, y1, z1, v1, ...]
        parsed_landmarks = {}
        for lm_idx in range(33):
            base_idx = lm_idx * 4
            x, y, z, vis = keypoints[base_idx:base_idx+4]
            
            # Skip if visibility is exactly 0.0 (usually means padding or no detection)
            if vis > 0.0:
                x_px = int(x * w)
                y_px = int(y * h)
                parsed_landmarks[lm_idx] = (x_px, y_px)
                
                # Draw the joint as a green circle
                cv2.circle(frame, (x_px, y_px), 5, (0, 255, 0), -1)
                
        # Draw the skeleton connections as yellow lines
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx in parsed_landmarks and end_idx in parsed_landmarks:
                cv2.line(frame, parsed_landmarks[start_idx], parsed_landmarks[end_idx], (0, 255, 255), 2)
                
        # Save output
        out_path = os.path.join(output_dir, f"rendered_{i:04d}.jpg")
        cv2.imwrite(out_path, frame)
        
    print(f"Selesai merender {num_frames_to_render} frame ke {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render .npy keypoints onto frames.")
    parser.add_argument("--npy", type=str, required=True, help="Path ke file .npy")
    parser.add_argument("--frames_dir", type=str, required=True, help="Direktori frame input")
    parser.add_argument("--output_dir", type=str, required=True, help="Direktori output untuk frame yang sudah di-render")
    args = parser.parse_args()
    
    render_keypoints(args.npy, args.frames_dir, args.output_dir)
