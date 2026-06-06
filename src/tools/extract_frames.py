import os
import cv2
import argparse

def extract_frames(video_path, output_dir):
    """
    Ekstrak video menjadi frame gambar (.jpg)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Tidak dapat membuka video {video_path}")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_idx += 1

    cap.release()
    print(f"Berhasil mengekstrak {frame_idx} frame ke {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ekstrak video menjadi frame.")
    parser.add_argument("--video", type=str, required=True, help="Path ke file video sumber")
    parser.add_argument("--output", type=str, required=True, help="Direktori output untuk menyimpan frame")
    args = parser.parse_args()
    
    extract_frames(args.video, args.output)
