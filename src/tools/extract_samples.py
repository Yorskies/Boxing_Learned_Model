import os
import subprocess

# Daftar 1 video sampel untuk tiap kelas
videos = {
    'Hook': 'Nested Sequence 17.mp4',
    'Jab': 'Nested Sequence 139.mp4',
    'Straight': 'Nested Sequence 01.mp4',
    'Uppercut': 'Nested Sequence 148.mp4'
}

print("Memulai ekstraksi dan rendering untuk dokumentasi skripsi...")

for cls_name, vid_name in videos.items():
    video_path = os.path.join("Dataset", cls_name, vid_name)
    frames_dir = os.path.join("Dokumentasi_Skripsi", "Frames", cls_name)
    
    # Nama file npy hasil preprocessing
    npy_name = vid_name.replace('.mp4', '.npy')
    npy_path = os.path.join("Dataset_Numpy_Normalized", "Original", cls_name, npy_name)
    
    render_dir = os.path.join("Dokumentasi_Skripsi", "Rendered", cls_name)
    
    print(f"\nMemproses kelas: {cls_name}")
    print(f"Video: {vid_name}")
    
    # 1. Ekstrak Frames
    subprocess.run([
        ".\\venv\\Scripts\\python", 
        "src\\tools\\extract_frames.py", 
        "--video", video_path, 
        "--output", frames_dir
    ])
    
    # 2. Render Keypoints (menggunakan file npy yang dinormalisasi)
    if os.path.exists(npy_path):
        subprocess.run([
            ".\\venv\\Scripts\\python", 
            "src\\tools\\render_keypoints.py", 
            "--npy", npy_path, 
            "--frames_dir", frames_dir, 
            "--output_dir", render_dir
        ])
    else:
        print(f"Peringatan: File {npy_path} tidak ditemukan!")

print("\nSeluruh proses dokumentasi selesai. Hasil dapat dilihat di folder 'Dokumentasi_Skripsi'.")
