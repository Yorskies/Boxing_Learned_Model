import os
import cv2
import mediapipe as mp

# Konfigurasi MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Standar Topologi Skeleton MediaPipe (33 Titik)
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

# Daftar 4 Video Sampel (Satu per kelas)
videos = {
    'Hook': 'Nested Sequence 17.mp4',
    'Jab': 'Nested Sequence 139.mp4',
    'Straight': 'Nested Sequence 01.mp4',
    'Uppercut': 'Nested Sequence 148.mp4'
}

# Load model dari folder preprocessing
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preprocessing', 'pose_landmarker.task'))

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

print("Memulai ekstraksi dan rendering RAW keypoints secara akurat...")

timestamp_ms = 0
with PoseLandmarker.create_from_options(options) as landmarker:
    for cls_name, vid_name in videos.items():
        video_path = os.path.join("Dataset", cls_name, vid_name)
        output_dir = os.path.join("Dokumentasi_Skripsi", "Rendered_Raw", cls_name)
        
        # Buat folder khusus Raw Extraction
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error membuka video: {video_path}")
            continue
            
        print(f"Memproses {cls_name} - {vid_name}...")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            h, w, _ = frame.shape
            
            # Konversi BGR ke RGB karena MediaPipe membutuhkan format RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            # Lakukan deteksi (RAW extraction)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += 41 # Perkiraan 24 FPS
            
            parsed_landmarks = {}
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0] # Ambil orang pertama
                
                # Gambar titik sendi
                for idx, lm in enumerate(landmarks):
                    # Filter visibilitas jika sendi tertutup badan
                    vis = lm.visibility if lm.visibility is not None else 1.0
                    if vis > 0.3:
                        # Raw coordinates (0.0 - 1.0) dikali dimensi asli gambar
                        x_px = int(lm.x * w)
                        y_px = int(lm.y * h)
                        parsed_landmarks[idx] = (x_px, y_px)
                        
                        # Titik sendi digambar warna hijau neon
                        cv2.circle(frame, (x_px, y_px), 5, (0, 255, 0), -1)
                        
                # Gambar garis kerangka (skeleton)
                for connection in POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx in parsed_landmarks and end_idx in parsed_landmarks:
                        # Garis koneksi digambar warna kuning
                        cv2.line(frame, parsed_landmarks[start_idx], parsed_landmarks[end_idx], (0, 255, 255), 2)
            
            # Simpan frame yang sudah ditimpa RAW skeleton
            out_filename = os.path.join(output_dir, f"frame_raw_{frame_idx:04d}.jpg")
            cv2.imwrite(out_filename, frame)
            frame_idx += 1
            
        cap.release()
        
print("Selesai! Gambar yang sangat presisi (RAW) dapat Anda ambil dari folder 'Dokumentasi_Skripsi/Rendered_Raw'.")
