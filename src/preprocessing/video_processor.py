import cv2

def get_video_frames(video_path):
    """
    Generator function to read a video frame by frame using OpenCV.
    Using a generator ensures strict memory management as frames are yielded
    one at a time instead of loading the entire video into memory.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Optionally convert BGR to RGB here, but usually mediapipe prefers RGB
        # We can do that conversion here to ensure consistency
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame_rgb
        
    cap.release()
