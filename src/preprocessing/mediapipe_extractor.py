import mediapipe as mp
import numpy as np
import os
import config

# Define MediaPipe Tasks API components
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Path to the downloaded model
model_path = os.path.join(os.path.dirname(__file__), 'pose_landmarker.task')

def normalize_landmarks(keypoints):
    """
    Normalizes keypoints spatially by:
    1. Centering to Mid-hip: (Left Hip (23) + Right Hip (24)) / 2
    2. Scaling by shoulder distance: distance between Left Shoulder (11) and Right Shoulder (12)
    Only x, y, z are normalized. Visibility is kept as is.
    """
    if sum(keypoints) == 0.0:
        return keypoints
        
    L_HIP = 23 * 4
    R_HIP = 24 * 4
    L_SHOULDER = 11 * 4
    R_SHOULDER = 12 * 4
    
    mid_hip_x = (keypoints[L_HIP] + keypoints[R_HIP]) / 2.0
    mid_hip_y = (keypoints[L_HIP+1] + keypoints[R_HIP+1]) / 2.0
    mid_hip_z = (keypoints[L_HIP+2] + keypoints[R_HIP+2]) / 2.0
    
    shoulder_dist = np.sqrt(
        (keypoints[L_SHOULDER] - keypoints[R_SHOULDER])**2 + 
        (keypoints[L_SHOULDER+1] - keypoints[R_SHOULDER+1])**2 + 
        (keypoints[L_SHOULDER+2] - keypoints[R_SHOULDER+2])**2
    )
    
    if shoulder_dist == 0:
        shoulder_dist = 1.0
        
    normalized = []
    for i in range(33):
        idx = i * 4
        nx = (keypoints[idx] - mid_hip_x) / shoulder_dist
        ny = (keypoints[idx+1] - mid_hip_y) / shoulder_dist
        nz = (keypoints[idx+2] - mid_hip_z) / shoulder_dist
        normalized.extend([nx, ny, nz, keypoints[idx+3]])
        
    return normalized

def extract_keypoints(frame_generator):
    """
    Takes a generator of RGB frames and returns a NumPy array of extracted keypoints.
    Shape will be (num_frames, FEATURE_DIM).
    """
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    sequence_keypoints = []
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        # We need a monotonically increasing timestamp for video mode
        timestamp_ms = 0
        
        for frame in frame_generator:
            # Convert NumPy array to MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            
            # Perform detection
            pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            # Increment timestamp (assuming ~24 fps -> ~41ms per frame)
            timestamp_ms += 41
            
            frame_keypoints = []
            if pose_landmarker_result.pose_landmarks:
                # Get landmarks for the first detected person
                landmarks = pose_landmarker_result.pose_landmarks[0]
                for landmark in landmarks:
                    # visibility might be None in some models, default to 0.0 if so
                    vis = landmark.visibility if landmark.visibility is not None else 0.0
                    frame_keypoints.extend([landmark.x, landmark.y, landmark.z, vis])
                    
                # Apply Spatial Normalization
                frame_keypoints = normalize_landmarks(frame_keypoints)
            else:
                # If no person is detected in this frame, append zeros to maintain temporal shape
                frame_keypoints = [0.0] * config.FEATURE_DIM
                
            sequence_keypoints.append(frame_keypoints)
            
    return np.array(sequence_keypoints, dtype=np.float32)
