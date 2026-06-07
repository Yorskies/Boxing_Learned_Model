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

# ---------------------------------------------------------------------------
# Engineered feature computation (12 additional features per frame)
# ---------------------------------------------------------------------------
# Landmark IDs used for feature engineering
_L_SHOULDER = 11
_R_SHOULDER = 12
_L_ELBOW    = 13
_R_ELBOW    = 14
_L_WRIST    = 15
_R_WRIST    = 16
_L_ANKLE    = 27
_R_ANKLE    = 28


def _get_xyz(frame, landmark_id):
    """Extract (x, y, z) for a given landmark from a 132-dim frame vector."""
    idx = landmark_id * 4
    return frame[idx], frame[idx + 1], frame[idx + 2]


def _euclidean_3d(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


def _angle_at_b(a, b, c):
    """Angle (radians) at vertex *b* formed by points a-b-c."""
    ba = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]])
    bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]])
    n1 = np.linalg.norm(ba)
    n2 = np.linalg.norm(bc)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_a = np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_a))


def compute_engineered_features(sequence: np.ndarray) -> np.ndarray:
    """Compute 12 engineered features and append them to each frame.

    Input shape  : (N, 132)   – raw / normalised keypoints
    Output shape : (N, 144)   – original 132 + 12 new features

    New features (per frame, columns 132 → 143):
        0  elbow_angle_left        – sudut siku kiri  (0‥π rad)
        1  elbow_angle_right       – sudut siku kanan (0‥π rad)
        2  hand_extension_diff     – d_right − d_left (positif → tangan kanan aktif)
        3  foot_forward_diff       – selisih posisi kaki depan (XZ plane)
        4  stance_relation         – sign(hand) × sign(foot) ; +1 same-side = Jab
        5  shoulder_rotation       – z_L_shoulder − z_R_shoulder
        6  wrist_vel_x_ratio_L     – rasio kecepatan horizontal tangan kiri  (Hook)
        7  wrist_vel_y_ratio_L     – rasio kecepatan vertikal↑ tangan kiri   (Uppercut)
        8  wrist_vel_z_ratio_L     – rasio kecepatan kedalaman tangan kiri   (Jab/Straight)
        9  wrist_vel_x_ratio_R     – rasio kecepatan horizontal tangan kanan (Hook)
       10  wrist_vel_y_ratio_R     – rasio kecepatan vertikal↑ tangan kanan  (Uppercut)
       11  wrist_vel_z_ratio_R     – rasio kecepatan kedalaman tangan kanan  (Jab/Straight)
    """
    N = sequence.shape[0]
    eng = np.zeros((N, config.ENGINEERED_FEATURE_DIM), dtype=np.float32)
    eps = 1e-8

    for t in range(N):
        frame = sequence[t]

        # Skip zero/padding frames
        if np.abs(frame).sum() < eps:
            continue

        # --- joint positions -------------------------------------------------
        l_sh = _get_xyz(frame, _L_SHOULDER)
        r_sh = _get_xyz(frame, _R_SHOULDER)
        l_el = _get_xyz(frame, _L_ELBOW)
        r_el = _get_xyz(frame, _R_ELBOW)
        l_wr = _get_xyz(frame, _L_WRIST)
        r_wr = _get_xyz(frame, _R_WRIST)
        l_an = _get_xyz(frame, _L_ANKLE)
        r_an = _get_xyz(frame, _R_ANKLE)

        # [0] elbow angle left  (shoulder→elbow→wrist)
        eng[t, 0] = _angle_at_b(l_sh, l_el, l_wr)

        # [1] elbow angle right
        eng[t, 1] = _angle_at_b(r_sh, r_el, r_wr)

        # [2] hand extension diff  (d_right − d_left)
        d_left  = _euclidean_3d(l_wr, l_sh)
        d_right = _euclidean_3d(r_wr, r_sh)
        eng[t, 2] = d_right - d_left

        # [3] foot forward diff (combined XZ plane for varied camera angles)
        #     positive → right foot lebih dekat ke kamera / di depan
        dx_foot = r_an[0] - l_an[0]
        dz_foot = r_an[2] - l_an[2]
        foot_mag = np.sqrt(dx_foot**2 + dz_foot**2)
        eng[t, 3] = foot_mag * np.sign(dz_foot) if foot_mag > eps else 0.0

        # [4] stance relation  (same side = +1 → Jab, cross = −1 → Straight)
        hand_sign = np.sign(eng[t, 2])
        foot_sign = np.sign(eng[t, 3])
        eng[t, 4] = hand_sign * foot_sign

        # [5] shoulder rotation  (z_left − z_right)
        eng[t, 5] = l_sh[2] - r_sh[2]

        # [6‥11] wrist velocity direction ratios ─────────────────────────────
        if t > 0:
            prev = sequence[t - 1]
            if np.abs(prev).sum() > eps:
                prev_l_wr = _get_xyz(prev, _L_WRIST)
                prev_r_wr = _get_xyz(prev, _R_WRIST)

                # Left wrist velocity components
                vx_l = l_wr[0] - prev_l_wr[0]
                vy_l = l_wr[1] - prev_l_wr[1]
                vz_l = l_wr[2] - prev_l_wr[2]
                tot_l = abs(vx_l) + abs(vy_l) + abs(vz_l) + eps

                eng[t, 6]  = abs(vx_l) / tot_l        # X ratio  → Hook
                eng[t, 7]  = -vy_l     / tot_l         # Y ratio↑ → Uppercut
                eng[t, 8]  = abs(vz_l) / tot_l        # Z ratio  → Jab/Straight

                # Right wrist velocity components
                vx_r = r_wr[0] - prev_r_wr[0]
                vy_r = r_wr[1] - prev_r_wr[1]
                vz_r = r_wr[2] - prev_r_wr[2]
                tot_r = abs(vx_r) + abs(vy_r) + abs(vz_r) + eps

                eng[t, 9]  = abs(vx_r) / tot_r
                eng[t, 10] = -vy_r     / tot_r
                eng[t, 11] = abs(vz_r) / tot_r

    return np.concatenate([sequence, eng], axis=1).astype(np.float32)


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
