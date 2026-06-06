import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths
dataset_dir = r"f:\Skripsi\Wahyu\Dataset"
output_image = r"f:\Skripsi\Wahyu\dataset_distribution.png"
output_json = r"f:\Skripsi\Wahyu\dataset_stats.json"

classes = ["Hook", "Jab", "Straight", "Uppercut"]

results = {}

for cls in classes:
    cls_path = os.path.join(dataset_dir, cls)
    if not os.path.exists(cls_path):
        continue
    
    video_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    durations = []
    fps_list = []
    frame_counts = []
    resolutions = []
    file_sizes = []
    corrupt_files = []
    
    for f in video_files:
        path = os.path.join(cls_path, f)
        # Size in MB
        size_mb = os.path.getsize(path) / (1024 * 1024)
        file_sizes.append(size_mb)
        
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            corrupt_files.append(f)
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        if fps > 0 and total_frames > 0:
            duration = total_frames / fps
            durations.append(duration)
            fps_list.append(fps)
            frame_counts.append(total_frames)
            resolutions.append(f"{width}x{height}")
        else:
            corrupt_files.append(f)
            
    results[cls] = {
        "count": len(video_files),
        "corrupt_count": len(corrupt_files),
        "corrupt_files": corrupt_files,
        "file_sizes_mb": file_sizes,
        "durations_sec": durations,
        "fps": fps_list,
        "resolutions": resolutions,
        "frame_counts": frame_counts
    }

# Compute stats
stats = {}
all_classes = list(results.keys())
counts = [results[c]["count"] for c in all_classes]

for cls, data in results.items():
    if data["durations_sec"]:
        stats[cls] = {
            "count": data["count"],
            "corrupt_count": data["corrupt_count"],
            "total_size_mb": sum(data["file_sizes_mb"]),
            "avg_size_mb": np.mean(data["file_sizes_mb"]),
            "min_size_mb": np.min(data["file_sizes_mb"]),
            "max_size_mb": np.max(data["file_sizes_mb"]),
            "total_duration_sec": sum(data["durations_sec"]),
            "avg_duration_sec": np.mean(data["durations_sec"]),
            "min_duration_sec": np.min(data["durations_sec"]),
            "max_duration_sec": np.max(data["durations_sec"]),
            "resolutions_dist": {r: data["resolutions"].count(r) for r in set(data["resolutions"])},
            "fps_dist": {float(f): data["fps"].count(f) for f in set(data["fps"])},
            "avg_fps": np.mean(data["fps"]),
            "avg_frames": np.mean(data["frame_counts"]),
        }
    else:
        stats[cls] = {
            "count": data["count"],
            "corrupt_count": data["corrupt_count"],
            "total_size_mb": 0,
            "avg_size_mb": 0,
            "min_size_mb": 0,
            "max_size_mb": 0,
            "total_duration_sec": 0,
            "avg_duration_sec": 0,
            "min_duration_sec": 0,
            "max_duration_sec": 0,
            "resolutions_dist": {},
            "fps_dist": {},
            "avg_fps": 0,
            "avg_frames": 0,
        }

# Save stats to JSON
with open(output_json, "w") as f:
    json.dump(stats, f, indent=4)

# Create figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1. Class Distribution (Bar chart)
colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2']
axs[0, 0].bar(all_classes, counts, color=colors, edgecolor='black', alpha=0.85)
axs[0, 0].set_title('Video Count per Class', fontsize=14, fontweight='bold', pad=10)
axs[0, 0].set_ylabel('Number of Videos', fontsize=12)
for i, v in enumerate(counts):
    axs[0, 0].text(i, v + max(counts)*0.01, str(v), ha='center', fontweight='bold')

# 2. Total Duration per Class (Bar chart)
durations_total = [stats[c]["total_duration_sec"] for c in all_classes]
axs[0, 1].bar(all_classes, durations_total, color=colors, edgecolor='black', alpha=0.85)
axs[0, 1].set_title('Total Duration per Class (Seconds)', fontsize=14, fontweight='bold', pad=10)
axs[0, 1].set_ylabel('Total Duration (s)', fontsize=12)
for i, v in enumerate(durations_total):
    axs[0, 1].text(i, v + max(durations_total)*0.01, f"{v:.1f}s", ha='center', fontweight='bold')

# 3. Average Video Duration per Class
durations_avg = [stats[c]["avg_duration_sec"] for c in all_classes]
axs[1, 0].bar(all_classes, durations_avg, color=colors, edgecolor='black', alpha=0.85)
axs[1, 0].set_title('Average Video Duration per Class', fontsize=14, fontweight='bold', pad=10)
axs[1, 0].set_ylabel('Average Duration (s)', fontsize=12)
for i, v in enumerate(durations_avg):
    axs[1, 0].text(i, v + max(durations_avg)*0.01, f"{v:.2f}s", ha='center', fontweight='bold')

# 4. Total Size in MB per Class
sizes_total = [stats[c]["total_size_mb"] for c in all_classes]
axs[1, 1].bar(all_classes, sizes_total, color=colors, edgecolor='black', alpha=0.85)
axs[1, 1].set_title('Total File Size per Class (MB)', fontsize=14, fontweight='bold', pad=10)
axs[1, 1].set_ylabel('Total Size (MB)', fontsize=12)
for i, v in enumerate(sizes_total):
    axs[1, 1].text(i, v + max(sizes_total)*0.01, f"{v:.1f} MB", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(output_image, dpi=150)
plt.close()

print("Analysis complete and artifacts saved.")
