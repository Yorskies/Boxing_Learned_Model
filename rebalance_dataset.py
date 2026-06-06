import os
import shutil
import json

# Paths
dataset_dir = r"f:\Skripsi\Wahyu\Dataset"
unused_dir = r"f:\Skripsi\Wahyu\Dataset_Unused"
mapping_json = r"f:\Skripsi\Wahyu\dataset_files.json"

# Target count (based on Jab class size)
K = 34

# Load sorted details
with open(mapping_json, "r") as f:
    details = json.load(f)

print(f"Starting dataset rebalancing. Target: {K} videos per class.\n")

for cls, files in details.items():
    cls_path = os.path.join(dataset_dir, cls)
    cls_unused_path = os.path.join(unused_dir, cls)
    
    total_files = len(files)
    if total_files <= K:
        print(f"Class '{cls}' has {total_files} files (<= target {K}). No balancing needed.")
        continue
    
    # Calculate middle K range
    start_idx = (total_files - K) // 2
    end_idx = start_idx + K
    
    files_to_keep = files[start_idx:end_idx]
    files_to_move = files[0:start_idx] + files[end_idx:]
    
    # Verify math
    assert len(files_to_keep) == K
    assert len(files_to_move) == total_files - K
    
    # Create unused directory if it doesn't exist
    os.makedirs(cls_unused_path, exist_ok=True)
    
    print(f"Class '{cls}' (total {total_files} files):")
    print(f"  Keeping indices {start_idx} to {end_idx-1} (Durations: {files_to_keep[0]['duration']:.3f}s to {files_to_keep[-1]['duration']:.3f}s)")
    print(f"  Moving {len(files_to_move)} files to {cls_unused_path}...")
    
    for f in files_to_move:
        filename = f['filename']
        src = os.path.join(cls_path, filename)
        dst = os.path.join(cls_unused_path, filename)
        
        try:
            shutil.move(src, dst)
        except Exception as e:
            print(f"    Failed to move {filename}: {e}")
            
    print(f"  Finished class '{cls}'.\n")

print("Rebalancing execution completed successfully!")
