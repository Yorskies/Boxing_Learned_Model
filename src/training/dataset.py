import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random

class BoxingDataset(Dataset):
    def __init__(self, data_pairs):
        """
        data_pairs is a list of tuples: (file_path, label_idx)
        """
        self.data_pairs = data_pairs

    def __len__(self):
        return len(self.data_pairs)
        
    def __getitem__(self, idx):
        file_path, label_idx = self.data_pairs[idx]
        sequence = np.load(file_path)
        
        # Konversi ke format PyTorch Tensors
        x = torch.tensor(sequence, dtype=torch.float32)
        y = torch.tensor(label_idx, dtype=torch.long)
        return x, y

def get_dataloaders(data_dir, batch_size=8, num_workers=0, val_split=0.2):
    """
    Memecah data menjadi Train dan Validation secara Group-based (berbasis video asli).
    Mencegah Data Leakage dari file augmentasi.
    """
    class_map = {'Hook': 0, 'Jab': 1, 'Straight': 2, 'Uppercut': 3}
    
    orig_dir = os.path.join(data_dir, 'Original')
    aug_dir = os.path.join(data_dir, 'Augmented')
    
    train_pairs = []
    val_pairs = []
    
    # Gunakan seed agar split dataset konsisten di tiap run
    random.seed(42)
    
    total_base_videos = 0
    
    for cls_name, cls_idx in class_map.items():
        cls_orig_dir = os.path.join(orig_dir, cls_name)
        cls_aug_dir = os.path.join(aug_dir, cls_name)
        
        if not os.path.exists(cls_orig_dir):
            continue
            
        # 1. Dapatkan semua Base Videos (hanya dari folder Original)
        base_files = [f for f in os.listdir(cls_orig_dir) if f.endswith('.npy')]
        base_files.sort()  # Sort untuk konsistensi sebelum shuffle
        
        # Shuffle base files
        random.shuffle(base_files)
        
        val_size = int(len(base_files) * val_split)
        val_base_files = set(base_files[:val_size])
        train_base_files = set(base_files[val_size:])
        
        total_base_videos += len(base_files)
        
        # 2. Bangun Validation Set (Hanya Original, TIDAK ADA augmentasi)
        for base_file in val_base_files:
            file_path = os.path.join(cls_orig_dir, base_file)
            val_pairs.append((file_path, cls_idx))
            
        # 3. Bangun Training Set (Original + Augmented)
        for base_file in train_base_files:
            # Masukkan file Original
            orig_path = os.path.join(cls_orig_dir, base_file)
            train_pairs.append((orig_path, cls_idx))
            
            # Cari dan masukkan versi Augmented
            if os.path.exists(cls_aug_dir):
                base_name = os.path.splitext(base_file)[0] # e.g. "Jab_01"
                
                # Sesuai output dari main_preprocessing.py
                aug_names = [
                    f"{base_name}.npy",         # flip
                    f"{base_name}_crop.npy",    # crop
                    f"{base_name}_speed.npy"    # speed
                ]
                
                for aname in aug_names:
                    apath = os.path.join(cls_aug_dir, aname)
                    if os.path.exists(apath):
                        train_pairs.append((apath, cls_idx))

    print(f"\n[DataLoader] Data Leakage ditutup! Menerapkan Group-based Split.")
    print(f"[DataLoader] Total Base Videos  : {total_base_videos}")
    print(f"[DataLoader] Total Train files  : {len(train_pairs)} (termasuk augmentasi)")
    print(f"[DataLoader] Total Val files    : {len(val_pairs)} (MURNI original, tanpa augmentasi)\n")

    train_dataset = BoxingDataset(train_pairs)
    val_dataset = BoxingDataset(val_pairs)
    
    # Penggunaan pin_memory=True jika CUDA tersedia untuk mempercepat transfer CPU ke GPU
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader
