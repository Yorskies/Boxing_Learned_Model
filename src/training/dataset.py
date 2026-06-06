import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class BoxingDataset(Dataset):
    def __init__(self, data_dir):
        """
        Membaca file .npy dari direktori Dataset_Numpy_Normalized.
        Memasukkan data dari 'Original' dan 'Augmented'.
        """
        self.data = []
        self.labels = []
        
        self.class_map = {'Hook': 0, 'Jab': 1, 'Straight': 2, 'Uppercut': 3}
        
        for variant in ['Original', 'Augmented']:
            variant_dir = os.path.join(data_dir, variant)
            if not os.path.exists(variant_dir):
                continue
                
            for cls_name, cls_idx in self.class_map.items():
                cls_dir = os.path.join(variant_dir, cls_name)
                if not os.path.exists(cls_dir):
                    continue
                    
                for file_name in os.listdir(cls_dir):
                    if file_name.endswith('.npy'):
                        file_path = os.path.join(cls_dir, file_name)
                        
                        # Load array numpy (30, 132)
                        sequence = np.load(file_path)
                        
                        self.data.append(sequence)
                        self.labels.append(cls_idx)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        # Konversi ke format PyTorch Tensors
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def get_dataloaders(data_dir, batch_size=8, num_workers=0, val_split=0.2):
    """
    Memecah data menjadi Train dan Validation secara acak, dan mengemasnya dalam DataLoader.
    Menerapkan num_workers yang disesuaikan untuk menghindari bottleneck prosesor (sesuai instruksi).
    """
    dataset = BoxingDataset(data_dir)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    # Gunakan seed agar split dataset konsisten di tiap run
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
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
