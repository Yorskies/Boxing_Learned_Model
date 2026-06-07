import os
import random

def analyze_split(data_dir, val_split=0.2):
    class_map = {'Hook': 0, 'Jab': 1, 'Straight': 2, 'Uppercut': 3}
    
    orig_dir = os.path.join(data_dir, 'Original')
    aug_dir = os.path.join(data_dir, 'Augmented')
    
    random.seed(42)
    
    print("="*60)
    print("ANALISIS DISTRIBUSI DATASET DAN PEMISAHAN (SPLIT) DATA")
    print("="*60)
    
    total_orig_all = 0
    total_aug_all = 0
    total_train_all = 0
    total_val_all = 0
    
    details = {}
    
    for cls_name, cls_idx in class_map.items():
        cls_orig_dir = os.path.join(orig_dir, cls_name)
        cls_aug_dir = os.path.join(aug_dir, cls_name)
        
        orig_files = []
        if os.path.exists(cls_orig_dir):
            orig_files = [f for f in os.listdir(cls_orig_dir) if f.endswith('.npy')]
            orig_files.sort()
            
        aug_files = []
        if os.path.exists(cls_aug_dir):
            aug_files = [f for f in os.listdir(cls_aug_dir) if f.endswith('.npy')]
            aug_files.sort()
            
        # Group-based split
        base_files = list(orig_files)
        random.shuffle(base_files)
        
        val_size = int(len(base_files) * val_split)
        val_base_files = set(base_files[:val_size])
        train_base_files = set(base_files[val_size:])
        
        # Count train pairs (Original + Augmented)
        train_count = 0
        val_count = len(val_base_files)
        
        # Original files that go to train
        train_count += len(train_base_files)
        
        # Augmented files linked to train base files
        linked_aug_count = 0
        for base_file in train_base_files:
            base_name = os.path.splitext(base_file)[0]
            aug_names = [
                f"{base_name}.npy",         # flip
                f"{base_name}_crop.npy",    # crop
                f"{base_name}_speed.npy"    # speed
            ]
            for aname in aug_names:
                apath = os.path.join(cls_aug_dir, aname)
                if os.path.exists(apath):
                    linked_aug_count += 1
                    
        train_count += linked_aug_count
        
        total_orig_all += len(orig_files)
        total_aug_all += len(aug_files)
        total_train_all += train_count
        total_val_all += val_count
        
        details[cls_name] = {
            'original': len(orig_files),
            'augmented_total': len(aug_files),
            'train': train_count,
            'val': val_count,
            'train_orig': len(train_base_files),
            'train_aug': linked_aug_count
        }
        
    print(f"{'Kelas':<12} | {'Original':<10} | {'Augmented':<10} | {'Train (Total)':<15} | {'Val (Murni)':<12}")
    print("-"*65)
    for cls_name, info in details.items():
        print(f"{cls_name:<12} | {info['original']:<10} | {info['augmented_total']:<10} | {info['train']:<15} | {info['val']:<12}")
    print("-"*65)
    print(f"{'TOTAL':<12} | {total_orig_all:<10} | {total_aug_all:<10} | {total_train_all:<15} | {total_val_all:<12}")
    print("="*60)
    print("\nDetail Komponen pada Train Set:")
    for cls_name, info in details.items():
        print(f"- Kelas {cls_name:<10}: {info['train_orig']} Original + {info['train_aug']} Augmented = {info['train']} total")

if __name__ == '__main__':
    data_dir = r"f:\Skripsi\Wahyu\Dataset_Numpy_Normalized"
    analyze_split(data_dir)
