import os
import json
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from dataset import get_dataloaders
from model import AttentionBiLSTM

def plot_history(history, output_dir):
    """Fungsi untuk menggambar grafik Loss, Accuracy, dan Metrik dari history training"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(18, 5))
    
    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot Metrics
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_precision'], 'g-', label='Precision')
    plt.plot(epochs, history['val_recall'], 'm-', label='Recall')
    plt.plot(epochs, history['val_f1'], 'c-', label='F1-Score')
    plt.title('Validation Metrics (Macro)')
    plt.xlabel('Epochs')
    plt.ylabel('Percentage (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Grafik pelatihan berhasil disimpan di: {plot_path}")

def train(scenario_id, data_dir, num_workers=0):
    # Mengonfigurasi Hyperparameter berdasarkan Pilihan Skenario
    if scenario_id == 1:
        print("Mengeksekusi Skenario 1: Baseline (Stabil & Aman)")
        hidden_size = 64
        num_layers = 1
        batch_size = 8
        lr = 0.001
        dropout = 0.3
        optimizer_name = 'Adam'
        epochs = 100
        use_scheduler = False
    elif scenario_id == 2:
        print("Mengeksekusi Skenario 2: Deep & Regularized (Ekstraksi Fitur Kompleks)")
        hidden_size = 128
        num_layers = 2
        batch_size = 16
        lr = 0.0005
        dropout = 0.5
        optimizer_name = 'AdamW'
        epochs = 150
        use_scheduler = False
    elif scenario_id == 3:
        print("Mengeksekusi Skenario 3: Lightweight & Fast Learner (Pencegahan Overfitting)")
        hidden_size = 32
        num_layers = 1
        batch_size = 8
        lr = 0.002
        dropout = 0.2
        optimizer_name = 'Adam'
        epochs = 120
        use_scheduler = True
    elif scenario_id == 4:
        print("Mengeksekusi Skenario 4: Skenario Ultimate (Kapasitas + Stabilitas)")
        hidden_size = 64
        num_layers = 1
        batch_size = 16
        lr = 0.0015
        dropout = 0.3
        optimizer_name = 'Adam'
        epochs = 120
        use_scheduler = True
    elif scenario_id == 5:
        print("Mengeksekusi Skenario 5: Skenario Ultimate (Batch Size 4)")
        hidden_size = 64
        num_layers = 1
        batch_size = 4
        lr = 0.0015
        dropout = 0.3
        optimizer_name = 'Adam'
        epochs = 120
        use_scheduler = True
    else:
        raise ValueError("Scenario ID tidak valid. Pilih 1, 2, 3, 4, atau 5.")

    # Penataan Direktori Output yang Rapi
    base_results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'results', f'scenario_{scenario_id}'))
    models_dir = os.path.join(base_results_dir, 'models')
    logs_dir = os.path.join(base_results_dir, 'logs')
    plots_dir = os.path.join(base_results_dir, 'plots')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    best_model_path = os.path.join(models_dir, 'best_model.pth')
    history_path = os.path.join(logs_dir, 'training_history.json')

    # Deteksi Hardware
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan perangkat komputasi: {device}")
    
    # Load Data
    print(f"Memuat data dengan batch_size={batch_size} dan num_workers={num_workers}...")
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=batch_size, num_workers=num_workers)
    
    # Inisialisasi Model
    model = AttentionBiLSTM(
        input_size=144, 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        num_classes=4, 
        dropout_rate=dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
    if use_scheduler:
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
        
    # Dictionary untuk menyimpan riwayat (Log)
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    patience = 25 if scenario_id in [4, 5] else 20
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    best_all_preds = []
    best_all_targets = []
    
    print("\nMemulai proses training...\n")
    for epoch in range(epochs):
        # --- FASE TRAINING ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
            
        train_loss = train_loss / train_total
        train_acc = 100 * train_correct / train_total
        
        # --- FASE VALIDATION ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
                
        val_loss = val_loss / val_total
        val_acc = 100 * val_correct / val_total
        
        # Hitung Metrics dengan Scikit-Learn (Macro Average)
        val_precision = precision_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        val_recall = recall_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        
        # Simpan ke history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        if use_scheduler:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = lr
            
        print(f"Epoch [{epoch+1:03d}/{epochs:03d}] | LR: {current_lr:.5f} | "
              f"T-Loss: {train_loss:.4f} T-Acc: {train_acc:.2f}% | "
              f"V-Loss: {val_loss:.4f} V-Acc: {val_acc:.2f}% | "
              f"F1: {val_f1:.2f}%")
        
        # Simpan state terbaik
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            best_all_preds = all_preds.copy()
            best_all_targets = all_targets.copy()
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\n[Early Stopping] Pelatihan dihentikan pada epoch {epoch+1}. Tidak ada peningkatan selama {patience} epoch.")
            break
            
    # Menyimpan file log JSON
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\nRiwayat pelatihan (JSON) berhasil disimpan di: {history_path}")
    
    # Menggambar dan menyimpan grafik
    plot_history(history, plots_dir)
    
    target_names = ['Hook', 'Jab', 'Straight', 'Uppercut']
    
    # Menampilkan dan menyimpan Classification Report dari Epoch Terbaik
    print(f"\n--- Laporan Klasifikasi (Epoch Terbaik: {best_epoch}) ---")
    best_report = classification_report(best_all_targets, best_all_preds, target_names=target_names, zero_division=0)
    print(best_report)
    
    report_path = os.path.join(logs_dir, 'best_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"Classification Report (Best Epoch: {best_epoch})\n")
        f.write(f"Validation Accuracy: {best_val_acc:.2f}%\n\n")
        f.write(best_report)
    print(f"Laporan klasifikasi disimpan di: {report_path}")
    
    # Menggambar dan menyimpan Confusion Matrix dari Epoch Terbaik
    cm = confusion_matrix(best_all_targets, best_all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Epoch {best_epoch})')
    cm_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix disimpan di: {cm_path}")
    
    print(f"\nTraining Selesai! Model terbaik (Epoch {best_epoch}) tersimpan di:\n{best_model_path}\nAkurasi Validasi Puncak: {best_val_acc:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch BiLSTM Training")
    parser.add_argument('--scenario', type=int, required=True, choices=[1, 2, 3, 4, 5], 
                        help="Pilih skenario: 1-Baseline, 2-Deep, 3-Lightweight, 4-Ultimate(BS16), 5-Ultimate(BS4)")
    parser.add_argument('--data_dir', type=str, default=r"f:\Skripsi\Wahyu\Dataset_Numpy_Normalized",
                        help="Direktori dataset numpy hasil normalisasi")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="Jumlah CPU workers untuk DataLoader")
    args = parser.parse_args()
    
    train(args.scenario, args.data_dir, args.num_workers)
