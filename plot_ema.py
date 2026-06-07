import os
import json
import matplotlib.pyplot as plt

def smooth_ema(scalars, weight=0.7):
    """
    Implementasi Exponential Moving Average (EMA) seperti di TensorBoard.
    weight: bobot perataan (0.0 hingga 0.99). Semakin tinggi, kurva makin mulus.
    """
    if not scalars:
        return scalars
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_history_with_ema(json_path, output_path, scenario_name):
    if not os.path.exists(json_path):
        print(f"File tidak ditemukan: {json_path}")
        return

    with open(json_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)
    
    t_loss = history['train_loss']
    v_loss = history['val_loss']
    t_acc = history['train_acc']
    v_acc = history['val_acc']

    # Smoothing factor
    weight = 0.75 
    
    t_loss_sm = smooth_ema(t_loss, weight)
    v_loss_sm = smooth_ema(v_loss, weight)
    t_acc_sm = smooth_ema(t_acc, weight)
    v_acc_sm = smooth_ema(v_acc, weight)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Grafik Loss ----
    ax1.plot(epochs, t_loss, alpha=0.25, color='#3498db', label='Train Loss (Raw)')
    ax1.plot(epochs, t_loss_sm, color='#2980b9', linewidth=2.5, label='Train Loss (EMA)')
    ax1.plot(epochs, v_loss, alpha=0.25, color='#e67e22', label='Val Loss (Raw)')
    ax1.plot(epochs, v_loss_sm, color='#d35400', linewidth=2.5, label='Val Loss (EMA)')
    
    ax1.set_title(f'{scenario_name} - Training & Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ---- Grafik Accuracy ----
    ax2.plot(epochs, t_acc, alpha=0.25, color='#2ecc71', label='Train Acc (Raw)')
    ax2.plot(epochs, t_acc_sm, color='#27ae60', linewidth=2.5, label='Train Acc (EMA)')
    ax2.plot(epochs, v_acc, alpha=0.25, color='#9b59b6', label='Val Acc (Raw)')
    ax2.plot(epochs, v_acc_sm, color='#8e44ad', linewidth=2.5, label='Val Acc (EMA)')
    
    ax2.set_title(f'{scenario_name} - Training & Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Selesai di-plot: {output_path}")

def main():
    base_dir = r"f:\Skripsi\Wahyu\results"
    
    for i in range(1, 6):
        scenario_dir = os.path.join(base_dir, f"scenario_{i}")
        json_path = os.path.join(scenario_dir, "logs", "training_history.json")
        plots_dir = os.path.join(scenario_dir, "plots")
        output_path = os.path.join(plots_dir, "training_curves_ema.png")
        
        if os.path.exists(json_path):
            os.makedirs(plots_dir, exist_ok=True)
            plot_history_with_ema(json_path, output_path, f"Skenario {i}")

if __name__ == "__main__":
    main()
