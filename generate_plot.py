import os
import json
import matplotlib.pyplot as plt

def plot_history(history, output_dir):
    """Fungsi untuk menggambar grafik Loss dan Accuracy dari history training"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Memastikan direktori output ada
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Grafik pelatihan berhasil disimpan di: {plot_path}")

if __name__ == '__main__':
    # Path ke file history json
    json_path = r'f:\Skripsi\Wahyu\results\scenario_1\logs\training_history.json'
    
    # Path ke direktori output yang terpisah
    output_dir = r'f:\Skripsi\Wahyu\generated_plot'
    
    if not os.path.exists(json_path):
        print(f"File {json_path} tidak ditemukan!")
    else:
        with open(json_path, 'r') as f:
            history = json.load(f)
            
        plot_history(history, output_dir)
