# Olympic Boxing Punch Classification using Attention-BiLSTM

Proyek ini bertujuan untuk mengklasifikasikan jenis-jenis pukulan dalam tinju olimpiade (*Olympic Boxing*) secara *real-time* menggunakan kombinasi ekstraksi fitur **MediaPipe Pose Landmarker** dan arsitektur Deep Learning **Attention-BiLSTM**.

Model mengklasifikasikan 4 jenis pukulan utama:
* **Hook**
* **Jab**
* **Straight**
* **Uppercut**

---

## 1. Arsitektur & Rekayasa Fitur
* **Ekstraktor Spasial**: MediaPipe Pose Landmarker digunakan untuk melacak koordinat sendi tubuh secara spasial.
* **Fitur Kinematik**: Vektor fitur diperluas hingga **144-dimensi** dengan memasukkan parameter turunan kinematik seperti kecepatan sendi (kinematic speed) dan perubahan sudut sendi untuk membedakan akselerasi pukulan secara temporal.
* **Arsitektur Klasifikasi**: Jaringan saraf **BiLSTM** dua arah dikombinasikan dengan mekanisme **Attention (Perhatian)** untuk memberikan bobot lebih pada frame-frame kunci saat pukulan dilepaskan.
* **Penyaring Temporal**: GUI aplikasi dilengkapi dengan **Temporal Non-Maximum Suppression (NMS)** untuk menyaring *flickering noise* (prediksi sesaat yang salah) agar transisi deteksi di layar tetap stabil dan konsisten.

---

## 2. Distribusi & Pemisahan Dataset (Group-Based Split)

Untuk mencegah kebocoran data (*data leakage*) selama proses latih, pembagian dataset dilakukan secara **Group-based Split** berbasis video asli (Original) dengan rasio **80% Training** dan **20% Validation**. 

* **Validation Set**: Murni berisi video original (tidak ada augmentasi) untuk menguji model pada *unseen subjects*.
* **Training Set**: Berisi data original + 3 jenis augmentasi (*flip*, *crop*, *speed variation*).

### Tabel Distribusi Dataset
| Kelas Pukulan | Original | Augmented | Train Set (Total) | Val Set (Murni Original) |
| :--- | :---: | :---: | :---: | :---: |
| **Hook** | 37 | 111 | 120 | 7 |
| **Jab** | 43 | 129 | 140 | 8 |
| **Straight** | 42 | 126 | 136 | 8 |
| **Uppercut** | 48 | 144 | 156 | 9 |
| **TOTAL** | **170** | **510** | **552** | **32** |

---

## 3. Hasil Skenario Eksperimen
Berikut adalah perbandingan performa teknis dari 5 skenario eksperimen yang diuji pada Validation Set:

| ID | Skenario Konfigurasi | Hidden Size | Layers | Batch Size | Epoch Optimal | Akurasi | F1-Score | Status |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **1** | **Baseline (Best Model)** | **64** | **1** | **8** | **14** | **90.62%** | **0.91** | **Dipilih untuk Produksi** |
| **2** | Deep & Regularized | 128 | 2 | 16 | 21 | 90.62% | 0.90 | Over-parameterized |
| **3** | Lightweight | 32 | 1 | 8 | 18 | 87.50% | 0.87 | Underfitting |
| **4** | Ultimate (BS 16) | 64 | 1 | 16 | 29 | 90.62% | 0.91 | Konvergensi lambat |
| **5** | Ultimate (BS 4) | 64 | 1 | 4 | 24 | 90.62% | 0.91 | Pelatihan fluktuatif |

### Justifikasi Pemilihan Skenario 1:
1. **Konvergensi Cepat**: Mencapai tingkat keakuratan optimal hanya dalam **14 epoch** (tercepat di antara semua model berakurasi 90%+).
2. **Ringan & Efisien**: Jumlah parameter yang lebih sedikit (`hidden_size=64`, `layer=1`) sangat menguntungkan untuk proses *sliding window* inferensi di sistem GUI *real-time*.
3. **Performa Stabil**: Menghasilkan akurasi **90.62%** dan F1-Score **0.91** dengan kemampuan deteksi pukulan lurus (*Jab/Straight*) yang sangat presisi.

---

## 4. Struktur Repositori

```text
├── Dataset/                   # Dataset video mentah (.mp4 / .avi) berdasarkan kelas
├── Dataset_Numpy_Normalized/  # Dataset koordinat fitur hasil ekstraksi (.npy)
├── results/                   # Log training, model terbaik (.pth), & grafik kurva per skenario
├── src/                       # Source code proyek
│   ├── preprocessing/         # Pipeline ekstraksi koordinat & augmentasi data
│   │   ├── augmentator.py     # Logika augmentasi (flip, crop, speed)
│   │   ├── config.py          # Konfigurasi parameter prapemrosesan
│   │   ├── mediapipe_extractor.py # Ekstrasi pose landmarker & kinematik koordinat
│   │   └── main_preprocessing.py  # Script utama pengeksekusi preprocessing
│   └── training/              # Script pelatihan model neural network
│       ├── dataset.py         # PyTorch Dataset & DataLoader (Group split)
│       ├── model.py           # Arsitektur Attention-BiLSTM
│       └── train.py           # Script utama pelatihan model
├── test_gui.py                # Aplikasi GUI Real-time berbasis OpenCV
├── plot_ema.py                # Visualisasi kurva loss & akurasi dengan smoothing EMA
├── laporan_skripsi_bilstm.txt # Draf laporan pembahasan skripsi (format txt)
└── requirements.txt           # Kebutuhan pustaka Python
```

---

## 5. Cara Menjalankan Proyek

### A. Instalasi Dependensi
Pastikan Anda menggunakan Python 3.8+ dan telah menginstal seluruh pustaka yang dibutuhkan:
```bash
pip install -r requirements.txt
```

### B. Preprocessing & Ekstraksi Fitur
Jalankan perintah berikut untuk mengekstrak pose landmark, menghitung percepatan kinematik, melakukan augmentasi data, dan menyimpannya ke format `.npy`:
```bash
python src/preprocessing/main_preprocessing.py
```

### C. Pelatihan Model (Training)
Untuk melatih model sesuai dengan skenario eksperimen yang diinginkan (misal Skenario 1):
```bash
python src/training/train.py --scenario 1 --data_dir Dataset_Numpy_Normalized
```
Hasil pelatihan (model terbaik, grafik, laporan klasifikasi) akan otomatis tersimpan dalam folder `results/scenario_1/`.

### D. Menjalankan Aplikasi Real-time GUI
Untuk melakukan inferensi secara langsung terhadap file video baru atau webcam menggunakan model terbaik yang telah dilatih:
```bash
python test_gui.py
```
*(GUI ini mengaplikasikan sliding window 30-frame dan filter Temporal NMS untuk menjamin kelancaran penayangan hasil prediksi)*
