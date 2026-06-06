# Olympic Boxing Punch Classification

Proyek ini bertujuan untuk mengklasifikasikan jenis-jenis pukulan dalam tinju olimpiade (Olympic Boxing) menggunakan Deep Learning. Terdapat 4 jenis pukulan yang diklasifikasikan, yaitu:
- **Hook**
- **Jab**
- **Straight**
- **Uppercut**

## Tahapan Penelitian & Pengujian

### 1. Persiapan Dataset
Dataset berupa rekaman video dikumpulkan untuk setiap jenis pukulan dengan rincian kurang lebih sebagai berikut:
- **Hook**: 38 video
- **Jab**: 44 video
- **Straight**: 42 video
- **Uppercut**: 48 video

Video-video ini di-preprocess (dijadikan frame, diekstrak fiturnya, dinormalisasi, dll) yang kemudian disimpan dalam format `.npy` untuk mempercepat proses *training*. Dataset mentah berupa video sengaja diabaikan (`.gitignore`) untuk menghindari beban penyimpanan berlebih pada repositori.

### 2. Skenario Pengujian
Pengujian *(testing)* dan Pelatihan *(training)* model dieksekusi dengan membagi dataset ke dalam 3 skenario yang berbeda (`scenario_1`, `scenario_2`, dan `scenario_3`). Eksperimen dilakukan dalam beberapa tahapan iterasi/perbaikan parameter, yang disimpan di direktori `results` dan `results_1`.

### 3. Hasil Pengujian
Berdasarkan log riwayat pelatihan (*training history*), model dievaluasi terhadap data validasi untuk mengukur tingkat akurasi yang dapat dicapai. Berikut adalah ringkasan akurasi validasi terbaik (*best validation accuracy*) dari masing-masing eksperimen:

#### Eksperimen Pertama (`results`)
- **Skenario 1**: 95.62%
- **Skenario 2**: 92.70%
- **Skenario 3**: 91.24%

#### Eksperimen Kedua (`results_1`)
- **Skenario 1**: 90.74%
- **Skenario 2**: 92.59%
- **Skenario 3**: 79.62%

Berdasarkan hasil di atas, model berhasil mengklasifikasikan gerakan pukulan dengan akurasi tertinggi mencapai **95.62%** pada Skenario 1 di percobaan pertama.

## Struktur Repositori
- `src/` : Kumpulan source code untuk preprocessing data, augmentasi, dll.
- `results/` & `results_1/` : Menyimpan log training, metrik performa, grafik kurva (*plots*), dan model tersimpan *(saved models)*.
- `test_gui.py` : Skrip untuk melakukan pengujian model dengan menggunakan antarmuka grafis (GUI).
- `dataset_stats.json` : Rincian statistik jumlah data per kelas, resolusi, durasi, dll.
- `.gitignore` : Mencegah agar file virtual environment dan video mentah tidak di-*commit* ke Git, namun mengizinkan file fitur (*Numpy*) yang penting.
