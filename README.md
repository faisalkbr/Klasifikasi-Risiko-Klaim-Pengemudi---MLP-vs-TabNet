# Klasifikasi Risiko Klaim Pengemudi — Perbandingan MLP vs TabNet + Deployment Flutter

**UAS Praktikum Machine Learning 2026 — Kelompok XXX**
Jalur: **Klasifikasi + Mobile**

Proyek ini membandingkan **dua metode Deep Learning** (Deep MLP dan TabNet) pada satu dataset
telematics asuransi untuk mengklasifikasikan pengemudi **berisiko** (pernah mengajukan klaim)
vs **tidak berisiko**, lalu men-deploy model terbaik ke aplikasi **Flutter** via TensorFlow Lite.

Referensi metodologi & dataset: McDonnell et al. (2023), *Deep learning in insurance: Accuracy
and model interpretability using TabNet*, Expert Systems With Applications 217, 119543.

---

## Ringkasan Hasil

Dataset sangat **imbalanced** (hanya 4.27% pengemudi berisiko), sehingga **akurasi menyesatkan**
dan evaluasi difokuskan pada F1, Recall, AUC, dan MCC.

| Metrik    | MLP        | TabNet |
|-----------|------------|--------|
| Accuracy  | **0.9403** | 0.9269 |
| Precision | **0.3338** | 0.2385 |
| Recall    | **0.3994** | 0.3245 |
| F1-Score  | **0.3636** | 0.2750 |
| AUC       | **0.8658** | 0.8262 |
| MCC       | **0.3340** | 0.2405 |

**Temuan utama:** MLP yang lebih sederhana **mengungguli** TabNet pada seluruh metrik. Ini
konsisten dengan literatur bahwa arsitektur Deep Learning kompleks tidak selalu unggul pada
data tabular. **Model terbaik = MLP**, dan model inilah yang di-deploy ke aplikasi mobile.

Perhatikan kontras Accuracy 0.94 vs F1 0.36: bukti kuat bahwa akurasi tinggi ≠ model bagus
pada data imbalanced.

---

## Struktur Proyek

```
UAS_KelompokXXX/
├── data/
│   └── telematics.csv              # dataset asli (100.000 baris, 52 kolom)
├── notebooks/
│   ├── 01_EDA_Preprocessing.ipynb  # eksplorasi, definisi target, preprocessing, split
│   ├── 02_Training_Modeling.ipynb  # arsitektur & training MLP + TabNet
│   └── 03_Evaluasi_Komparasi.ipynb # evaluasi, confusion matrix, ROC/PR, ekspor TFLite
├── src/
│   ├── models.py                   # definisi arsitektur MLP & TabNet
│   └── train_eval.py               # pipeline lengkap sekali jalan
├── outputs/
│   ├── figures/                    # semua grafik hasil
│   └── models/                     # .keras, .tflite, metrics.json, preprocessing.json
├── mobile_flutter/                 # aplikasi Flutter (inferensi on-device)
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan

```bash
pip install -r requirements.txt

# Opsi A — pipeline sekali jalan (menghasilkan semua grafik + model + tflite)
python src/train_eval.py

# Opsi B — jalankan notebook berurutan 01 -> 02 -> 03
jupyter notebook
```

---

## Metodologi

1. **Input** — dataset telematics sintetis (So et al., 2021): 100.000 pengemudi, 50 fitur
   (data tradisional: usia, gender, skor kredit, dll + telematics: total miles, harsh braking,
   turn intensity, dll).
2. **Preprocessing** — cek missing value (0), one-hot encoding 4 kolom kategorikal,
   **pembuangan kolom bocor** (`NB_Claim`, `AMT_Claim` yang mendefinisikan target).
3. **Transformation** — standardisasi (StandardScaler) di-fit hanya pada train.
4. **Data Splitting** — stratified 70% train / 15% validation / 15% test.
5. **Penanganan Imbalance** — class weight balanced saat training.
6. **Modeling** — dua metode Deep Learning paralel:
   - **MLP**: 3 hidden layer (128-64-32) + BatchNorm + Dropout.
   - **TabNet**: multi-step decision + attentive transformer (mask softmax) + feature
     transformer GLU. Mask memakai softmax (bukan sparsemax) agar kompatibel TFLite.
7. **Evaluasi** — threshold dioptimalkan pada validation (maksimalkan F1), diterapkan ke test.
   Metrik: Accuracy, Precision, Recall, F1, AUC, MCC + confusion matrix + ROC/PR.
8. **Output/Deployment** — model terbaik diekspor ke `.tflite` dan dijalankan on-device di
   aplikasi Flutter.

---

## Anggota Kelompok

| NIM | Nama | Peran |
|-----|------|-------|
| ... | ...  | ...   |

## Lisensi
Dataset di bawah lisensi asli So et al. (2021). Kode untuk keperluan akademik.
