# Credit Card Fraud Detection - Machine Learning Pipeline

Proyek ini bertujuan untuk mendeteksi transaksi penipuan (fraud) pada kartu kredit menggunakan algoritma Machine Learning. Tantangan utama dalam proyek ini adalah menangani ketidakseimbangan data yang sangat ekstrem (imbalanced dataset), di mana kasus *fraud* hanya sebesar 0.17% dari total data.

## Deskripsi Proyek
Model yang digunakan dalam proyek ini adalah **XGBoost Classifier**. Untuk memastikan model dapat mengenali kelas minoritas (*fraud*) dengan baik, dilakukan penyesuaian bobot menggunakan parameter `scale_pos_weight` dan pencarian hyperparameter terbaik melalui `RandomizedSearchCV`.

Metrik utama yang menjadi fokus evaluasi adalah **Recall**, karena dalam kasus deteksi penipuan, jauh lebih penting untuk meminimalkan *False Negatives* (transaksi penipuan yang gagal dideteksi).

## Tahapan Pipeline
1. **Data Preprocessing:** Pembersihan data, scaling fitur, dan pembagian dataset (Train-Test Split).
2. **Modeling & Tuning:** Pelatihan model XGBoost awal dan optimasi parameter menggunakan RandomizedSearchCV dengan Stratified K-Fold Cross Validation.
3. **Evaluasi Model:** Menggunakan Classification Report, ROC-AUC Score, dan visualisasi Confusion Matrix.

## Cara Menjalankan Proyek
1. Clone repositori ini:
   `git clone https://github.com/kaixinsho/credit-card-fraud-detection`
2. Install semua library yang dibutuhkan:
   `pip install -r requirements.txt`
3. Jalankan file `notebook.ipynb` menggunakan Jupyter Notebook, VS Code, atau Google Colab.

## Author
* Gabriella Annabelle Antoniette