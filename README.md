# Credit Card Fraud Detection System

Deteksi transaksi kartu kredit fraud (dataset Kaggle *Credit Card Fraud Detection*, 284.807 transaksi, 0,17% fraud).

**Komponen**
- `notebooks/fraud_detection_v2.ipynb` — EDA, preprocessing, XGBoost (default vs tuning), Deep Learning (Keras DNN), perbandingan model, SHAP, export artefak.
- `src/fraud/` — paket inti: preprocessing, inferensi model (XGBoost/DNN), service prediksi, database SQLite.
- `api/main.py` — REST API (FastAPI).
- `app/streamlit_app.py` — antarmuka Streamlit (memanggil API, atau berjalan lokal bila API tidak diset).
- `tests/` — tes otomatis (pytest). `.github/workflows/ci.yml` — CI/CD (ruff, pytest, build Docker, push ke GHCR).

## Arsitektur
```
Streamlit UI  ──HTTP──>  FastAPI (/predict, /predict/batch, /history, /stats)
                              │
                     FraudService (model + preprocessing)  ──>  SQLite (predictions.db)
                              │
                     artifacts/ (model_*.json|npz, preprocessing.json, model_meta.json)
```

## Menjalankan lokal
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q                                              # jalankan tes
uvicorn api.main:create_app --factory --reload         # API di http://127.0.0.1:8000/docs
# terminal kedua:
set FRAUD_API_URL=http://127.0.0.1:8000                # Windows CMD; Mac/Linux: export FRAUD_API_URL=...
streamlit run app/streamlit_app.py                     # tanpa FRAUD_API_URL, Streamlit jalan mode lokal
```

## Docker
```bash
docker compose up --build        # API :8000, UI :8501
```

## Endpoint
| Method | Path | Fungsi |
|---|---|---|
| GET | `/health` | status layanan |
| GET | `/model/info` | metadata model |
| POST | `/predict?save=true&explain=false&threshold=0.5` | prediksi 1 transaksi |
| POST | `/predict/batch` | prediksi banyak transaksi |
| GET | `/history`, `/stats` | riwayat & ringkasan dari database |
| DELETE | `/history` | hapus riwayat |

## Mengganti model
Jalankan notebook v2 di Colab, salin `model_*`, `preprocessing.json`, `model_meta.json` dari `artifacts_v2/` ke `artifacts/`, lalu `pytest` dan jalankan ulang API. Model yang tersimpan di repo saat ini: XGBoost tuned (v2), threshold 0.963 (dipilih dari data validation).

## Hasil (data test, setelah duplikat dibuang)
| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost baseline | 0.948 | 0.768 | 0.849 | 0.815 | 0.965 |
| XGBoost tuned | 0.986 | 0.726 | 0.836 | 0.819 | 0.975 |
| DNN (Keras) | 0.878 | 0.758 | 0.814 | 0.701 | 0.960 |

Threshold tiap model dipilih dari data validation (F1 terbaik), bukan dari test. Tuning menaikkan PR-AUC/ROC-AUC sedikit dan memangkas false alarm (10 → 1), tetapi recall sedikit turun; DNN tidak mengalahkan XGBoost pada data tabular ini. Detail di `artifacts/model_comparison.csv` dan notebook.

## Catatan
- Fitur V1–V28 adalah hasil PCA yang dianonimkan; aplikasi ini untuk demonstrasi/portofolio, bukan sistem produksi.
- Penjelasan SHAP membutuhkan library `xgboost` terpasang dan model bertipe XGBoost.
