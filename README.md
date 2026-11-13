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
Jalankan notebook v2 di Colab, salin `model_*`, `preprocessing.json`, `model_meta.json` dari `artifacts_v2/` ke `artifacts/`, lalu `pytest` dan jalankan ulang API. Model yang tersimpan di repo saat ini adalah baseline v1 sampai artefak v2 dimasukkan.

## Catatan
- Fitur V1–V28 adalah hasil PCA yang dianonimkan; aplikasi ini untuk demonstrasi/portofolio, bukan sistem produksi.
- Penjelasan SHAP membutuhkan library `xgboost` terpasang dan model bertipe XGBoost.
