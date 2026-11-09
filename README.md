# Credit Card Fraud Detector (Streamlit)

## Cara menjalankan (di VS Code / PowerShell)
```
cd folder_ini
pip install -U -r requirements.txt
streamlit run app.py
```
Browser akan terbuka di http://localhost:8501

## File
- app.py              : tampilan Streamlit
- fraud_engine.py     : load model, scaling, prediksi, SHAP
- ui_parts.py         : CSS & komponen tampilan
- xgboost_fraud_model.json : model kamu, versi portabel (pengganti .pkl yang error)
- scaler_params.json  : parameter standardisasi Amount & Time

## Kenapa .pkl error "input stream corrupted"?
File .pkl kamu TIDAK corrupt. Model disimpan di Colab dengan XGBoost 3.4.1,
sedangkan XGBoost di laptop lebih lama, jadi tidak bisa membaca format biner pickle-nya.
Solusi: pakai format JSON (sudah disediakan) + `pip install -U xgboost`.
Ke depan simpan model dengan `xgb_model.save_model('model.json')`, bukan joblib/pickle.

## Bug kecil di notebook (opsional diperbaiki)
`scaler` dipakai ulang untuk Amount lalu Time, jadi scaler.pkl hanya berisi scaler Time.
Perbaikan:
    scaler_amount = StandardScaler(); scaler_time = StandardScaler()
    df['scaled_amount'] = scaler_amount.fit_transform(df[['Amount']])
    df['scaled_time']   = scaler_time.fit_transform(df[['Time']])
Aplikasi ini tidak bergantung pada scaler.pkl (parameternya ada di scaler_params.json).
