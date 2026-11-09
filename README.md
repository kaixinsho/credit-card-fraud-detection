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