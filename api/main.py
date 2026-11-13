"""REST API prediksi fraud kartu kredit (FastAPI).

Jalankan dari root repo:
    uvicorn api.main:create_app --factory --reload
Dokumentasi interaktif: http://127.0.0.1:8000/docs
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typing import Annotated  # noqa: E402

import pandas as pd  # noqa: E402
from fastapi import Body, FastAPI, HTTPException, Query, Request  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field, create_model  # noqa: E402

from fraud import __version__  # noqa: E402
from fraud.config import DB_PATH, MAX_BATCH  # noqa: E402
from fraud.database import PredictionDB  # noqa: E402
from fraud.features import RAW_COLS, V_COLS  # noqa: E402
from fraud.service import FraudService  # noqa: E402

# ---------------------------------------------------------------- skema data
_fields = {
    "Time": (float, Field(..., ge=0, description="Detik sejak transaksi pertama di dataset")),
    "Amount": (float, Field(..., ge=0, description="Nominal transaksi")),
}
_fields.update({c: (float, Field(..., description="Fitur PCA (anonim)")) for c in V_COLS})
Transaction = create_model("Transaction", **_fields)


class TopFactor(BaseModel):
    feature: str
    value: float
    contribution: float = Field(..., description="Kontribusi SHAP (log-odds); positif = ke arah fraud")


class PredictionResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    fraud_probability: float
    is_fraud: bool
    threshold: float
    risk_level: str = Field(..., description="low / medium / high")
    model_kind: str
    model_version: str
    top_factors: list[TopFactor] | None = None


class PredictResponse(PredictionResult):
    saved: bool = False
    transaction_id: int | None = None


class BatchRequest(BaseModel):
    transactions: list[Transaction]


class BatchItem(BaseModel):
    fraud_probability: float
    is_fraud: bool


class BatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    count: int
    fraud_count: int
    threshold: float
    model_kind: str
    model_version: str
    saved: bool
    results: list[BatchItem]


EXAMPLE_NORMAL = {
    "Time": 0.0, "Amount": 149.62,
    "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155, "V5": -0.338321, "V6": 0.462388,
    "V7": 0.239599, "V8": 0.098698, "V9": 0.363787, "V10": 0.090794, "V11": -0.5516, "V12": -0.6178,
    "V13": -0.99139, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401, "V17": 0.207971,
    "V18": 0.025791, "V19": 0.403993, "V20": 0.251412, "V21": -0.018307, "V22": 0.277838,
    "V23": -0.110474, "V24": 0.066928, "V25": 0.128539, "V26": -0.189115, "V27": 0.133558,
    "V28": -0.021053,
}  # fmt: skip
# Profil sintetis (bukan data asli) yang dinilai sangat berisiko oleh model; untuk demo Swagger.
EXAMPLE_SUSPICIOUS = {
    **{c: 0.0 for c in V_COLS}, "Time": 406.0, "Amount": 1.0,
    "V1": -3.0, "V2": 3.0, "V3": -6.0, "V4": 4.0, "V5": -3.0, "V7": -5.0, "V9": -2.0, "V10": -6.0,
    "V11": 3.0, "V12": -8.0, "V14": -9.0, "V16": -4.0, "V17": -7.0, "V18": -3.0,
}  # fmt: skip


# ------------------------------------------------------------------- aplikasi
def create_app(service: FraudService | None = None, db: PredictionDB | None = None) -> FastAPI:
    """Buat aplikasi FastAPI. Argumen service/db bisa diisi saat testing."""
    app = FastAPI(
        title="Credit Card Fraud Detection API",
        version=__version__,
        description="Memprediksi probabilitas fraud sebuah transaksi kartu kredit dan menyimpan riwayatnya.",
    )
    app.state.service = service or FraudService()
    app.state.db = db or PredictionDB(DB_PATH)

    def svc(request: Request) -> FraudService:
        return request.app.state.service

    def database(request: Request) -> PredictionDB:
        return request.app.state.db

    @app.get("/", tags=["info"])
    def root():
        return {"name": app.title, "version": __version__, "docs": "/docs", "health": "/health"}

    @app.get("/health", tags=["info"])
    def health(request: Request):
        s = svc(request)
        return {
            "status": "ok",
            "api_version": __version__,
            "model_kind": s.kind,
            "model_version": s.version,
            "threshold": s.threshold,
            "backend": s.backend,
            "explain_available": s.explain_available,
        }

    @app.get("/model/info", tags=["info"])
    def model_info(request: Request):
        return svc(request).meta

    @app.post("/predict", response_model=PredictResponse, response_model_exclude_none=True, tags=["prediction"])
    def predict(
        request: Request,
        tx: Annotated[Transaction, Body(examples=[EXAMPLE_NORMAL, EXAMPLE_SUSPICIOUS])],
        save: Annotated[bool, Query(description="Simpan hasil ke database")] = True,
        explain: Annotated[bool, Query(description="Sertakan faktor SHAP (jika tersedia)")] = False,
        threshold: Annotated[float | None, Query(gt=0, lt=1, description="Timpa threshold bawaan model")] = None,
    ):
        data = tx.model_dump()
        df = pd.DataFrame([data])[RAW_COLS]
        result = svc(request).predict(df, threshold=threshold, explain=explain)[0]
        response = {**result, "saved": False}
        if save:
            response["transaction_id"] = database(request).add(data, result)
            response["saved"] = True
        return response

    @app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
    def predict_batch(
        request: Request,
        body: BatchRequest,
        save: Annotated[bool, Query(description="Simpan semua hasil ke database")] = False,
        threshold: Annotated[float | None, Query(gt=0, lt=1)] = None,
    ):
        if len(body.transactions) == 0:
            raise HTTPException(status_code=422, detail="Daftar transaksi kosong.")
        if len(body.transactions) > MAX_BATCH:
            raise HTTPException(status_code=413, detail=f"Maksimal {MAX_BATCH} transaksi per permintaan.")
        rows = [t.model_dump() for t in body.transactions]
        df = pd.DataFrame(rows)[RAW_COLS]
        s = svc(request)
        results = s.predict(df, threshold=threshold)
        if save:
            database(request).add_many(rows, results)
        return {
            "count": len(results),
            "fraud_count": sum(r["is_fraud"] for r in results),
            "threshold": results[0]["threshold"],
            "model_kind": s.kind,
            "model_version": s.version,
            "saved": save,
            "results": [
                {"fraud_probability": r["fraud_probability"], "is_fraud": r["is_fraud"]} for r in results
            ],
        }

    @app.get("/history", tags=["database"])
    def history(request: Request, limit: Annotated[int, Query(ge=1, le=1000)] = 50):
        return database(request).recent(limit)

    @app.get("/stats", tags=["database"])
    def stats(request: Request):
        return database(request).stats()

    @app.delete("/history", tags=["database"])
    def clear_history(request: Request):
        return {"deleted": database(request).clear()}

    return app
