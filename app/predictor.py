"""Penghubung Streamlit ke model: lewat REST API (RemotePredictor) atau langsung (LocalPredictor).

Keduanya punya method yang sama, jadi tampilan tidak perlu tahu sedang memakai yang mana.
"""
import pandas as pd
import requests

from fraud.config import DB_PATH
from fraud.database import PredictionDB
from fraud.features import RAW_COLS
from fraud.service import FraudService


class PredictorError(RuntimeError):
    """Gagal memanggil model / API (pesan sudah siap ditampilkan ke pengguna)."""


class LocalPredictor:
    mode = "Lokal"

    def __init__(self):
        self.service = FraudService()
        self.db = PredictionDB(DB_PATH)

    def info(self) -> dict:
        s = self.service
        return {
            "model_kind": s.kind, "model_version": s.version, "threshold": s.threshold,
            "backend": s.backend, "explain_available": s.explain_available, "meta": s.meta,
        }  # fmt: skip

    def predict_one(self, tx: dict, threshold=None, explain=True, save=False) -> dict:
        df = pd.DataFrame([tx])[RAW_COLS]
        result = self.service.predict(df, threshold=threshold, explain=explain)[0]
        if save:
            result["transaction_id"] = self.db.add(tx, result)
        return result

    def predict_batch_proba(self, df: pd.DataFrame, progress=None):
        probs = self.service.predict_proba(df[RAW_COLS])
        if progress:
            progress(1.0)
        return probs

    def history(self, limit=100) -> list[dict]:
        return self.db.recent(limit)

    def stats(self) -> dict:
        return self.db.stats()


class RemotePredictor:
    mode = "REST API"

    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _call(self, method: str, path: str, **kwargs):
        try:
            r = requests.request(method, self.base + path, timeout=self.timeout, **kwargs)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError as e:
            raise PredictorError(
                f"API tidak bisa dihubungi di {self.base}. Jalankan dulu: "
                "uvicorn api.main:create_app --factory"
            ) from e
        except requests.exceptions.RequestException as e:
            raise PredictorError(f"Permintaan ke API gagal: {e}") from e

    def info(self) -> dict:
        health = self._call("GET", "/health")
        health["meta"] = self._call("GET", "/model/info")
        return health

    def predict_one(self, tx: dict, threshold=None, explain=True, save=False) -> dict:
        params = {"save": str(save).lower(), "explain": str(explain).lower()}
        if threshold is not None:
            params["threshold"] = threshold
        return self._call("POST", "/predict", params=params, json=tx)

    def predict_batch_proba(self, df: pd.DataFrame, progress=None, chunk: int = 5000):
        probs = []
        rows = df[RAW_COLS].to_dict(orient="records")
        for start in range(0, len(rows), chunk):
            body = {"transactions": rows[start : start + chunk]}
            out = self._call("POST", "/predict/batch", params={"save": "false"}, json=body)
            probs.extend(item["fraud_probability"] for item in out["results"])
            if progress:
                progress(min(1.0, (start + chunk) / len(rows)))
        return pd.Series(probs).to_numpy()

    def history(self, limit=100) -> list[dict]:
        return self._call("GET", "/history", params={"limit": limit})

    def stats(self) -> dict:
        return self._call("GET", "/stats")
