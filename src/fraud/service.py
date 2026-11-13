"""Layanan prediksi: memuat artefak (model + preprocessing) lalu memberi probabilitas fraud."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR
from .features import FEATURES, load_params, preprocess
from .models import load_model


def risk_level(prob: float, threshold: float) -> str:
    """low / medium / high relatif terhadap threshold."""
    if prob >= threshold:
        return "high"
    if prob >= 0.5 * threshold:
        return "medium"
    return "low"


class FraudService:
    def __init__(self, artifact_dir=None):
        self.dir = Path(artifact_dir or ARTIFACT_DIR)
        self.meta = json.loads((self.dir / "model_meta.json").read_text(encoding="utf-8"))
        self.params = load_params(self.dir / "preprocessing.json")
        self.kind = self.meta["model_kind"]
        self.version = str(self.meta.get("version", "unknown"))
        self.threshold = float(self.meta.get("threshold", 0.5))
        self.model = load_model(self.dir, self.meta)
        self._xgb = None
        self._booster = None
        if self.kind == "xgboost":  # XGBoost asli (opsional) dipakai untuk SHAP
            try:
                import xgboost as xgb

                booster = xgb.Booster()
                booster.load_model(str(self.dir / self.meta["model_file"]))
                self._xgb, self._booster = xgb, booster
            except Exception:  # library tidak ada / beda versi: pakai fallback numpy
                self._xgb = self._booster = None

    @property
    def backend(self) -> str:
        if self._booster is not None:
            return f"xgboost {self._xgb.__version__}"
        return "numpy"

    @property
    def explain_available(self) -> bool:
        return self._booster is not None

    def prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        return preprocess(raw, self.params)

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        X = self.prepare(raw)
        if self._booster is not None:
            d = self._xgb.DMatrix(X.to_numpy(dtype=np.float32), feature_names=FEATURES)
            return np.asarray(self._booster.predict(d), dtype=np.float64)
        return self.model.predict_proba(X.to_numpy())

    def explain(self, raw: pd.DataFrame, top_k: int = 8):
        """Kontribusi SHAP per transaksi (log-odds). None bila XGBoost tidak tersedia."""
        if self._booster is None:
            return None
        X = self.prepare(raw)
        d = self._xgb.DMatrix(X.to_numpy(dtype=np.float32), feature_names=FEATURES)
        contrib = self._booster.predict(d, pred_contribs=True)[:, :-1]  # kolom terakhir = bias
        result = []
        for i in range(len(X)):
            order = np.argsort(-np.abs(contrib[i]))[:top_k]
            result.append(
                [
                    {
                        "feature": FEATURES[j],
                        "value": float(X.iloc[i, j]),
                        "contribution": float(contrib[i, j]),
                    }
                    for j in order
                ]
            )
        return result

    def predict(self, raw: pd.DataFrame, threshold: float | None = None, explain: bool = False, top_k: int = 8):
        """Daftar hasil per baris: probabilitas, keputusan, level risiko, (opsional) faktor SHAP."""
        thr = self.threshold if threshold is None else float(threshold)
        probs = self.predict_proba(raw)
        factors = self.explain(raw, top_k) if explain else None
        results = []
        for i, p in enumerate(probs):
            item = {
                "fraud_probability": float(p),
                "is_fraud": bool(p >= thr),
                "threshold": thr,
                "risk_level": risk_level(float(p), thr),
                "model_kind": self.kind,
                "model_version": self.version,
            }
            if factors is not None:
                item["top_factors"] = factors[i]
            results.append(item)
        return results
