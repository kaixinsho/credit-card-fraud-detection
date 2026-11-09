"""Logika model (tanpa Streamlit): load model, scaling, prediksi, penjelasan."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
FEATURES = ["scaled_amount", "scaled_time"] + [f"V{i}" for i in range(1, 29)]
RAW_COLS = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]


class NumpyTreeModel:
    """Fallback murni numpy: baca model XGBoost (JSON) tanpa library xgboost."""

    def __init__(self, path):
        m = json.load(open(path))["learner"]
        self.n_features = int(m["learner_model_param"]["num_feature"])
        base = float(str(m["learner_model_param"]["base_score"]).strip("[]"))
        self.base_margin = float(np.log(base / (1 - base)))
        self.trees = []
        for t in m["gradient_booster"]["model"]["trees"]:
            self.trees.append(dict(
                left=np.asarray(t["left_children"]),
                right=np.asarray(t["right_children"]),
                feat=np.asarray(t["split_indices"]),
                cond=np.asarray(t["split_conditions"], dtype=np.float32),
                dleft=np.asarray(t["default_left"]),
            ))

    def margin(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.full(len(X), self.base_margin, dtype=np.float64)
        for t in self.trees:
            node = np.zeros(len(X), dtype=np.int64)
            for _ in range(64):
                left = t["left"][node]
                leaf = left == -1
                if leaf.all():
                    break
                x = X[np.arange(len(X)), t["feat"][node]]
                go_left = np.where(np.isnan(x), t["dleft"][node] == 1, x < t["cond"][node])
                nxt = np.where(go_left, left, t["right"][node])
                node = np.where(leaf, node, nxt)
            out += t["cond"][node]
        return out

    def predict_proba1(self, X):
        return 1 / (1 + np.exp(-self.margin(X)))


class FraudEngine:
    def __init__(self):
        sp = json.load(open(BASE / "scaler_params.json"))
        self.amt = sp["amount"]
        self.tim = sp["time"]
        self.backend = "numpy"
        self.booster = None
        self.np_model = NumpyTreeModel(BASE / "xgboost_fraud_model.json")
        try:  # pakai XGBoost asli bila tersedia (dan mengaktifkan SHAP)
            import xgboost as xgb
            b = xgb.Booster()
            b.load_model(str(BASE / "xgboost_fraud_model.json"))
            self.xgb = xgb
            self.booster = b
            self.backend = f"xgboost {xgb.__version__}"
        except Exception:
            pass

    def prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        """raw: kolom Time, Amount, V1..V28 (nilai asli) -> fitur model."""
        X = pd.DataFrame(index=raw.index)
        X["scaled_amount"] = (raw["Amount"] - self.amt["mean"]) / self.amt["scale"]
        X["scaled_time"] = (raw["Time"] - self.tim["mean"]) / self.tim["scale"]
        for i in range(1, 29):
            X[f"V{i}"] = raw[f"V{i}"].astype(float)
        return X[FEATURES]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.booster is not None:
            d = self.xgb.DMatrix(X.values.astype(np.float32), feature_names=FEATURES)
            return self.booster.predict(d)
        return self.np_model.predict_proba1(X.values)

    def contributions(self, X: pd.DataFrame):
        """Kontribusi SHAP per fitur (log-odds). None jika xgboost tidak tersedia."""
        if self.booster is None:
            return None
        d = self.xgb.DMatrix(X.values.astype(np.float32), feature_names=FEATURES)
        c = self.booster.predict(d, pred_contribs=True)
        return pd.DataFrame(c[:, :-1], columns=FEATURES, index=X.index)
