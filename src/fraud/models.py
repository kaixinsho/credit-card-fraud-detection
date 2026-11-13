"""Model inferensi berbasis numpy, sehingga tidak bergantung pada versi XGBoost/TensorFlow.

- NumpyXGBoost : membaca file model XGBoost (JSON) dan menghitung prediksi pohon.
- NumpyMLP     : forward pass jaringan saraf (Dense + ReLU, output sigmoid) dari bobot .npz.
"""
import json
from pathlib import Path

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


class NumpyXGBoost:
    """Inferensi XGBoost (objective binary:logistic) langsung dari file JSON."""

    def __init__(self, path):
        learner = json.loads(Path(path).read_text(encoding="utf-8"))["learner"]
        base = float(str(learner["learner_model_param"]["base_score"]).strip("[]"))
        self.base_margin = float(np.log(base / (1.0 - base)))
        self.trees = []
        for t in learner["gradient_booster"]["model"]["trees"]:
            self.trees.append(
                {
                    "left": np.asarray(t["left_children"]),
                    "right": np.asarray(t["right_children"]),
                    "feat": np.asarray(t["split_indices"]),
                    "cond": np.asarray(t["split_conditions"], dtype=np.float32),
                    "dleft": np.asarray(t["default_left"]),
                }
            )

    def margin(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        rows = np.arange(len(X))
        out = np.full(len(X), self.base_margin, dtype=np.float64)
        for t in self.trees:
            node = np.zeros(len(X), dtype=np.int64)
            for _ in range(128):  # batas kedalaman pengaman
                left = t["left"][node]
                is_leaf = left == -1
                if is_leaf.all():
                    break
                x = X[rows, t["feat"][node]]
                go_left = np.where(np.isnan(x), t["dleft"][node] == 1, x < t["cond"][node])
                nxt = np.where(go_left, left, t["right"][node])
                node = np.where(is_leaf, node, nxt)
            out += t["cond"][node]  # nilai daun disimpan di split_conditions
        return out

    def predict_proba(self, X) -> np.ndarray:
        return _sigmoid(self.margin(X))


class NumpyMLP:
    """Inferensi MLP: hidden layer ReLU, output 1 neuron sigmoid.

    File .npz berisi W0,b0,W1,b1,... (urutan sama dengan model.get_weights() di Keras).
    """

    def __init__(self, path):
        data = np.load(path)
        n = len([k for k in data.files if k.startswith("W")])
        if n < 1:
            raise ValueError("File bobot tidak valid: tidak ada matriks W0, W1, ...")
        self.weights = [data[f"W{i}"].astype(np.float64) for i in range(n)]
        self.biases = [data[f"b{i}"].astype(np.float64) for i in range(n)]

    def predict_proba(self, X) -> np.ndarray:
        h = np.asarray(X, dtype=np.float64)
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            h = np.maximum(h @ W + b, 0.0)
        z = h @ self.weights[-1] + self.biases[-1]
        return _sigmoid(z).reshape(-1)


def load_model(artifact_dir, meta: dict):
    """Pilih dan muat model sesuai model_meta.json ("xgboost" atau "dnn")."""
    path = Path(artifact_dir) / meta["model_file"]
    kind = meta["model_kind"]
    if kind == "xgboost":
        return NumpyXGBoost(path)
    if kind == "dnn":
        return NumpyMLP(path)
    raise ValueError(f"model_kind tidak dikenal: {kind}")
