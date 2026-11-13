"""Preprocessing fitur: harus identik dengan yang dipakai saat training di notebook."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_COLS = ["Time", "Amount", *V_COLS]
FEATURES = ["scaled_amount", "scaled_time", *V_COLS]


def load_params(path) -> dict:
    """Baca parameter standardisasi (mean/scale) yang disimpan notebook."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preprocess(raw: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Ubah kolom mentah (Time, Amount, V1..V28) menjadi fitur model (urutan FEATURES).

    Amount boleh di-log1p dulu (params["amount"]["log1p"]) sebelum standardisasi.
    """
    missing = [c for c in RAW_COLS if c not in raw.columns]
    if missing:
        raise ValueError(f"Kolom tidak ditemukan: {', '.join(missing)}")

    amount = raw["Amount"].astype(float)
    if params["amount"].get("log1p", False):
        amount = np.log1p(amount)

    out = pd.DataFrame(index=raw.index)
    out["scaled_amount"] = (amount - params["amount"]["mean"]) / params["amount"]["scale"]
    out["scaled_time"] = (raw["Time"].astype(float) - params["time"]["mean"]) / params["time"]["scale"]
    for col in V_COLS:
        out[col] = raw[col].astype(float)
    return out[FEATURES]
