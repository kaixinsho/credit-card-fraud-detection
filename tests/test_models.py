import json

import numpy as np
import pandas as pd
import pytest

from fraud.features import FEATURES, RAW_COLS, preprocess
from fraud.models import NumpyMLP, NumpyXGBoost, load_model
from tests.samples import NORMAL, SUSPICIOUS


def _features(xgb_dir, rows):
    params = json.loads((xgb_dir / "preprocessing.json").read_text())
    return preprocess(pd.DataFrame(rows)[RAW_COLS], params).to_numpy()


def test_mlp_matches_manual_forward(tmp_path):
    rng = np.random.default_rng(1)
    W0, b0 = rng.normal(size=(30, 6)), rng.normal(size=6)
    W1, b1 = rng.normal(size=(6, 1)), rng.normal(size=1)
    np.savez(tmp_path / "w.npz", W0=W0, b0=b0, W1=W1, b1=b1)
    x = rng.normal(size=(5, 30))
    expected = (1 / (1 + np.exp(-(np.maximum(x @ W0 + b0, 0) @ W1 + b1)))).reshape(-1)
    np.testing.assert_allclose(NumpyMLP(tmp_path / "w.npz").predict_proba(x), expected, rtol=1e-9)


def test_mlp_probabilities_in_range(dnn_dir):
    meta = json.loads((dnn_dir / "model_meta.json").read_text())
    model = load_model(dnn_dir, meta)
    p = model.predict_proba(np.random.default_rng(2).normal(size=(50, 30)) * 50)
    assert p.shape == (50,)
    assert np.all((p >= 0) & (p <= 1))


def test_numpy_xgboost_separates_normal_and_suspicious(xgb_dir):
    model = NumpyXGBoost(xgb_dir / "model_xgboost.json")
    p = model.predict_proba(_features(xgb_dir, [NORMAL, SUSPICIOUS]))
    assert p[0] < 0.05
    assert p[1] > 0.9


def test_numpy_xgboost_matches_native_xgboost(xgb_dir):
    xgb = pytest.importorskip("xgboost")
    model = NumpyXGBoost(xgb_dir / "model_xgboost.json")
    booster = xgb.Booster()
    booster.load_model(str(xgb_dir / "model_xgboost.json"))
    X = np.random.default_rng(3).normal(size=(300, 30)) * 3
    native = booster.predict(xgb.DMatrix(X.astype(np.float32), feature_names=FEATURES))
    np.testing.assert_allclose(model.predict_proba(X), native, atol=1e-4)


def test_unknown_model_kind_raises(tmp_path):
    with pytest.raises(ValueError):
        load_model(tmp_path, {"model_kind": "svm", "model_file": "x"})
