import json

import numpy as np
import pandas as pd
import pytest

from fraud.features import FEATURES, RAW_COLS, preprocess
from tests.conftest import FIXTURE_XGB
from tests.samples import NORMAL

PARAMS = json.loads((FIXTURE_XGB / "preprocessing.json").read_text())


def _df(**overrides):
    return pd.DataFrame([{**NORMAL, **overrides}])[RAW_COLS]


def test_feature_order_and_count():
    X = preprocess(_df(), PARAMS)
    assert list(X.columns) == FEATURES
    assert X.shape == (1, 30)


def test_standardization_formula():
    mean_a, scale_a = PARAMS["amount"]["mean"], PARAMS["amount"]["scale"]
    mean_t = PARAMS["time"]["mean"]
    X = preprocess(_df(Amount=mean_a + scale_a, Time=mean_t), PARAMS)
    assert X["scaled_amount"].iloc[0] == pytest.approx(1.0)
    assert X["scaled_time"].iloc[0] == pytest.approx(0.0)


def test_pca_columns_unchanged():
    X = preprocess(_df(V14=-9.5), PARAMS)
    assert X["V14"].iloc[0] == -9.5


def test_log1p_option():
    params = {**PARAMS, "amount": {**PARAMS["amount"], "log1p": True}}
    X = preprocess(_df(Amount=0.0), params)
    expected = (np.log1p(0.0) - PARAMS["amount"]["mean"]) / PARAMS["amount"]["scale"]
    assert X["scaled_amount"].iloc[0] == pytest.approx(expected)


def test_missing_column_raises():
    with pytest.raises(ValueError):
        preprocess(_df().drop(columns=["V3"]), PARAMS)
