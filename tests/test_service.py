import pandas as pd

from fraud.features import RAW_COLS
from fraud.service import FraudService, risk_level
from tests.samples import NORMAL, SUSPICIOUS


def _df():
    return pd.DataFrame([NORMAL, SUSPICIOUS])[RAW_COLS]


def test_xgboost_service_decisions(xgb_dir):
    s = FraudService(xgb_dir)
    normal, suspicious = s.predict(_df())
    assert normal["is_fraud"] is False and normal["risk_level"] == "low"
    assert suspicious["is_fraud"] is True and suspicious["risk_level"] == "high"
    assert suspicious["model_kind"] == "xgboost"


def test_threshold_override_changes_decision(xgb_dir):
    s = FraudService(xgb_dir)
    assert s.predict(_df(), threshold=0.5)[1]["is_fraud"] is True
    assert s.predict(_df(), threshold=0.999999)[1]["is_fraud"] is False


def test_dnn_service_uses_meta_threshold_and_has_no_explain(dnn_dir):
    s = FraudService(dnn_dir)
    results = s.predict(_df(), explain=True)
    assert len(results) == 2
    assert results[0]["threshold"] == 0.3
    assert s.explain_available is False
    assert "top_factors" not in results[0]
    assert s.backend == "numpy"


def test_risk_level_boundaries():
    assert risk_level(0.5, 0.5) == "high"
    assert risk_level(0.25, 0.5) == "medium"
    assert risk_level(0.1, 0.5) == "low"
