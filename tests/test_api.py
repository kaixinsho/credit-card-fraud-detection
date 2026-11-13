import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

import api.main as api_main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from fraud.database import PredictionDB  # noqa: E402
from fraud.service import FraudService  # noqa: E402
from tests.samples import NORMAL, SUSPICIOUS  # noqa: E402


@pytest.fixture
def client(xgb_dir, tmp_path):
    app = api_main.create_app(service=FraudService(xgb_dir), db=PredictionDB(tmp_path / "t.db"))
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_kind"] == "xgboost"


def test_predict_normal_and_suspicious(client):
    normal = client.post("/predict?save=false", json=NORMAL).json()
    fraud = client.post("/predict?save=false", json=SUSPICIOUS).json()
    assert normal["is_fraud"] is False and normal["saved"] is False
    assert fraud["is_fraud"] is True and fraud["fraud_probability"] > 0.9


def test_predict_saves_to_database(client):
    r = client.post("/predict", json=SUSPICIOUS).json()
    assert r["saved"] is True and isinstance(r["transaction_id"], int)
    history = client.get("/history?limit=5").json()
    assert history[0]["id"] == r["transaction_id"]
    stats = client.get("/stats").json()
    assert stats["total"] == 1 and stats["fraud"] == 1


def test_threshold_override(client):
    r = client.post("/predict?save=false&threshold=0.999999", json=SUSPICIOUS).json()
    assert r["is_fraud"] is False and r["threshold"] == 0.999999


def test_validation_error_on_missing_feature(client):
    bad = {k: v for k, v in NORMAL.items() if k != "V7"}
    assert client.post("/predict", json=bad).status_code == 422


def test_negative_amount_rejected(client):
    assert client.post("/predict", json={**NORMAL, "Amount": -5}).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"transactions": [NORMAL, SUSPICIOUS, NORMAL]}).json()
    assert r["count"] == 3 and r["fraud_count"] == 1 and r["saved"] is False
    assert [x["is_fraud"] for x in r["results"]] == [False, True, False]
    assert client.get("/stats").json()["total"] == 0  # tidak disimpan


def test_batch_save_and_clear(client):
    client.post("/predict/batch?save=true", json={"transactions": [NORMAL, SUSPICIOUS]})
    assert client.get("/stats").json()["total"] == 2
    assert client.delete("/history").json()["deleted"] == 2
    assert client.get("/stats").json()["total"] == 0


def test_batch_limits(client, monkeypatch):
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422
    monkeypatch.setattr(api_main, "MAX_BATCH", 1)
    assert client.post("/predict/batch", json={"transactions": [NORMAL, NORMAL]}).status_code == 413
