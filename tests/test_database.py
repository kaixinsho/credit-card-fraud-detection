from fraud.database import PredictionDB
from tests.samples import NORMAL, SUSPICIOUS


def _result(prob, fraud):
    return {
        "fraud_probability": prob, "is_fraud": fraud, "threshold": 0.5,
        "model_kind": "xgboost", "model_version": "t",
    }  # fmt: skip


def test_add_recent_and_stats(tmp_path):
    db = PredictionDB(tmp_path / "sub" / "p.db")
    assert db.stats()["total"] == 0
    id1 = db.add(NORMAL, _result(0.01, False))
    id2 = db.add(SUSPICIOUS, _result(0.99, True))
    assert id2 == id1 + 1
    rows = db.recent(10)
    assert [r["id"] for r in rows] == [id2, id1]  # terbaru dulu
    stats = db.stats()
    assert (stats["total"], stats["fraud"], stats["normal"]) == (2, 1, 1)
    assert stats["fraud_rate"] == 0.5


def test_add_many_and_clear(tmp_path):
    db = PredictionDB(tmp_path / "p.db")
    n = db.add_many([NORMAL, SUSPICIOUS, NORMAL], [_result(0.1, False), _result(0.9, True), _result(0.2, False)])
    assert n == 3
    assert db.stats()["total"] == 3
    assert db.clear() == 3
    assert db.stats()["total"] == 0


def test_recent_respects_limit(tmp_path):
    db = PredictionDB(tmp_path / "p.db")
    for _ in range(5):
        db.add(NORMAL, _result(0.1, False))
    assert len(db.recent(2)) == 2
