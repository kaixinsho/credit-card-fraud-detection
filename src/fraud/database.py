"""Penyimpanan riwayat prediksi di SQLite (satu koneksi baru per operasi, aman untuk thread)."""
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,
    time_sec      REAL    NOT NULL,
    amount        REAL    NOT NULL,
    features_json TEXT    NOT NULL,
    probability   REAL    NOT NULL,
    is_fraud      INTEGER NOT NULL,
    threshold     REAL    NOT NULL,
    model_kind    TEXT    NOT NULL,
    model_version TEXT    NOT NULL
);
"""


class PredictionDB:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def add(self, transaction: dict, result: dict) -> int:
        """Simpan satu prediksi, kembalikan id baris."""
        with closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "INSERT INTO predictions (created_at, time_sec, amount, features_json, probability, "
                "is_fraud, threshold, model_kind, model_version) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    float(transaction["Time"]),
                    float(transaction["Amount"]),
                    json.dumps({k: float(v) for k, v in transaction.items()}),
                    float(result["fraud_probability"]),
                    int(result["is_fraud"]),
                    float(result["threshold"]),
                    str(result["model_kind"]),
                    str(result["model_version"]),
                ),
            )
            return int(cur.lastrowid)

    def add_many(self, transactions: list[dict], results: list[dict]) -> int:
        """Simpan banyak prediksi sekaligus dalam satu transaksi database (untuk batch)."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = [
            (
                now,
                float(t["Time"]),
                float(t["Amount"]),
                json.dumps({k: float(v) for k, v in t.items()}),
                float(r["fraud_probability"]),
                int(r["is_fraud"]),
                float(r["threshold"]),
                str(r["model_kind"]),
                str(r["model_version"]),
            )
            for t, r in zip(transactions, results)
        ]
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT INTO predictions (created_at, time_sec, amount, features_json, probability, "
                "is_fraud, threshold, model_kind, model_version) VALUES (?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return len(rows)

    def recent(self, limit: int = 50) -> list[dict]:
        """Prediksi terbaru lebih dulu (tanpa kolom fitur mentah agar ringkas)."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, created_at, time_sec, amount, probability, is_fraud, threshold, "
                "model_kind, model_version FROM predictions ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total, COALESCE(SUM(is_fraud), 0) AS fraud, "
                "AVG(probability) AS avg_prob, MAX(created_at) AS last_at FROM predictions"
            ).fetchone()
        total, fraud = int(row["total"]), int(row["fraud"])
        return {
            "total": total,
            "fraud": fraud,
            "normal": total - fraud,
            "fraud_rate": (fraud / total) if total else 0.0,
            "avg_probability": float(row["avg_prob"]) if row["avg_prob"] is not None else 0.0,
            "last_prediction_at": row["last_at"],
        }

    def clear(self) -> int:
        with closing(self._connect()) as conn, conn:
            return conn.execute("DELETE FROM predictions").rowcount
