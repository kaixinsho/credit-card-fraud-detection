"""Konfigurasi path dan batas lewat environment variable (nilai default aman untuk lokal)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = Path(os.getenv("FRAUD_ARTIFACT_DIR", ROOT / "artifacts"))
DB_PATH = Path(os.getenv("FRAUD_DB_PATH", ROOT / "data" / "predictions.db"))
MAX_BATCH = int(os.getenv("FRAUD_MAX_BATCH", "50000"))
