import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

FIXTURE_XGB = ROOT / "tests" / "fixtures" / "xgb"


@pytest.fixture
def xgb_dir():
    """Artefak XGBoost statis (model baseline) agar tes tidak ikut berubah saat model diganti."""
    return FIXTURE_XGB


@pytest.fixture
def dnn_dir(tmp_path):
    """Artefak DNN buatan (bobot acak) untuk menguji jalur model 'dnn'."""
    rng = np.random.default_rng(0)
    np.savez(
        tmp_path / "model_dnn.npz",
        W0=rng.normal(size=(30, 8)), b0=rng.normal(size=8),
        W1=rng.normal(size=(8, 1)), b1=rng.normal(size=1),
    )  # fmt: skip
    shutil.copy(FIXTURE_XGB / "preprocessing.json", tmp_path / "preprocessing.json")
    meta = {"model_kind": "dnn", "model_file": "model_dnn.npz", "version": "test-dnn", "threshold": 0.3}
    (tmp_path / "model_meta.json").write_text(json.dumps(meta))
    return tmp_path
