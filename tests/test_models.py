"""Tests for the model layer: fit, predict, and serialization round-trip."""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from pdm.models.xgboost_model import XGBoostModel
from pdm.models.random_forest import RandomForestModel
from pdm.models.autoencoder import AutoencoderModel
from pdm.models.lstm import LSTMModel
from pdm.models.rul import RULEstimator


@pytest.fixture
def tiny_dataset():
    """Create a tiny deterministic dataset for testing model fitting and prediction."""
    rng = np.random.default_rng(42)
    n_samples = 60
    n_features = 25

    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y_cls = rng.choice(["none", "bearing", "gear_wear", "motor_overheat", "cable_fault"], size=n_samples)
    y_rul = rng.uniform(10.0, 300.0, size=n_samples).astype(np.float32)

    return X, y_cls, y_rul


def test_xgboost_fit_and_roundtrip(tiny_dataset) -> None:
    """XGBoost fits on tiny dataset and save/load produces identical predictions."""
    X, y_cls, y_rul = tiny_dataset
    model = XGBoostModel()
    model.fit((X, y_cls, y_rul), feature_names=[f"f_{i}" for i in range(X.shape[1])])

    preds = model.predict(X)
    probs = model.predict_proba(X)
    ruls = model.predict_rul(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 5)
    assert len(ruls) == len(X)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "xgb.joblib"
        model.save(save_path)
        assert save_path.exists()
        assert model.size_bytes() > 0

        loaded = XGBoostModel.load(save_path)
        loaded_preds = loaded.predict(X)
        loaded_probs = loaded.predict_proba(X)
        loaded_ruls = loaded.predict_rul(X)

        np.testing.assert_array_equal(preds, loaded_preds)
        np.testing.assert_array_almost_equal(probs, loaded_probs, decimal=5)
        np.testing.assert_array_almost_equal(ruls, loaded_ruls, decimal=5)


def test_random_forest_fit_and_roundtrip(tiny_dataset) -> None:
    """Random Forest fits on tiny dataset and save/load produces identical predictions."""
    X, y_cls, _ = tiny_dataset
    model = RandomForestModel()
    model.fit((X, y_cls), feature_names=[f"f_{i}" for i in range(X.shape[1])])

    preds = model.predict(X)
    probs = model.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 5)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "rf.joblib"
        model.save(save_path)
        assert save_path.exists()
        assert model.size_bytes() > 0

        loaded = RandomForestModel.load(save_path)
        loaded_preds = loaded.predict(X)
        loaded_probs = loaded.predict_proba(X)

        np.testing.assert_array_equal(preds, loaded_preds)
        np.testing.assert_array_almost_equal(probs, loaded_probs, decimal=5)


def test_autoencoder_fit_and_roundtrip(tiny_dataset) -> None:
    """Autoencoder fits on healthy windows and save/load produces identical anomaly scores."""
    X, y_cls, _ = tiny_dataset
    model = AutoencoderModel()
    model.config.epochs = 5  # fast test
    model.fit((X, y_cls), feature_names=[f"f_{i}" for i in range(X.shape[1])])

    scores = model.compute_anomaly_score(X)
    anom_preds = model.predict(X)
    probs = model.predict_proba(X)

    assert len(scores) == len(X)
    assert len(anom_preds) == len(X)
    assert probs.shape == (len(X), 2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "ae.pt"
        model.save(save_path)
        assert save_path.exists()
        assert model.size_bytes() > 0

        loaded = AutoencoderModel.load(save_path)
        loaded_scores = loaded.compute_anomaly_score(X)
        loaded_preds = loaded.predict(X)

        np.testing.assert_array_almost_equal(scores, loaded_scores, decimal=5)
        np.testing.assert_array_equal(anom_preds, loaded_preds)


def test_lstm_fit_and_roundtrip() -> None:
    """LSTM fits on sequence tensor and save/load produces identical predictions."""
    rng = np.random.default_rng(42)
    n_seq = 20
    seq_len = 10
    n_feat = 15

    X_seq = rng.standard_normal((n_seq, seq_len, n_feat)).astype(np.float32)
    y_rul = rng.uniform(20.0, 300.0, size=n_seq).astype(np.float32)
    y_anom = rng.choice([0, 1], size=n_seq)

    model = LSTMModel()
    model.config.epochs = 5
    model.fit((X_seq, y_rul, y_anom))

    preds = model.predict_rul(X_seq)
    probs = model.predict_proba(X_seq)

    assert len(preds) == n_seq
    assert probs.shape == (n_seq, 2)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "lstm.pt"
        model.save(save_path)
        assert save_path.exists()
        assert model.size_bytes() > 0

        loaded = LSTMModel.load(save_path)
        loaded_preds = loaded.predict_rul(X_seq)
        loaded_probs = loaded.predict_proba(X_seq)

        np.testing.assert_array_almost_equal(preds, loaded_preds, decimal=5)
        np.testing.assert_array_almost_equal(probs, loaded_probs, decimal=5)


def test_rul_estimator_wrapper(tiny_dataset) -> None:
    """RULEstimator wrapper correctly clips and returns predictions."""
    X, y_cls, y_rul = tiny_dataset
    model = XGBoostModel()
    model.fit((X, y_cls, y_rul))

    rul_est = RULEstimator(backend_model=model, rul_cap=300.0)
    preds = rul_est.predict_rul(X)

    assert len(preds) == len(X)
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 300.0)
