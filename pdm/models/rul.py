"""RULEstimator wrapper that uniformly exposes RUL predictions from XGBoost or LSTM."""

from typing import Any, Optional, Union
import numpy as np

from pdm.models.xgboost_model import XGBoostModel
from pdm.models.lstm import LSTMModel


class RULEstimator:
    """Wrapper that exposes Remaining Useful Life (RUL) estimation uniformly."""

    def __init__(self, backend_model: Union[XGBoostModel, LSTMModel], rul_cap: float = 300.0) -> None:
        self.backend = backend_model
        self.rul_cap = rul_cap

    def predict_rul(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL and clip to [0, rul_cap]."""
        if hasattr(self.backend, "predict_rul"):
            raw_rul = self.backend.predict_rul(X)
        else:
            raw_rul = self.backend.predict(X)
        return np.clip(raw_rul, 0.0, self.rul_cap)
