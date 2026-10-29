"""Random Forest model for fault classification."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from pdm.config import RandomForestConfig
from pdm.models.base import BaseModel, register_model
from pdm.models.xgboost_model import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS


@register_model("random_forest")
class RandomForestModel(BaseModel):
    """Supervised Random Forest baseline classifier for robot fault classification."""

    name: str = "random_forest"
    task: Literal["classification", "anomaly", "rul"] = "classification"

    def __init__(self, config: Optional[RandomForestConfig] = None) -> None:
        self.config = config or RandomForestConfig()
        self.clf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=1,
        )
        self.feature_names: List[str] = []
        self._saved_size_bytes: int = 0
        self._is_fitted: bool = False

    def fit(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Fit random forest on training dataset."""
        if feature_names is not None:
            self.feature_names = list(feature_names)

        X_train, y_train_cls = self._unpack_data(train_data)
        self.clf.fit(X_train, y_train_cls)
        self._is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted.")
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted.")
        probs = self.clf.predict_proba(X)
        # Ensure probs has shape (N, 5) even if some classes were unobserved in small fits
        if probs.shape[1] == len(CLASS_NAMES):
            return probs
        full_probs = np.zeros((len(X), len(CLASS_NAMES)), dtype=np.float32)
        for i, c_idx in enumerate(self.clf.classes_):
            full_probs[:, c_idx] = probs[:, i]
        return full_probs

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return feature importance scores (MDI)."""
        if not self._is_fitted:
            return np.zeros(len(self.feature_names))
        return self.clf.feature_importances_

    def save(self, path: str | Path) -> None:
        """Serialize model to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "config": self.config,
            "clf": self.clf,
            "feature_names": self.feature_names,
            "is_fitted": self._is_fitted,
        }
        joblib.dump(bundle, p)
        self._saved_size_bytes = p.stat().st_size

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestModel":
        """Load serialized model from disk."""
        p = Path(path)
        bundle = joblib.load(p)
        model = cls(config=bundle["config"])
        model.clf = bundle["clf"]
        model.feature_names = bundle.get("feature_names", [])
        model._is_fitted = bundle.get("is_fitted", True)
        model._saved_size_bytes = p.stat().st_size
        return model

    def size_bytes(self) -> int:
        """Return serialized model size in bytes."""
        if self._saved_size_bytes > 0:
            return self._saved_size_bytes
        import sys
        return sys.getsizeof(self.clf)

    def _unpack_data(self, data: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Convert input data to (X, y_class_idx)."""
        if isinstance(data, tuple):
            X = data[0]
            y_cls = data[1]
        else:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                label_col = "y_fault" if "y_fault" in data.columns else "label_fault"
                meta_cols = ["episode_id", "robot_id", "t", "joint", "window_idx", "health", "damage", "y_fault", "y_anomaly", "y_rul", "label_fault", "label_joint", "rul_steps"]
                f_cols = [c for c in data.columns if c not in meta_cols]
                X = data[f_cols].values
                y_cls = data[label_col].values
            else:
                raise ValueError("Unsupported data format.")

        if len(y_cls) > 0 and isinstance(y_cls[0], str):
            y_cls_idx = np.array([CLASS_TO_IDX.get(c, 0) for c in y_cls], dtype=int)
        else:
            y_cls_idx = np.array(y_cls, dtype=int)

        return np.asarray(X, dtype=np.float32), y_cls_idx
