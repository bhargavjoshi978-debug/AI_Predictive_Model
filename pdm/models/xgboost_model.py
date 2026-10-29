"""XGBoost model for fault classification and RUL regression."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import joblib
import numpy as np
import xgboost as xgb

from pdm.config import XGBoostConfig
from pdm.models.base import BaseModel, register_model

CLASS_NAMES: List[str] = ["none", "bearing", "gear_wear", "motor_overheat", "cable_fault"]
CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {i: name for i, name in enumerate(CLASS_NAMES)}


@register_model("xgboost")
class XGBoostModel(BaseModel):
    """Proposed lightweight XGBoost model with multi-class classification and RUL regression."""

    name: str = "xgboost"
    task: Literal["classification", "anomaly", "rul"] = "classification"

    def __init__(self, config: Optional[XGBoostConfig] = None) -> None:
        self.config = config or XGBoostConfig()
        self.clf: Optional[xgb.XGBClassifier] = None
        self.reg: Optional[xgb.XGBRegressor] = None
        self.feature_names: List[str] = []
        self._saved_size_bytes: int = 0

    def fit(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Fit classifier on full dataset and RUL regressor on degrading windows."""
        if feature_names is not None:
            self.feature_names = list(feature_names)

        X_train, y_train_cls, y_train_rul = self._unpack_data(train_data)

        # Ensure all 5 classes (0..4) exist in training data so XGBoost
        # never fails with missing intermediate class indices (e.g. only [0, 3] in small splits)
        unique_classes = set(y_train_cls)
        missing_classes = [c_idx for c_idx in range(len(CLASS_NAMES)) if c_idx not in unique_classes]
        if missing_classes:
            dummy_X = np.zeros((len(missing_classes), X_train.shape[1]), dtype=np.float32)
            dummy_y = np.array(missing_classes, dtype=int)
            dummy_w = np.full(len(missing_classes), 1e-6, dtype=np.float32)

            orig_len = len(y_train_cls)
            X_train = np.vstack([X_train, dummy_X])
            y_train_cls = np.concatenate([y_train_cls, dummy_y])
            y_train_rul = np.concatenate([y_train_rul, np.full(len(missing_classes), np.nan, dtype=np.float32)])
            sample_weight = np.concatenate([np.ones(orig_len, dtype=np.float32), dummy_w])
        else:
            sample_weight = None

        # 1. Train classifier
        self.clf = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            tree_method=self.config.tree_method,
            early_stopping_rounds=self.config.early_stopping_rounds if val_data is not None else None,
            eval_metric="mlogloss",
            device="cpu",
            n_jobs=1,
            random_state=42,
        )

        eval_set = None
        if val_data is not None:
            X_val, y_val_cls, _ = self._unpack_data(val_data)
            v_missing = [c_idx for c_idx in range(len(CLASS_NAMES)) if c_idx not in set(y_val_cls)]
            if v_missing:
                v_dummy_X = np.zeros((len(v_missing), X_val.shape[1]), dtype=np.float32)
                v_dummy_y = np.array(v_missing, dtype=int)
                X_val = np.vstack([X_val, v_dummy_X])
                y_val_cls = np.concatenate([y_val_cls, v_dummy_y])
            eval_set = [(X_val, y_val_cls)]

        self.clf.fit(X_train, y_train_cls, sample_weight=sample_weight, eval_set=eval_set, verbose=False)

        # 2. Train RUL regressor on fault-active samples
        fault_mask = (y_train_cls != CLASS_TO_IDX["none"]) & (~np.isnan(y_train_rul))
        if np.any(fault_mask) and np.count_nonzero(fault_mask) >= 10:
            X_rul = X_train[fault_mask]
            y_rul = y_train_rul[fault_mask]

            self.reg = xgb.XGBRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                tree_method=self.config.tree_method,
                early_stopping_rounds=self.config.early_stopping_rounds if val_data is not None else None,
                eval_metric="rmse",
                device="cpu",
                n_jobs=1,
                random_state=42,
            )

            eval_reg_set = None
            if val_data is not None:
                X_v, y_v_cls, y_v_rul = self._unpack_data(val_data)
                v_mask = (y_v_cls != CLASS_TO_IDX["none"]) & (~np.isnan(y_v_rul))
                if np.any(v_mask):
                    eval_reg_set = [(X_v[v_mask], y_v_rul[v_mask])]

            self.reg.fit(X_rul, y_rul, eval_set=eval_reg_set, verbose=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class indices or names."""
        if self.clf is None:
            raise RuntimeError("Model is not fitted.")
        return self.clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict class probabilities of shape (N, n_classes)."""
        if self.clf is None:
            raise RuntimeError("Model is not fitted.")
        return self.clf.predict_proba(X)

    def predict_rul(self, X: np.ndarray) -> np.ndarray:
        """Predict remaining useful life in steps."""
        if self.reg is None:
            return np.full(len(X), 300.0)
        preds = self.reg.predict(X)
        return np.clip(preds, 0.0, 300.0)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Return feature importance scores (gain-based)."""
        if self.clf is None:
            return np.zeros(len(self.feature_names))
        return self.clf.feature_importances_

    def get_top_features(self, k: int = 15) -> List[Tuple[str, float]]:
        """Get top k features sorted by importance."""
        imps = self.feature_importances_
        if len(self.feature_names) != len(imps):
            names = [f"feat_{i}" for i in range(len(imps))]
        else:
            names = self.feature_names

        pairs = sorted(zip(names, imps), key=lambda p: p[1], reverse=True)
        return [(str(k), float(v)) for k, v in pairs[:k]]

    def save(self, path: str | Path) -> None:
        """Serialize model to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "config": self.config,
            "clf": self.clf,
            "reg": self.reg,
            "feature_names": self.feature_names,
        }
        joblib.dump(bundle, p)
        self._saved_size_bytes = p.stat().st_size

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostModel":
        """Load serialized model from disk."""
        p = Path(path)
        bundle = joblib.load(p)
        model = cls(config=bundle["config"])
        model.clf = bundle["clf"]
        model.reg = bundle["reg"]
        model.feature_names = bundle.get("feature_names", [])
        model._saved_size_bytes = p.stat().st_size
        return model

    def size_bytes(self) -> int:
        """Return serialized model size in bytes."""
        if self._saved_size_bytes > 0:
            return self._saved_size_bytes
        # Temporary memory estimate
        import sys
        return sys.getsizeof(self.clf)

    def _unpack_data(self, data: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert input dataset/tuple/DataFrame to (X, y_class_idx, y_rul)."""
        if isinstance(data, tuple):
            X = data[0]
            y_cls = data[1]
            y_rul = data[2] if len(data) > 2 else np.full(len(X), np.nan)
        else:
            # Assume pandas DataFrame
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                label_col = "y_fault" if "y_fault" in data.columns else "label_fault"
                rul_col = "y_rul" if "y_rul" in data.columns else "rul_steps"
                meta_cols = ["episode_id", "robot_id", "t", "joint", "window_idx", "health", "damage", "y_fault", "y_anomaly", "y_rul", "label_fault", "label_joint", "rul_steps"]
                f_cols = [c for c in data.columns if c not in meta_cols]
                X = data[f_cols].values
                y_cls = data[label_col].values
                y_rul = data[rul_col].values if rul_col in data.columns else np.full(len(X), np.nan)
            else:
                raise ValueError("Unsupported data format.")

        # Convert class names to numeric indices if strings
        if len(y_cls) > 0 and isinstance(y_cls[0], str):
            y_cls_idx = np.array([CLASS_TO_IDX.get(c, 0) for c in y_cls], dtype=int)
        else:
            y_cls_idx = np.array(y_cls, dtype=int)

        return np.asarray(X, dtype=np.float32), y_cls_idx, np.asarray(y_rul, dtype=np.float32)
