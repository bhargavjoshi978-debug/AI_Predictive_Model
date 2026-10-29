"""Base model definitions and registry."""

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Protocol, Type, runtime_checkable
import numpy as np


@runtime_checkable
class BaseModel(Protocol):
    """Common model interface across all predictive maintenance models."""

    name: str
    task: Literal["classification", "anomaly", "rul"]

    def fit(self, train_data: Any, val_data: Optional[Any] = None) -> None:
        """Fit model on training dataset with optional validation data."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions (class labels, anomaly scores, or RUL values)."""
        ...

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Return predicted class probabilities, or None if not applicable."""
        ...

    def save(self, path: str | Path) -> None:
        """Serialize model to disk."""
        ...

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        """Load serialized model from disk."""
        ...

    def size_bytes(self) -> int:
        """Return serialized model size in bytes."""
        ...


MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(name: str):
    """Decorator to register a model class into MODEL_REGISTRY."""
    def decorator(cls: Type[BaseModel]):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator
