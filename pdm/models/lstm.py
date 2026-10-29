"""PyTorch CPU small 2-layer LSTM model for RUL estimation and anomaly detection."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from pdm.config import LSTMConfig
from pdm.models.base import BaseModel, register_model


class LSTMNet(nn.Module):
    """2-layer LSTM with linear heads for RUL regression and anomaly classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_rul = nn.Linear(hidden_dim, 1)
        self.fc_anom = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.lstm(x)
        # Use last step representation
        last_repr = out[:, -1, :]
        rul_pred = self.fc_rul(last_repr).squeeze(-1)
        anom_logits = self.fc_anom(last_repr)
        return rul_pred, anom_logits


@register_model("lstm")
class LSTMModel(BaseModel):
    """Small CPU-only LSTM baseline for sequence-based RUL regression and anomaly detection."""

    name: str = "lstm"
    task: Literal["classification", "anomaly", "rul"] = "rul"

    def __init__(self, config: Optional[LSTMConfig] = None) -> None:
        self.config = config or LSTMConfig()
        self.net: Optional[LSTMNet] = None
        self.input_dim: int = 0
        self.feature_names: List[str] = []
        self._saved_size_bytes: int = 0
        self.device = torch.device("cpu")

    def fit(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Fit LSTM on sequences of feature windows."""
        if feature_names is not None:
            self.feature_names = list(feature_names)

        X_train, y_train_rul, y_train_anom = self._unpack_data(train_data)
        if len(X_train) == 0:
            raise ValueError("Empty training data for LSTM.")

        self.input_dim = X_train.shape[2]
        self.net = LSTMNet(
            input_dim=self.input_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.device)

        # Replace NaNs in RUL for healthy data with 300.0 (capped)
        y_train_rul_clean = np.where(np.isnan(y_train_rul), 300.0, y_train_rul)

        dataset = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train_rul_clean).float(),
            torch.from_numpy(y_train_anom).long(),
        )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        optimizer = optim.Adam(self.net.parameters(), lr=self.config.learning_rate)
        loss_fn_rul = nn.MSELoss()
        loss_fn_anom = nn.CrossEntropyLoss()

        val_tensor: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
        if val_data is not None:
            X_v, y_v_rul, y_v_anom = self._unpack_data(val_data)
            if len(X_v) > 0:
                y_v_rul_clean = np.where(np.isnan(y_v_rul), 300.0, y_v_rul)
                val_tensor = (
                    torch.from_numpy(X_v).float().to(self.device),
                    torch.from_numpy(y_v_rul_clean).float().to(self.device),
                    torch.from_numpy(y_v_anom).long().to(self.device),
                )

        best_loss = float("inf")
        patience_counter = 0

        self.net.train()
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            for bx, by_rul, by_anom in loader:
                bx, by_rul, by_anom = bx.to(self.device), by_rul.to(self.device), by_anom.to(self.device)
                optimizer.zero_grad()
                pred_rul, pred_anom = self.net(bx)

                loss = loss_fn_rul(pred_rul / 100.0, by_rul / 100.0) + loss_fn_anom(pred_anom, by_anom)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(bx)

            epoch_loss /= len(X_train)

            # Validation early stopping
            if val_tensor is not None:
                self.net.eval()
                with torch.no_grad():
                    vx, vy_r, vy_a = val_tensor
                    vp_r, vp_a = self.net(vx)
                    v_loss = (loss_fn_rul(vp_r / 100.0, vy_r / 100.0) + loss_fn_anom(vp_a, vy_a)).item()
                self.net.train()

                if v_loss < best_loss - 1e-4:
                    best_loss = v_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL values."""
        return self.predict_rul(X)

    def predict_rul(self, X: np.ndarray) -> np.ndarray:
        """Predict remaining useful life for sequences."""
        if self.net is None:
            raise RuntimeError("Model is not fitted.")
        self.net.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(self._ensure_3d(X)).float().to(self.device)
            rul_pred, _ = self.net(x_t)
            res = rul_pred.cpu().numpy()
        return np.clip(res, 0.0, 300.0)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Predict anomaly probabilities: [P(healthy), P(anomaly)]."""
        if self.net is None:
            raise RuntimeError("Model is not fitted.")
        self.net.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(self._ensure_3d(X)).float().to(self.device)
            _, anom_logits = self.net(x_t)
            probs = torch.softmax(anom_logits, dim=-1).cpu().numpy()
        return probs

    def save(self, path: str | Path) -> None:
        """Serialize LSTM model to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "config": self.config,
            "input_dim": self.input_dim,
            "feature_names": self.feature_names,
            "state_dict": self.net.state_dict() if self.net is not None else None,
        }
        torch.save(bundle, p)
        self._saved_size_bytes = p.stat().st_size

    @classmethod
    def load(cls, path: str | Path) -> "LSTMModel":
        """Load serialized model from disk."""
        p = Path(path)
        bundle = torch.load(p, map_location=torch.device("cpu"), weights_only=False)
        model = cls(config=bundle["config"])
        model.input_dim = bundle["input_dim"]
        model.feature_names = bundle.get("feature_names", [])

        if bundle.get("state_dict") is not None:
            model.net = LSTMNet(
                input_dim=model.input_dim,
                hidden_dim=model.config.hidden_dim,
                num_layers=model.config.num_layers,
                dropout=model.config.dropout,
            )
            model.net.load_state_dict(bundle["state_dict"])
            model.net.eval()

        model._saved_size_bytes = p.stat().st_size
        return model

    def size_bytes(self) -> int:
        """Return serialized model size in bytes."""
        if self._saved_size_bytes > 0:
            return self._saved_size_bytes
        if self.net is None:
            return 0
        param_bytes = sum(p.numel() * p.element_size() for p in self.net.parameters())
        return int(param_bytes)

    def _ensure_3d(self, X: np.ndarray) -> np.ndarray:
        """Ensure input is 3D (batch, seq_len, feat_dim) and matches self.input_dim."""
        arr = np.asarray(X, dtype=np.float32)

        # Handle 1D or 2D inputs
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim == 2:
            arr = np.repeat(arr[:, np.newaxis, :], 20, axis=1)

        # Match input_dim
        if self.input_dim > 0:
            current_dim = arr.shape[-1]
            if current_dim > self.input_dim:
                arr = arr[..., :self.input_dim]
            elif current_dim < self.input_dim:
                pad_width = [(0, 0)] * arr.ndim
                pad_width[-1] = (0, self.input_dim - current_dim)
                arr = np.pad(arr, pad_width)

        return arr

    def _unpack_data(self, data: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unpack sequence tuple (X_seq, y_rul, y_anom)."""
        if isinstance(data, tuple):
            X = data[0]
            y_rul = data[1]
            y_anom = data[2] if len(data) > 2 else (y_rul < 250).astype(int)
            return np.asarray(X, dtype=np.float32), np.asarray(y_rul, dtype=np.float32), np.asarray(y_anom, dtype=int)
        raise ValueError("LSTM requires sequence tensor data tuple (X_seq, y_rul, y_anom).")
