"""PyTorch CPU Dense Autoencoder for zero-day anomaly detection."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from pdm.config import AutoencoderConfig
from pdm.models.base import BaseModel, register_model


class AutoencoderNet(nn.Module):
    """Dense autoencoder: input -> 64 -> 32 -> 8 -> 32 -> 64 -> input."""

    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dims: Optional[List[int]] = None) -> None:
        super().__init__()
        h = hidden_dims or [64, 32]

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h[0]),
            nn.ReLU(),
            nn.Linear(h[0], h[1]),
            nn.ReLU(),
            nn.Linear(h[1], latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h[1]),
            nn.ReLU(),
            nn.Linear(h[1], h[0]),
            nn.ReLU(),
            nn.Linear(h[0], input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


@register_model("autoencoder")
class AutoencoderModel(BaseModel):
    """Unsupervised Autoencoder for zero-day anomaly detection trained only on healthy windows."""

    name: str = "autoencoder"
    task: Literal["classification", "anomaly", "rul"] = "anomaly"

    def __init__(self, config: Optional[AutoencoderConfig] = None) -> None:
        self.config = config or AutoencoderConfig()
        self.net: Optional[AutoencoderNet] = None
        self.input_dim: int = 0
        self.threshold: float = 1.0
        self.norm_factor: float = 1.0
        self.feature_names: List[str] = []
        self._saved_size_bytes: int = 0
        self.device = torch.device("cpu")

    def fit(
        self,
        train_data: Any,
        val_data: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """Fit autoencoder on healthy training windows only, with early stopping."""
        if feature_names is not None:
            self.feature_names = list(feature_names)

        X_train_healthy = self._extract_healthy(train_data)
        if len(X_train_healthy) == 0:
            raise ValueError("No healthy data provided to train Autoencoder.")

        self.input_dim = X_train_healthy.shape[1]
        self.net = AutoencoderNet(
            input_dim=self.input_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims,
        ).to(self.device)

        dataset = TensorDataset(torch.from_numpy(X_train_healthy).float())
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        optimizer = optim.Adam(self.net.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        # Validation set for early stopping and threshold calculation
        val_tensor: Optional[torch.Tensor] = None
        if val_data is not None:
            X_val_healthy = self._extract_healthy(val_data)
            if len(X_val_healthy) > 0:
                val_tensor = torch.from_numpy(X_val_healthy).float().to(self.device)

        best_loss = float("inf")
        patience_counter = 0

        self.net.train()
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            for batch in loader:
                x_b = batch[0].to(self.device)
                optimizer.zero_grad()
                out = self.net(x_b)
                loss = criterion(out, x_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(x_b)

            epoch_loss /= len(X_train_healthy)

            # Evaluate on validation
            if val_tensor is not None:
                self.net.eval()
                with torch.no_grad():
                    val_out = self.net(val_tensor)
                    val_loss = criterion(val_out, val_tensor).item()
                self.net.train()

                if val_loss < best_loss - 1e-5:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        break

        # Compute threshold and normalization factor on healthy validation set (or train set)
        self.net.eval()
        calib_data = val_tensor if val_tensor is not None else torch.from_numpy(X_train_healthy).float()
        with torch.no_grad():
            calib_out = self.net(calib_data)
            mse_per_sample = torch.mean((calib_out - calib_data) ** 2, dim=1).cpu().numpy()

        p99 = float(np.percentile(mse_per_sample, self.config.threshold_percentile))
        self.norm_factor = max(1e-8, p99)
        self.threshold = 1.0  # normalized threshold

    def compute_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Compute normalized reconstruction MSE score."""
        if self.net is None:
            raise RuntimeError("Model is not fitted.")
        self.net.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
            out = self.net(x_t)
            mse = torch.mean((out - x_t) ** 2, dim=1).cpu().numpy()
        return mse / self.norm_factor

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary anomaly label (1 = anomaly, 0 = healthy)."""
        scores = self.compute_anomaly_score(X)
        return (scores >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """Return proxy probability array [P(healthy), P(anomaly)]."""
        scores = self.compute_anomaly_score(X)
        p_anom = np.clip(scores / 2.0, 0.0, 1.0)
        p_health = 1.0 - p_anom
        return np.stack([p_health, p_anom], axis=1)

    def save(self, path: str | Path) -> None:
        """Serialize PyTorch autoencoder to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "config": self.config,
            "input_dim": self.input_dim,
            "threshold": self.threshold,
            "norm_factor": self.norm_factor,
            "feature_names": self.feature_names,
            "state_dict": self.net.state_dict() if self.net is not None else None,
        }
        torch.save(bundle, p)
        self._saved_size_bytes = p.stat().st_size

    @classmethod
    def load(cls, path: str | Path) -> "AutoencoderModel":
        """Load serialized model from disk."""
        p = Path(path)
        bundle = torch.load(p, map_location=torch.device("cpu"), weights_only=False)
        model = cls(config=bundle["config"])
        model.input_dim = bundle["input_dim"]
        model.threshold = bundle["threshold"]
        model.norm_factor = bundle["norm_factor"]
        model.feature_names = bundle.get("feature_names", [])

        if bundle.get("state_dict") is not None:
            model.net = AutoencoderNet(
                input_dim=model.input_dim,
                latent_dim=model.config.latent_dim,
                hidden_dims=model.config.hidden_dims,
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

    def _extract_healthy(self, data: Any) -> np.ndarray:
        """Extract healthy rows from dataset."""
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            label_col = "y_fault" if "y_fault" in data.columns else "label_fault"
            healthy_mask = (data[label_col] == "none")
            if "health" in data.columns:
                healthy_mask = healthy_mask & (data["health"] >= 0.95)
            meta_cols = ["episode_id", "robot_id", "t", "joint", "window_idx", "health", "damage", "y_fault", "y_anomaly", "y_rul", "label_fault", "label_joint", "rul_steps"]
            f_cols = [c for c in data.columns if c not in meta_cols]
            return data.loc[healthy_mask, f_cols].values.astype(np.float32)
        elif isinstance(data, tuple):
            X = data[0]
            y = data[1]
            healthy_mask = (y == 0) | (y == "none")
            return X[healthy_mask].astype(np.float32)
        return np.asarray(data, dtype=np.float32)
