"""Live streaming inference detector and alert management."""

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np

from pdm.config import AlertsConfig, Config, load_config
from pdm.features.extract import JointFeatureExtractor
from pdm.models.autoencoder import AutoencoderModel
from pdm.models.lstm import LSTMModel
from pdm.models.random_forest import RandomForestModel
from pdm.models.xgboost_model import CLASS_NAMES, XGBoostModel
from pdm.simulator.robot import JointObservation, Observation
from pdm.streaming.alerts import Alert, AlertStateMachine


@dataclass
class Prediction:
    """Model inference prediction output for a single joint step."""
    t: int
    joint: int
    fault_probs: Dict[str, float]
    anomaly_score: float
    rul_estimate: Optional[float]
    latency_ms: float
    model: str


class LiveDetector:
    """Maintains feature extraction buffers, executes model inference, and tracks joint alerts."""

    def __init__(
        self,
        config: Optional[Config] = None,
        models_dir: str | Path = "models",
        active_models: Optional[List[str]] = None,
    ) -> None:
        self.config = config or load_config()
        self.models_dir = Path(models_dir)
        self.active_models = active_models or ["xgboost", "autoencoder"]

        # Models storage
        self.xgb: Optional[XGBoostModel] = None
        self.rf: Optional[RandomForestModel] = None
        self.ae: Optional[AutoencoderModel] = None
        self.lstm: Optional[LSTMModel] = None
        self.scaler: Optional[Any] = None

        self._load_available_models()

        # Per-joint extractors and alert state machines
        n_joints = self.config.simulator.n_joints
        self.extractors = [
            JointFeatureExtractor(
                sample_rate_hz=self.config.simulator.sample_rate_hz,
                n_bands=self.config.features.fft_bands,
                trend_window=self.config.features.trend_windows,
            )
            for _ in range(n_joints)
        ]
        self.alert_machines = [
            AlertStateMachine(self.config.alerts)
            for _ in range(n_joints)
        ]

    def _load_available_models(self) -> None:
        """Load trained models from disk with graceful fallback."""
        scaler_p = self.models_dir / "scaler.joblib"
        if scaler_p.exists():
            try:
                self.scaler = joblib.load(scaler_p)
            except Exception:
                self.scaler = None

        xgb_p = self.models_dir / "xgboost.joblib"
        if xgb_p.exists():
            try:
                self.xgb = XGBoostModel.load(xgb_p)
            except Exception:
                self.xgb = None

        rf_p = self.models_dir / "random_forest.joblib"
        if rf_p.exists():
            try:
                self.rf = RandomForestModel.load(rf_p)
            except Exception:
                self.rf = None

        ae_p = self.models_dir / "autoencoder.pt"
        if ae_p.exists():
            try:
                self.ae = AutoencoderModel.load(ae_p)
            except Exception:
                self.ae = None

        lstm_p = self.models_dir / "lstm.pt"
        if lstm_p.exists():
            try:
                self.lstm = LSTMModel.load(lstm_p)
            except Exception:
                self.lstm = None

    def process_step(self, obs: Observation) -> Tuple[List[Prediction], List[Alert], float]:
        """Process observation across all joints, perform model inference, and return predictions & alerts."""
        predictions: List[Prediction] = []
        alerts: List[Alert] = []
        highest_fused_score = 0.0

        for j_obs in obs.joints:
            j_idx = j_obs.joint
            pred, alert = self.process_joint(obs.t, j_obs)
            predictions.append(pred)
            alerts.append(alert)
            if alert.fused_score > highest_fused_score:
                highest_fused_score = alert.fused_score

        return predictions, alerts, highest_fused_score

    def process_joint(self, t: int, j_obs: JointObservation) -> Tuple[Prediction, Alert]:
        """Extract features, run inference, calculate fused risk score, and update alert state machine."""
        j_idx = j_obs.joint
        extractor = self.extractors[j_idx]
        sm = self.alert_machines[j_idx]

        # 1. Feature extraction
        f_rot = self.config.simulator.f_rot_base
        raw_feats = extractor.extract_step(
            signals=j_obs.signals,
            slow=j_obs.slow,
            f_rot=f_rot,
        )

        # Scale features if scaler available
        feat_keys = list(raw_feats.keys())
        feat_vals = np.array([raw_feats[k] for k in feat_keys], dtype=np.float32).reshape(1, -1)
        if self.scaler is not None and hasattr(self.scaler, "transform"):
            try:
                feat_scaled = self.scaler.transform(feat_vals)
            except Exception:
                feat_scaled = feat_vals
        else:
            feat_scaled = feat_vals

        # 2. Run active models
        t0 = time.perf_counter()
        fault_probs: Dict[str, float] = {name: (1.0 if name == "none" else 0.0) for name in CLASS_NAMES}
        anomaly_score = 0.0
        rul_estimate: Optional[float] = None
        model_name = "fallback"

        # XGBoost inference
        if "xgboost" in self.active_models and self.xgb is not None:
            model_name = "xgboost"
            probs = self.xgb.predict_proba(feat_scaled)
            if probs is not None:
                for idx, c in enumerate(CLASS_NAMES):
                    fault_probs[c] = float(probs[0, idx])
            rul_estimate = float(self.xgb.predict_rul(feat_scaled)[0])

        # Random Forest inference (if active and XGBoost not overriding)
        elif "random_forest" in self.active_models and self.rf is not None:
            model_name = "random_forest"
            probs = self.rf.predict_proba(feat_scaled)
            if probs is not None:
                for idx, c in enumerate(CLASS_NAMES):
                    fault_probs[c] = float(probs[0, idx])

        # Autoencoder anomaly score
        if "autoencoder" in self.active_models and self.ae is not None:
            anomaly_score = float(self.ae.compute_anomaly_score(feat_scaled)[0])
            if model_name == "fallback":
                model_name = "autoencoder"

        # LSTM sequence inference (if active and loaded)
        if "lstm" in self.active_models and self.lstm is not None:
            rul_lstm = float(self.lstm.predict_rul(feat_scaled)[0])
            if rul_estimate is None:
                rul_estimate = rul_lstm
            if model_name == "fallback":
                model_name = "lstm"

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # 3. Fused risk score R in [0, 1] from DESIGN.md §6
        p_none = fault_probs.get("none", 1.0)
        p_fault = 1.0 - p_none

        anom_term = min(1.0, anomaly_score / 2.0)

        fault_suspected = 1.0 if (p_fault > 0.4 or anomaly_score >= 1.0) else 0.0
        rul_cap = float(self.config.dataset.degradation.rul_cap)
        if rul_estimate is not None:
            rul_term = (1.0 - min(1.0, rul_estimate / rul_cap)) * fault_suspected
        else:
            rul_term = 0.0

        fused_score = float(max(p_fault, anom_term, rul_term))
        fused_score = float(np.clip(fused_score, 0.0, 1.0))

        # Predicted fault label
        pred_fault = max(fault_probs.items(), key=lambda x: x[1])[0]

        # 4. Update alert state machine
        alert = sm.update(
            t=t,
            joint=j_idx,
            fused_score=fused_score,
            fault_type=pred_fault,
            rul_estimate=rul_estimate,
        )

        prediction = Prediction(
            t=t,
            joint=j_idx,
            fault_probs=fault_probs,
            anomaly_score=anomaly_score,
            rul_estimate=rul_estimate,
            latency_ms=latency_ms,
            model=model_name,
        )

        return prediction, alert
