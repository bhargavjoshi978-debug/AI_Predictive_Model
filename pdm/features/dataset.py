"""Dataset management: episode-level splitting, healthy-only scaling, and sequence formatting."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def split_by_episode(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset by episode ID to prevent data leakage across time windows.

    Stratified by fault type across distinct episodes.
    """
    # Extract unique episode metadata (strictly one row per episode_id)
    if "label_fault" in df.columns:
        ep_df = df[["episode_id", "label_fault"]].drop_duplicates("episode_id").reset_index(drop=True)
        fault_col = "label_fault"
    else:
        # Determine the primary fault of the episode (non-none if present)
        def _get_primary_fault(series):
            non_none = [v for v in series if v != "none"]
            return non_none[0] if non_none else "none"
        ep_df = df.groupby("episode_id")["y_fault"].agg(_get_primary_fault).reset_index()
        fault_col = "y_fault"

    rng = np.random.default_rng(seed)
    train_eps: List[int] = []
    val_eps: List[int] = []
    test_eps: List[int] = []

    # Stratified split per fault type
    for fault_type, group in ep_df.groupby(fault_col):
        eps = group["episode_id"].values.copy()
        rng.shuffle(eps)
        n = len(eps)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        # Ensure at least 1 in each split if possible
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))

        train_eps.extend(eps[:n_train])
        val_eps.extend(eps[n_train : n_train + n_val])
        test_eps.extend(eps[n_train + n_val :])

    train_set = set(train_eps)
    val_set = set(val_eps)
    test_set = set(test_eps)

    # Sanity check: strictly disjoint
    assert train_set.isdisjoint(val_set), "Train and validation splits share episodes!"
    assert train_set.isdisjoint(test_set), "Train and test splits share episodes!"
    assert val_set.isdisjoint(test_set), "Validation and test splits share episodes!"

    train_df = df[df["episode_id"].isin(train_set)].copy().reset_index(drop=True)
    val_df = df[df["episode_id"].isin(val_set)].copy().reset_index(drop=True)
    test_df = df[df["episode_id"].isin(test_set)].copy().reset_index(drop=True)

    return train_df, val_df, test_df


def fit_healthy_scaler(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    scaler_path: Optional[str | Path] = None,
) -> StandardScaler:
    """Fit StandardScaler ONLY on healthy training windows and persist it."""
    fault_col = "y_fault" if "y_fault" in train_df.columns else "label_fault"
    healthy_mask = (train_df[fault_col] == "none")
    if "health" in train_df.columns:
        healthy_mask = healthy_mask & (train_df["health"] >= 0.95)

    healthy_data = train_df.loc[healthy_mask, feature_cols].values
    if len(healthy_data) == 0:
        # Fallback to all training data with warning if no healthy rows found
        healthy_data = train_df[feature_cols].values

    scaler = StandardScaler()
    scaler.fit(healthy_data)

    if scaler_path is not None:
        p = Path(scaler_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, p)

    return scaler


def build_sequence_tensor(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 3D tensor (n_sequences, seq_len, n_features) per episode.

    Returns:
    - X_seq: np.ndarray shape (N, seq_len, len(feature_cols))
    - y_rul: np.ndarray shape (N,)
    - y_anomaly: np.ndarray shape (N,)
    """
    sequences: List[np.ndarray] = []
    ruls: List[float] = []
    anomalies: List[int] = []

    rul_col = "y_rul" if "y_rul" in df.columns else "rul_steps"
    anom_col = "y_anomaly" if "y_anomaly" in df.columns else "label_fault"

    for (ep_id, j_id), group in df.groupby(["episode_id", "joint"]):
        group = group.sort_values("t")
        vals = group[feature_cols].values
        r_vals = group[rul_col].values if rul_col in group.columns else np.zeros(len(group))
        if anom_col == "y_anomaly":
            a_vals = group[anom_col].values
        else:
            a_vals = (group[anom_col] != "none").astype(int).values

        n_rows = len(group)
        if n_rows < seq_len:
            continue

        for i in range(seq_len - 1, n_rows):
            seq = vals[i - seq_len + 1 : i + 1]
            sequences.append(seq)
            ruls.append(r_vals[i])
            anomalies.append(a_vals[i])

    if not sequences:
        return (
            np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    return (
        np.array(sequences, dtype=np.float32),
        np.array(ruls, dtype=np.float32),
        np.array(anomalies, dtype=np.int32),
    )
