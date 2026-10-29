"""Tests for dataset splitting to verify no episode leakage."""

import numpy as np
import pandas as pd
import pytest
from pdm.features.dataset import split_by_episode


def test_split_by_episode_no_leakage() -> None:
    """Verify that no episode_id appears in more than one split."""
    # Create mock dataset with 30 episodes, 5 steps each, multiple fault classes
    records = []
    faults = ["none", "bearing", "gear_wear", "motor_overheat", "cable_fault"]
    for ep_id in range(30):
        fault = faults[ep_id % len(faults)]
        for t in range(5):
            records.append({
                "episode_id": ep_id,
                "robot_id": 1,
                "t": t,
                "joint": 0,
                "label_fault": fault,
                "feat_1": np.random.randn(),
                "feat_2": np.random.randn(),
            })

    df = pd.DataFrame(records)

    train_df, val_df, test_df = split_by_episode(
        df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    train_eps = set(train_df["episode_id"].unique())
    val_eps = set(val_df["episode_id"].unique())
    test_eps = set(test_df["episode_id"].unique())

    # Check non-empty
    assert len(train_eps) > 0
    assert len(val_eps) > 0
    assert len(test_eps) > 0

    # Check complete union equals original
    assert (train_eps | val_eps | test_eps) == set(df["episode_id"].unique())

    # Check disjointness (strict zero leakage)
    assert train_eps.isdisjoint(val_eps), f"Leakage between train and val: {train_eps & val_eps}"
    assert train_eps.isdisjoint(test_eps), f"Leakage between train and test: {train_eps & test_eps}"
    assert val_eps.isdisjoint(test_eps), f"Leakage between val and test: {val_eps & test_eps}"
