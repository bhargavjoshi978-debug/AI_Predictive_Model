"""Command-Line Interface for pdm: generate, train, evaluate."""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tabulate import tabulate

from pdm.config import load_config
from pdm.evaluation.report import ExperimentRunner
from pdm.features.dataset import build_sequence_tensor, fit_healthy_scaler, split_by_episode
from pdm.models.autoencoder import AutoencoderModel
from pdm.models.lstm import LSTMModel
from pdm.models.random_forest import RandomForestModel
from pdm.models.xgboost_model import XGBoostModel
from pdm.simulator.generator import generate_dataset

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate synthetic robot telemetry episodes and extracted feature table."""
    config = load_config(args.config)
    print(f"=== Generating Synthetic Telemetry Dataset ({config.dataset.n_episodes} episodes) ===")
    t0 = time.perf_counter()

    episodes_file, features_file = generate_dataset(config, output_dir=args.output)
    dur = time.perf_counter() - t0

    df_feat = pd.read_parquet(features_file)
    print(f"[OK] Generated {episodes_file} and {features_file} in {dur:.1f}s.")
    print(f"Total window samples: {len(df_feat)}")
    print("Class distribution:\n" + tabulate(
        df_feat["y_fault"].value_counts(normalize=True).reset_index(),
        headers=["Class", "Proportion"],
        tablefmt="simple",
    ))


def cmd_train(args: argparse.Namespace) -> None:
    """Train all models (Random Forest, XGBoost, Autoencoder, LSTM) on synthetic dataset."""
    config = load_config(args.config)
    data_dir = Path(args.data)
    features_file = data_dir / "features.parquet"

    if not features_file.exists():
        print(f"ERROR: Dataset not found at {features_file}.")
        print("Please run 'python -m pdm.cli generate' first to create the dataset.")
        sys.exit(1)

    print("=== Training Predictive Maintenance Models ===")
    t0 = time.perf_counter()

    df = pd.read_parquet(features_file)
    meta_cols = [
        "episode_id", "robot_id", "t", "joint", "window_idx",
        "health", "damage", "y_fault", "y_anomaly", "y_rul",
        "label_fault", "label_joint", "rul_steps"
    ]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # Split by episode
    train_df, val_df, _ = split_by_episode(
        df,
        train_ratio=config.dataset.split["train"],
        val_ratio=config.dataset.split["val"],
        test_ratio=config.dataset.split["test"],
        seed=config.seed,
    )

    models_dir = Path(args.models)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fit healthy-only scaler
    print("Fitting healthy StandardScaler...")
    scaler = fit_healthy_scaler(train_df, feature_cols, scaler_path=models_dir / "scaler.joblib")
    train_scaled = train_df.copy()
    val_scaled = val_df.copy()
    train_scaled[feature_cols] = scaler.transform(train_df[feature_cols].values)
    val_scaled[feature_cols] = scaler.transform(val_df[feature_cols].values)

    X_train = train_scaled[feature_cols].values
    y_train_cls = train_scaled["y_fault"].values
    y_train_rul = train_scaled["y_rul"].values

    X_val = val_scaled[feature_cols].values
    y_val_cls = val_scaled["y_fault"].values
    y_val_rul = val_scaled["y_rul"].values

    # 2. Train Random Forest
    print("Training Random Forest baseline...")
    rf = RandomForestModel(config.models.random_forest)
    rf.fit((X_train, y_train_cls), feature_names=feature_cols)
    rf.save(models_dir / "random_forest.joblib")

    # 3. Train XGBoost
    print("Training proposed XGBoost model (classification + RUL regression)...")
    xgb = XGBoostModel(config.models.xgboost)
    xgb.fit((X_train, y_train_cls, y_train_rul), val_data=(X_val, y_val_cls, y_val_rul), feature_names=feature_cols)
    xgb.save(models_dir / "xgboost.joblib")

    # 4. Train Autoencoder
    print("Training Autoencoder on healthy windows...")
    ae = AutoencoderModel(config.models.autoencoder)
    ae.fit(train_scaled, val_scaled, feature_names=feature_cols)
    ae.save(models_dir / "autoencoder.pt")

    # 5. Train LSTM
    print("Training sequence LSTM baseline...")
    top30_cols = feature_cols[:30]
    X_train_seq, y_train_s_rul, y_train_s_anom = build_sequence_tensor(train_scaled, top30_cols, seq_len=20)
    X_val_seq, y_val_s_rul, y_val_s_anom = build_sequence_tensor(val_scaled, top30_cols, seq_len=20)
    lstm = LSTMModel(config.models.lstm)
    if len(X_train_seq) > 0:
        lstm.fit((X_train_seq, y_train_s_rul, y_train_s_anom), val_data=(X_val_seq, y_val_s_rul, y_val_s_anom))
        lstm.save(models_dir / "lstm.pt")

    total_dur = time.perf_counter() - t0
    print(f"[OK] All models successfully trained and saved to '{models_dir}' in {total_dur:.1f}s.")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run full evaluation suite (experiments E1-E6) and generate benchmark reports."""
    config = load_config(args.config)
    data_dir = Path(args.data)
    features_file = data_dir / "features.parquet"

    if not features_file.exists():
        print(f"ERROR: Dataset not found at {features_file}.")
        print("Please run 'python -m pdm.cli generate' first to create the dataset.")
        sys.exit(1)

    print("=== Running Evaluation & Benchmark Experiments (E1-E6) ===")
    t0 = time.perf_counter()

    runner = ExperimentRunner(config=config, data_dir=args.data, reports_dir=args.reports)
    results_df = runner.run_all()

    total_dur = time.perf_counter() - t0
    print(f"\n[OK] Evaluation completed in {total_dur:.1f}s.")
    print(f"Reports and figures saved to: '{args.reports}'")
    print("\n" + tabulate(results_df, headers="keys", tablefmt="github", showindex=False))


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="pdm",
        description="AI-Based Predictive Maintenance Simulator for Robotic Equipment",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML file")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: generate
    p_gen = subparsers.add_parser("generate", help="Generate synthetic robot dataset")
    p_gen.add_argument("--output", default="data", help="Output directory for parquet files")

    # Subcommand: train
    p_train = subparsers.add_parser("train", help="Train all predictive maintenance models")
    p_train.add_argument("--data", default="data", help="Data directory containing features.parquet")
    p_train.add_argument("--models", default="models", help="Output directory for saved models")

    # Subcommand: evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run experiments E1-E6 and generate report")
    p_eval.add_argument("--data", default="data", help="Data directory containing features.parquet")
    p_eval.add_argument("--reports", default="reports", help="Output directory for reports and figures")

    parsed = parser.parse_args()
    if parsed.subcommand == "generate":
        cmd_generate(parsed)
    elif parsed.subcommand == "train":
        cmd_train(parsed)
    elif parsed.subcommand == "evaluate":
        cmd_evaluate(parsed)


if __name__ == "__main__":
    main()
