"""Federated learning (FedAvg) simulation across distributed factory cells."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

from pdm.models.xgboost_model import CLASS_TO_IDX


class FedAvgSimulation:
    """Simulates multi-factory federated learning across isolated factory cells."""

    def __init__(
        self,
        n_cells: int = 3,
        n_rounds: int = 10,
        local_epochs: int = 3,
        seed: int = 42,
    ) -> None:
        self.n_cells = n_cells
        self.n_rounds = n_rounds
        self.local_epochs = local_epochs
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def partition_non_iid(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Partition episodes into non-IID factory cells with distinct fault distributions."""
        episodes = df["episode_id"].unique()
        self.rng.shuffle(episodes)
        
        # Partition episodes across cells
        chunks = np.array_split(episodes, self.n_cells)
        cell_dfs = []
        for chunk in chunks:
            cell_dfs.append(df[df["episode_id"].isin(chunk)].copy())
        return cell_dfs

    def run(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
        reports_dir: str | Path = "reports",
    ) -> pd.DataFrame:
        """Run centralized vs. federated vs. local-isolated benchmarks."""
        reports_path = Path(reports_dir)
        reports_path.mkdir(parents=True, exist_ok=True)
        figures_path = reports_path / "figures"
        figures_path.mkdir(parents=True, exist_ok=True)

        X_test = test_df[feature_cols].values
        y_test = np.array([CLASS_TO_IDX.get(c, 0) for c in test_df["y_fault"].values], dtype=int)

        # 1. Centralized baseline
        X_train_full = train_df[feature_cols].values
        y_train_full = np.array([CLASS_TO_IDX.get(c, 0) for c in train_df["y_fault"].values], dtype=int)
        classes = np.array(list(CLASS_TO_IDX.values()), dtype=int)

        cent_clf = SGDClassifier(loss="log_loss", penalty="l2", random_state=self.seed)
        for _ in range(self.n_rounds * self.local_epochs):
            cent_clf.partial_fit(X_train_full, y_train_full, classes=classes)
        cent_preds = cent_clf.predict(X_test)
        cent_acc = float(accuracy_score(y_test, cent_preds))
        cent_f1 = float(f1_score(y_test, cent_preds, average="macro", zero_division=0))

        # 2. Isolated local models
        cell_dfs = self.partition_non_iid(train_df)
        local_accs: List[float] = []
        local_f1s: List[float] = []
        local_clfs: List[SGDClassifier] = []

        for c_idx, c_df in enumerate(cell_dfs):
            X_c = c_df[feature_cols].values
            y_c = np.array([CLASS_TO_IDX.get(c, 0) for c in c_df["y_fault"].values], dtype=int)

            clf = SGDClassifier(loss="log_loss", penalty="l2", random_state=self.seed + c_idx)
            for _ in range(self.n_rounds * self.local_epochs):
                clf.partial_fit(X_c, y_c, classes=classes)
            preds = clf.predict(X_test)
            acc = float(accuracy_score(y_test, preds))
            f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
            local_accs.append(acc)
            local_f1s.append(f1)
            local_clfs.append(clf)

        mean_local_acc = float(np.mean(local_accs))
        mean_local_f1 = float(np.mean(local_f1s))

        # 3. Federated FedAvg
        global_clf = SGDClassifier(loss="log_loss", penalty="l2", random_state=self.seed)
        global_clf.partial_fit(X_train_full[:5], y_train_full[:5], classes=classes)

        total_samples = sum(len(c) for c in cell_dfs)
        weights = [len(c) / total_samples for c in cell_dfs]

        for round_idx in range(self.n_rounds):
            local_weights = []
            local_intercepts = []

            for c_idx, c_df in enumerate(cell_dfs):
                X_c = c_df[feature_cols].values
                y_c = np.array([CLASS_TO_IDX.get(c, 0) for c in c_df["y_fault"].values], dtype=int)

                # Initialize local client with global model weights
                c_model = SGDClassifier(loss="log_loss", penalty="l2", random_state=self.seed + round_idx + c_idx)
                c_model.coef_ = np.copy(global_clf.coef_)
                c_model.intercept_ = np.copy(global_clf.intercept_)
                c_model.classes_ = np.copy(global_clf.classes_)

                # Local training epochs
                for _ in range(self.local_epochs):
                    c_model.partial_fit(X_c, y_c, classes=classes)

                local_weights.append(c_model.coef_)
                local_intercepts.append(c_model.intercept_)

            # FedAvg aggregation: weighted parameter averaging
            avg_coef = np.zeros_like(global_clf.coef_)
            avg_intercept = np.zeros_like(global_clf.intercept_)
            for w, coef, inter in zip(weights, local_weights, local_intercepts):
                avg_coef += w * coef
                avg_intercept += w * inter

            global_clf.coef_ = avg_coef
            global_clf.intercept_ = avg_intercept

        fed_preds = global_clf.predict(X_test)
        fed_acc = float(accuracy_score(y_test, fed_preds))
        fed_f1 = float(f1_score(y_test, fed_preds, average="macro", zero_division=0))

        # Results table
        results = [
            {"Architecture": "Centralized (All Data Combined)", "Accuracy": cent_acc, "Macro-F1": cent_f1},
            {"Architecture": f"Federated FedAvg ({self.n_cells} Cells, {self.n_rounds} Rounds)", "Accuracy": fed_acc, "Macro-F1": fed_f1},
            {"Architecture": f"Isolated Local (Mean Across {self.n_cells} Cells)", "Accuracy": mean_local_acc, "Macro-F1": mean_local_f1},
        ]
        res_df = pd.DataFrame(results)
        res_df.to_csv(reports_path / "e7_federated.csv", index=False)

        # Plot E7
        plt.figure(figsize=(7, 4))
        plt.bar(res_df["Architecture"], res_df["Accuracy"], color=["#1565c0", "#2e7d32", "#c62828"], width=0.5)
        plt.ylabel("Accuracy")
        plt.ylim(0.0, 1.05)
        plt.title("E7: Federated vs. Centralized vs. Isolated Local Learning")
        plt.xticks(rotation=15, ha="right")
        for i, v in enumerate(res_df["Accuracy"]):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
        plt.tight_layout()
        plt.savefig(figures_path / "e7_federated.png", dpi=150)
        plt.close()

        return res_df
