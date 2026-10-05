"""Train and evaluate six classifiers.

Protocol
  1. Stratified 80/20 split (seed 42).
  2. 5-fold stratified CV on the TRAINING part only, for all six models  -> model selection.
  3. Fit each model on the full training part, evaluate ONCE on the held-out test part.
"""
import argparse
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate

from .data_utils import RANDOM_STATE, load_csv, split_data
from .models import build_models

warnings.filterwarnings("ignore")


def evaluate(name, model, X_train, y_train, X_test, y_test, fig_dir):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, prob),
        "PR_AUC": average_precision_score(y_test, prob),
        "Log_Loss": log_loss(y_test, np.clip(prob, 1e-7, 1 - 1e-7)),
        "Fit_Time_s": fit_time,
    }
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"{name}\nConfusion Matrix", fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["Non-bankrupt", "Bankrupt"])
    ax.set_yticks([0, 1], ["Non-bankrupt", "Bankrupt"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{name}_confusion_matrix.png", dpi=180)
    plt.close(fig)
    return metrics


def plot_comparison(results_df, fig_dir):
    df = results_df.sort_values("F1")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(y - 0.2, df["F1"], 0.4, label="F1", color="#1f4e79")
    ax.barh(y + 0.2, df["PR_AUC"], 0.4, label="PR-AUC", color="#e07b39")
    ax.set_yticks(y, df["Model"])
    ax.set_xlim(0, 1)
    ax.set_title("Test-set F1 and PR-AUC (bankrupt class)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "model_comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/data.csv")
    parser.add_argument("--output", default=".")
    args = parser.parse_args()
    root = Path(args.output)
    (root / "models").mkdir(exist_ok=True)
    (root / "figures").mkdir(exist_ok=True)

    X, y, df = load_csv(args.data)
    print(f"Dataset shape: {df.shape} | features used: {X.shape[1]}")
    print("Class distribution:")
    print(y.value_counts().sort_index())
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {len(X_train)} rows ({int(y_train.sum())} bankrupt) | "
          f"Test: {len(X_test)} rows ({int(y_test.sum())} bankrupt)")
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    models = build_models(pos_weight)

    # 1) cross-validation on the training part only (used for model selection)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_rows = []
    for name, model in models.items():
        print(f"\nCross-validating {name} ...")
        s = cross_validate(model, X_train, y_train, cv=cv, n_jobs=1,
                           scoring={"f1": "f1", "pr_auc": "average_precision"})
        cv_rows.append({"Model": name,
                        "CV_F1_mean": s["test_f1"].mean(), "CV_F1_std": s["test_f1"].std(),
                        "CV_PR_AUC_mean": s["test_pr_auc"].mean(),
                        "CV_PR_AUC_std": s["test_pr_auc"].std()})
    cv_df = pd.DataFrame(cv_rows).sort_values("CV_F1_mean", ascending=False)
    cv_df.to_csv(root / "cross_validation_results.csv", index=False)

    # 2) final fit on the full training part, single evaluation on the test part
    results = []
    for name, model in models.items():
        print(f"Training + testing {name} ...")
        results.append(evaluate(name, model, X_train, y_train, X_test, y_test, root / "figures"))
        joblib.dump({"model": model, "feature_names": list(X.columns)},
                    root / "models" / f"{name}.joblib")
    results_df = pd.DataFrame(results).sort_values(["F1", "PR_AUC"], ascending=False)
    results_df.to_csv(root / "model_results.csv", index=False)
    plot_comparison(results_df, root / "figures")

    fmt = lambda v: f"{v:.4f}"
    print("\n=== 5-fold CV on training data (used to choose the model) ===")
    print(cv_df.to_string(index=False, float_format=fmt))
    print(f"Model selected by CV F1: {cv_df.iloc[0]['Model']}")
    print("\n=== Held-out test-set results ===")
    print(results_df.to_string(index=False, float_format=fmt))


if __name__ == "__main__":
    main()
