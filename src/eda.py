"""Exploratory data analysis: class balance and the features most correlated with bankruptcy."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_utils import TARGET, load_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/data.csv")
    parser.add_argument("--output", default="figures")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(exist_ok=True)
    X, y, df = load_csv(args.data)

    counts = y.value_counts().sort_index()
    print(f"Rows: {len(df)} | predictors used: {X.shape[1]} | missing cells: {int(df.isna().sum().sum())}")
    print(f"Non-bankrupt: {counts[0]} | Bankrupt: {counts[1]} ({counts[1] / len(y):.2%})")

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    bars = ax.bar(["Non-bankrupt", "Bankrupt"], counts.values, color=["#1f4e79", "#e07b39"])
    ax.bar_label(bars)
    ax.set_title("Class distribution (3.23% bankrupt)")
    ax.set_ylabel("Companies")
    fig.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=180)
    plt.close(fig)

    corr = X.corrwith(y).dropna()
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(10).iloc[::-1]
    print("\nTop 10 features by |correlation| with", TARGET)
    print(top.iloc[::-1].round(3).to_string())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh([n[:38] for n in top.index], top.values,
            color=["#e07b39" if v > 0 else "#1f4e79" for v in top.values])
    ax.set_title("Top 10 features by correlation with bankruptcy")
    ax.set_xlabel("Pearson correlation")
    fig.tight_layout()
    fig.savefig(out / "top_feature_correlations.png", dpi=180)
    plt.close(fig)

    c = X.corr().abs().values
    print(f"\nFeature pairs with |r| > 0.95: {int(((c > 0.95).sum() - X.shape[1]) // 2)}")


if __name__ == "__main__":
    main()
