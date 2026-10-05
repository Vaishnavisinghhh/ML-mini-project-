"""Live demo: predict bankruptcy for companies from the HELD-OUT test split (never seen in training)."""
import argparse
from pathlib import Path

import joblib

from .data_utils import load_csv, split_data
from .models import OversampledMLP  # noqa: F401  (needed so joblib can unpickle NeuralNetwork.joblib)


def show(model, X_test, y_test, pos):
    row = X_test.iloc[[pos]]
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0, 1])
    actual = int(y_test.iloc[pos])
    verdict = "CORRECT" if pred == actual else "WRONG"
    print(f"  Test row #{pos} (original dataset row {X_test.index[pos]})")
    print(f"    Predicted: {pred} ({'BANKRUPT' if pred else 'NON-BANKRUPT'})   "
          f"P(bankrupt) = {prob:.4f}")
    print(f"    Actual:    {actual} ({'BANKRUPT' if actual else 'NON-BANKRUPT'})   -> {verdict}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/data.csv")
    parser.add_argument("--model", default="models/LightGradientBoostingMachine.joblib")
    parser.add_argument("--index", type=int, default=None,
                        help="position inside the test split (0..n-1); default shows one bankrupt "
                             "and one non-bankrupt company")
    args = parser.parse_args()
    if not Path(args.model).exists():
        raise SystemExit(f"Model file not found: {args.model}. Run: python -m src.train")
    X, y, _ = load_csv(args.data)
    _, X_test, _, y_test = split_data(X, y)
    bundle = joblib.load(args.model)
    model, expected = bundle["model"], bundle["feature_names"]
    X_test = X_test[expected]

    print("Corporate Bankruptcy Prediction Demo")
    print("------------------------------------")
    print(f"Model: {Path(args.model).stem} | held-out test companies: {len(X_test)}")
    if args.index is not None:
        if not 0 <= args.index < len(X_test):
            raise SystemExit(f"--index must be between 0 and {len(X_test) - 1}.")
        show(model, X_test, y_test, args.index)
    else:
        show(model, X_test, y_test, int((y_test.values == 1).argmax()))
        show(model, X_test, y_test, int((y_test.values == 0).argmax()))


if __name__ == "__main__":
    main()
