# Corporate Bankruptcy Prediction (Kaggle Company Bankruptcy dataset)

**Course:** UE24CS352A Machine Learning, Mini-Project
**Team:** Vaishnavi Singh (PES2UG24AM177), Yashavantha G (PES2UG24AM189)

## Dataset
Kaggle *Company Bankruptcy Prediction* (Taiwan Economic Journal data):
https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction
6,819 companies, 95 predictors (94 after dropping the constant `Net Income Flag`), target `Bankrupt?`
(220 bankrupt = 3.23%, 6,599 non-bankrupt), no missing values. Stored at `data/data.csv`.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run (from the repository root)
```bash
python -m src.eda   --data data/data.csv     # class balance + feature correlations -> figures/
python -m src.train --data data/data.csv     # CV, test metrics, models/, figures/, CSVs
python -m src.demo  --data data/data.csv     # predictions on held-out test companies
python -m src.demo  --model models/SVM.joblib --index 7   # a specific test row
```

## Method
1. Stratified 80/20 split (seed 42), shared by all scripts via `src/data_utils.py`.
2. Median imputation and (for LR, SVM, MLP) standardisation inside each Pipeline.
3. Imbalance handling: class weights (LR, SVM, LightGBM), `scale_pos_weight` (XGBoost),
   minority oversampling inside `fit` (MLP). Naive Bayes has no balancing mechanism.
4. 5-fold stratified CV on the training split for all six models (model selection).
5. One evaluation on the held-out test split; confusion matrices and metrics saved.

## Results (held-out test set, 1,364 companies, 44 bankrupt)
CV F1 is the 5-fold mean on the training split (used to pick the model); the other columns are test-set metrics for the bankrupt class (accuracy is overall).

| Model | CV F1 | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|---|
| LightGBM | 0.443 | 0.972 | 0.583 | 0.477 | 0.525 | 0.553 |
| XGBoost | 0.430 | 0.955 | 0.390 | 0.727 | 0.508 | 0.487 |
| Neural Network | 0.368 | 0.956 | 0.357 | 0.455 | 0.400 | 0.331 |
| SVM | 0.289 | 0.900 | 0.188 | 0.636 | 0.290 | 0.344 |
| Logistic Regression | 0.278 | 0.878 | 0.186 | 0.818 | 0.303 | 0.319 |
| Naive Bayes | 0.062 | 0.062 | 0.032 | 0.955 | 0.062 | 0.052 |

LightGBM is selected by cross-validation. Full numbers: `model_results.csv`, `cross_validation_results.csv`. Re-running on another machine can change XGBoost slightly; the CSVs in the repo are the source of record.

## Model names
Report and slides use short names; files, CSVs and figures use the code names from `src/models.py`:
LightGBM = `LightGradientBoostingMachine`, XGBoost = `ExtremeGradientBoosting`,
Neural Network = `NeuralNetwork` (MLP), SVM = `SVM`, Logistic Regression = `LogisticRegression`, Naive Bayes = `NaiveBayes`.

## Troubleshooting
- Run every command from the repository root (they use `python -m src.<script>`).
- If `pip install -r requirements.txt` fails on a pinned version, install the unpinned packages instead:
  `pip install numpy pandas scipy scikit-learn matplotlib xgboost lightgbm joblib`.
  The saved `models/*.joblib` files were made with the pinned versions; if loading them fails, regenerate them with `python -m src.train`.
- `python -m src.demo` needs `models/` to exist; it prints a clear message if it does not.

## Structure
`src/` code, `data/` dataset, `models/` fitted pipelines, `figures/` plots, `reports/` write-ups (one per member), slide deck, speaker notes and viva Q&A.
