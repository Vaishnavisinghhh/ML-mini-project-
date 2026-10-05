"""Model definitions. Every model is a Pipeline so preprocessing is fitted on training data only."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .data_utils import RANDOM_STATE


class OversampledMLP(ClassifierMixin, BaseEstimator):
    """MLP that randomly oversamples the minority class (to 1:1) inside fit().

    scikit-learn's MLPClassifier has no class_weight option. Doing the oversampling inside
    fit() means it only ever touches the training part of each cross-validation fold,
    so there is no leakage into the validation fold or the test set.
    """

    def __init__(self, hidden_layer_sizes=(32, 16), alpha=1e-1, max_iter=60,
                 random_state=RANDOM_STATE):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        extra = rng.choice(pos, size=max(len(neg) - len(pos), 0), replace=True)
        idx = rng.permutation(np.concatenate([np.arange(len(y)), extra]))
        self.mlp_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha,
            max_iter=self.max_iter, random_state=self.random_state,
        ).fit(X[idx], y[idx])
        self.classes_ = self.mlp_.classes_
        return self

    def predict_proba(self, X):
        return self.mlp_.predict_proba(np.asarray(X))

    def predict(self, X):
        return self.mlp_.predict(np.asarray(X))


def _prep(scale):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return steps


def build_models(pos_weight):
    """pos_weight = (#non-bankrupt / #bankrupt) in the training set, used by XGBoost."""
    return {
        "LogisticRegression": Pipeline(_prep(True) + [("model", LogisticRegression(
            max_iter=3000, class_weight="balanced", C=1.0, random_state=RANDOM_STATE))]),
        "SVM": Pipeline(_prep(True) + [("model", SVC(
            kernel="rbf", probability=True, class_weight="balanced", C=1.0,
            random_state=RANDOM_STATE))]),
        "NeuralNetwork": Pipeline(_prep(True) + [("model", OversampledMLP())]),
        "NaiveBayes": Pipeline(_prep(False) + [("model", GaussianNB())]),
        "ExtremeGradientBoosting": Pipeline(_prep(False) + [("model", XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, objective="binary:logistic", eval_metric="logloss",
            scale_pos_weight=pos_weight, random_state=RANDOM_STATE, n_jobs=4))]),
        "LightGradientBoostingMachine": Pipeline(_prep(False) + [("model", LGBMClassifier(
            n_estimators=300, num_leaves=31, learning_rate=0.05, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=4, verbosity=-1))]),
    }
