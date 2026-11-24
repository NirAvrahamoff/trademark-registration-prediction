"""Preprocessing pipeline, models and search spaces for the classification part."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42


def make_preprocessor(numeric, categorical, text=None, svd_components=50,
                      attorney_onehot=False, max_attorneys=100):
    """Scale numeric columns, one-hot encode categories, TF-IDF + SVD on the goods text.

    attorney_onehot=True adds one-hot columns for the most frequent attorneys
    (used in the encoding experiment instead of the failure-rate feature).
    """
    parts = [
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ]
    if text is not None:
        parts.append(("text", Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", min_df=5, max_features=20000,
                                      sublinear_tf=True)),
            ("svd", TruncatedSVD(n_components=svd_components, random_state=RANDOM_STATE)),
        ]), text))
    if attorney_onehot:
        parts.append(("attorney", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                                max_categories=max_attorneys), ["attorney_name"]))
    return ColumnTransformer(parts, sparse_threshold=0)


def make_models():
    """The six classifiers (plus the baseline) with their default settings."""
    return {
        "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                n_jobs=-1, random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                                   verbose=-1, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_jobs=-1),
        "MLP": MLPClassifier(early_stopping=True, max_iter=200, random_state=RANDOM_STATE),
    }


# search spaces (prefix "model__" refers to the model step of the pipeline)
PARAM_SPACES = {
    "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10]},
    "Decision Tree": {"model__max_depth": [6, 10, 15, 20],
                      "model__min_samples_leaf": [20, 50, 100, 200]},
    "Random Forest": {"model__max_depth": [10, 20, None],
                      "model__min_samples_leaf": [5, 20, 50],
                      "model__max_features": ["sqrt", 0.3]},
    "LightGBM": {"model__num_leaves": [31, 63, 127],
                 "model__learning_rate": [0.03, 0.05, 0.1],
                 "model__n_estimators": [300, 600],
                 "model__min_child_samples": [20, 100]},
    "KNN": {"model__n_neighbors": [15, 51, 101],
            "model__weights": ["uniform", "distance"]},
    "MLP": {"model__hidden_layer_sizes": [(64,), (128, 64), (64, 32)],
            "model__alpha": [1e-4, 1e-3, 1e-2],
            "model__learning_rate_init": [1e-3, 3e-3]},
}


def best_f1_threshold(y_true, proba):
    """Threshold that maximizes F1 for the failure class."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = np.nanargmax(f1[:-1])
    return float(thresholds[best])


def evaluate(y_true, proba, threshold=0.5):
    """Threshold-free metrics plus precision / recall / F1 at a given threshold."""
    pred = (proba >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, proba) if len(np.unique(proba)) > 1 else 0.5,
        "PR-AUC": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "F1": f1_score(y_true, pred),
        "threshold": threshold,
    }
