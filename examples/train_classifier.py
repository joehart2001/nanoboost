"""
Example: train an XGBoost classifier on two labelled recordings.

Builds on preprocess_and_extract.py — runs the full pipeline:
  1. load_to_event_data_nofeatures  — baseline correction + event isolation (x2)
  2. DWT_and_features_thresh_trace  — per-event DWT + feature extraction  (x2)
  3. hyperparam_op                  — SMOTE + RobustScaler + XGBoost with random search

Usage:
    python examples/train_classifier.py
    python examples/train_classifier.py path/to/5nm.abf path/to/10nm.abf
"""

import sys
import pickle
from pathlib import Path

import numpy as np

from nanoboost.scripts.discrete_wavelet_transform.novel_DWT import (
    DWT_and_features_thresh_trace,
    load_to_event_data_nofeatures,
)
from nanoboost.scripts.ml_model_building.supervised_learning.training_and_hyperparameter_op import (
    hyperparam_op,
)

# --- configuration -----------------------------------------------------------
TESTS_DIR = Path(__file__).resolve().parent.parent / "tests"
ABF_5NM  = TESTS_DIR / "5nm_test.abf"
ABF_10NM = TESTS_DIR / "10nm_test.abf"
WAVELET   = "bior3.3"
THRESHOLD = 0.2
MODEL     = "XG"   # XG | RF | SVM | DT
SEARCH    = "random"

if len(sys.argv) >= 3:
    ABF_5NM  = Path(sys.argv[1])
    ABF_10NM = Path(sys.argv[2])

# --- helper ------------------------------------------------------------------
def preprocess_and_extract(abf_path: Path, label: int):
    print(f"\nLoading:  {abf_path.name}  (label={label})")
    event_time, event_data, _, sd_threshold, sd_threshold_lower, mean_noise = \
        load_to_event_data_nofeatures(str(abf_path), resistive=False)
    print(f"Isolated: {len(event_data)} events")

    if not event_data:
        raise RuntimeError(f"No events found in {abf_path}")

    _, _, _, _, features_list, labels, _ = DWT_and_features_thresh_trace(
        event_time, event_data, mean_noise,
        sd_threshold, sd_threshold_lower,
        label, WAVELET, THRESHOLD,
    )
    return features_list, labels

# --- stage 1 & 2: preprocess + extract both recordings ----------------------
X_5,  y_5  = preprocess_and_extract(ABF_5NM,  label=5)
X_10, y_10 = preprocess_and_extract(ABF_10NM, label=10)

X = np.array(X_5  + X_10)
y = np.concatenate([y_5, y_10])
print(f"\nDataset: {len(X)} events  ({int(y.sum())} × 10 nm, {int((y == 0).sum())} × 5 nm)")

# --- stage 3: train ----------------------------------------------------------
print(f"\nTraining {MODEL} with {SEARCH} search …")
_, X_test, _, y_test, y_pred, search, best_params = hyperparam_op(MODEL, SEARCH, X, y)

# --- save model --------------------------------------------------------------
model_path = Path("model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(search.best_estimator_, f)
print(f"\nModel saved → {model_path}")

# --- quick evaluation on held-out test set -----------------------------------
if X_test is not None:
    test_acc = search.best_estimator_.score(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
