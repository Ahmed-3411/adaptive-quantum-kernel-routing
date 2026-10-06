# HISTORICAL DIAGNOSTIC: narrative interpretations below are superseded.
# Use docs/RESEARCH_RECORD.md and experiments/*_audited.py for current conclusions.
"""
experiments/oracle_predictability.py

User's proposed diagnostic #5 (their priority order puts it last, but it's
the cheapest and most fundamental): is "which kernel is correct" even
PREDICTABLE from raw features x, using leakage-free labels?

If a simple classifier (logistic regression / linear SVM / tiny MLP) can
predict the CV-derived best-kernel label from x with accuracy well above
chance, the information needed for routing clearly exists in x, and the
neural router's failure to reach Oracle is a training-mechanism problem.
If even a simple classifier can't beat chance, the routing signal may not
be linearly/simply recoverable from x with this little data, which would
revive "information limitation" as a live hypothesis alongside
optimization/mechanism failure.

Procedure (leakage-free):
    1. K-fold cross-validation WITHIN the training set: for each fold, fit
       each single kernel's SVM on the other folds, predict on the held-out
       fold, and record which kernel(s) are correct for each training
       point -- WITHOUT ever letting a kernel's classifier see the point
       it's being scored on.
    2. Build a single best-kernel label per training point (the CV-correct
       kernel with the highest decision-function margin; points where no
       kernel is CV-correct are excluded from this diagnostic, since there
       is no informative label to learn).
    3. Train a simple classifier (logistic regression) x -> best-kernel
       label, evaluated via its own nested cross-validation (never the
       real test set).
    4. Compare its cross-validated accuracy to chance (1/n_kernels) and to
       the majority-class baseline.

Run:
    python3 -m experiments.oracle_predictability --dataset wine --n_qubits 4 --seeds 10
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.dummy import DummyClassifier

from data.datasets import load_dataset
from models.adaptive_qkernel import AdaptiveQuantumKernel


def cv_best_kernel_labels(model, X_train, y_train, n_folds=5, seed=0):
    """
    Leakage-free per-training-point best-kernel labels via k-fold CV.
    Returns (labels, valid_mask): labels[i] in {0,...,n_kernels-1} (only
    meaningful where valid_mask[i] is True -- i.e. at least one kernel was
    CV-correct on point i).
    """
    n = len(X_train)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n_kernels = model.n_kernels
    correct = np.zeros((n, n_kernels), dtype=bool)
    margin = np.zeros((n, n_kernels))

    # Precompute each kernel's FULL train-train matrix once (fixed circuit, theta already trained)
    full_mats = {name: model.single_kernel_matrix(name, X_train, X_train, symmetric=True)
                 for name in model.kernel_names}

    for train_idx, held_idx in kf.split(X_train):
        for k_idx, name in enumerate(model.kernel_names):
            K_full = full_mats[name]
            K_fit = K_full[np.ix_(train_idx, train_idx)]
            K_eval = K_full[np.ix_(held_idx, train_idx)]
            clf = SVC(kernel="precomputed").fit(K_fit, y_train[train_idx])
            pred = clf.predict(K_eval)
            dec = clf.decision_function(K_eval)
            correct[held_idx, k_idx] = (pred == y_train[held_idx])
            margin[held_idx, k_idx] = np.abs(dec)

    valid_mask = correct.any(axis=1)
    # best-kernel label = CV-correct kernel with highest margin
    masked_margin = np.where(correct, margin, -np.inf)
    labels = np.argmax(masked_margin, axis=1)
    return labels, valid_mask


def run_one(dataset, n_qubits, seed, max_train=40, max_test=20, probe="linear"):
    X_train, X_test, y_train, y_test = load_dataset(
        dataset, n_qubits=n_qubits, max_train=max_train, max_test=max_test, seed=seed
    )
    model = AdaptiveQuantumKernel(n_qubits=n_qubits, seed=seed, router_type="sample_level")
    # NOTE: theta is at its random INIT here (deliberately not trained) so this
    # diagnostic measures predictability of a fixed, well-defined kernel bank,
    # not a moving target. This matches how kernel-aware diagnostics were
    # computed earlier in this project for consistency.

    labels, valid_mask = cv_best_kernel_labels(model, X_train, y_train, n_folds=5, seed=seed)
    X_valid = X_train[valid_mask]
    y_valid = labels[valid_mask]

    if len(np.unique(y_valid)) < 2 or len(X_valid) < 10:
        return None  # not enough signal/variety to test meaningfully

    # cross-validated accuracy of a classifier predicting the best kernel from x
    if probe == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=3000, random_state=seed)
    else:
        clf = LogisticRegression(max_iter=2000)
    n_splits = min(5, len(X_valid) // 3)
    cv_preds = cross_val_predict(clf, X_valid, y_valid, cv=n_splits)
    simple_acc = (cv_preds == y_valid).mean()

    dummy = DummyClassifier(strategy="most_frequent")
    dummy_preds = cross_val_predict(dummy, X_valid, y_valid, cv=n_splits)
    majority_acc = (dummy_preds == y_valid).mean()

    chance_acc = 1.0 / model.n_kernels

    return {
        "n_valid": int(valid_mask.sum()),
        "n_total": len(X_train),
        "simple_classifier_acc": simple_acc,
        "majority_baseline_acc": majority_acc,
        "chance_acc": chance_acc,
    }


def run(dataset, n_qubits, seeds, probe="linear"):
    rows = [r for r in (run_one(dataset, n_qubits, s, probe=probe) for s in seeds) if r is not None]
    if not rows:
        print(f"{dataset}: not enough valid points to test in any seed")
        return None

    simple = np.array([r["simple_classifier_acc"] for r in rows])
    majority = np.array([r["majority_baseline_acc"] for r in rows])
    chance = rows[0]["chance_acc"]
    frac_valid = np.mean([r["n_valid"] / r["n_total"] for r in rows])

    probe_label = "MLP (16,16)" if probe == "mlp" else "logistic regression"
    print(f"\n{'='*70}\nOracle-kernel predictability from x — {dataset} (n={len(rows)} seeds, probe={probe_label})\n{'='*70}")
    print(f"  mean fraction of training points with a valid (CV-correct) label: {frac_valid:.1%}")
    print(f"  {probe_label} CV accuracy: {simple.mean():.3f} ± {simple.std():.3f}")
    print(f"  majority-class baseline accuracy:                    {majority.mean():.3f} ± {majority.std():.3f}")
    print(f"  chance accuracy (1/n_kernels):                       {chance:.3f}")
    verdict = ("PREDICTABLE above majority baseline -- information exists in x"
               if simple.mean() > majority.mean() + 0.05
               else "NOT clearly predictable above majority baseline -- information may be limited")
    print(f"  -> {verdict}")
    return {"simple": simple.tolist(), "majority": majority.tolist(), "chance": chance}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wine")
    p.add_argument("--n_qubits", type=int, default=4)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--probe", default="linear", choices=["linear", "mlp"])
    args = p.parse_args()
    run(args.dataset, args.n_qubits, list(range(args.seeds)), probe=args.probe)
