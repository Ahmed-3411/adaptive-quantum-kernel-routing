"""
evaluation/oracle.py

Roadmap Phase 6 (Oracle vs Adaptive vs Global) + a Phase 0 fix this exposed.

BUG THIS FILE FIXES: every earlier experiment script picked "best single
kernel" by computing all 3 individual kernels' accuracy ON THE TEST SET and
reporting the max. That is test-set model selection -- exactly the failure
mode the roadmap prohibits ("Never choose the best kernel or hyperparameters
using the test set"). It biases the single-kernel baseline upward (you get
to pick the best of 3 draws per seed using the test labels), which is
unfair to the adaptive model in any adaptive-vs-single comparison.

FIX: `select_best_kernel_by_validation` carves a validation split out of
the TRAINING set only, picks the kernel with the best validation accuracy,
and only THEN reports that kernel's (already-decided, not cherry-picked)
test accuracy. The test set is touched exactly once, after the kernel
choice is frozen.

ORACLE (a genuinely different, intentionally leakage-tolerant quantity):
for each test point, count it correct if ANY of the individually-trained
single-kernel SVMs got it right. This is NOT a deployable method -- no real
system can know in advance which expert would have been correct on an
unseen point -- it is a theoretical CEILING used only to measure headroom
("Oracle - Adaptive gap"), exactly as roadmap Phase 6 specifies. Reporting
it as a baseline you could deploy would itself be a leakage/overclaim error;
it is reported here only as an upper bound.
"""

import numpy as np
from sklearn.svm import SVC


def _fit_predict(K_fit, y_fit, K_eval):
    clf = SVC(kernel="precomputed")
    clf.fit(K_fit, y_fit)
    return clf.predict(K_eval)


def select_best_kernel_by_validation(kernel_mats_fit, y_fit, kernel_mats_val, y_val,
                                      kernel_mats_test, kernel_names):
    """
    Leakage-free single-kernel selection.

    kernel_mats_fit:  dict name -> (n_fit, n_fit) kernel matrix (train-fit vs train-fit)
    kernel_mats_val:  dict name -> (n_val, n_fit) kernel matrix (val vs train-fit)
    kernel_mats_test: dict name -> (n_test, n_fit) kernel matrix (test vs train-fit)

    Returns: (chosen_kernel_name, val_accuracy_of_chosen, test_predictions_of_chosen)
    """
    val_accs = {}
    for name in kernel_names:
        clf = SVC(kernel="precomputed")
        clf.fit(kernel_mats_fit[name], y_fit)
        val_accs[name] = clf.score(kernel_mats_val[name], y_val)

    chosen = max(val_accs, key=val_accs.get)

    clf = SVC(kernel="precomputed")
    clf.fit(kernel_mats_fit[chosen], y_fit)
    test_preds = clf.predict(kernel_mats_test[chosen])

    return chosen, val_accs[chosen], test_preds, val_accs


def oracle_accuracy(per_kernel_test_preds, y_test):
    """
    per_kernel_test_preds: dict name -> array of predictions on the test set
    (each from a model trained on the training set only -- no test leakage
    in how these individual models were fit, only in how we now combine
    their predictions post-hoc to measure a ceiling).

    Returns: (oracle_accuracy, per_point_any_correct_bool_array)
    """
    y_test = np.asarray(y_test)
    names = list(per_kernel_test_preds.keys())
    preds_matrix = np.stack([per_kernel_test_preds[n] for n in names], axis=0)  # (M, n_test)
    correct_matrix = (preds_matrix == y_test[None, :])  # (M, n_test)
    any_correct = correct_matrix.any(axis=0)
    return float(any_correct.mean()), any_correct
