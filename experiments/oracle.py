"""
experiments/oracle.py

Roadmap Phase 6 / Section 6: Oracle baseline and the Oracle-Adaptive gap.

    "Oracle: How much theoretical headroom exists? -> Oracle-Adaptive gap"
    "Oracle protocol is leakage-free."

Definition used here (standard in classifier-ensemble literature): each of
the 3 single kernels is trained independently on the TRAINING set only (no
test-set access during training or model selection -- leakage-free in the
sense the roadmap cares about: no tuning/selection using test labels). At
test time, a point counts as an Oracle success if AT LEAST ONE of the 3
single-kernel SVMs classifies it correctly. This measures the theoretical
CEILING an ideal (impossible in practice) per-sample selector could reach --
it deliberately uses test labels to compute that ceiling, which is standard
for an oracle upper-bound diagnostic and is NOT the same thing as using test
labels to train or select a deployable model.

    Oracle gap = Score_oracle - Score_adaptive

A small gap means the adaptive router is already close to the best any
combination strategy could possibly achieve. A large gap means there is
real headroom -- the individual kernels disagree a lot and get different
points right, but the router isn't exploiting that disagreement well.

Run:
    python3 -m experiments.oracle --dataset wine --n_qubits 4 --seeds 25
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.svm import SVC

from data.datasets import load_dataset
from models.adaptive_qkernel import AdaptiveQuantumKernel
from baselines.svm import precomputed_kernel_svm_accuracy
from evaluation.stats import paired_comparison, format_result


def run_one(dataset, n_qubits, epochs, max_train, max_test, seed):
    X_train, X_test, y_train, y_test = load_dataset(
        dataset, n_qubits=n_qubits, max_train=max_train, max_test=max_test, seed=seed
    )
    model = AdaptiveQuantumKernel(n_qubits=n_qubits, seed=seed, router_type="sample_level")
    model.fit(X_train, y_train, epochs=epochs, lr=0.05, verbose=False)

    # adaptive accuracy (for reference / paired comparison)
    K_tr = model.kernel_matrix(X_train, X_train, symmetric=True)
    K_te = model.kernel_matrix(X_test, X_train, symmetric=False)
    adaptive_acc = precomputed_kernel_svm_accuracy(K_tr, y_train, K_te, y_test)

    # per-kernel predictions on the test set, each trained independently
    # (this IS the leakage-free part: each SVC only ever sees X_train/y_train
    # during .fit(); test labels are used only afterwards, to check the oracle)
    per_kernel_correct = []  # list of boolean arrays, one per kernel, len(y_test)
    per_kernel_acc = {}
    for name in model.kernel_names:
        Ktr_m = model.single_kernel_matrix(name, X_train, X_train, symmetric=True)
        Kte_m = model.single_kernel_matrix(name, X_test, X_train, symmetric=False)
        clf = SVC(kernel="precomputed")
        clf.fit(Ktr_m, y_train)
        preds = clf.predict(Kte_m)
        correct = (preds == y_test)
        per_kernel_correct.append(correct)
        per_kernel_acc[name] = float(correct.mean())

    oracle_correct = np.any(np.stack(per_kernel_correct, axis=0), axis=0)  # OR across kernels
    oracle_acc = float(oracle_correct.mean())

    return {
        "adaptive": adaptive_acc,
        "oracle": oracle_acc,
        "per_kernel": per_kernel_acc,
        "best_single": max(per_kernel_acc.values()),
    }


def run(dataset, n_qubits, epochs, max_train, max_test, seeds):
    print(f"\n{'='*70}\nOracle baseline — {dataset} (n_qubits={n_qubits}), {len(seeds)} seeds\n{'='*70}")
    adaptive_accs, oracle_accs, best_single_accs = [], [], []
    for s in seeds:
        r = run_one(dataset, n_qubits, epochs, max_train, max_test, s)
        adaptive_accs.append(r["adaptive"])
        oracle_accs.append(r["oracle"])
        best_single_accs.append(r["best_single"])
        print(f"  seed {s:2d}: adaptive={r['adaptive']:.3f}  oracle={r['oracle']:.3f}  "
              f"best_single={r['best_single']:.3f}  per_kernel={ {k: round(v,3) for k,v in r['per_kernel'].items()} }")

    adaptive_accs = np.array(adaptive_accs)
    oracle_accs = np.array(oracle_accs)
    best_single_accs = np.array(best_single_accs)

    print(f"\n{'='*70}\nSummary\n{'='*70}")
    print(f"  adaptive:    {adaptive_accs.mean():.3f} ± {adaptive_accs.std():.3f}")
    print(f"  oracle:      {oracle_accs.mean():.3f} ± {oracle_accs.std():.3f}")
    print(f"  best_single: {best_single_accs.mean():.3f} ± {best_single_accs.std():.3f}")

    gap_result = paired_comparison(oracle_accs, adaptive_accs, seed=0)
    print("\n" + format_result("Oracle - Adaptive gap", gap_result))

    headroom_result = paired_comparison(oracle_accs, best_single_accs, seed=0)
    print(format_result("Oracle - BestSingle gap (total complementarity available)", headroom_result))

    captured_pct = (
        (adaptive_accs.mean() - best_single_accs.mean()) /
        (oracle_accs.mean() - best_single_accs.mean()) * 100
        if oracle_accs.mean() > best_single_accs.mean() else float("nan")
    )
    print(f"\n  Fraction of available oracle headroom captured by the adaptive router: {captured_pct:.1f}%")

    return {"adaptive": adaptive_accs.tolist(), "oracle": oracle_accs.tolist(), "best_single": best_single_accs.tolist()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wine", choices=["xor", "iris", "wine", "breast_cancer", "digits", "moons", "circles"])
    p.add_argument("--n_qubits", type=int, default=4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--max_train", type=int, default=40)
    p.add_argument("--max_test", type=int, default=20)
    p.add_argument("--seeds", type=int, default=25, help="number of seeds, 0..seeds-1")
    args = p.parse_args()
    run(args.dataset, args.n_qubits, args.epochs, args.max_train, args.max_test, list(range(args.seeds)))
