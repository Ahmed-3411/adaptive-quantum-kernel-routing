# HISTORICAL DIAGNOSTIC: narrative interpretations below are superseded.
# Use docs/RESEARCH_RECORD.md and experiments/*_audited.py for current conclusions.
"""
experiments/perfect_router.py

User's proposed diagnostic #6: complete the ceiling decomposition

    Best Single -> Learned Router -> Perfect Router -> Oracle

"Perfect Router" plugs PERFECT (oracle-derived) gates into the exact same
combination mechanism used everywhere else in this project,
K_A(xi,xj) = sum_m g_m(xi) g_m(xj) Km(xi,xj), instead of the learned
g(x) = softmax(f_phi(x)). This isolates: given the best possible routing
DECISIONS, how well does the combination mechanism + downstream SVM do?

IMPORTANT: this is an explicit, hindsight-using DIAGNOSTIC CEILING, not a
deployable model or a fair accuracy claim -- exactly like the existing
"oracle" baseline (experiments/oracle.py), which also uses per-point
correctness knowledge no real deployed model would have in advance. Perfect
gates are built directly from each single kernel's correctness at every
point (train AND test), using each kernel's own SVM (fit on the training
Gram matrix only -- no cross-kernel or router leakage, just the same kind
of hindsight the "oracle" metric already uses).

If perfect_router is close to oracle, the combination mechanism itself is
essentially "perfect" -- confirming Diagnostic 1's finding that
soft-mixing does not lose information. If perfect_router is well BELOW
oracle, that would newly show the combination mechanism has a ceiling of
its own below oracle's "any kernel correct" definition, even with perfect
gating -- a more precise mechanism-level ceiling than "oracle" (whose
"any correct" definition is more permissive than what one Gram matrix +
one SVM decision boundary can necessarily reproduce).

Run:
    python3 -m experiments.perfect_router --dataset wine --n_qubits 4 --seeds 10
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.svm import SVC

from data.datasets import load_dataset
from models.adaptive_qkernel import AdaptiveQuantumKernel
from baselines.svm import precomputed_kernel_svm_accuracy


def build_perfect_gates(per_kernel_correct, per_kernel_margin, n_kernels):
    """
    per_kernel_correct: dict name -> bool array (n,)
    per_kernel_margin:  dict name -> float array (n,) (abs decision-function margin)
    Returns (n, n_kernels) gate array: weight concentrated on correct kernels,
    proportional to their margin (more confident correct answers get more
    weight); uniform fallback where no kernel is correct.
    """
    names = list(per_kernel_correct.keys())
    n = len(per_kernel_correct[names[0]])
    gates = np.zeros((n, n_kernels))
    for i in range(n):
        correct_idx = [k for k, name in enumerate(names) if per_kernel_correct[name][i]]
        if not correct_idx:
            gates[i, :] = 1.0 / n_kernels  # no correct kernel: uninformative fallback
        else:
            margins = np.array([per_kernel_margin[names[k]][i] for k in correct_idx])
            margins = np.clip(margins, 1e-6, None)
            weights = margins / margins.sum()
            for k, w in zip(correct_idx, weights):
                gates[i, k] = w
    return gates


def run_one(dataset, n_qubits, seed, max_train=40, max_test=20):
    X_train, X_test, y_train, y_test = load_dataset(
        dataset, n_qubits=n_qubits, max_train=max_train, max_test=max_test, seed=seed
    )
    model = AdaptiveQuantumKernel(n_qubits=n_qubits, seed=seed, router_type="sample_level")
    model.fit(X_train, y_train, epochs=15, lr=0.05, verbose=False)  # trains theta + learned router together

    # -- best single --
    single_accs = {}
    per_kernel_correct_train, per_kernel_margin_train = {}, {}
    per_kernel_correct_test, per_kernel_margin_test = {}, {}
    for name in model.kernel_names:
        K_tr_m = model.single_kernel_matrix(name, X_train, X_train, symmetric=True)
        K_te_m = model.single_kernel_matrix(name, X_test, X_train, symmetric=False)
        clf = SVC(kernel="precomputed").fit(K_tr_m, y_train)
        pred_tr, pred_te = clf.predict(K_tr_m), clf.predict(K_te_m)
        dec_tr, dec_te = clf.decision_function(K_tr_m), clf.decision_function(K_te_m)
        per_kernel_correct_train[name] = (pred_tr == y_train)
        per_kernel_margin_train[name] = np.abs(dec_tr)
        per_kernel_correct_test[name] = (pred_te == y_test)
        per_kernel_margin_test[name] = np.abs(dec_te)
        single_accs[name] = (pred_te == y_test).mean()
    best_single_acc = max(single_accs.values())

    # -- learned router (the deployed model, already trained above) --
    K_tr_learned = model.kernel_matrix(X_train, X_train, symmetric=True)
    K_te_learned = model.kernel_matrix(X_test, X_train, symmetric=False)
    learned_acc = precomputed_kernel_svm_accuracy(K_tr_learned, y_train, K_te_learned, y_test)

    # -- perfect router: same combination formula, oracle-derived gates --
    gates_train = build_perfect_gates(per_kernel_correct_train, per_kernel_margin_train, model.n_kernels)
    gates_test = build_perfect_gates(per_kernel_correct_test, per_kernel_margin_test, model.n_kernels)
    kernel_mats_train = {name: model.single_kernel_matrix(name, X_train, X_train, symmetric=True) for name in model.kernel_names}
    kernel_mats_test = {name: model.single_kernel_matrix(name, X_test, X_train, symmetric=False) for name in model.kernel_names}
    stacked_train = np.stack([kernel_mats_train[n] for n in model.kernel_names], axis=-1)
    stacked_test = np.stack([kernel_mats_test[n] for n in model.kernel_names], axis=-1)
    K_tr_perfect = np.sum(gates_train[:, None, :] * gates_train[None, :, :] * stacked_train, axis=-1)
    K_te_perfect = np.sum(gates_test[:, None, :] * gates_train[None, :, :] * stacked_test, axis=-1)
    perfect_acc = precomputed_kernel_svm_accuracy(K_tr_perfect, y_train, K_te_perfect, y_test)

    # -- oracle (any-kernel-correct ceiling, unchanged definition) --
    oracle_correct = np.any(np.stack(list(per_kernel_correct_test.values())), axis=0)
    oracle_acc = oracle_correct.mean()

    return {
        "best_single": best_single_acc,
        "learned_router": learned_acc,
        "perfect_router": perfect_acc,
        "oracle": oracle_acc,
    }


def run(dataset, n_qubits, seeds):
    rows = [run_one(dataset, n_qubits, s) for s in seeds]
    print(f"\n{'='*70}\nCeiling decomposition — {dataset} (n={len(seeds)} seeds)\n{'='*70}")
    for key in ["best_single", "learned_router", "perfect_router", "oracle"]:
        accs = np.array([r[key] for r in rows])
        print(f"  {key:16s}: {accs.mean():.3f} ± {accs.std():.3f}")

    best_single = np.mean([r["best_single"] for r in rows])
    learned = np.mean([r["learned_router"] for r in rows])
    perfect = np.mean([r["perfect_router"] for r in rows])
    oracle = np.mean([r["oracle"] for r in rows])
    print(f"\n  Chain: Best Single ({best_single:.3f}) -> Learned Router ({learned:.3f}) "
          f"-> Perfect Router ({perfect:.3f}) -> Oracle ({oracle:.3f})")
    print(f"  Learned captures {100*(learned-best_single)/(perfect-best_single):.1f}% of the "
          f"(Perfect - Best Single) gap achievable by the combination mechanism itself"
          if abs(perfect - best_single) > 1e-6 else "")
    print(f"  Perfect Router vs. Oracle gap: {perfect-oracle:+.3f}  "
          f"({'mechanism ceiling is close to Oracle' if abs(perfect-oracle) < 0.03 else 'mechanism has its OWN ceiling below Oracle'})")

    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wine")
    p.add_argument("--n_qubits", type=int, default=4)
    p.add_argument("--seeds", type=int, default=10)
    args = p.parse_args()
    run(args.dataset, args.n_qubits, list(range(args.seeds)))
