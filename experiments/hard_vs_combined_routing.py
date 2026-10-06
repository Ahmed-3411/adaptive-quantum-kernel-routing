# HISTORICAL DIAGNOSTIC: narrative interpretations below are superseded.
# Use docs/RESEARCH_RECORD.md and experiments/*_audited.py for current conclusions.
"""
experiments/hard_vs_combined_routing.py

User's proposed diagnostic #2 (highest priority after expert-imitation):
does the router "know" which kernel to trust, even if the combined-kernel
Gram matrix mechanism doesn't turn that knowledge into a prediction
advantage?

For an already-trained sample_level model, compare three things on the
SAME test points:

    1. combined   -- the deployed model: K_A = sum_m g_m(xi)g_m(xj)Km(xi,xj),
                      then a precomputed-kernel SVM on K_A (what we've been
                      calling "adaptive" everywhere else in this project).
    2. hard_route -- take m*(x) = argmax_m g_m(x) per TEST point, and use
                      THAT single kernel's own SVM prediction for that point
                      (a different single-kernel classifier per test point,
                      chosen by the router's own top choice).
    3. oracle     -- correct if ANY single-kernel SVM is correct (unchanged
                      from experiments/oracle.py).

If hard_route is close to oracle but combined is not, the router's
DECISIONS are good and the bottleneck is the combination mechanism
(soft-mixing into one Gram matrix loses the information hard selection
would have used). If hard_route is no better than combined (both far below
oracle), the bottleneck is the router's decision quality itself, not how
decisions get turned into a Gram matrix.

Run:
    python3 -m experiments.hard_vs_combined_routing --dataset wine --n_qubits 4 --seeds 10
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


def run_one(dataset, n_qubits, epochs, max_train, max_test, seed):
    X_train, X_test, y_train, y_test = load_dataset(
        dataset, n_qubits=n_qubits, max_train=max_train, max_test=max_test, seed=seed
    )
    model = AdaptiveQuantumKernel(n_qubits=n_qubits, seed=seed, router_type="sample_level")
    model.fit(X_train, y_train, epochs=epochs, lr=0.05, verbose=False)

    # 1. combined (the deployed model)
    K_tr = model.kernel_matrix(X_train, X_train, symmetric=True)
    K_te = model.kernel_matrix(X_test, X_train, symmetric=False)
    combined_acc = precomputed_kernel_svm_accuracy(K_tr, y_train, K_te, y_test)

    # per-kernel SVMs + predictions, needed for both hard-routing and oracle
    per_kernel_pred = {}
    per_kernel_correct = {}
    for name in model.kernel_names:
        K_tr_m = model.single_kernel_matrix(name, X_train, X_train, symmetric=True)
        K_te_m = model.single_kernel_matrix(name, X_test, X_train, symmetric=False)
        clf = SVC(kernel="precomputed").fit(K_tr_m, y_train)
        pred = clf.predict(K_te_m)
        per_kernel_pred[name] = pred
        per_kernel_correct[name] = (pred == y_test)

    # 2. hard_route: argmax_m g_m(x) per test point, use that kernel's own prediction
    gates_test = model.router.gate_matrix(X_test).detach().numpy()  # (n_test, n_kernels)
    best_kernel_idx = np.argmax(gates_test, axis=1)
    hard_pred = np.array([
        per_kernel_pred[model.kernel_names[best_kernel_idx[i]]][i] for i in range(len(y_test))
    ])
    hard_acc = (hard_pred == y_test).mean()

    # 3. oracle
    oracle_correct = np.any(np.stack(list(per_kernel_correct.values())), axis=0)
    oracle_acc = oracle_correct.mean()

    # bonus: does the router's hard choice actually match an oracle-correct kernel when possible?
    is_hard_choice_correct = np.array([
        per_kernel_correct[model.kernel_names[best_kernel_idx[i]]][i] for i in range(len(y_test))
    ])
    # among points where oracle succeeds, how often did the router's top-1 choice land on a correct kernel?
    routing_precision = is_hard_choice_correct[oracle_correct].mean() if oracle_correct.sum() > 0 else float("nan")

    return {
        "combined_acc": combined_acc,
        "hard_acc": hard_acc,
        "oracle_acc": oracle_acc,
        "routing_precision": routing_precision,  # P(top-1 kernel choice is correct | oracle succeeds)
    }


def run(dataset, n_qubits, epochs, max_train, max_test, seeds):
    rows = [run_one(dataset, n_qubits, epochs, max_train, max_test, s) for s in seeds]
    combined = np.array([r["combined_acc"] for r in rows])
    hard = np.array([r["hard_acc"] for r in rows])
    oracle = np.array([r["oracle_acc"] for r in rows])
    precision = np.array([r["routing_precision"] for r in rows])

    print(f"\n{'='*70}\nHard-routed vs. Combined-kernel vs. Oracle — {dataset} (n={len(seeds)} seeds)\n{'='*70}")
    print(f"  combined (deployed model):        {combined.mean():.3f} ± {combined.std():.3f}")
    print(f"  hard_route (argmax-g, own SVM):    {hard.mean():.3f} ± {hard.std():.3f}")
    print(f"  oracle (any-kernel-correct):       {oracle.mean():.3f} ± {oracle.std():.3f}")
    print(f"  routing precision (top-1 correct | oracle succeeds): {np.nanmean(precision):.3f}")
    print(f"\n  Best Single -> Combined -> Hard-route -> Oracle chain:")
    print(f"    combined - hard_route = {combined.mean() - hard.mean():+.3f}  "
          f"({'combination mechanism LOSES information vs. hard routing' if combined.mean() < hard.mean() - 0.01 else 'combination mechanism is not the loss point' if combined.mean() > hard.mean() + 0.01 else 'roughly equivalent'})")
    print(f"    hard_route - oracle   = {hard.mean() - oracle.mean():+.3f}  "
          f"(remaining gap is the router's decision-quality problem)")

    return {"combined": combined.tolist(), "hard": hard.tolist(), "oracle": oracle.tolist(), "precision": precision.tolist()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wine")
    p.add_argument("--n_qubits", type=int, default=4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--max_train", type=int, default=40)
    p.add_argument("--max_test", type=int, default=20)
    p.add_argument("--seeds", type=int, default=10)
    args = p.parse_args()
    run(args.dataset, args.n_qubits, args.epochs, args.max_train, args.max_test, list(range(args.seeds)))
