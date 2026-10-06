# HISTORICAL DIAGNOSTIC: narrative interpretations below are superseded.
# Use docs/RESEARCH_RECORD.md and experiments/*_audited.py for current conclusions.
"""
experiments/optimization_stability.py

Final diagnostic in the reviewer's proposed order: is the router's
under-capture of the Perfect-Router ceiling (ROADMAP_STATUS.md Diagnostic
3) an OPTIMIZATION problem -- i.e. do good solutions exist but the current
training procedure (Adam, fixed lr, fixed init, fixed epoch count) fails
to reliably find them -- or does capture stay low regardless of training
randomness/hyperparameters, which would argue against optimization being
the limiting factor?

Design: fix the DATA split (one dataset seed, so X_train/X_test/y_train/
y_test never change), and vary only the TRAINING randomness (router+theta
initialization seed) and, separately, the learning rate. If accuracy
varies widely across training seeds at a fixed data split and lr, that is
direct evidence of an unstable optimization landscape (some runs find a
much better router than others, purely by initialization luck). If
accuracy is essentially flat across restarts, optimization instability is
not the story -- the ceiling is being hit consistently, whatever that
ceiling's cause.

Run:
    python3 -m experiments.optimization_stability --dataset wine --n_qubits 4 --restarts 15
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.datasets import load_dataset
from models.adaptive_qkernel import AdaptiveQuantumKernel
from baselines.svm import precomputed_kernel_svm_accuracy


def run_restart(X_train, X_test, y_train, y_test, n_qubits, epochs, lr, train_seed):
    model = AdaptiveQuantumKernel(n_qubits=n_qubits, seed=train_seed, router_type="sample_level")
    model.fit(X_train, y_train, epochs=epochs, lr=lr, verbose=False)
    K_tr = model.kernel_matrix(X_train, X_train, symmetric=True)
    K_te = model.kernel_matrix(X_test, X_train, symmetric=False)
    return precomputed_kernel_svm_accuracy(K_tr, y_train, K_te, y_test)


def run(dataset, n_qubits, epochs, max_train, max_test, data_seed, n_restarts, lrs):
    X_train, X_test, y_train, y_test = load_dataset(
        dataset, n_qubits=n_qubits, max_train=max_train, max_test=max_test, seed=data_seed
    )
    print(f"\n{'='*70}\nOptimization stability — {dataset} (FIXED data split, seed={data_seed})\n{'='*70}")

    results = {}
    for lr in lrs:
        accs = [run_restart(X_train, X_test, y_train, y_test, n_qubits, epochs, lr, train_seed=s)
                for s in range(n_restarts)]
        accs = np.array(accs)
        results[lr] = accs
        print(f"\n  lr={lr}: {n_restarts} restarts, SAME data split")
        print(f"    mean={accs.mean():.3f}  std={accs.std():.3f}  min={accs.min():.3f}  max={accs.max():.3f}  range={accs.max()-accs.min():.3f}")
        print(f"    individual: {[f'{a:.2f}' for a in accs]}")

    print(f"\n{'='*70}\nInterpretation\n{'='*70}")
    all_accs = np.concatenate(list(results.values()))
    overall_range = all_accs.max() - all_accs.min()
    print(f"  Overall range across all restarts/lrs: {overall_range:.3f} (min={all_accs.min():.3f}, max={all_accs.max():.3f})")
    if overall_range > 0.10:
        print("  -> LARGE spread: consistent with an unstable optimization landscape "
              "(some initializations/lr find much better routers than others on IDENTICAL data)")
    else:
        print("  -> SMALL spread: training consistently converges to a similar outcome regardless of "
              "initialization/lr -- optimization instability is not well supported as the limiting factor")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wine")
    p.add_argument("--n_qubits", type=int, default=4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--max_train", type=int, default=40)
    p.add_argument("--max_test", type=int, default=20)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--restarts", type=int, default=15)
    p.add_argument("--lrs", type=str, default="0.05")
    args = p.parse_args()
    lrs = [float(x) for x in args.lrs.split(",")]
    run(args.dataset, args.n_qubits, args.epochs, args.max_train, args.max_test, args.data_seed, args.restarts, lrs)
