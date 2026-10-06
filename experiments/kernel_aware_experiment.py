# HISTORICAL DIAGNOSTIC: narrative interpretations below are superseded.
# Use docs/RESEARCH_RECORD.md and experiments/*_audited.py for current conclusions.
"""
experiments/kernel_aware_experiment.py

THE decisive experiment for Q5: does giving the router leakage-free
per-kernel behavior diagnostics (confidence + disagreement, computed from
training-set-only SVMs -- see routing/kernel_aware_router.py) let it
capture more of the Oracle headroom than routing on raw features x alone?

Two possible outcomes, both informative:
    A) Kernel-aware routing closes some of the gap -> the raw-feature
       router was "blind" to the signal that reveals which kernel to
       trust; feeding it directly helps. Mechanistic bottleneck: features
       available to the router, not the routing mechanism itself.
    B) Kernel-aware routing does NOT help -> even with direct access to
       kernel behavior, neural routing can't turn it into reliable
       sample-level decisions. Mechanistic bottleneck: optimization/
       objective, not information availability.

Run:
    python3 -m experiments.kernel_aware_experiment --dataset wine --n_qubits 4 --seeds 10
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.datasets import load_dataset
from models.adaptive_qkernel import AdaptiveQuantumKernel
from routing.kernel_aware_router import KernelAwareGate, compute_kernel_diagnostics
from baselines.svm import precomputed_kernel_svm_accuracy


def run_one(dataset, n_qubits, epochs, max_train, max_test, seed, router_type):
    X_train, X_test, y_train, y_test = load_dataset(
        dataset, n_qubits=n_qubits, max_train=max_train, max_test=max_test, seed=seed
    )
    model = AdaptiveQuantumKernel(n_qubits=n_qubits, seed=seed, router_type="sample_level")

    if router_type == "kernel_aware":
        X_all = np.vstack([X_train, X_test])
        diagnostics = compute_kernel_diagnostics(model, X_train, y_train, X_all)  # uses INITIAL theta
        gate = KernelAwareGate(n_features=n_qubits, n_kernels=model.n_kernels)
        gate.set_diagnostics(X_all, diagnostics)
        model.router = gate  # interface-compatible swap; router_type stays "sample_level" downstream
    elif router_type == "global":
        from routing.global_router import GlobalRouter
        model.router = GlobalRouter(n_features=n_qubits, n_kernels=model.n_kernels)
        model.router_type = "global"
    elif router_type == "uniform":
        from routing.global_router import UniformRouter
        model.router = UniformRouter(n_features=n_qubits, n_kernels=model.n_kernels)
        model.router_type = "uniform"
    # else: router_type == "current" -> leave the default sample_level (raw-x) router as-is

    model.fit(X_train, y_train, epochs=epochs, lr=0.05, verbose=False)
    K_tr = model.kernel_matrix(X_train, X_train, symmetric=True)
    K_te = model.kernel_matrix(X_test, X_train, symmetric=False)
    return precomputed_kernel_svm_accuracy(K_tr, y_train, K_te, y_test)


def run(dataset, n_qubits, epochs, max_train, max_test, seeds):
    variants = ["global", "uniform", "current", "kernel_aware"]
    results = {v: [] for v in variants}
    print(f"\n{'='*70}\nKernel-aware router experiment — {dataset} (n_qubits={n_qubits}), {len(seeds)} seeds\n{'='*70}")
    for s in seeds:
        row = {}
        for v in variants:
            acc = run_one(dataset, n_qubits, epochs, max_train, max_test, s, v)
            results[v].append(acc)
            row[v] = acc
        print(f"  seed {s:2d}: " + "  ".join(f"{v}={row[v]:.3f}" for v in variants))

    print(f"\n{'='*70}\nSummary (n={len(seeds)} seeds)\n{'='*70}")
    for v in variants:
        accs = np.array(results[v])
        print(f"  {v:14s}: {accs.mean():.3f} ± {accs.std():.3f}")

    return results


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
