"""
experiments/kernel_diversity.py

Runs evaluation/kernel_analysis.py across xor/wine/breast_cancer to answer
the open question from README.md's priority #1: are Kernel 1 (angle),
Kernel 2 (entangled), Kernel 3 (trainable, untrained init) actually
DIFFERENT from each other as functions on these datasets, or are they
redundant -- which would fully explain why the adaptive router never beat
the best single kernel in the multi-seed results?

Run:
    python3 -m experiments.kernel_diversity
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.datasets import load_dataset
from quantum.feature_maps import FEATURE_MAPS, TRAINABLE_THETA_SHAPE
from quantum.kernels import build_kernel_bank
from evaluation.kernel_analysis import analyze_kernel_bank


def run(dataset, n_qubits, max_train=40, seed=0):
    print(f"\n{'='*64}\nKernel diversity check — {dataset} (n_qubits={n_qubits})\n{'='*64}")
    X_train, _, y_train, _ = load_dataset(dataset, n_qubits=n_qubits, max_train=max_train, max_test=2, seed=seed)

    bank = build_kernel_bank(FEATURE_MAPS, n_qubits)
    torch.manual_seed(seed)
    theta = 0.1 * torch.randn(n_qubits, *TRAINABLE_THETA_SHAPE, dtype=torch.float64)  # same random init fit() starts from

    kernel_matrices = {}
    for name, k in bank.items():
        if k.trainable:
            K = k.matrix(X_train, X_train, theta=theta, symmetric=True).detach().numpy()
        else:
            K = k.matrix(X_train, X_train, symmetric=True)
        kernel_matrices[name] = K

    report = analyze_kernel_bank(kernel_matrices, y_train)

    print(f"\n{'kernel':22s} | {'target_align':>12} | {'eff_dim':>9} | {'concentration_std':>18}")
    print("-" * 70)
    for name, m in report["per_kernel"].items():
        print(f"{name:22s} | {m['target_alignment']:12.4f} | {m['effective_dimension']:9.2f} "
              f"(of {m['n']}) | {m['concentration_std']:18.4f}")

    print(f"\n{'pair':30s} | {'kernel-kernel similarity':>24}")
    print("-" * 58)
    for pair, sim in report["pairwise_kernel_alignment"].items():
        flag = "  <-- NEARLY REDUNDANT" if sim > 0.9 else ("  <-- genuinely different" if sim < 0.5 else "")
        print(f"{pair:30s} | {sim:24.4f}{flag}")

    return report


if __name__ == "__main__":
    results = {}
    for dataset, n_qubits in [("xor", 2), ("wine", 4), ("breast_cancer", 4),
                               ("digits", 4), ("moons", 2), ("circles", 2)]:
        results[dataset] = run(dataset, n_qubits)

    print(f"\n{'='*64}\nSummary across datasets: mean pairwise kernel-kernel similarity\n{'='*64}")
    for dataset, report in results.items():
        sims = list(report["pairwise_kernel_alignment"].values())
        print(f"  {dataset:15s}: mean={np.mean(sims):.4f}  (pairs: {[f'{s:.3f}' for s in sims]})")
