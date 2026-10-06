# HISTORICAL DIAGNOSTIC: narrative interpretations below are superseded.
# Use docs/RESEARCH_RECORD.md and experiments/*_audited.py for current conclusions.
"""
experiments/analyze_kernels.py

Runs evaluation/kernel_analysis.py on each dataset's kernel bank to test the
leading hypothesis from the README: are Kernel 1 (angle), Kernel 2
(entangled), Kernel 3 (trainable, untrained here) actually different enough
on real data for a router to gain anything by choosing between them?

If pairwise_kernel_alignment is close to 1.0 for every pair, the three
kernels are near-redundant as functions -- which alone would explain why
the adaptive kernel never beat the best single kernel in the multi-seed
results, with no need to touch the training objective.

Run:
    python3 -m experiments.analyze_kernels --dataset xor --n_qubits 2
    python3 -m experiments.analyze_kernels --dataset wine --n_qubits 4
    python3 -m experiments.analyze_kernels --dataset breast_cancer --n_qubits 4
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import load_dataset
from quantum.feature_maps import FEATURE_MAPS, TRAINABLE_THETA_SHAPE
from quantum.kernels import build_kernel_bank
from evaluation.kernel_analysis import analyze_kernel_bank


def run(dataset, n_qubits, max_train=40, seed=0):
    print(f"\n=== Kernel analysis — {dataset} (n_qubits={n_qubits}) ===")
    X_train, _, y_train, _ = load_dataset(dataset, n_qubits=n_qubits, max_train=max_train, max_test=4, seed=seed)
    print(f"train={len(X_train)}")

    bank = build_kernel_bank(FEATURE_MAPS, n_qubits)
    kernel_matrices = {}
    for name, kobj in bank.items():
        if kobj.trainable:
            # untrained theta (small random init) -- same starting point every fit() run begins from
            import torch
            theta = 0.1 * torch.randn(n_qubits, *TRAINABLE_THETA_SHAPE, dtype=torch.float64)
            K = kobj.matrix(X_train, X_train, theta=theta, symmetric=True).detach().numpy()
        else:
            K = kobj.matrix(X_train, X_train, symmetric=True)
        kernel_matrices[name] = K

    report = analyze_kernel_bank(kernel_matrices, y_train)

    print("\n-- per-kernel metrics --")
    print(f"  {'kernel':22s} {'target_align':>13} {'eff_dim':>9} {'concentration_std':>19}")
    for name, m in report["per_kernel"].items():
        print(f"  {name:22s} {m['target_alignment']:13.4f} {m['effective_dimension']:9.2f} {m['concentration_std']:19.4f}")

    print("\n-- pairwise kernel-kernel alignment (1.0 = redundant, 0.0 = orthogonal) --")
    for pair, val in report["pairwise_kernel_alignment"].items():
        print(f"  {pair:40s} {val:.4f}")

    avg_pairwise = sum(report["pairwise_kernel_alignment"].values()) / len(report["pairwise_kernel_alignment"])
    print(f"\n  average pairwise alignment: {avg_pairwise:.4f}")
    if avg_pairwise > 0.9:
        print("  -> VERY HIGH: the 3 kernels are near-redundant on this dataset.")
        print("     This alone would explain why the router can't beat the best single kernel.")
    elif avg_pairwise > 0.7:
        print("  -> MODERATE-HIGH: kernels overlap substantially but aren't identical.")
    else:
        print("  -> the kernels ARE meaningfully different -- diversity is not the bottleneck here.")

    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="xor", choices=["xor", "iris", "wine", "breast_cancer", "digits", "moons", "circles"])
    p.add_argument("--n_qubits", type=int, default=2)
    p.add_argument("--max_train", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args.dataset, args.n_qubits, args.max_train, args.seed)
