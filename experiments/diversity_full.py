# HISTORICAL DIAGNOSTIC: narrative interpretations below are superseded.
# Use docs/RESEARCH_RECORD.md and experiments/*_audited.py for current conclusions.
"""
experiments/diversity_full.py

Roadmap Phase 4: compute all 4 diversity metrics (alignment similarity,
Frobenius distance, eigenspectrum distance, prediction disagreement) across
the 6 datasets, and check whether they agree with each other AND with the
known outcome (only Wine has a statistically confirmed adaptive-kernel
benefit; results/*_v2_diverse.json). Relying on just one metric (as
experiments/kernel_diversity.py did) risks a hypothesis built on that one
metric's quirks -- if multiple, conceptually distinct diversity notions
all point the same direction, that's much stronger evidence.

Run:
    python3 -m experiments.diversity_full
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy import stats

from data.datasets import load_dataset
from quantum.feature_maps import FEATURE_MAPS, TRAINABLE_THETA_SHAPE
from quantum.kernels import build_kernel_bank
from evaluation.diversity_metrics import all_pairwise_diversity

# known outcome from the confirmed 25-seed PSD-preserving multi-seed runs
KNOWN_ADAPTIVE_BENEFIT = {
    "digits": 0.016, "wine": 0.046, "circles": -0.012,
    "breast_cancer": 0.004, "xor": 0.004, "moons": -0.030,
}
# (wine value updated to the PSD-preserving re-confirmed number; others are
#  the pre-PSD-fix numbers since those datasets weren't re-run with
#  sample_level yet -- treat as approximate / to be refreshed later)


def compute_for_dataset(dataset, n_qubits, max_train=40, seed=0):
    X_train, X_test, y_train, y_test = load_dataset(
        dataset, n_qubits=n_qubits, max_train=max_train, max_test=20, seed=seed
    )
    bank = build_kernel_bank(FEATURE_MAPS, n_qubits)
    torch.manual_seed(seed)
    theta = 0.1 * torch.randn(n_qubits, *TRAINABLE_THETA_SHAPE, dtype=torch.float64)

    kernel_mats_train, kernel_mats_test = {}, {}
    for name, k in bank.items():
        if k.trainable:
            kernel_mats_train[name] = k.matrix(X_train, X_train, theta=theta, symmetric=True).detach().numpy()
            kernel_mats_test[name] = k.matrix(X_test, X_train, theta=theta, symmetric=False).detach().numpy()
        else:
            kernel_mats_train[name] = k.matrix(X_train, X_train, symmetric=True)
            kernel_mats_test[name] = k.matrix(X_test, X_train, symmetric=False)

    pairwise = all_pairwise_diversity(kernel_mats_train, kernel_mats_test, y_train, list(bank.keys()))
    # average across the 3 pairs -> one summary number per metric per dataset
    metrics = ["alignment_similarity", "frobenius_distance", "eigenspectrum_distance", "prediction_disagreement"]
    summary = {m: float(np.mean([pairwise[pair][m] for pair in pairwise])) for m in metrics}
    return summary, pairwise


def run():
    datasets = [("digits", 4), ("wine", 4), ("circles", 2), ("breast_cancer", 4), ("xor", 2), ("moons", 2)]
    all_summaries = {}
    print(f"{'dataset':15s} | {'align_sim':>10} | {'frobenius':>10} | {'eigenspec':>10} | {'disagree':>9} | benefit")
    print("-" * 80)
    for name, nq in datasets:
        summary, _ = compute_for_dataset(name, nq)
        all_summaries[name] = summary
        benefit = KNOWN_ADAPTIVE_BENEFIT[name]
        print(f"{name:15s} | {summary['alignment_similarity']:10.4f} | {summary['frobenius_distance']:10.4f} | "
              f"{summary['eigenspectrum_distance']:10.4f} | {summary['prediction_disagreement']:9.4f} | {benefit:+.3f}")

    print(f"\n{'='*80}\nSpearman correlation of each diversity metric with the known adaptive benefit (n=6 datasets)\n{'='*80}")
    benefits = [KNOWN_ADAPTIVE_BENEFIT[name] for name, _ in datasets]
    for metric in ["alignment_similarity", "frobenius_distance", "eigenspectrum_distance", "prediction_disagreement"]:
        values = [all_summaries[name][metric] for name, _ in datasets]
        rho, p = stats.spearmanr(values, benefits)
        # note expected sign: alignment_similarity should correlate NEGATIVELY with benefit
        # (higher similarity = less diverse = less benefit); the other 3 are "distance/
        # disagreement" metrics so should correlate POSITIVELY with benefit.
        print(f"  {metric:24s}: rho={rho:+.3f}  p={p:.3f}")

    print("\nCross-metric agreement (do the 4 diversity notions rank datasets similarly?):")
    align = [all_summaries[name]["alignment_similarity"] for name, _ in datasets]
    frob = [all_summaries[name]["frobenius_distance"] for name, _ in datasets]
    eig = [all_summaries[name]["eigenspectrum_distance"] for name, _ in datasets]
    dis = [all_summaries[name]["prediction_disagreement"] for name, _ in datasets]
    # alignment is a SIMILARITY (opposite sign convention from the other 3, which are distances)
    pairs = [("align(-1) vs frobenius", [-a for a in align], frob),
             ("align(-1) vs eigenspec", [-a for a in align], eig),
             ("align(-1) vs disagreement", [-a for a in align], dis),
             ("frobenius vs eigenspec", frob, eig),
             ("frobenius vs disagreement", frob, dis),
             ("eigenspec vs disagreement", eig, dis)]
    for label, x, y in pairs:
        rho, p = stats.spearmanr(x, y)
        print(f"  {label:28s}: rho={rho:+.3f}  p={p:.3f}")

    return all_summaries


if __name__ == "__main__":
    run()
