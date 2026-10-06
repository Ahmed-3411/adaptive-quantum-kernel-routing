"""
evaluation/kernel_analysis.py

Section 16 of the plan: is a kernel actually informative, and are the
kernels in the bank actually DIFFERENT from each other? This is the cheap
diagnostic to run before touching the training objective or router
architecture -- if Kernel 1/2/3 are near-identical as functions on a given
dataset, no router can meaningfully choose between them, and the flat
"adaptive doesn't beat single kernel" result we saw across all 3 datasets
would be explained without needing to touch Objective A/B at all.

Metrics implemented:
    - kernel_target_alignment(K, y)      : how well K aligns with the labels
    - kernel_kernel_alignment(K_a, K_b)   : how similar two kernels are AS
                                             FUNCTIONS (same formula, applied
                                             to two kernel matrices instead
                                             of one kernel + the label matrix)
    - spectrum(K)                         : sorted eigenvalues (kernel matrix
                                             spectrum)
    - effective_dimension(K)              : inverse participation ratio of
                                             the normalized eigenvalue
                                             distribution -- how many
                                             "directions" the kernel actually
                                             uses. Low effective dimension on
                                             an n x n matrix is the textbook
                                             symptom of kernel concentration
                                             (RQ2 / RQ4's "does representation
                                             richness matter" question).
    - concentration(K)                    : std of the off-diagonal entries.
                                             Kernels that concentrate toward
                                             a constant (all pairs equally
                                             similar) have LOW std here and
                                             carry little discriminative
                                             information regardless of the
                                             router.
"""

import numpy as np


def kernel_target_alignment(K, y):
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    Y = y @ y.T
    num = np.sum(K * Y)
    denom = np.linalg.norm(K) * np.linalg.norm(Y) + 1e-12
    return float(num / denom)


def kernel_kernel_alignment(K_a, K_b):
    num = np.sum(K_a * K_b)
    denom = np.linalg.norm(K_a) * np.linalg.norm(K_b) + 1e-12
    return float(num / denom)


def spectrum(K):
    eigvals = np.linalg.eigvalsh((K + K.T) / 2)  # symmetrize for numerical safety
    return np.sort(eigvals)[::-1]


def effective_dimension(K):
    eig = spectrum(K)
    eig = np.clip(eig, 0, None)
    total = eig.sum()
    if total <= 1e-12:
        return 0.0
    p = eig / total
    return float(1.0 / np.sum(p ** 2))  # inverse participation ratio


def concentration(K):
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(K[mask].std())


def analyze_kernel_bank(kernel_matrices: dict, y):
    """
    kernel_matrices: dict name -> (n,n) numpy kernel matrix (train-train)
    y: labels for that same set, in {-1,+1}

    Returns a dict report with per-kernel metrics and pairwise kernel-kernel
    alignment (how similar the kernels are to EACH OTHER, not to the labels).
    """
    names = list(kernel_matrices.keys())
    report = {"per_kernel": {}, "pairwise_kernel_alignment": {}}

    for name, K in kernel_matrices.items():
        report["per_kernel"][name] = {
            "target_alignment": kernel_target_alignment(K, y),
            "effective_dimension": effective_dimension(K),
            "concentration_std": concentration(K),
            "n": K.shape[0],
        }

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            report["pairwise_kernel_alignment"][f"{a} vs {b}"] = kernel_kernel_alignment(
                kernel_matrices[a], kernel_matrices[b]
            )

    return report
