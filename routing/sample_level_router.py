"""
routing/sample_level_router.py

Phase 1 of the research roadmap: PSD-PRESERVING ADAPTIVE KERNEL.

The original per-pair router computes weight(xi, xj) directly from the pair
(via concat(xi, xj, |xi-xj|)). Nothing forces weight(xi, xj) == weight(xj, xi),
so the resulting combined kernel K_adaptive is generally NOT symmetric and
NOT positive semi-definite -- confirmed numerically: max|K-K.T| = 0.338,
min eigenvalue = -0.135 on a real run. A non-PSD, asymmetric matrix is not a
valid kernel; SVC(kernel="precomputed") is not guaranteed to behave sensibly
on it, and every earlier result in this project used this construction.

Fix (exactly the roadmap's Phase 1 recipe):

    g(x) = softmax(f_phi(x))            -- a SAMPLE-level gate, not a pair-level one
    K_A(xi, xj) = sum_m g_m(xi) g_m(xj) K_m(xi, xj)
                = sum_m [D_m K_m D_m]_{ij},   D_m = diag(g_m(x_1), ..., g_m(x_n))

Why this is symmetric: g_m(xi) g_m(xj) K_m(xi,xj) is manifestly symmetric
under swapping i and j (K_m itself is symmetric, and the scalar product
g_m(xi)*g_m(xj) doesn't care about order).

Why this is PSD: each K_m is a valid quantum fidelity kernel
K_m(x,x') = |<psi_m(x)|psi_m(x')>|^2 = Tr[rho_m(x) rho_m(x')], a genuine
inner product of density operators, hence PSD. For any real diagonal D_m,
v^T (D_m K_m D_m) v = (D_m v)^T K_m (D_m v) >= 0, so D_m K_m D_m is PSD.
A sum of PSD matrices is PSD, so K_A is PSD. This is the standard
"positive combination of kernels is a kernel" argument, made pair-dependent
through g(x) instead of a single global scalar.
"""

import torch
import torch.nn as nn


class SampleLevelGate(nn.Module):
    def __init__(self, n_features, n_kernels, hidden=16, architecture="mlp"):
        super().__init__()
        self.n_kernels = n_kernels
        self.architecture = architecture
        if architecture == "linear":
            # near-linear gate: single Linear layer, no hidden layers/nonlinearity.
            # Used to test whether the current 2-hidden-layer MLP's capacity is
            # itself the bottleneck (ROADMAP_STATUS.md optimization ablation,
            # step (b)) -- if a linear gate performs the same, capacity isn't
            # the issue either.
            self.net = nn.Linear(n_features, n_kernels)
        else:
            self.net = nn.Sequential(
                nn.Linear(n_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_kernels),
            )

    def forward(self, x):
        """x: (n_features,) -> (n_kernels,) softmax gate."""
        return torch.softmax(self.net(x), dim=-1)

    def gate_matrix(self, X):
        """X: (n, n_features) numpy or tensor -> (n, n_kernels) gate tensor.
        O(n) forward passes (vs. O(n^2) for the old pair-level router)."""
        X_t = torch.as_tensor(X, dtype=torch.float32)
        return self.net(X_t).softmax(dim=-1)


def psd_combine(gates1, gates2, kernel_mats, kernel_names):
    """
    gates1: (n1, M) gate tensor for the first sample set
    gates2: (n2, M) gate tensor for the second sample set
    kernel_mats: dict name -> (n1, n2) tensor
    Returns: (n1, n2) combined kernel tensor, symmetric and PSD when
             gates1 is gates2 (i.e. X1 is X2) and kernel_mats are exact.
    """
    stacked = torch.stack([kernel_mats[n] for n in kernel_names], dim=-1)  # (n1, n2, M)
    g1 = gates1.unsqueeze(1)  # (n1, 1, M)
    g2 = gates2.unsqueeze(0)  # (1, n2, M)
    weights = g1 * g2         # (n1, n2, M) -- symmetric when g1 is g2
    return (weights * stacked).sum(dim=-1)
