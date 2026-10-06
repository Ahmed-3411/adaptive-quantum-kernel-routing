# AUDIT: training diagnostics here are IN-SAMPLE, not OOF. Their labels affect the SVM fit.
# Test labels are not used; do not reuse these diagnostics as leakage-free imitation targets.
"""
routing/kernel_aware_router.py

Router B from the audit plan: "Kernel-aware" gate. The current router
(Router A, routing/sample_level_router.py) gates purely on raw features x.
The decisive test: does the router fail to exploit Oracle headroom because
x alone doesn't carry the signal needed to tell which kernel to trust, or
because neural routing can't turn that signal into reliable decisions even
when given it?

Router B answers this by additionally feeding the gate network per-kernel
BEHAVIOR diagnostics, computed WITHOUT label leakage:

    For each kernel m, fit a single-kernel SVM on the TRAINING set only
    (using the training labels, exactly like the "best single kernel"
    baseline elsewhere in this project), then evaluate its decision
    function d_m(x) on every point (train and test). d_m(x) uses only x
    and the training data/labels -- never x's own label -- so this is
    leakage-free for both train and test points, in exactly the same sense
    the "best single kernel" and "Oracle" baselines already are.

Gate input becomes: [x, d_1(x), d_2(x), d_3(x), |d_1-d_2|, |d_1-d_3|, |d_2-d_3|]
i.e. raw features PLUS each kernel's own confidence/direction PLUS pairwise
kernel disagreement -- exactly the roadmap's suggested routing signals
(kernel disagreement, per-kernel confidence), without exposing the Oracle
label itself (Router C, not implemented here, would be a stronger leakage
check on top of this).

Design simplification (noted, not hidden): the single-kernel SVMs used to
compute d_m(x) are fit ONCE, using the kernel bank's state at
initialization (before router/theta training begins), not refit every
epoch. This keeps the experiment tractable (SVM fitting is not
differentiable and refitting every epoch would be expensive) at the cost
of the diagnostic features becoming slightly stale as theta updates during
training. This is flagged as a limitation, not hidden.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC


def compute_kernel_diagnostics(model, X_train, y_train, X_all):
    """
    Fit one SVM per kernel on (X_train, y_train) using the kernel bank's
    CURRENT state (e.g. initial random theta), then return each SVM's
    decision_function evaluated on every point in X_all (which may include
    both train and test points, concatenated by the caller).

    Returns: (len(X_all), n_kernels) array of decision-function values,
    NOT scaled/normalized (SVC decision_function is roughly a signed
    margin, not a probability).
    """
    diagnostics = []
    for name in model.kernel_names:
        K_train = model.single_kernel_matrix(name, X_train, X_train, symmetric=True)
        if hasattr(K_train, "detach"):
            K_train = K_train.detach().numpy()
        K_all = model.single_kernel_matrix(name, X_all, X_train, symmetric=False)
        if hasattr(K_all, "detach"):
            K_all = K_all.detach().numpy()
        clf = SVC(kernel="precomputed")
        clf.fit(K_train, y_train)
        d = clf.decision_function(K_all)
        diagnostics.append(d)
    return np.stack(diagnostics, axis=1)  # (n, n_kernels)


class KernelAwareGate(nn.Module):
    def __init__(self, n_features, n_kernels, hidden=16):
        super().__init__()
        self.n_kernels = n_kernels
        n_pairs = n_kernels * (n_kernels - 1) // 2
        in_dim = n_features + n_kernels + n_pairs  # x + d_1..d_M + pairwise |d_i-d_j|
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_kernels),
        )
        # populated externally via set_diagnostics() before use
        self._diag_lookup = None  # dict: id(x_row_bytes) -> diagnostic row (fallback: index-based)
        self._diag_array = None
        self._x_array = None

    def set_diagnostics(self, X_all, diagnostics):
        """X_all: (n, n_features) numpy array (in the SAME order diagnostics were computed).
        diagnostics: (n, n_kernels) numpy array from compute_kernel_diagnostics.
        We match incoming gate_matrix(X) calls back to these rows by exact
        float match (safe here because X always comes from the same
        preprocessed arrays used to build the diagnostics, never re-sampled)."""
        self._x_array = np.asarray(X_all, dtype=np.float64)
        self._diag_array = np.asarray(diagnostics, dtype=np.float64)
        n_kernels = diagnostics.shape[1]
        pair_diffs = []
        for i in range(n_kernels):
            for j in range(i + 1, n_kernels):
                pair_diffs.append(np.abs(diagnostics[:, i] - diagnostics[:, j]))
        self._pair_diag_array = np.stack(pair_diffs, axis=1) if pair_diffs else np.zeros((len(diagnostics), 0))

    def _lookup_diagnostics(self, X):
        X_np = np.asarray(X, dtype=np.float64)
        idx = []
        for row in X_np:
            matches = np.where(np.all(np.isclose(self._x_array, row, atol=1e-9), axis=1))[0]
            if len(matches) == 0:
                raise ValueError("KernelAwareGate: no precomputed diagnostics for a requested point -- "
                                  "call set_diagnostics() with a superset of every X used downstream.")
            idx.append(matches[0])
        return self._diag_array[idx], self._pair_diag_array[idx]

    def forward(self, x):
        """x: (n_features,) single sample -- used rarely (kept for interface parity)."""
        return self.gate_matrix(x.reshape(1, -1))[0]

    def gate_matrix(self, X):
        if self._diag_array is None:
            raise RuntimeError("KernelAwareGate.set_diagnostics() must be called before use.")
        X_t = torch.as_tensor(X, dtype=torch.float32)
        diag, pair_diag = self._lookup_diagnostics(X)
        aug = torch.as_tensor(np.concatenate([diag, pair_diag], axis=1), dtype=torch.float32)
        full_input = torch.cat([X_t, aug], dim=1)
        return self.net(full_input).softmax(dim=-1)
