"""
routing/classical_router.py

Version A - Classical Router (Phase 3):

    (xi, xj) --> Dense --> Dense --> Softmax --> [alpha_1, ..., alpha_M]

Input representation for a pair (xi, xj): concat(xi, xj, |xi - xj|).
Including the absolute difference gives the router an explicit notion of
"how far apart / in what direction" the pair is, which is what should
drive kernel selection (RQ1 / Experiment 3: sample-dependent weights).
"""

import torch
import torch.nn as nn


class ClassicalRouter(nn.Module):
    def __init__(self, n_features, n_kernels, hidden=16):
        super().__init__()
        in_dim = 3 * n_features  # xi, xj, |xi - xj|
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_kernels),
        )

    def pair_features(self, xi, xj):
        return torch.cat([xi, xj, torch.abs(xi - xj)], dim=-1)

    def forward(self, xi, xj):
        """xi, xj: (n_features,) tensors -> returns (n_kernels,) softmax weights"""
        feats = self.pair_features(xi, xj)
        logits = self.net(feats)
        return torch.softmax(logits, dim=-1)

    def weight_matrix(self, X):
        """
        Compute alpha_m(xi, xj) for every pair (i, j) in X (n x n).
        Returns tensor of shape (n, n, n_kernels).
        """
        X_t = torch.as_tensor(X, dtype=torch.float32)
        n = X_t.shape[0]
        out = torch.zeros(n, n, self.net[-1].out_features)
        for i in range(n):
            for j in range(n):
                out[i, j] = self.forward(X_t[i], X_t[j])
        return out

    def weight_matrix_cross(self, X1, X2):
        """Same as weight_matrix but for two possibly-different sets (train vs test)."""
        X1_t = torch.as_tensor(X1, dtype=torch.float32)
        X2_t = torch.as_tensor(X2, dtype=torch.float32)
        n1, n2 = X1_t.shape[0], X2_t.shape[0]
        out = torch.zeros(n1, n2, self.net[-1].out_features)
        for i in range(n1):
            for j in range(n2):
                out[i, j] = self.forward(X1_t[i], X2_t[j])
        return out
