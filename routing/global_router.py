"""
routing/global_router.py

Ablation variants of the router, matching the same interface as
ClassicalRouter (forward / weight_matrix / weight_matrix_cross) so
AdaptiveQuantumKernel can swap between them without other code changes.

    GlobalRouter  -- ONE learned (n_kernels,) softmax weight vector, shared
                     by every pair. This isolates "does learning the mixture
                     weights help" from "does making the mixture SAMPLE-
                     DEPENDENT help" (the plan's own Experiment 3: Global vs
                     Local Adaptation). If GlobalRouter performs as well as
                     the per-pair ClassicalRouter, the benefit isn't coming
                     from adaptivity -- just from finding good fixed weights.

    UniformRouter -- fixed 1/n_kernels weights, NOT trained at all. The
                     "dumbest possible" baseline: plain averaging of the
                     kernel bank. If this alone matches the adaptive
                     router's benefit, neither training nor routing is
                     doing anything -- the kernel bank itself (e.g. having
                     Kernel 3 in the mix) is what matters.
"""

import torch
import torch.nn as nn


class GlobalRouter(nn.Module):
    def __init__(self, n_features, n_kernels, hidden=None):
        super().__init__()
        self.n_kernels = n_kernels
        self.logits = nn.Parameter(torch.zeros(n_kernels))  # start uniform

    def forward(self, xi=None, xj=None):
        return torch.softmax(self.logits, dim=-1)

    def weight_matrix(self, X):
        n = len(X)
        w = self.forward()
        return w.view(1, 1, -1).expand(n, n, -1)

    def weight_matrix_cross(self, X1, X2):
        n1, n2 = len(X1), len(X2)
        w = self.forward()
        return w.view(1, 1, -1).expand(n1, n2, -1)


class UniformRouter(nn.Module):
    def __init__(self, n_features, n_kernels, hidden=None):
        super().__init__()
        self.n_kernels = n_kernels
        self.register_buffer("weights", torch.full((n_kernels,), 1.0 / n_kernels))

    def parameters(self, recurse=True):
        return iter([])  # nothing to train

    def forward(self, xi=None, xj=None):
        return self.weights

    def weight_matrix(self, X):
        n = len(X)
        return self.weights.view(1, 1, -1).expand(n, n, -1)

    def weight_matrix_cross(self, X1, X2):
        n1, n2 = len(X1), len(X2)
        return self.weights.view(1, 1, -1).expand(n1, n2, -1)
