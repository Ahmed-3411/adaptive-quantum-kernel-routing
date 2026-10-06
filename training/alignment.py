"""
training/alignment.py

Objective A - Kernel-Target Alignment:

    A(K, Y) = <K, Y>_F / (||K||_F * ||Y||_F)

Y is the ideal "label kernel": Y[i,j] = y_i * y_j  (y in {-1, +1}).
Maximizing A(K, Y) pulls K towards grouping same-label pairs together and
pushing different-label pairs apart -- without needing an SVM in the loop,
which keeps the objective differentiable and cheap during training.
"""

import torch


def label_kernel(y):
    y_t = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    return y_t @ y_t.T


def kernel_alignment(K, Y):
    """K, Y: (n, n) torch tensors. Returns scalar alignment in [-1, 1]."""
    K = K.to(torch.float32)
    Y = Y.to(torch.float32)
    num = torch.sum(K * Y)
    denom = torch.norm(K) * torch.norm(Y) + 1e-12
    return num / denom


def alignment_loss(K, y):
    """Loss to minimize = -alignment (so gradient ascent on alignment)."""
    Y = label_kernel(y)
    return -kernel_alignment(K, Y)
