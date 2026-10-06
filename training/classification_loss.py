"""
training/classification_loss.py

Objective B (see plan section 10, Objective B — Classification Loss):
train the router + kernel3 theta against a loss that's directly tied to
CLASSIFICATION, not kernel-target alignment (Objective A). This matters
because Objective A optimizes a proxy (how well does K's structure match
same/different-label pairs) that may not correlate tightly with the
downstream SVM's actual decision boundary -- part of why the Oracle
analysis shows the router only captures ~42% of available headroom.

SVM training itself is a QP and not directly differentiable end-to-end
inside a PyTorch loop, so this uses the standard differentiable surrogate
from kernel-classifier literature: a leave-one-out WEIGHTED-NEIGHBOR
classifier score,

    f_i = sum_{j != i} K(i,j) * y_j  /  sum_{j != i} |K(i,j)|

(i.e. each training point casts a similarity-weighted vote using every
OTHER point's label; the point's own diagonal entry is masked out to avoid
the trivial self-vote), followed by a smooth classification loss (logistic
or hinge) on y_i * f_i. This is differentiable through K (hence through
the router weights and kernel3's theta) and is a much closer proxy to "is
this kernel actually good for classification" than kernel-target alignment.
"""

import torch


def classification_surrogate_loss(K, y, loss_type="logistic"):
    y_t = torch.as_tensor(y, dtype=torch.float32)
    n = K.shape[0]
    mask = 1.0 - torch.eye(n, dtype=K.dtype)
    K_masked = K * mask
    scores = K_masked @ y_t
    denom = K_masked.abs().sum(dim=1) + 1e-6
    scores = scores / denom

    margin = y_t * scores
    if loss_type == "logistic":
        loss = torch.nn.functional.softplus(-margin).mean()
    elif loss_type == "hinge":
        loss = torch.clamp(1.0 - margin, min=0.0).mean()
    else:
        raise ValueError(f"unknown loss_type: {loss_type}")
    return loss
