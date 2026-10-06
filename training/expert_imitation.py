"""Leakage-free, fold-local expert targets and gate cross entropy.

The API deliberately accepts raw training features/labels only. Each held-out
prediction excludes that fold's features/labels from every fitted transform,
quantum teacher and SVM. Targets may use held-out training labels to score
those predictions; no outer-test arrays enter this API.
"""
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from data.fold_preprocessing import AnglePreprocessor
from models.adaptive_qkernel import AdaptiveQuantumKernel


def make_targets(decisions, y, variant):
    decisions, y = np.asarray(decisions, float), np.asarray(y, float)
    if decisions.ndim != 2 or decisions.shape[0] != len(y) or not np.isfinite(decisions).all():
        raise ValueError("Finite (n_samples, n_experts) decisions required")
    if not np.isin(y, [-1, 1]).all():
        raise ValueError("Labels must be -1/+1")
    signed = decisions * y[:, None]
    correct = signed > 0  # exact zero is ambiguous: do not assign positive correctness
    valid = correct.any(axis=1)
    target = np.full_like(decisions, 1 / decisions.shape[1])
    if variant == 'hard':
        labels = np.argmax(np.where(correct, signed, -np.inf), axis=1)
        target[valid] = np.eye(decisions.shape[1])[labels[valid]]
    elif variant == 'soft':
        target[valid] = correct[valid] / correct[valid].sum(axis=1, keepdims=True)
    elif variant == 'margin_weighted':
        weights = np.maximum(signed[valid], 0)
        target[valid] = weights / weights.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown target variant: {variant}")
    return target, valid.astype(float)


def imitation_loss(logits, targets, weights):
    target = torch.as_tensor(targets, dtype=logits.dtype, device=logits.device)
    weight = torch.as_tensor(weights, dtype=logits.dtype, device=logits.device)
    if logits.shape != target.shape or weight.shape != logits.shape[:1]:
        raise ValueError("Target/weight shape mismatch")
    row_loss = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
    return (weight * row_loss).sum() / weight.sum().clamp_min(1)


def fit_fold_experts(X_raw, y, fit_idx, held_idx, config, seed):
    fit_idx, held_idx = np.asarray(fit_idx), np.asarray(held_idx)
    if len(np.intersect1d(fit_idx, held_idx)):
        raise ValueError("Expert fit and held-out indices overlap")
    prep = AnglePreprocessor(config['n_qubits'], seed).fit(X_raw[fit_idx])
    xf, xh = prep.transform(X_raw[fit_idx]), prep.transform(X_raw[held_idx])
    model = AdaptiveQuantumKernel(config['n_qubits'], seed=seed, kernel_backend=config['kernel_backend'])
    model.fit(xf, y[fit_idx], epochs=config['teacher_epochs'], lr=config['teacher_lr'], verbose=False)
    decisions = []
    for name in model.kernel_names:
        kt = model.single_kernel_matrix(name, xf, xf, symmetric=True)
        kh = model.single_kernel_matrix(name, xh, xf)
        svm = SVC(kernel='precomputed', C=config['svm_C']).fit(kt, y[fit_idx])
        decisions.append(svm.decision_function(kh))
    return np.stack(decisions, axis=1), {'fit_indices':fit_idx.tolist(), 'held_indices':held_idx.tolist(),
            'teacher_seed':seed, 'preprocessing':prep.state(), 'theta':model.theta.detach().tolist()}


def oof_expert_decisions(X_raw, y, config, seed):
    cv = StratifiedKFold(config['inner_folds'], shuffle=True, random_state=seed)
    d = np.full((len(y), 3), np.nan)
    folds = []
    for fit, held in cv.split(X_raw, y):
        d[held], provenance = fit_fold_experts(X_raw, y, fit, held, config, seed)
        folds.append(provenance)
    if not np.isfinite(d).all():
        raise RuntimeError("Incomplete OOF coverage")
    return d, folds
