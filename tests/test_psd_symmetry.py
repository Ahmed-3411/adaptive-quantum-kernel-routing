"""
tests/test_psd_symmetry.py

Roadmap Phase 0 requirement: "Add unit tests for symmetry, PSD, tensor
shapes, and gradients." Phase 1: "Numerically verify K ~ K^T. Inspect
Gram-matrix eigenvalues within a documented tolerance. Run a
quantum-parameter gradient/sensitivity test to ensure theta is actually
influential."

Run:
    cd adaptive-qkernel
    python3 -m pytest tests/ -v
    # or, without pytest:
    python3 tests/test_psd_symmetry.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.datasets import load_dataset
from models.adaptive_qkernel import AdaptiveQuantumKernel

TOL = 1e-6


def _make_model_and_data(router_type, seed=0):
    X_train, X_test, y_train, y_test = load_dataset(
        "wine", n_qubits=4, max_train=16, max_test=8, seed=seed
    )
    model = AdaptiveQuantumKernel(n_qubits=4, seed=seed, router_type=router_type)
    model.fit(X_train, y_train, epochs=8, lr=0.05, verbose=False)
    return model, X_train, X_test, y_train, y_test


def test_sample_level_is_symmetric():
    model, X_train, X_test, *_ = _make_model_and_data("sample_level")
    K = model.kernel_matrix(X_train, X_train, symmetric=True)
    asymmetry = np.abs(K - K.T).max()
    assert asymmetry < TOL, f"sample_level router produced asymmetric K: max|K-K.T|={asymmetry}"


def test_sample_level_is_psd():
    model, X_train, X_test, *_ = _make_model_and_data("sample_level")
    K = model.kernel_matrix(X_train, X_train, symmetric=True)
    eigvals = np.linalg.eigvalsh((K + K.T) / 2)
    assert eigvals.min() >= -TOL, f"sample_level router produced a non-PSD K: min eigenvalue={eigvals.min()}"


def test_per_pair_router_is_known_broken():
    """
    Documents the bug that motivated Phase 1, so a future refactor that
    accidentally "fixes" per_pair without realizing it was already broken
    doesn't silently invalidate this test. per_pair is NOT expected to be
    symmetric or PSD -- it is kept only for historical ablation comparison
    and must never be used for new headline results.
    """
    model, X_train, X_test, *_ = _make_model_and_data("per_pair")
    K = model.kernel_matrix(X_train, X_train, symmetric=True)
    asymmetry = np.abs(K - K.T).max()
    # This assertion is intentionally checking for the KNOWN bug, not asserting correctness.
    assert asymmetry > 1e-3, (
        "per_pair router is now symmetric -- if this was fixed intentionally, "
        "update this test and consider promoting per_pair back to production use."
    )


def test_global_and_uniform_are_symmetric_and_psd():
    for router_type in ["global", "uniform"]:
        model, X_train, X_test, *_ = _make_model_and_data(router_type)
        K = model.kernel_matrix(X_train, X_train, symmetric=True)
        asymmetry = np.abs(K - K.T).max()
        eigvals = np.linalg.eigvalsh((K + K.T) / 2)
        assert asymmetry < TOL, f"{router_type}: max|K-K.T|={asymmetry}"
        assert eigvals.min() >= -TOL, f"{router_type}: min eigenvalue={eigvals.min()}"


def test_cross_shape_train_test():
    model, X_train, X_test, *_ = _make_model_and_data("sample_level")
    K_tr = model.kernel_matrix(X_train, X_train, symmetric=True)
    K_te = model.kernel_matrix(X_test, X_train, symmetric=False)
    assert K_tr.shape == (len(X_train), len(X_train))
    assert K_te.shape == (len(X_test), len(X_train))


def test_theta_is_gradient_sensitive():
    """theta must actually change the kernel -- guards against the earlier
    unitary-cancellation bug (see quantum/feature_maps.py docstring) where
    theta had zero effect on every kernel value."""
    model, X_train, X_test, *_ = _make_model_and_data("sample_level")
    K_before = model.kernel_matrix(X_train, X_train, symmetric=True)
    with torch.no_grad():
        model.theta += 0.5
    K_after = model.kernel_matrix(X_train, X_train, symmetric=True)
    max_change = np.abs(K_after - K_before).max()
    assert max_change > 1e-3, f"theta perturbation had negligible effect on K: max_change={max_change}"


def test_theta_receives_nonzero_gradient():
    model, X_train, X_test, *_ = _make_model_and_data("sample_level")
    K = model.kernel_matrix(X_train, X_train, symmetric=True)  # detached, just for shape
    # recompute with grad enabled
    mats = model._all_kernel_matrices(X_train, X_train, symmetric=True)
    K_t, _ = model.adaptive_matrix(X_train, X_train, kernel_mats=mats, symmetric=True)
    loss = -(K_t.sum())
    loss.backward()
    assert model.theta.grad is not None
    grad_norm = model.theta.grad.norm().item()
    assert grad_norm > 1e-6, f"theta.grad is effectively zero: norm={grad_norm}"


ALL_TESTS = [
    test_sample_level_is_symmetric,
    test_sample_level_is_psd,
    test_per_pair_router_is_known_broken,
    test_global_and_uniform_are_symmetric_and_psd,
    test_cross_shape_train_test,
    test_theta_is_gradient_sensitive,
    test_theta_receives_nonzero_gradient,
]

if __name__ == "__main__":
    passed, failed = 0, 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
