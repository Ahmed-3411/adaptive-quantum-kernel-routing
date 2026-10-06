"""
models/adaptive_qkernel.py

Kadaptive(xi, xj) = sum_m alpha_m(xi, xj) * K_m(xi, xj)

Joint training (Phase 4 - End-to-End Training):
    trainable params = router weights (phi) + Kernel-3 circuit params (theta)
    objective        = kernel-target alignment (Objective A)

This is the first working prototype requested: Experiment 2 (Single best
kernel vs Adaptive) and Experiment 3 (global vs sample-dependent weights)
both fall out of this same class.
"""

import numpy as np
import torch

from quantum.feature_maps import FEATURE_MAPS, TRAINABLE_THETA_SHAPE
from quantum.kernels import build_kernel_bank
from routing.classical_router import ClassicalRouter
from routing.global_router import GlobalRouter, UniformRouter
from routing.sample_level_router import SampleLevelGate, psd_combine
from training.alignment import alignment_loss, label_kernel, kernel_alignment

import functools

ROUTER_TYPES = {
    "per_pair": ClassicalRouter,      # ORIGINAL router (Version A). Kept for
                                       # comparison only -- confirmed NOT
                                       # symmetric/PSD (max|K-K.T|=0.338,
                                       # min eigenvalue=-0.135 on a real run).
                                       # Do not use for new results.
    "global": GlobalRouter,           # ablation: one learned weight vector, not sample-dependent
    "uniform": UniformRouter,         # ablation: fixed 1/M weights, nothing trained at all
    "sample_level": SampleLevelGate,  # PSD-preserving (roadmap Phase 1, default going forward)
    "sample_level_linear": functools.partial(SampleLevelGate, architecture="linear"),
                                       # optimization ablation: same PSD-preserving
                                       # construction, but a single Linear layer
                                       # instead of a 2-hidden-layer MLP -- tests
                                       # whether gate CAPACITY is the bottleneck.
}


class AdaptiveQuantumKernel:
    def __init__(self, n_qubits, seed=0, router_type="sample_level", kernel_subset=None, kernel_backend="overlap"):
        torch.manual_seed(seed)
        self.n_qubits = n_qubits
        self.router_type = router_type
        full_bank = build_kernel_bank(FEATURE_MAPS, n_qubits, backend=kernel_backend)
        if kernel_subset is not None:
            # ablation: restrict the kernel bank to a subset (e.g. drop Kernel 3)
            self.bank = {k: v for k, v in full_bank.items() if k in kernel_subset}
        else:
            self.bank = full_bank
        self.kernel_names = list(self.bank.keys())
        self.n_kernels = len(self.kernel_names)
        router_cls = ROUTER_TYPES[router_type]
        self.router = router_cls(n_features=n_qubits, n_kernels=self.n_kernels)
        # trainable circuit params for kernel3 -- shape (n_qubits, *TRAINABLE_THETA_SHAPE).
        # Kept in sync with quantum/feature_maps.py's deep_trainable_feature_map,
        # which now uses 5 angles per wire (2 local layers + 1 entangling IsingZZ angle)
        # instead of the original 2 (see feature_maps.py for the full redesign story).
        self.theta = torch.nn.Parameter(
            0.1 * torch.randn(n_qubits, *TRAINABLE_THETA_SHAPE, dtype=torch.float64)
        )

    # ---- fixed kernels are cheap to precompute once per dataset ----
    def _fixed_matrix(self, name, X1, X2, symmetric):
        return torch.as_tensor(
            self.bank[name].matrix(X1, X2, symmetric=symmetric), dtype=torch.float32
        )

    def _all_kernel_matrices(self, X1, X2, symmetric):
        mats = {}
        for name in self.kernel_names:
            if self.bank[name].trainable:
                mats[name] = self.bank[name].matrix(
                    X1, X2, theta=self.theta, symmetric=symmetric
                ).to(torch.float32)
            else:
                mats[name] = self._fixed_matrix(name, X1, X2, symmetric)
        return mats

    def adaptive_matrix(self, X1, X2, kernel_mats=None, symmetric=False):
        if kernel_mats is None:
            kernel_mats = self._all_kernel_matrices(X1, X2, symmetric)
        if self.router_type in ("sample_level", "sample_level_linear"):
            gates1 = self.router.gate_matrix(X1)
            gates2 = gates1 if symmetric else self.router.gate_matrix(X2)
            K = psd_combine(gates1, gates2, kernel_mats, self.kernel_names)
        else:
            weights = self.router.weight_matrix_cross(X1, X2)  # (n1, n2, M)
            stacked = torch.stack([kernel_mats[n] for n in self.kernel_names], dim=-1)
            K = torch.sum(weights * stacked, dim=-1)
        return K, kernel_mats

    def fit(self, X_train, y_train, epochs=25, lr=0.05, lambda_complexity=0.0,
            noise_type=None, noise_level=0.0, verbose=True, objective="alignment"):
        """
        Joint training of router phi + kernel3 theta.

        objective="alignment" (default, Objective A): kernel-target alignment
            (training/alignment.py). Optimizes a mathematical property of K
            correlated with, but not identical to, downstream SVM accuracy.
        objective="classification" (Objective B): differentiable
            classification-surrogate loss (training/classification_loss.py)
            -- a leave-one-out weighted-neighbor classifier score, trained
            with a logistic loss. More directly tied to "is this kernel
            actually good for classification" -- see if it closes some of
            the Oracle gap that Objective A leaves on the table
            (experiments/oracle.py: only ~42% of headroom captured).

        lambda_complexity>0 lightly penalizes router entropy
        collapse-to-single-kernel-everywhere, i.e. encourages using diversity
        only where it helps (a cheap stand-in for Objective C's complexity term).

        noise_level > 0 turns this into NOISE-AWARE training (Phase 5's actual
        intent): the router and theta see NOISY kernel values during training
        itself, via quantum.noise.NoisyQuantumKernel, instead of training on
        the ideal simulator and only testing under noise afterwards (which is
        what experiments/noise.py does). Comparing a model fit() with
        noise_level=0 against one fit() with noise_level>0, then evaluating
        BOTH under a noise sweep, answers "does training under noise actually
        make the learned kernel more robust" -- not just "does noise hurt".
        """
        if objective not in ("alignment", "classification"):
            raise ValueError(f"Unknown objective: {objective}; use training.expert_imitation for imitation")
        params = list(self.router.parameters())
        trainable_names = [name for name in self.kernel_names if self.bank[name].trainable]
        if trainable_names:
            params = params + [self.theta]
        opt = torch.optim.Adam(params, lr=lr)

        if noise_level > 0:
            from quantum.noise import build_noisy_kernel_bank
            bank = build_noisy_kernel_bank(
                FEATURE_MAPS, self.n_qubits,
                noise_type=noise_type or "depolarizing", noise_level=noise_level,
            )
            bank = {k: v for k, v in bank.items() if k in self.kernel_names}  # respect kernel_subset
        else:
            bank = self.bank

        # Fixed kernels don't change across epochs -> precompute once.
        fixed_cache = {}
        for name in self.kernel_names:
            if not bank[name].trainable:
                fixed_cache[name] = torch.as_tensor(
                    bank[name].matrix(X_train, X_train, symmetric=True), dtype=torch.float32
                )

        history = []
        for ep in range(epochs):
            opt.zero_grad()
            mats = dict(fixed_cache)
            # generic over which kernel (if any) is trainable -- ablations may
            # drop Kernel 3 entirely, in which case trainable_names is empty
            # and this loop is a no-op (mats already has every kernel from
            # fixed_cache above).
            for name in trainable_names:
                mats[name] = bank[name].matrix(
                    X_train, X_train, theta=self.theta, symmetric=True
                ).to(torch.float32)

            K, _ = self.adaptive_matrix(X_train, X_train, kernel_mats=mats, symmetric=True)
            if objective == "classification":
                from training.classification_loss import classification_surrogate_loss
                loss = classification_surrogate_loss(K, y_train, loss_type="logistic")
            else:
                loss = alignment_loss(K, y_train)

            if lambda_complexity > 0:
                if self.router_type in ("sample_level", "sample_level_linear"):
                    g = self.router.gate_matrix(X_train)
                    entropy = -(g * torch.log(g + 1e-9)).sum(-1).mean()
                else:
                    w = self.router.weight_matrix(X_train)
                    entropy = -(w * torch.log(w + 1e-9)).sum(-1).mean()
                loss = loss - lambda_complexity * entropy  # encourage some diversity

            loss.backward()
            opt.step()
            history.append(loss.item())
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"  epoch {ep:3d}  {objective} loss = {loss.item():.4f}")
        return history

    def kernel_matrix(self, X1, X2, symmetric=False):
        with torch.no_grad():
            K, _ = self.adaptive_matrix(X1, X2, symmetric=symmetric)
        return K.numpy().astype(np.float64)

    def single_kernel_matrix(self, name, X1, X2, symmetric=False):
        with torch.no_grad():
            if self.bank[name].trainable:
                K = self.bank[name].matrix(X1, X2, theta=self.theta, symmetric=symmetric)
                return K.numpy().astype(np.float64)
            return self.bank[name].matrix(X1, X2, symmetric=symmetric)
