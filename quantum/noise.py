"""
quantum/noise.py

Phase 5 - Noise-Aware Training / Experiment 4 (Ideal -> Low -> Medium -> High).

We model NISQ noise using PennyLane's `default.mixed` density-matrix device
with its built-in noise channels (DepolarizingChannel, BitFlip, PhaseFlip),
applied per-qubit right after each half of the overlap circuit (after
encoding x1, and again after the adjoint-encoding of x2). This mirrors the
noise types the plan calls for (Qiskit Aer's depolarizing / bit-flip /
phase-flip models) while staying inside the PennyLane+PyTorch pipeline --
routing to a real Qiskit Aer backend would break autodiff for the trainable
Kernel 3 theta, since Aer circuits aren't differentiable through PyTorch.
If the project later needs Aer-specific profiles (e.g. a real IBM device's
calibrated noise), swap the device in `build_noisy_overlap_qnode` for a
`qiskit.aer` device via pennylane-qiskit -- nothing else in the codebase
needs to change.

NOTE: this module deliberately does NOT reuse the batched/vectorized
computation from quantum/kernels.py. `default.mixed` is a density-matrix
simulator (state size 4^n instead of 2^n) and noise experiments run on the
same small (~20-30 sample) prototyping datasets, so a plain loop is simplest
and correct; vectorize this the same way if/when noise experiments need to
scale up.
"""

import numpy as np
import torch
import pennylane as qml

NOISE_CHANNELS = {
    "depolarizing": qml.DepolarizingChannel,
    "bit_flip": qml.BitFlip,
    "phase_flip": qml.PhaseFlip,
}

NOISE_LEVELS = [0.00, 0.01, 0.03, 0.05, 0.10]  # matches section 14 of the plan


def _apply_noise(noise_type, noise_level, wires):
    if noise_level <= 0:
        return
    channel = NOISE_CHANNELS[noise_type]
    for w in wires:
        channel(noise_level, wires=w)


def build_noisy_overlap_qnode(feature_map_fn, n_wires, trainable,
                               noise_type="depolarizing", noise_level=0.0):
    dev = qml.device("default.mixed", wires=n_wires)
    interface = "torch" if trainable else "autograd"

    @qml.qnode(dev, interface=interface)
    def circuit(x1, x2, theta=None):
        wires = list(range(n_wires))
        if theta is not None:
            feature_map_fn(x1, theta, wires=wires)
        else:
            feature_map_fn(x1, wires=wires)
        _apply_noise(noise_type, noise_level, wires)
        if theta is not None:
            qml.adjoint(feature_map_fn)(x2, theta, wires=wires)
        else:
            qml.adjoint(feature_map_fn)(x2, wires=wires)
        _apply_noise(noise_type, noise_level, wires)
        return qml.probs(wires=wires)

    return circuit


class NoisyQuantumKernel:
    """
    Same batched-broadcasting strategy as quantum.kernels.QuantumKernel
    (see that file's docstring), applied to the noisy density-matrix circuit.
    `default.mixed` supports PennyLane parameter broadcasting, so we can
    stack every (xi, xj) pair that needs evaluating into one vectorized
    QNode call instead of one Python-level call per pair -- verified to give
    numerically identical results to the per-pair loop, and much faster.
    """

    def __init__(self, name, feature_map_fn, n_wires, trainable,
                 noise_type="depolarizing", noise_level=0.0, batch_size=128):
        self.name = name
        self.trainable = trainable
        self.batch_size = batch_size
        self.qnode = build_noisy_overlap_qnode(
            feature_map_fn, n_wires, trainable, noise_type, noise_level
        )

    def matrix(self, X1, X2, theta=None, symmetric=False):
        n1, n2 = len(X1), len(X2)
        pairs = [(i, j) for i in range(n1) for j in range((i if symmetric else 0), n2)]

        if self.trainable:
            K = torch.zeros((n1, n2), dtype=torch.float32)
            for start in range(0, len(pairs), self.batch_size):
                chunk = pairs[start:start + self.batch_size]
                x1_batch = torch.as_tensor(np.stack([X1[i] for i, _ in chunk]), dtype=torch.float64)
                x2_batch = torch.as_tensor(np.stack([X2[j] for _, j in chunk]), dtype=torch.float64)
                probs = self.qnode(x1_batch, x2_batch, theta=theta)
                vals = probs[:, 0]
                for (i, j), v in zip(chunk, vals):
                    K[i, j] = v
                    if symmetric:
                        K[j, i] = v
            return K
        else:
            K = np.zeros((n1, n2))
            for start in range(0, len(pairs), self.batch_size):
                chunk = pairs[start:start + self.batch_size]
                x1_batch = np.stack([X1[i] for i, _ in chunk])
                x2_batch = np.stack([X2[j] for _, j in chunk])
                probs = self.qnode(x1_batch, x2_batch)
                vals = np.asarray(probs)[:, 0]
                for (i, j), v in zip(chunk, vals):
                    K[i, j] = v
                    if symmetric:
                        K[j, i] = v
            return K


def build_noisy_kernel_bank(feature_maps_dict, n_qubits, noise_type="depolarizing", noise_level=0.0):
    return {
        name: NoisyQuantumKernel(name, spec["fn"], n_qubits, spec["trainable"], noise_type, noise_level)
        for name, spec in feature_maps_dict.items()
    }
