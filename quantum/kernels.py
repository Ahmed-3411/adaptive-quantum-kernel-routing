"""
quantum/kernels.py

Implements  K(xi, xj) = |<0| U(xj)^dagger U(xi) |0>|^2  (the overlap/adjoint
test) for each feature map in the kernel bank.

Two modes:
    - fixed kernels (Kernel 1, Kernel 2): plain numpy, computed once and cached.
    - trainable kernel (Kernel 3): torch interface with `diff_method="backprop"`
      so its `theta` parameters receive gradients from the training objective.
"""

import numpy as np
import torch
import pennylane as qml


def _build_qnode(feature_map_fn, n_wires, trainable, x1_is_input=True):
    dev = qml.device("default.qubit", wires=n_wires)
    interface = "torch" if trainable else "autograd"
    diff_method = "backprop"

    @qml.qnode(dev, interface=interface, diff_method=diff_method)
    def circuit(x1, x2, theta=None):
        if theta is not None:
            feature_map_fn(x1, theta, wires=list(range(n_wires)))
            qml.adjoint(feature_map_fn)(x2, theta, wires=list(range(n_wires)))
        else:
            feature_map_fn(x1, wires=list(range(n_wires)))
            qml.adjoint(feature_map_fn)(x2, wires=list(range(n_wires)))
        return qml.probs(wires=list(range(n_wires)))

    return circuit


def _pair_indices(n1, n2, symmetric):
    """Return the (i, j) index pairs that actually need a circuit evaluation."""
    if symmetric:
        return [(i, j) for i in range(n1) for j in range(i, n2)]
    return [(i, j) for i in range(n1) for j in range(n2)]


class QuantumKernel:
    """
    Wraps a feature map into a callable kernel object.

    fixed kernels  -> .matrix(X1, X2) returns a numpy array (no grad).
    trainable kernel -> .matrix(X1, X2, theta) returns a torch tensor with
                         gradients flowing into `theta`.

    Both branches use PennyLane's *parameter broadcasting*: instead of calling
    the QNode once per pair (Python-level loop with per-call tracing overhead),
    we stack every (xi, xj) pair that needs evaluating into a single batched
    call. This is the same circuit executed many times, just dispatched as one
    vectorized simulator call instead of n1*n2 separate ones -- large speedup
    for the O(n^2) kernel-matrix workload, with identical numerical results.
    """

    def __init__(self, name, feature_map_fn, n_wires, trainable, batch_size=256, backend="overlap"):
        if backend not in ("overlap", "statevector"):
            raise ValueError("Unknown kernel backend")
        self.backend = backend
        self.name = name
        self.n_wires = n_wires
        self.trainable = trainable
        self.batch_size = batch_size
        self.qnode = _build_qnode(feature_map_fn, n_wires, trainable)
        dev = qml.device("default.qubit", wires=n_wires)
        @qml.qnode(dev, interface="torch" if trainable else "autograd", diff_method="backprop")
        def state_node(x, theta=None):
            if theta is None:
                feature_map_fn(x, wires=list(range(n_wires)))
            else:
                feature_map_fn(x, theta, wires=list(range(n_wires)))
            return qml.state()
        self.state_node = state_node

    def matrix(self, X1, X2, theta=None, symmetric=False):
        if symmetric and (np.shape(X1) != np.shape(X2) or not np.array_equal(X1, X2)):
            raise ValueError("symmetric=True requires identical sample arrays")
        if self.backend == "statevector":
            if self.trainable:
                a = self.state_node(torch.as_tensor(X1, dtype=torch.float64), theta)
                b = a if symmetric else self.state_node(torch.as_tensor(X2, dtype=torch.float64), theta)
                return (a @ b.conj().T).abs().square().to(torch.float32)
            a = np.asarray(self.state_node(np.asarray(X1)))
            b = a if symmetric else np.asarray(self.state_node(np.asarray(X2)))
            return np.abs(a @ b.conj().T) ** 2
        n1, n2 = len(X1), len(X2)
        pairs = _pair_indices(n1, n2, symmetric)

        if self.trainable:
            K = torch.zeros((n1, n2), dtype=torch.float32)
            for start in range(0, len(pairs), self.batch_size):
                chunk = pairs[start:start + self.batch_size]
                x1_batch = torch.as_tensor(
                    np.stack([X1[i] for i, _ in chunk]), dtype=torch.float64
                )
                x2_batch = torch.as_tensor(
                    np.stack([X2[j] for _, j in chunk]), dtype=torch.float64
                )
                probs = self.qnode(x1_batch, x2_batch, theta=theta)  # (batch, 2**n_wires)
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
                probs = self.qnode(x1_batch, x2_batch)  # (batch, 2**n_wires)
                vals = np.asarray(probs)[:, 0]
                for (i, j), v in zip(chunk, vals):
                    K[i, j] = v
                    if symmetric:
                        K[j, i] = v
            return K


def build_kernel_bank(feature_maps_dict, n_qubits, backend="overlap"):
    """
    feature_maps_dict: quantum.feature_maps.FEATURE_MAPS
    Returns dict name -> QuantumKernel
    """
    bank = {}
    for name, spec in feature_maps_dict.items():
        bank[name] = QuantumKernel(name, spec["fn"], n_qubits, spec["trainable"], backend=backend)
    return bank
