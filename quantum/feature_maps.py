"""
quantum/feature_maps.py

REDESIGNED kernel bank (see README.md "kernel diversity" section for the
full story of why): the original three feature maps all shared the same
basic RY-encoding skeleton and only differed in small details (an extra
CNOT ring, a trainable rotation layer). `evaluation/kernel_analysis.py`
showed they were 90-96% similar to each other as KERNEL FUNCTIONS (pairwise
kernel-kernel alignment) on every dataset tested, even after fixing two
earlier unitary-cancellation bugs -- which fully explained why the adaptive
router could never beat the best single kernel: there was nothing genuinely
different to route between.

This version makes the three feature maps structurally different, not just
detail-different, along THREE independent axes at once (basis, entangling
topology, and what "trainable" means):

    Kernel 1 - IQP-style encoding (Havlicek et al. 2019 flavor):
        Z-basis + Hadamard + pairwise PRODUCT-of-feature interactions
        (RZ(x_i), then RZ((pi-x_i)(pi-x_j)) on entangled pairs). This is a
        genuinely different function class from a pure rotation encoding --
        it depends on feature PRODUCTS, not just individual feature values,
        and is the standard example of a feature map believed to be hard to
        evaluate classically.

    Kernel 2 - Combined-axis encoding + star entanglement:
        RY(x_i) AND RX(x_i) per qubit (a richer single-qubit trajectory than
        one axis), entangled with a STAR topology (qubit 0 controls every
        other qubit) instead of the nearest-neighbor ring used elsewhere --
        a different entanglement GRAPH, not just a different gate count.

    Kernel 3 - Deep trainable re-uploading with TRAINABLE two-qubit gates:
        Two data re-uploading layers, but the trainable block between them
        uses entangling IsingZZ(theta) gates (not just local RY/RZ rotations
        like the old version) -- so training can reshape which qubits are
        correlated, not just their local orientation. theta shape is
        (n_wires, 5): columns are [layer1_RY, layer1_RZ, ZZ_angle,
        layer2_RY, layer2_RZ].

Each feature map is a function  U(x, params) -> applies gates to `wires`.
The kernel value between two points is computed in quantum/kernels.py via
the standard overlap/adjoint test:  K(xi, xj) = |<0|U(xj)^dagger U(xi)|0>|^2

`x[..., i]` (not `x[i]`) is required everywhere to correctly select feature i
under PennyLane's parameter broadcasting (see quantum/kernels.py docstring).
"""

import numpy as np
import pennylane as qml


def iqp_encoding(x, wires):
    """Kernel 1: IQP-style encoding -- Hadamard basis + single-feature RZ +
    pairwise PRODUCT-of-feature RZ interactions on a ring of entangled pairs.
    Depends on feature PRODUCTS (x_i * x_j), not just individual features --
    a fundamentally different function class from a plain rotation encoding.
    """
    n = len(wires)
    for w in wires:
        qml.Hadamard(wires=w)
    for i, w in enumerate(wires):
        qml.RZ(x[..., i], wires=w)
    for i in range(n):
        j = (i + 1) % n
        wi, wj = wires[i], wires[j]
        qml.CNOT(wires=[wi, wj])
        qml.RZ((np.pi - x[..., i]) * (np.pi - x[..., j]), wires=wj)
        qml.CNOT(wires=[wi, wj])


def xy_star_encoding(x, wires):
    """Kernel 2: combined RY+RX single-qubit encoding (two rotation axes
    instead of one), entangled with a STAR graph (wire 0 controls every
    other wire) -- a different entanglement topology from the ring used in
    Kernel 1 and Kernel 3, not just a different gate count.
    """
    n = len(wires)
    for i, w in enumerate(wires):
        qml.RY(x[..., i], wires=w)
    for i, w in enumerate(wires):
        qml.RX(x[..., i], wires=w)
    for i in range(1, n):
        qml.CNOT(wires=[wires[0], wires[i]])


def deep_trainable_feature_map(x, theta, wires):
    """Kernel 3: two data re-uploading layers, with a trainable block in
    between that includes TRAINABLE TWO-QUBIT gates (IsingZZ), not just
    local single-qubit rotations -- so training can reshape entanglement
    structure, not just per-qubit orientation. This is structurally
    different from Kernel 1 (fixed IQP entangling angles, driven by data
    products) and Kernel 2 (fixed star topology, no training at all).

    theta shape: (len(wires), 5) ->
        [:,0] = layer-1 trainable RY,  [:,1] = layer-1 trainable RZ
        [:,2] = trainable IsingZZ angle for ring edge (i, i+1)
        [:,3] = layer-2 trainable RY,  [:,4] = layer-2 trainable RZ
    (theta is never batched, only x is, so theta[i, k] indexing is fine.)

    Design note (same lesson as the original trainable_feature_map bug):
    the trainable block sits BETWEEN two data-dependent layers (RY(x) then
    RZ(x)), so it cannot cancel out of the overlap the way a trailing
    unitary-only block would.
    """
    n = len(wires)
    # data re-upload layer 1
    for i, w in enumerate(wires):
        qml.RY(x[..., i], wires=w)
    # trainable local rotations, layer 1
    for i, w in enumerate(wires):
        qml.RY(theta[i, 0], wires=w)
        qml.RZ(theta[i, 1], wires=w)
    # trainable two-qubit entangling layer (this is the structurally new part)
    for i in range(n):
        j = (i + 1) % n
        qml.IsingZZ(theta[i, 2], wires=[wires[i], wires[j]])
    # data re-upload layer 2 (different axis from layer 1, more re-uploading richness)
    for i, w in enumerate(wires):
        qml.RZ(x[..., i], wires=w)
    # trainable local rotations, layer 2
    for i, w in enumerate(wires):
        qml.RY(theta[i, 3], wires=w)
        qml.RZ(theta[i, 4], wires=w)
    # fixed entangling ring to mix in the layer-2 trainable rotations too
    for i in range(n):
        qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])


FEATURE_MAPS = {
    # dict keys kept stable (legacy names) so the rest of the codebase
    # (model.fit(), experiments/*.py) doesn't need to change -- only the
    # circuits themselves were redesigned.
    "kernel1_angle": {"fn": iqp_encoding, "trainable": False},
    "kernel2_entangled": {"fn": xy_star_encoding, "trainable": False},
    "kernel3_trainable": {"fn": deep_trainable_feature_map, "trainable": True},
}

# theta shape needed by the (now deeper) trainable feature map -- imported
# by models/adaptive_qkernel.py so the two stay in sync.
TRAINABLE_THETA_SHAPE = (5,)  # per-wire: 5 trainable angles, see docstring above
