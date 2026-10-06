"""
experiments/adaptive.py

Experiment 2 (the project's core experiment): Best Single Kernel vs Adaptive Kernel.
Also reports the classical baselines (Phase 12) for context.

Run:
    cd adaptive-qkernel
    python3 -m experiments.adaptive --dataset xor --n_qubits 2 --epochs 20
    python3 -m experiments.adaptive --dataset iris --n_qubits 4 --epochs 20
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import load_dataset
from models.adaptive_qkernel import AdaptiveQuantumKernel
from baselines.svm import classical_baselines, precomputed_kernel_svm_accuracy


def run(dataset, n_qubits, epochs, max_train, max_test, seed=0):
    print(f"\n=== Dataset: {dataset}  (n_qubits={n_qubits}) ===")
    X_train, X_test, y_train, y_test = load_dataset(
        dataset, n_qubits=n_qubits, max_train=max_train, max_test=max_test, seed=seed
    )
    print(f"train={len(X_train)}  test={len(X_test)}")

    model = AdaptiveQuantumKernel(n_qubits=n_qubits, seed=seed)

    print("\n-- classical baselines --")
    cb = classical_baselines(X_train, y_train, X_test, y_test)
    for k, v in cb.items():
        print(f"  {k:12s}: {v:.3f}")

    print("\n-- fixed single quantum kernels (untrained theta for kernel3) --")
    single_results = {}
    for name in model.kernel_names:
        K_tr = model.single_kernel_matrix(name, X_train, X_train, symmetric=True)
        K_te = model.single_kernel_matrix(name, X_test, X_train, symmetric=False)
        acc = precomputed_kernel_svm_accuracy(K_tr, y_train, K_te, y_test)
        single_results[name] = acc
        print(f"  {name:20s}: {acc:.3f}")

    print("\n-- training adaptive router + kernel3 (kernel-target alignment) --")
    model.fit(X_train, y_train, epochs=epochs, lr=0.05, verbose=True)

    print("\n-- kernel3 AFTER training (trainable feature map) --")
    K_tr = model.single_kernel_matrix("kernel3_trainable", X_train, X_train, symmetric=True)
    K_te = model.single_kernel_matrix("kernel3_trainable", X_test, X_train, symmetric=False)
    acc_k3 = precomputed_kernel_svm_accuracy(K_tr, y_train, K_te, y_test)
    print(f"  kernel3_trainable   : {acc_k3:.3f}")

    print("\n-- adaptive kernel (router-weighted combination) --")
    K_tr_adapt = model.kernel_matrix(X_train, X_train, symmetric=True)
    K_te_adapt = model.kernel_matrix(X_test, X_train, symmetric=False)
    acc_adapt = precomputed_kernel_svm_accuracy(K_tr_adapt, y_train, K_te_adapt, y_test)
    print(f"  adaptive_kernel     : {acc_adapt:.3f}")

    best_single = max(list(single_results.values()) + [acc_k3])
    print("\n=== Summary ===")
    print(f"  best classical baseline : {max(cb.values()):.3f}")
    print(f"  best single quantum     : {best_single:.3f}")
    print(f"  adaptive quantum kernel : {acc_adapt:.3f}")
    verdict = "ADAPTIVE WINS" if acc_adapt >= best_single else "single kernel still ahead"
    print(f"  -> {verdict}")

    return {
        "classical_baselines": cb,
        "single_kernels": {**single_results, "kernel3_trainable": acc_k3},
        "adaptive": acc_adapt,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="xor", choices=["xor", "iris", "wine", "breast_cancer", "digits", "moons", "circles"])
    p.add_argument("--n_qubits", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max_train", type=int, default=30)
    p.add_argument("--max_test", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args.dataset, args.n_qubits, args.epochs, args.max_train, args.max_test, args.seed)
