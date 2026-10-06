"""
experiments/noise.py

Experiment 4 - Noise (RQ3): does the trained ADAPTIVE kernel degrade more
gracefully under NISQ noise than the best single fixed/trainable kernel?

Router weights + Kernel-3 theta are trained ONCE on the ideal (noiseless)
simulator (exactly like experiments/adaptive.py), then both the best single
kernel and the adaptive combination are RE-EVALUATED at each noise level
without retraining. This isolates "does what we learned hold up under noise"
(Experiment 4) from "does training under noise itself help" (a natural
follow-up -- see README, Phase 5 extension: train_under_noise()).

Run:
    python3 -m experiments.noise --dataset xor --n_qubits 2 --epochs 20
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from data.datasets import load_dataset
from models.adaptive_qkernel import AdaptiveQuantumKernel
from quantum.feature_maps import FEATURE_MAPS
from quantum.noise import build_noisy_kernel_bank, NOISE_LEVELS
from baselines.svm import precomputed_kernel_svm_accuracy


def _noisy_adaptive_matrix(model, bank_noisy, X1, X2, symmetric):
    """Combine noisy per-kernel matrices using the ALREADY-TRAINED router weights."""
    with torch.no_grad():
        mats = {}
        for name in model.kernel_names:
            k = bank_noisy[name]
            if k.trainable:
                m = k.matrix(X1, X2, theta=model.theta, symmetric=symmetric)
                mats[name] = m.to(torch.float32)
            else:
                mats[name] = torch.as_tensor(k.matrix(X1, X2, symmetric=symmetric), dtype=torch.float32)
        K, _ = model.adaptive_matrix(X1, X2, kernel_mats=mats, symmetric=symmetric)
    return K.numpy().astype(np.float64)


def run(dataset, n_qubits, epochs, max_train, max_test, noise_type="depolarizing", seed=0):
    print(f"\n=== Experiment 4: Noise robustness — {dataset} (n_qubits={n_qubits}, noise={noise_type}) ===")
    X_train, X_test, y_train, y_test = load_dataset(
        dataset, n_qubits=n_qubits, max_train=max_train, max_test=max_test, seed=seed
    )
    print(f"train={len(X_train)}  test={len(X_test)}")

    model = AdaptiveQuantumKernel(n_qubits=n_qubits, seed=seed)
    print("\nTraining router + kernel3 on the IDEAL (noiseless) simulator...")
    model.fit(X_train, y_train, epochs=epochs, lr=0.05, verbose=False)

    # Pick the best single kernel on the ideal test set as the noise-robustness comparison point.
    best_name, best_acc = None, -1.0
    for name in model.kernel_names:
        K_tr = model.single_kernel_matrix(name, X_train, X_train, symmetric=True)
        K_te = model.single_kernel_matrix(name, X_test, X_train, symmetric=False)
        acc = precomputed_kernel_svm_accuracy(K_tr, y_train, K_te, y_test)
        if acc > best_acc:
            best_acc, best_name = acc, name
    print(f"Best single kernel on ideal test set: {best_name} (acc={best_acc:.3f})")

    print(f"\n{'noise':>6} | {'single_best':>12} | {'adaptive':>9}")
    print("-" * 34)
    results = {"noise_level": [], "single_best": [], "adaptive": []}
    for level in NOISE_LEVELS:
        bank_noisy = build_noisy_kernel_bank(FEATURE_MAPS, n_qubits, noise_type=noise_type, noise_level=level)

        k = bank_noisy[best_name]
        theta_arg = model.theta if k.trainable else None
        K_tr = k.matrix(X_train, X_train, theta=theta_arg, symmetric=True)
        K_te = k.matrix(X_test, X_train, theta=theta_arg, symmetric=False)
        if k.trainable:
            K_tr, K_te = K_tr.detach().numpy(), K_te.detach().numpy()
        acc_single = precomputed_kernel_svm_accuracy(K_tr, y_train, K_te, y_test)

        K_tr_a = _noisy_adaptive_matrix(model, bank_noisy, X_train, X_train, symmetric=True)
        K_te_a = _noisy_adaptive_matrix(model, bank_noisy, X_test, X_train, symmetric=False)
        acc_adapt = precomputed_kernel_svm_accuracy(K_tr_a, y_train, K_te_a, y_test)

        results["noise_level"].append(level)
        results["single_best"].append(acc_single)
        results["adaptive"].append(acc_adapt)
        print(f"{level:6.2f} | {acc_single:12.3f} | {acc_adapt:9.3f}")

    drop_single = results["single_best"][0] - results["single_best"][-1]
    drop_adapt = results["adaptive"][0] - results["adaptive"][-1]
    print("\n=== Summary ===")
    print(f"  accuracy drop (ideal -> high noise), single best kernel : {drop_single:+.3f}")
    print(f"  accuracy drop (ideal -> high noise), adaptive kernel    : {drop_adapt:+.3f}")
    verdict = "ADAPTIVE MORE ROBUST" if drop_adapt < drop_single else "single kernel degrades no worse (or better)"
    print(f"  -> {verdict}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="xor", choices=["xor", "iris", "wine", "breast_cancer", "digits", "moons", "circles"])
    p.add_argument("--n_qubits", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max_train", type=int, default=40)
    p.add_argument("--max_test", type=int, default=20)
    p.add_argument("--noise_type", default="depolarizing", choices=["depolarizing", "bit_flip", "phase_flip"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args.dataset, args.n_qubits, args.epochs, args.max_train, args.max_test, args.noise_type, args.seed)
