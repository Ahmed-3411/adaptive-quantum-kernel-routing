"""
experiments/noise_aware_training.py

Phase 5 (actual intent): does training the router+theta WHILE SEEING NOISE
produce a more robust adaptive kernel than training on the ideal simulator
and only meeting noise at test time (which is all experiments/noise.py
checks)?

Two models, same data / same init seed:
    Model A ("ideal-trained")       : model.fit(..., noise_level=0.0)
    Model B ("noise-aware-trained") : model.fit(..., noise_level=train_noise)

Both are then evaluated across the same NOISE_LEVELS sweep (Experiment 4's
Ideal -> Low -> Medium -> High), using the noisy kernel bank at each level
combined with each model's own trained router weights + theta.

Run:
    python3 -m experiments.noise_aware_training --dataset xor --n_qubits 2 \
        --epochs 20 --train_noise 0.05
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from data.datasets import load_dataset
from models.adaptive_qkernel import AdaptiveQuantumKernel
from quantum.feature_maps import FEATURE_MAPS
from quantum.noise import build_noisy_kernel_bank, NOISE_LEVELS
from baselines.svm import precomputed_kernel_svm_accuracy


def _adaptive_acc_under_noise(model, X_train, y_train, X_test, y_test, noise_type, level):
    bank_noisy = build_noisy_kernel_bank(FEATURE_MAPS, model.n_qubits, noise_type=noise_type, noise_level=level)
    with torch.no_grad():
        def combo(X1, X2, symmetric):
            mats = {}
            for name in model.kernel_names:
                k = bank_noisy[name]
                if k.trainable:
                    mats[name] = k.matrix(X1, X2, theta=model.theta, symmetric=symmetric).to(torch.float32)
                else:
                    mats[name] = torch.as_tensor(k.matrix(X1, X2, symmetric=symmetric), dtype=torch.float32)
            K, _ = model.adaptive_matrix(X1, X2, kernel_mats=mats, symmetric=symmetric)
            return K.numpy().astype(np.float64)

        K_tr = combo(X_train, X_train, True)
        K_te = combo(X_test, X_train, False)
    return precomputed_kernel_svm_accuracy(K_tr, y_train, K_te, y_test)


def run(dataset, n_qubits, epochs, max_train, max_test, train_noise, noise_type="depolarizing", seed=0):
    print(f"\n=== Noise-aware vs ideal training — {dataset} (n_qubits={n_qubits}) ===")
    X_train, X_test, y_train, y_test = load_dataset(
        dataset, n_qubits=n_qubits, max_train=max_train, max_test=max_test, seed=seed
    )
    print(f"train={len(X_train)}  test={len(X_test)}  train_noise_level={train_noise}")

    print("\nTraining Model A (ideal simulator, noise_level=0.0)...")
    model_ideal = AdaptiveQuantumKernel(n_qubits=n_qubits, seed=seed)
    model_ideal.fit(X_train, y_train, epochs=epochs, lr=0.05, verbose=False)

    print(f"Training Model B (noise-aware, noise_level={train_noise}, {noise_type})...")
    model_noisy = AdaptiveQuantumKernel(n_qubits=n_qubits, seed=seed)
    model_noisy.fit(X_train, y_train, epochs=epochs, lr=0.05,
                     noise_type=noise_type, noise_level=train_noise, verbose=False)

    print(f"\n{'noise':>6} | {'ideal-trained':>13} | {'noise-aware-trained':>19}")
    print("-" * 46)
    rows = []
    for level in NOISE_LEVELS:
        acc_ideal = _adaptive_acc_under_noise(model_ideal, X_train, y_train, X_test, y_test, noise_type, level)
        acc_noisy = _adaptive_acc_under_noise(model_noisy, X_train, y_train, X_test, y_test, noise_type, level)
        rows.append((level, acc_ideal, acc_noisy))
        print(f"{level:6.2f} | {acc_ideal:13.3f} | {acc_noisy:19.3f}")

    mean_ideal = np.mean([r[1] for r in rows])
    mean_noisy = np.mean([r[2] for r in rows])
    print("\n=== Summary ===")
    print(f"  mean accuracy across noise sweep, ideal-trained       : {mean_ideal:.3f}")
    print(f"  mean accuracy across noise sweep, noise-aware-trained : {mean_noisy:.3f}")
    verdict = "NOISE-AWARE TRAINING HELPS" if mean_noisy > mean_ideal else "no clear benefit from noise-aware training here"
    print(f"  -> {verdict}")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="xor", choices=["xor", "iris", "wine", "breast_cancer", "digits", "moons", "circles"])
    p.add_argument("--n_qubits", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max_train", type=int, default=40)
    p.add_argument("--max_test", type=int, default=20)
    p.add_argument("--train_noise", type=float, default=0.05)
    p.add_argument("--noise_type", default="depolarizing", choices=["depolarizing", "bit_flip", "phase_flip"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args.dataset, args.n_qubits, args.epochs, args.max_train, args.max_test,
        args.train_noise, args.noise_type, args.seed)
