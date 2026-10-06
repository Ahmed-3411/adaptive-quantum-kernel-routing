"""
experiments/run_kernel_analysis.py

Answers the diagnostic question before touching Objective B or the router:
are Kernel 1 (angle), Kernel 2 (entangled), and Kernel 3 (trained) actually
DIFFERENT as functions on each dataset, or do they collapse toward the same
thing (kernel concentration)? If they collapse, no router architecture or
training objective can make "adaptive" beat "single kernel" -- there's
nothing to route between.

Run:
    python3 -m experiments.run_kernel_analysis
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import load_dataset
from models.adaptive_qkernel import AdaptiveQuantumKernel
from evaluation.kernel_analysis import analyze_kernel_bank

DATASETS = [
    ("xor", 2, 60),
    ("wine", 4, 40),
    ("breast_cancer", 4, 40),
]


def run(seed=0, epochs=15):
    for dataset, n_qubits, max_train in DATASETS:
        print(f"\n{'='*64}\n{dataset} (n_qubits={n_qubits}, train={max_train})\n{'='*64}")
        X_train, X_test, y_train, y_test = load_dataset(
            dataset, n_qubits=n_qubits, max_train=max_train, max_test=max_train // 2, seed=seed
        )

        model = AdaptiveQuantumKernel(n_qubits=n_qubits, seed=seed)
        # Train briefly so kernel3's theta reflects a REAL trained state
        # (not just its random initialization), matching what the router
        # actually sees during Experiment 2 / noise-aware training.
        model.fit(X_train, y_train, epochs=epochs, lr=0.05, verbose=False)

        kernel_matrices = {
            name: model.single_kernel_matrix(name, X_train, X_train, symmetric=True)
            for name in model.kernel_names
        }

        report = analyze_kernel_bank(kernel_matrices, y_train)

        print(f"\n{'kernel':22s} {'target_align':>13} {'eff_dim':>9} {'concentration_std':>19}")
        n = len(X_train)
        for name, m in report["per_kernel"].items():
            print(f"{name:22s} {m['target_alignment']:13.4f} {m['effective_dimension']:9.2f} "
                  f"{m['concentration_std']:19.4f}   (n={n}, max possible eff_dim={n})")

        print(f"\nPairwise kernel-kernel alignment (1.0 = identical as functions):")
        for pair, val in report["pairwise_kernel_alignment"].items():
            flag = "  <-- VERY SIMILAR (near-collapse)" if val > 0.95 else ""
            print(f"  {pair:40s} {val:.4f}{flag}")


if __name__ == "__main__":
    run()
