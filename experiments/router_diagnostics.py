"""
experiments/router_diagnostics.py

Q5 follow-up: inspect the router's learned sample-level gates g(x) directly,
rather than only its downstream accuracy, to understand WHY capture of the
Oracle headroom (experiments/oracle.py) is so inconsistent across datasets
(see ROADMAP_STATUS.md Q3).

For each test point we record:
    - which single-kernel SVMs got it right (per_kernel_correct)
    - whether Oracle got it right (any kernel correct)
    - whether the deployed adaptive model got it right
    - the router's gate vector g(x) for that point

Then we specifically inspect "Oracle-rescuable failures": points where at
least one single kernel is correct (so Oracle succeeds) but the adaptive
model is wrong. For these points, was the gate weight on the CORRECT
kernel(s) low (router "should have" trusted it more but didn't -- a
gating/training failure) or comparable to the wrong kernel(s) (suggesting
the failure is in how the SVM combines already-reasonable gates, not in
the gating decision itself)?

We also compute average gate entropy per dataset as a measure of how
"decisive" vs. "confused" the router's gating is overall.

Run:
    python3 -m experiments.router_diagnostics --dataset wine --n_qubits 4
    python3 -m experiments.router_diagnostics --dataset xor --n_qubits 2
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.svm import SVC

from data.datasets import load_dataset
from models.adaptive_qkernel import AdaptiveQuantumKernel
from baselines.svm import precomputed_kernel_svm_accuracy


def run_one(dataset, n_qubits, epochs, max_train, max_test, seed):
    X_train, X_test, y_train, y_test = load_dataset(
        dataset, n_qubits=n_qubits, max_train=max_train, max_test=max_test, seed=seed
    )
    model = AdaptiveQuantumKernel(n_qubits=n_qubits, seed=seed, router_type="sample_level")
    model.fit(X_train, y_train, epochs=epochs, lr=0.05, verbose=False)

    # deployed adaptive model predictions
    K_tr = model.kernel_matrix(X_train, X_train, symmetric=True)
    K_te = model.kernel_matrix(X_test, X_train, symmetric=False)
    adaptive_clf = SVC(kernel="precomputed").fit(K_tr, y_train)
    adaptive_pred_test = adaptive_clf.predict(K_te)
    adaptive_pred_train = adaptive_clf.predict(K_tr)
    adaptive_correct_test = (adaptive_pred_test == y_test)
    adaptive_correct_train = (adaptive_pred_train == y_train)

    # per-single-kernel predictions, on BOTH train and test (for the overfitting check)
    per_kernel_correct_test = {}
    per_kernel_correct_train = {}
    for name in model.kernel_names:
        K_tr_m = model.single_kernel_matrix(name, X_train, X_train, symmetric=True)
        K_te_m = model.single_kernel_matrix(name, X_test, X_train, symmetric=False)
        clf = SVC(kernel="precomputed").fit(K_tr_m, y_train)
        per_kernel_correct_test[name] = (clf.predict(K_te_m) == y_test)
        per_kernel_correct_train[name] = (clf.predict(K_tr_m) == y_train)

    oracle_correct_test = np.any(np.stack(list(per_kernel_correct_test.values())), axis=0)
    oracle_correct_train = np.any(np.stack(list(per_kernel_correct_train.values())), axis=0)

    # gate values on both train and test
    gates_test = model.router.gate_matrix(X_test).detach().numpy()
    gates_train = model.router.gate_matrix(X_train).detach().numpy()
    entropy_test = -(gates_test * np.log(gates_test + 1e-12)).sum(axis=1)
    entropy_train = -(gates_train * np.log(gates_train + 1e-12)).sum(axis=1)

    return {
        "kernel_names": model.kernel_names,
        "gates_test": gates_test, "gates_train": gates_train,
        "entropy_test": entropy_test, "entropy_train": entropy_train,
        "adaptive_correct": adaptive_correct_test, "adaptive_correct_train": adaptive_correct_train,
        "oracle_correct": oracle_correct_test, "oracle_correct_train": oracle_correct_train,
        "per_kernel_correct": per_kernel_correct_test, "per_kernel_correct_train": per_kernel_correct_train,
        "adaptive_acc": adaptive_correct_test.mean(),
        "oracle_acc": oracle_correct_test.mean(),
    }


def summarize(dataset, results_list):
    print(f"\n{'='*70}\nRouter diagnostics — {dataset} (n={len(results_list)} seeds)\n{'='*70}")

    all_entropy = np.concatenate([r["entropy_test"] for r in results_list])
    max_entropy = np.log(len(results_list[0]["kernel_names"]))
    print(f"Mean gate entropy: {all_entropy.mean():.3f} (max possible = {max_entropy:.3f}, "
          f"i.e. {100*all_entropy.mean()/max_entropy:.1f}% of maximum 'confusion')")

    # Oracle-rescuable failures: oracle right, adaptive wrong
    n_rescuable = 0
    n_total_test = 0
    gate_on_correct_kernel = []
    gate_on_wrong_kernel = []
    for r in results_list:
        rescuable = r["oracle_correct"] & (~r["adaptive_correct"])
        n_rescuable += rescuable.sum()
        n_total_test += len(r["adaptive_correct"])
        for i in np.where(rescuable)[0]:
            gates_i = r["gates_test"][i]
            for k_idx, name in enumerate(r["kernel_names"]):
                if r["per_kernel_correct"][name][i]:
                    gate_on_correct_kernel.append(gates_i[k_idx])
                else:
                    gate_on_wrong_kernel.append(gates_i[k_idx])

    print(f"\nOracle-rescuable failures (oracle right, adaptive wrong): "
          f"{n_rescuable}/{n_total_test} test points ({100*n_rescuable/n_total_test:.1f}%)")
    if gate_on_correct_kernel:
        print(f"  Mean gate weight the router put on kernels that WOULD have been correct: "
              f"{np.mean(gate_on_correct_kernel):.3f}")
        print(f"  Mean gate weight the router put on kernels that would have been WRONG:    "
              f"{np.mean(gate_on_wrong_kernel):.3f}")
        gap = np.mean(gate_on_correct_kernel) - np.mean(gate_on_wrong_kernel)
        print(f"  Gap (correct - wrong): {gap:+.3f}  "
              f"{'(router DOES favor the right kernel but SVM combination still fails)' if gap > 0.02 else '(router does NOT reliably favor the right kernel here)'}")

    print(f"\nDataset-level: mean adaptive_acc={np.mean([r['adaptive_acc'] for r in results_list]):.3f}  "
          f"mean oracle_acc={np.mean([r['oracle_acc'] for r in results_list]):.3f}")


def run(dataset, n_qubits, epochs, max_train, max_test, seeds):
    results_list = [run_one(dataset, n_qubits, epochs, max_train, max_test, s) for s in seeds]
    summarize(dataset, results_list)
    return results_list


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wine")
    p.add_argument("--n_qubits", type=int, default=4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--max_train", type=int, default=40)
    p.add_argument("--max_test", type=int, default=20)
    p.add_argument("--seeds", type=int, default=10)
    args = p.parse_args()
    run(args.dataset, args.n_qubits, args.epochs, args.max_train, args.max_test, list(range(args.seeds)))
