"""
experiments/q4_full_table.py

Revised Q4: "What measurable property distinguishes datasets where the
router CAN exploit kernel complementarity from datasets where it cannot?"
(not "which datasets benefit" -- see ROADMAP_STATUS.md Q2/Q3 for why that
framing was replaced).

Computes, per dataset, in one consistent pass (same seeds for everything
so all 9 quantities are directly comparable):

    1. Oracle - Best Single       (available complementarity)
    2. Adaptive - Best Single     (actual adaptive gain)
    3. Capture % = (2)/(1)        (exploitation efficiency)
    4. Recoverable failure rate   (% Oracle-correct, Adaptive-wrong)
    5. Gate entropy               (router decisiveness; low = confident)
    6. Gate-Oracle agreement      (does the router's top kernel choice
                                    match a kernel that's actually correct,
                                    on points where at least one kernel is
                                    correct?)
    7. Kernel disagreement        (mean pairwise prediction disagreement
                                    between the 3 single-kernel classifiers)
    8. Kernel accuracy spread     (std of the 3 kernels' individual
                                    accuracies -- do kernels specialize
                                    globally, i.e. is one clearly best?)
    9. Local disagreement proxy   (does the ranking of per-kernel accuracy
                                    change between two halves of the test
                                    set split by a simple feature threshold
                                    -- i.e. is specialization sample-
                                    dependent, or the same everywhere?)

Run:
    python3 -m experiments.q4_full_table
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from sklearn.svm import SVC

from data.datasets import load_dataset
from models.adaptive_qkernel import AdaptiveQuantumKernel
from evaluation.diversity_metrics import prediction_disagreement

QUBITS = {
    "xor": 2, "moons": 2, "circles": 2,
    "iris": 4, "wine": 4, "wine_12": 4, "breast_cancer": 4,
    "digits": 4, "digits_01": 4, "digits_69": 4, "digits_45": 4,
    "synth_easy": 4, "synth_medium": 4, "synth_hard": 4, "gaussian_quantiles": 4,
}


def run_one(dataset, n_qubits, epochs, max_train, max_test, seed):
    X_train, X_test, y_train, y_test = load_dataset(
        dataset, n_qubits=n_qubits, max_train=max_train, max_test=max_test, seed=seed
    )
    model = AdaptiveQuantumKernel(n_qubits=n_qubits, seed=seed, router_type="sample_level")
    model.fit(X_train, y_train, epochs=epochs, lr=0.05, verbose=False)

    # deployed adaptive model
    K_tr = model.kernel_matrix(X_train, X_train, symmetric=True)
    K_te = model.kernel_matrix(X_test, X_train, symmetric=False)
    adaptive_clf = SVC(kernel="precomputed").fit(K_tr, y_train)
    adaptive_pred = adaptive_clf.predict(K_te)
    adaptive_correct = (adaptive_pred == y_test)

    # per-kernel
    per_kernel_correct = {}
    per_kernel_acc = {}
    kmats_train, kmats_test = {}, {}
    for name in model.kernel_names:
        K_tr_m = model.single_kernel_matrix(name, X_train, X_train, symmetric=True)
        K_te_m = model.single_kernel_matrix(name, X_test, X_train, symmetric=False)
        kmats_train[name] = K_tr_m
        kmats_test[name] = K_te_m
        clf = SVC(kernel="precomputed").fit(K_tr_m, y_train)
        pred = clf.predict(K_te_m)
        per_kernel_correct[name] = (pred == y_test)
        per_kernel_acc[name] = float((pred == y_test).mean())

    oracle_correct = np.any(np.stack(list(per_kernel_correct.values())), axis=0)

    # gates + entropy
    gates_test = model.router.gate_matrix(X_test).detach().numpy()
    entropy_test = -(gates_test * np.log(gates_test + 1e-12)).sum(axis=1)

    # gate-oracle agreement: on oracle-correct points, does argmax(gate) fall
    # among the kernels that were actually correct for that point?
    names = model.kernel_names
    argmax_kernel = np.argmax(gates_test, axis=1)
    agree = []
    for i in range(len(X_test)):
        if not oracle_correct[i]:
            continue
        correct_kernels = [k for k, n in enumerate(names) if per_kernel_correct[n][i]]
        agree.append(argmax_kernel[i] in correct_kernels)
    gate_oracle_agreement = float(np.mean(agree)) if agree else float("nan")

    # kernel disagreement (mean pairwise prediction disagreement, test set)
    pair_disagreements = []
    kname_list = list(model.kernel_names)
    for i in range(len(kname_list)):
        for j in range(i + 1, len(kname_list)):
            a, b = kname_list[i], kname_list[j]
            d = prediction_disagreement(kmats_train[a], kmats_test[a], kmats_train[b], kmats_test[b], y_train)
            pair_disagreements.append(d)
    kernel_disagreement = float(np.mean(pair_disagreements))

    # kernel accuracy spread
    acc_spread = float(np.std(list(per_kernel_acc.values())))

    # local disagreement proxy: split test set by median of first feature,
    # compare per-kernel accuracy ranking between the two halves
    feat0 = X_test[:, 0]
    median = np.median(feat0)
    half_a = feat0 <= median
    half_b = ~half_a
    local_disagreement = float("nan")
    if half_a.sum() >= 2 and half_b.sum() >= 2:
        acc_a = {n: float((per_kernel_correct[n][half_a]).mean()) for n in kname_list}
        acc_b = {n: float((per_kernel_correct[n][half_b]).mean()) for n in kname_list}
        # L1 distance between the two accuracy vectors (order preserved by kname_list)
        local_disagreement = float(np.mean([abs(acc_a[n] - acc_b[n]) for n in kname_list]))

    recoverable_failure_rate = float((oracle_correct & (~adaptive_correct)).mean())

    return {
        "adaptive_acc": float(adaptive_correct.mean()),
        "oracle_acc": float(oracle_correct.mean()),
        "best_single_acc": max(per_kernel_acc.values()),
        "recoverable_failure_rate": recoverable_failure_rate,
        "gate_entropy": float(entropy_test.mean()),
        "gate_oracle_agreement": gate_oracle_agreement,
        "kernel_disagreement": kernel_disagreement,
        "kernel_acc_spread": acc_spread,
        "local_disagreement": local_disagreement,
    }


def run(seeds_per_dataset=5, epochs=15, max_train=40, max_test=20):
    out_path = "results/q4_full_table.json"
    if os.path.exists(out_path):
        with open(out_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for name, nq in QUBITS.items():
        if name in all_results and len(all_results[name]) >= seeds_per_dataset:
            continue
        per_seed = all_results.get(name, [])
        for s in range(len(per_seed), seeds_per_dataset):
            r = run_one(name, nq, epochs, max_train, max_test, s)
            per_seed.append(r)
            all_results[name] = per_seed
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)
        print(f"{name}: {len(all_results[name])} seeds done")

    return all_results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=None, help="run just one dataset")
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()
    if args.dataset:
        QUBITS_SUBSET = {args.dataset: QUBITS[args.dataset]}
        QUBITS.clear()
        QUBITS.update(QUBITS_SUBSET)
    run(seeds_per_dataset=args.seeds)
