"""
experiments/ablation.py

Section 18 of the plan: strip one component at a time and see what breaks.
Specifically answers the question left open after the diversity finding:
does Wine's statistically significant benefit come from (a) the router
being genuinely ADAPTIVE (sample-dependent weights), (b) just having
Kernel 3 (deep trainable) in the mix at all, or (c) something else?

IMPORTANT: variants now default to the PSD-preserving `sample_level` router
(see routing/sample_level_router.py, models/adaptive_qkernel.py). The
original `per_pair` router was found to produce non-symmetric, non-PSD
kernel matrices (max|K-K.T|=0.338, min eigenvalue=-0.135) and is kept here
ONLY as `per_pair_legacy`, for historical comparison -- it must not be used
for any new headline result. See ROADMAP_STATUS.md for the full story.

Variants (all trained with the same objective, epochs, data, seeds):
    full_adaptive     -- sample_level router (g(x)=softmax(f(x))) + all 3
                          kernels. PSD-preserving. The model used everywhere
                          else in this project as of the PSD fix.
    global_router     -- ONE learned weight vector for all pairs (not
                          sample-dependent) + all 3 kernels. This is exactly
                          the plan's own Experiment 3 (Global vs Local
                          Adaptation). If this matches full_adaptive, the
                          benefit isn't from adaptivity. (Already symmetric/
                          PSD by construction -- see ROADMAP_STATUS.md.)
    uniform_router    -- fixed 1/3, 1/3, 1/3 average, NOTHING trained in the
                          router at all (kernel3's theta still trains via the
                          alignment objective). If this matches full_adaptive,
                          neither training nor routing the WEIGHTS is doing
                          anything -- just averaging the kernel bank is enough.
    no_kernel3        -- sample_level router over ONLY Kernel 1 + Kernel 2
                          (drop the deep trainable kernel entirely). If this
                          is much worse than full_adaptive, Kernel 3 itself
                          is doing most of the work, not the router's
                          ability to route.
    per_pair_legacy   -- the ORIGINAL (broken, non-PSD) router. Kept only so
                          "did the fix change the answer" stays checkable.

Run:
    python3 -m experiments.ablation --dataset wine --n_qubits 4 --seeds 25
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from data.datasets import load_dataset
from models.adaptive_qkernel import AdaptiveQuantumKernel
from baselines.svm import precomputed_kernel_svm_accuracy
from evaluation.stats import paired_comparison, holm_correction, format_result

VARIANTS = {
    "full_adaptive": dict(router_type="sample_level", kernel_subset=None),
    "global_router": dict(router_type="global", kernel_subset=None),
    "uniform_router": dict(router_type="uniform", kernel_subset=None),
    "no_kernel3": dict(router_type="sample_level", kernel_subset=["kernel1_angle", "kernel2_entangled"]),
    "per_pair_legacy": dict(router_type="per_pair", kernel_subset=None),
}


def run_one(dataset, n_qubits, epochs, max_train, max_test, seed, router_type, kernel_subset):
    X_train, X_test, y_train, y_test = load_dataset(
        dataset, n_qubits=n_qubits, max_train=max_train, max_test=max_test, seed=seed
    )
    model = AdaptiveQuantumKernel(n_qubits=n_qubits, seed=seed, router_type=router_type, kernel_subset=kernel_subset)
    model.fit(X_train, y_train, epochs=epochs, lr=0.05, verbose=False)
    K_tr = model.kernel_matrix(X_train, X_train, symmetric=True)
    K_te = model.kernel_matrix(X_test, X_train, symmetric=False)
    return precomputed_kernel_svm_accuracy(K_tr, y_train, K_te, y_test)


def run(dataset, n_qubits, epochs, max_train, max_test, seeds):
    print(f"\n{'='*70}\nAblation study — {dataset} (n_qubits={n_qubits}), {len(seeds)} seeds\n{'='*70}")
    results = {name: [] for name in VARIANTS}

    for variant_name, cfg in VARIANTS.items():
        print(f"\n-- {variant_name} --")
        for s in seeds:
            acc = run_one(dataset, n_qubits, epochs, max_train, max_test, s, **cfg)
            results[variant_name].append(acc)
        accs = np.array(results[variant_name])
        print(f"  {variant_name:16s}: {accs.mean():.3f} ± {accs.std():.3f}  (n={len(accs)})")

    print(f"\n{'='*70}\nSummary (paired t-test + Wilcoxon + bootstrap 95% CI, vs. full_adaptive)\n{'='*70}")
    full = np.array(results["full_adaptive"])
    comparisons = {}
    for name in VARIANTS:
        if name == "full_adaptive":
            continue
        accs = np.array(results[name])
        r = paired_comparison(full, accs, seed=0)  # full - variant: positive means full is better
        comparisons[name] = r
        print(" " + format_result(name, r))

    print(f"\n{'='*70}\nHolm correction across the 3 ablation comparisons (excludes per_pair_legacy, which is a sanity check, not a hypothesis test)\n{'='*70}")
    core_pvals = {k: v["p_ttest"] for k, v in comparisons.items() if k != "per_pair_legacy"}
    holm = holm_correction(core_pvals)
    for name, h in holm.items():
        print(f"  {name:16s}: p={h['p_value']:.5f}  Holm threshold={h['holm_threshold']:.5f}  "
              f"reject H0={'YES' if h['reject_null'] else 'no'}")

    return results, comparisons


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="wine", choices=["xor", "iris", "wine", "breast_cancer", "digits", "moons", "circles"])
    p.add_argument("--n_qubits", type=int, default=4)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--max_train", type=int, default=40)
    p.add_argument("--max_test", type=int, default=20)
    p.add_argument("--seeds", type=int, default=10, help="number of seeds, 0..seeds-1")
    args = p.parse_args()
    run(args.dataset, args.n_qubits, args.epochs, args.max_train, args.max_test, list(range(args.seeds)))

