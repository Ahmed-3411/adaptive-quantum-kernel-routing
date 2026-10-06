"""
experiments/multiseed.py

The methodological step the plan itself insists on before any claim: every
result so far (Experiment 2, Experiment 4, noise-aware training) was a
SINGLE seed. This script re-runs each of them across multiple seeds and
reports mean +/- std, so we can tell a real effect from noise in the random
initialization / train-test split.

Run:
    python3 -m experiments.multiseed --dataset xor --seeds 5
    python3 -m experiments.multiseed --dataset wine --n_qubits 4 --seeds 5
"""

import argparse
import contextlib
import io
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from experiments.adaptive import run as run_experiment2
from experiments.noise import run as run_experiment4
from experiments.noise_aware_training import run as run_noise_aware
from quantum.noise import NOISE_LEVELS


def _quiet(fn, *args, **kwargs):
    """Run fn while swallowing its internal print() calls; return its return value."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result


def _mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def aggregate_experiment2(dataset, n_qubits, epochs, max_train, max_test, seeds):
    print(f"\n{'='*60}\nExperiment 2 (single best kernel vs adaptive) — {dataset}, {len(seeds)} seeds\n{'='*60}")
    classical, single, adaptive = [], [], []
    for s in seeds:
        t0 = time.time()
        res = _quiet(run_experiment2, dataset, n_qubits, epochs, max_train, max_test, s)
        classical.append(max(res["classical_baselines"].values()))
        single.append(max(res["single_kernels"].values()))
        adaptive.append(res["adaptive"])
        print(f"  seed {s}: classical={classical[-1]:.3f}  single_best={single[-1]:.3f}  "
              f"adaptive={adaptive[-1]:.3f}  ({time.time()-t0:.1f}s)")

    mc, sc = _mean_std(classical)
    ms, ss = _mean_std(single)
    ma, sa = _mean_std(adaptive)
    print(f"\n  {'':22s}  mean ± std")
    print(f"  {'best classical':22s}  {mc:.3f} ± {sc:.3f}")
    print(f"  {'best single quantum':22s}  {ms:.3f} ± {ss:.3f}")
    print(f"  {'adaptive':22s}  {ma:.3f} ± {sa:.3f}")
    # a crude but honest significance check: do the mean±std bands overlap?
    overlap = (ma - sa) <= (ms + ss) and (ms - ss) <= (ma + sa)
    verdict = "bands overlap -> NOT a clear win either way" if overlap else (
        "ADAPTIVE clearly ahead (no overlap)" if ma > ms else "single kernel clearly ahead (no overlap)"
    )
    print(f"  -> {verdict}")
    return {"classical": classical, "single": single, "adaptive": adaptive}


def aggregate_experiment4(dataset, n_qubits, epochs, max_train, max_test, noise_type, seeds):
    print(f"\n{'='*60}\nExperiment 4 (noise robustness) — {dataset}, {len(seeds)} seeds\n{'='*60}")
    per_level_single = {lvl: [] for lvl in NOISE_LEVELS}
    per_level_adaptive = {lvl: [] for lvl in NOISE_LEVELS}
    for s in seeds:
        t0 = time.time()
        res = _quiet(run_experiment4, dataset, n_qubits, epochs, max_train, max_test, noise_type, s)
        for lvl, acc_s, acc_a in zip(res["noise_level"], res["single_best"], res["adaptive"]):
            per_level_single[lvl].append(acc_s)
            per_level_adaptive[lvl].append(acc_a)
        print(f"  seed {s} done ({time.time()-t0:.1f}s)")

    print(f"\n  {'noise':>6} | {'single_best (mean±std)':>24} | {'adaptive (mean±std)':>22}")
    print("  " + "-" * 58)
    for lvl in NOISE_LEVELS:
        ms, ss = _mean_std(per_level_single[lvl])
        ma, sa = _mean_std(per_level_adaptive[lvl])
        print(f"  {lvl:6.2f} | {ms:.3f} ± {ss:.3f}{'':>10} | {ma:.3f} ± {sa:.3f}")

    drop_single = np.mean(per_level_single[NOISE_LEVELS[0]]) - np.mean(per_level_single[NOISE_LEVELS[-1]])
    drop_adapt = np.mean(per_level_adaptive[NOISE_LEVELS[0]]) - np.mean(per_level_adaptive[NOISE_LEVELS[-1]])
    print(f"\n  mean accuracy drop (ideal->high noise), single best : {drop_single:+.3f}")
    print(f"  mean accuracy drop (ideal->high noise), adaptive     : {drop_adapt:+.3f}")
    return {"single": per_level_single, "adaptive": per_level_adaptive}


def aggregate_noise_aware(dataset, n_qubits, epochs, max_train, max_test, train_noise, noise_type, seeds):
    print(f"\n{'='*60}\nNoise-aware vs ideal training — {dataset}, {len(seeds)} seeds\n{'='*60}")
    per_level_ideal = {lvl: [] for lvl in NOISE_LEVELS}
    per_level_noisy = {lvl: [] for lvl in NOISE_LEVELS}
    for s in seeds:
        t0 = time.time()
        rows = _quiet(run_noise_aware, dataset, n_qubits, epochs, max_train, max_test, train_noise, noise_type, s)
        for lvl, acc_ideal, acc_noisy in rows:
            per_level_ideal[lvl].append(acc_ideal)
            per_level_noisy[lvl].append(acc_noisy)
        print(f"  seed {s} done ({time.time()-t0:.1f}s)")

    print(f"\n  {'noise':>6} | {'ideal-trained (mean±std)':>25} | {'noise-aware (mean±std)':>24}")
    print("  " + "-" * 60)
    all_ideal, all_noisy = [], []
    for lvl in NOISE_LEVELS:
        mi, si = _mean_std(per_level_ideal[lvl])
        mn, sn = _mean_std(per_level_noisy[lvl])
        all_ideal.extend(per_level_ideal[lvl])
        all_noisy.extend(per_level_noisy[lvl])
        print(f"  {lvl:6.2f} | {mi:.3f} ± {si:.3f}{'':>11} | {mn:.3f} ± {sn:.3f}")

    m_ideal, s_ideal = _mean_std(all_ideal)
    m_noisy, s_noisy = _mean_std(all_noisy)
    print(f"\n  overall mean±std, ideal-trained       : {m_ideal:.3f} ± {s_ideal:.3f}")
    print(f"  overall mean±std, noise-aware-trained : {m_noisy:.3f} ± {s_noisy:.3f}")
    overlap = (m_noisy - s_noisy) <= (m_ideal + s_ideal) and (m_ideal - s_ideal) <= (m_noisy + s_noisy)
    verdict = "bands overlap -> NOT a clear win either way" if overlap else (
        "NOISE-AWARE TRAINING clearly helps (no overlap)" if m_noisy > m_ideal
        else "ideal training clearly better here (no overlap)"
    )
    print(f"  -> {verdict}")
    return {"ideal": per_level_ideal, "noise_aware": per_level_noisy}


def run(dataset, n_qubits, epochs, max_train, max_test, n_seeds, train_noise, noise_type="depolarizing"):
    seeds = list(range(n_seeds))
    print(f"Running all 3 experiments across seeds {seeds} on dataset={dataset} (n_qubits={n_qubits})")
    t_start = time.time()

    r2 = aggregate_experiment2(dataset, n_qubits, epochs, max_train, max_test, seeds)
    r4 = aggregate_experiment4(dataset, n_qubits, epochs, max_train, max_test, noise_type, seeds)
    r5 = aggregate_noise_aware(dataset, n_qubits, epochs, max_train, max_test, train_noise, noise_type, seeds)

    print(f"\nTotal wall-clock time: {time.time()-t_start:.1f}s")
    return {"experiment2": r2, "experiment4": r4, "noise_aware": r5}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="xor", choices=["xor", "iris", "wine", "breast_cancer", "digits", "moons", "circles"])
    p.add_argument("--n_qubits", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max_train", type=int, default=40)
    p.add_argument("--max_test", type=int, default=20)
    p.add_argument("--seeds", type=int, default=5, help="number of seeds, 0..seeds-1")
    p.add_argument("--train_noise", type=float, default=0.05)
    p.add_argument("--noise_type", default="depolarizing", choices=["depolarizing", "bit_flip", "phase_flip"])
    args = p.parse_args()
    run(args.dataset, args.n_qubits, args.epochs, args.max_train, args.max_test,
        args.seeds, args.train_noise, args.noise_type)
