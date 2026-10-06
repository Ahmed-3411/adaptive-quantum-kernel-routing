"""
experiments/report.py

Reads a JSON file built up by experiments/chunk_runner.py (possibly across
many separate invocations) and prints the same mean +/- std summary that
experiments/multiseed.py prints, but for however many seeds have been
accumulated so far -- so we can inspect partial progress at any time.

Run:
    python3 -m experiments.report --file results_wine.json
"""

import argparse
import json
import numpy as np
from evaluation.stats import paired_comparison, format_result


def _mean_std(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def report(path):
    with open(path) as f:
        data = json.load(f)

    cfg = data.get("config", {})
    print(f"\n{'='*64}")
    print(f"Report for {path}")
    print(f"config: {cfg}")
    print(f"{'='*64}")

    e2 = data.get("experiment2", [])
    if e2:
        classical = [r["classical"] for r in e2]
        single = [r["single"] for r in e2]
        adaptive = [r["adaptive"] for r in e2]
        print(f"\n--- Experiment 2 (n={len(e2)} seeds) ---")
        mc, sc = _mean_std(classical)
        ms, ss = _mean_std(single)
        ma, sa = _mean_std(adaptive)
        print(f"  best classical        {mc:.3f} ± {sc:.3f}")
        print(f"  best single quantum    {ms:.3f} ± {ss:.3f}")
        print(f"  adaptive               {ma:.3f} ± {sa:.3f}")
        if len(e2) >= 2:
            print(format_result("adaptive - test-best single (historical)", paired_comparison(adaptive, single)))
            print("Nominal seed-paired inference; split dependence and original provenance remain limitations.")

    e4 = data.get("experiment4", [])
    if e4:
        print(f"\n--- Experiment 4 / noise robustness (n={len(e4)} seeds) ---")
        levels = e4[0]["noise_level"]
        print(f"  {'noise':>6} | {'single_best (mean±std)':>24} | {'adaptive (mean±std)':>22}")
        print("  " + "-" * 58)
        for idx, lvl in enumerate(levels):
            s_vals = [r["single_best"][idx] for r in e4]
            a_vals = [r["adaptive"][idx] for r in e4]
            ms, ss = _mean_std(s_vals)
            ma, sa = _mean_std(a_vals)
            print(f"  {lvl:6.2f} | {ms:.3f} ± {ss:.3f}{'':>10} | {ma:.3f} ± {sa:.3f}")

    na = data.get("noise_aware", [])
    if na:
        print(f"\n--- Noise-aware vs ideal training (n={len(na)} seeds) ---")
        levels = [row[0] for row in na[0]["rows"]]
        print(f"  {'noise':>6} | {'ideal-trained (mean±std)':>25} | {'noise-aware (mean±std)':>24}")
        print("  " + "-" * 60)
        all_ideal, all_noisy = [], []
        for idx, lvl in enumerate(levels):
            i_vals = [r["rows"][idx][1] for r in na]
            n_vals = [r["rows"][idx][2] for r in na]
            all_ideal.extend(i_vals)
            all_noisy.extend(n_vals)
            mi, si = _mean_std(i_vals)
            mn, sn = _mean_std(n_vals)
            print(f"  {lvl:6.2f} | {mi:.3f} ± {si:.3f}{'':>11} | {mn:.3f} ± {sn:.3f}")
        m_ideal, s_ideal = _mean_std(all_ideal)
        m_noisy, s_noisy = _mean_std(all_noisy)
        print(f"\n  overall: ideal={m_ideal:.3f}±{s_ideal:.3f}  noise-aware={m_noisy:.3f}±{s_noisy:.3f}")
        if len(na) >= 2:
            # Repeated noise levels are not independent replicates: average WITHIN seed first.
            seed_ideal = [np.mean([r[1] for r in row["rows"]]) for row in na]
            seed_noisy = [np.mean([r[2] for r in row["rows"]]) for row in na]
            print(format_result("noise-aware - ideal (one noise-average per seed)", paired_comparison(seed_noisy, seed_ideal)))



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True)
    args = p.parse_args()
    report(args.file)
