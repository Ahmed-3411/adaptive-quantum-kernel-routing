"""
experiments/chunk_runner.py

Same experiments as multiseed.py, but runs a SUBSET of seeds per invocation
and appends raw per-seed results to a JSON file. This lets a large seed
sweep (10-20 seeds x 3 experiments x multiple datasets) be split across
several shorter runs instead of one long one, then aggregated at the end
with experiments/report.py.

Run (repeat with different --seeds to build up the full sweep):
    python3 -m experiments.chunk_runner --dataset wine --n_qubits 4 \
        --experiments 2,4,na --seeds 0,1,2,3,4 --out results_wine.json
    python3 -m experiments.chunk_runner --dataset wine --n_qubits 4 \
        --experiments 2,4,na --seeds 5,6,7,8,9 --out results_wine.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.adaptive import run as run_experiment2
from experiments.noise import run as run_experiment4
from experiments.noise_aware_training import run as run_noise_aware


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"experiment2": [], "experiment4": [], "noise_aware": [], "config": {}}


def _save(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _done_seeds(entries):
    return {e["seed"] for e in entries}


def run(dataset, n_qubits, epochs, max_train, max_test, seeds, train_noise, noise_type, experiments, out_path):
    data = _load(out_path)
    requested_config = {
        "dataset": dataset, "n_qubits": n_qubits, "epochs": epochs,
        "max_train": max_train, "max_test": max_test,
        "train_noise": train_noise, "noise_type": noise_type,
    }

    if data.get("config") and data["config"] != requested_config:
        raise ValueError("Refusing to append incompatible configuration to an existing result file")
    data["config"] = requested_config

    if "2" in experiments:
        done = _done_seeds(data["experiment2"])
        for s in seeds:
            if s in done:
                continue
            t0 = time.time()
            res = run_experiment2(dataset, n_qubits, epochs, max_train, max_test, s)
            data["experiment2"].append({
                "seed": s,
                "classical": max(res["classical_baselines"].values()),
                "single": max(res["single_kernels"].values()),
                "adaptive": res["adaptive"],
            })
            _save(out_path, data)
            print(f"[experiment2] seed {s} done in {time.time()-t0:.1f}s")

    if "4" in experiments:
        done = _done_seeds(data["experiment4"])
        for s in seeds:
            if s in done:
                continue
            t0 = time.time()
            res = run_experiment4(dataset, n_qubits, epochs, max_train, max_test, noise_type, s)
            data["experiment4"].append({"seed": s, **res})
            _save(out_path, data)
            print(f"[experiment4] seed {s} done in {time.time()-t0:.1f}s")

    if "na" in experiments:
        done = _done_seeds(data["noise_aware"])
        for s in seeds:
            if s in done:
                continue
            t0 = time.time()
            rows = run_noise_aware(dataset, n_qubits, epochs, max_train, max_test, train_noise, noise_type, s)
            data["noise_aware"].append({"seed": s, "rows": rows})
            _save(out_path, data)
            print(f"[noise_aware] seed {s} done in {time.time()-t0:.1f}s")

    print(f"\nSaved to {out_path}")
    print(f"  experiment2 seeds done : {sorted(_done_seeds(data['experiment2']))}")
    print(f"  experiment4 seeds done : {sorted(_done_seeds(data['experiment4']))}")
    print(f"  noise_aware seeds done : {sorted(_done_seeds(data['noise_aware']))}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="xor", choices=["xor", "iris", "wine", "breast_cancer", "digits", "moons", "circles"])
    p.add_argument("--n_qubits", type=int, default=2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--max_train", type=int, default=40)
    p.add_argument("--max_test", type=int, default=20)
    p.add_argument("--seeds", type=str, required=True, help="comma-separated seed list, e.g. 0,1,2,3,4")
    p.add_argument("--train_noise", type=float, default=0.05)
    p.add_argument("--noise_type", default="depolarizing", choices=["depolarizing", "bit_flip", "phase_flip"])
    p.add_argument("--experiments", type=str, default="2,4,na", help="comma-separated subset of: 2,4,na")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    experiments = set(args.experiments.split(","))
    run(args.dataset, args.n_qubits, args.epochs, args.max_train, args.max_test,
        seeds, args.train_noise, args.noise_type, experiments, args.out)
