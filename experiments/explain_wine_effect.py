# HISTORICAL DIAGNOSTIC: narrative interpretations below are superseded.
# Use docs/RESEARCH_RECORD.md and experiments/*_audited.py for current conclusions.
"""
experiments/explain_wine_effect.py

Combines evaluation/kernel_analysis.py's diversity metrics with the
confirmed 25-seed multi-seed results (results/*_v2_diverse.json) to look
for what distinguishes Wine (the one dataset with a statistically
significant adaptive-kernel benefit) from XOR and Breast Cancer (no
benefit).

FINDING: mean pairwise kernel-kernel similarity ranks the 3 datasets in
EXACTLY the same order as the adaptive-kernel benefit:
    wine (similarity=0.649, LOWEST/most diverse) -> benefit p=0.00076
    breast_cancer (similarity=0.675)              -> no benefit p=0.792
    xor (similarity=0.691, HIGHEST/least diverse)  -> no benefit p=0.765

Two other candidate explanations were checked and RULED OUT because they
do NOT distinguish wine from breast_cancer (both have similarly high
values): the standard deviation of individual kernel-label alignment
across the 3 kernels, and the "dominance ratio" (best kernel's alignment
divided by the second-best's).

CAVEAT (important): this is 3 datasets, not a statistically testable
sample. This is a plausible, mechanistically sensible HYPOTHESIS --
"the adaptive router helps when the kernel bank is genuinely diverse,
not when one kernel dominates or all are similarly-good" -- not a proven
law. It needs testing on more datasets before it can be trusted as a
predictive rule for "when to use adaptive quantum kernels."

Run:
    python3 -m experiments.explain_wine_effect
"""

import numpy as np

# Hardcoded from evaluation/kernel_analysis.py (via experiments/kernel_diversity.py)
# and the confirmed 25-seed paired t-test results (results/*_v2_diverse.json).
DATA = {
    "xor": {
        "similarity": 0.6914,
        "alignments": [0.0894, 0.0640, 0.0335],
        "adaptive_effect": 0.004, "p_value": 0.765,
    },
    "wine": {
        "similarity": 0.6488,
        "alignments": [0.1307, 0.2525, 0.1072],
        "adaptive_effect": 0.056, "p_value": 0.00076,
    },
    "breast_cancer": {
        "similarity": 0.6747,
        "alignments": [0.1673, 0.1453, 0.3201],
        "adaptive_effect": 0.004, "p_value": 0.792,
    },
}


def run():
    print(f"{'dataset':15s} | {'kernel similarity':>18} | {'align std':>10} | "
          f"{'dominance ratio':>16} | {'adaptive effect':>16} | {'p-value':>8}")
    print("-" * 100)
    for name, d in DATA.items():
        align = np.array(d["alignments"])
        align_std = align.std()
        sorted_align = np.sort(align)[::-1]
        dominance = sorted_align[0] / sorted_align[1]
        print(f"{name:15s} | {d['similarity']:18.4f} | {align_std:10.4f} | "
              f"{dominance:16.2f} | {d['adaptive_effect']:+16.4f} | {d['p_value']:8.5f}")

    print("\nRanked by kernel-kernel similarity (LOW = more diverse):")
    ranked = sorted(DATA.items(), key=lambda kv: kv[1]["similarity"])
    for name, d in ranked:
        print(f"  {name:15s}: similarity={d['similarity']:.4f}  ->  "
              f"adaptive_effect={d['adaptive_effect']:+.4f} (p={d['p_value']:.5f})")

    print("\nFINDING: similarity ranking matches the adaptive-benefit ranking exactly.")
    print("align_std and dominance_ratio do NOT distinguish wine from breast_cancer")
    print("(both are similarly high for both datasets) -- ruled out as explanations.")
    print("\nCAVEAT: n=3 datasets. This is a hypothesis for further testing, not a")
    print("statistically confirmed law. See README.md for the full discussion.")


if __name__ == "__main__":
    run()
