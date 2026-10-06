# Reproduction guide

## Scope and environment

Tested on Linux x86_64 CPU with CPython 3.12.14. Core versions: torch 2.14.0+cpu, PennyLane 0.45.1, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, Matplotlib 3.10.8, pytest 9.1.1. The exact installed dependency closure (46 packages) is `requirements-lock.txt`; a shorter direct-dependency list is `requirements.txt`. Actual platform/package metadata is in `docs/environment.json` and each experiment manifest. No GPU or downloaded datasets are needed; Wine and synthetic data come from sklearn.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest tests -q
python -m experiments.verify_results
```

The locked torch wheel is a Linux CPU build. Other operating systems or dependency versions constitute a new environment; use a new output directory and compare numerical outputs before claiming exact reproduction.

## Frozen experiments and reliable commands

Run all commands from the repository root. Each runner writes an atomic JSON checkpoint after every seed or restart. Existing outputs are skipped only after exact config/source/environment manifest agreement. Never merge different runs by copying arrays into the same file. Use a new `--out` directory when making changes.

```bash
# Independent fresh reproduction of the complete 25-seed Wine comparison:
python -m experiments.expert_imitation \
  --config configs/expert_imitation_wine.json \
  --out results/reproduction/wine

# Verify/regenerate analysis from completed canonical Wine checkpoints:
python -m experiments.expert_imitation --analyze-only

# 576 fixed-split optimization runs, default completed directory is resumable:
python -m experiments.stability_audited
# Independent fresh output:
python -m experiments.stability_audited --out results/reproduction/stability

# Nested probes and alignment-only diagnostic decompositions (30 seed/dataset runs):
python -m experiments.diagnostics_audited
# Independent fresh output:
python -m experiments.diagnostics_audited --out results/reproduction/diagnostics

# Analyze original archived JSON, same-gate interventions, and noisy overlap validity:
python -m experiments.audit_stored_results
python -m experiments.mechanism_audit
python -m experiments.noise_validity

# Regenerate all figures ONLY from canonical saved experimental data:
python -m experiments.publication_figures
python -m experiments.verify_results
```

Figure and mechanism scripts intentionally read canonical `results/continuation` paths. Compare an independent reproduction against those files first; do not overwrite the original evidence to make a plot. `experiments.verify_results` checks the canonical evidence delivered in this archive, not arbitrary reproduction directories.

## Protocol details

- Wine uses binary classes 0/1, labels -1/+1, four qubits. XOR uses two qubits; synth_hard uses four. Outer split fraction is 0.3, followed by stratified caps at 40 training / 20 test points, matching the supplied loader.
- Raw dataset indices are saved. Outer preprocessing is fitted only on the capped training subset. Inner folds independently fit StandardScaler, PCA if needed, and MinMax angle scaling. Held-out angles are not clipped; they may lie outside [-pi,pi].
- Wine outer seeds are exactly 0–24. Five stratified inner folds use the same seed. For every inner fold, both quantum alignment teacher and expert SVMs are fitted on its 32 training points; predictions on the other eight produce target margins. Fold fit/held IDs and fitted states are saved.
- All expert targets are fixed before gate training. Hard targets choose the correct expert with the largest positive signed margin; exact ties use the first kernel. Soft targets distribute mass uniformly among correct experts. Margin targets normalize positive signed margins. Rows with no positive signed margin have zero loss weight and an unused uniform placeholder target.
- A full-training alignment teacher supplies the shared outer theta. Imitation and matched alignment/classification controls start from identical fresh gate weights and train 100 Adam steps at lr=.01 with this bank frozen. Original joint alignment trains 15 steps at lr=.05. No early stopping or test-selected epochs.
- A teacher trained under alignment favors a particular bank; freezing it isolates gate objectives conditional on that bank, not all possible joint expert-imitation systems. Fold and full teachers differ in sample count/preprocessing. Raw margins are uncalibrated and not directly comparable probabilities.
- SVC uses C=1, precomputed kernels, otherwise sklearn defaults. The deployable single kernel is chosen by OOF expert accuracy; test-best is a separate hindsight quantity.
- Quantum fidelity is computed through an optional exact statevector implementation: |Psi(X) Psi(Z)^H|². The original overlap backend remains available and remains the model default. Tests compare values and gradients. This improves simulation speed without changing the feature maps or claiming hardware efficiency.
- Perfect Router uses in-sample expert margins for training gates and **test-label-informed** margins for test gates; it is diagnostic only. Additional heuristic and OOF-training-gate variants demonstrate that it is not a mathematical optimum.

## Statistics and branching

The primary family consists of the three imitation-minus-original-alignment comparisons. Both paired t and two-sided Wilcoxon signed-rank tests are reported, separately Holm-adjusted across three variants. Differences are rounded to 12 decimals before ranks to remove floating-point pseudo-ties; Wilcoxon explicitly uses asymptotic/Wilcox-zero handling. Reported SD uses ddof=1. Paired bootstrap: 20,000 seed-pair resamples, RNG seed 20260918. Cohen dz, small-sample Hedges gz, and matched rank-biserial effects are included. Individual bootstrap intervals are not simultaneous family intervals.

Matched fixed-bank alignment and classification are secondary comparator families. Nine-comparison Holm t values are supplied as a sensitivity analysis for readers considering all controls together. Mechanism interventions form a separate exploratory 18-comparison family; probes form a six-comparison exploratory family. No cross-family omnibus claim is made.

Because Wine splits reuse observations, nominal tests and seed-bootstrap intervals describe variability under this split protocol and do not provide independent-sample population evidence. A corrected-resampled-t sensitivity calculation inflates variance by `(1/n + n_test/n_train)`; the capped split design makes this an approximation, not a universal correction. Stability restarts are not pooled as independent dataset replications.

Expansion required a mean improvement ≥.02, positive nominal bootstrap lower limit, both Holm-adjusted tests and corrected-resampled-t ≤.05, and positive improvement over matched fixed-bank alignment. **No variant passed**, including the simpler requirement of positive observed mean gain. No imitation was run on other datasets. Alignment-only mechanism/probe runs on XOR/synth_hard answer separate diagnostic questions.

## Historical reproducibility limits

Every original result file is hash-checked and unchanged. Several lack explicit IDs/configuration/environment. No previous implementation snapshots for the original broken circuits/router are present; their old outputs cannot all be exactly regenerated from current code. Existing original experiment scripts remain for history, with notices on superseded interpretations. Prefer the continuation commands above for current claims. Original docs are archived verbatim instead of erasing the negative record.

References for implementation choices: [sklearn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html), [SciPy Wilcoxon rounding and ties](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html), and [sklearn corrected resampled test example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html).
