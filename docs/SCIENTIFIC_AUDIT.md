# Final scientific audit

## What the evidence supports

The 25-seed Wine Expert-Imitation test is a negative result under one fixed protocol: hard/soft/margin variants average 0.878/0.852/0.872 versus 0.898 for original alignment. No variant has a positive observed mean difference, so expansion was stopped without relying on a debatable significance threshold. The fixed-bank alignment and classification controls also outperform imitation descriptively. This does **not** prove all expert-imitation strategies ineffective.

The original Wine baseline was reproduced exactly per seed for all 25 alignment accuracies and test-best single accuracies. Every one of the 38 original result JSON files is unchanged by SHA-256. New results are distinguished by dedicated directories and explicit manifests. Original README/roadmap text is archived verbatim and marked superseded.

## Leakage and target generation

| Component | Audit result | Remaining qualification |
|---|---|---|
| Outer train/test split | Disjoint raw indices stored | Repeated splits reuse the same finite dataset |
| Outer preprocessing | Fitted on training only | Held-out angle values may exceed the fitted range; intentional |
| OOF target preprocessing | StandardScaler/PCA/angle scaling refitted per inner fold | Small folds produce unstable transforms |
| OOF kernel learning | Alignment teacher trained within each fold | Fold bank differs from final outer-trained bank |
| OOF expert predictions | SVM fitted only on inner fit points | Margins are uncalibrated across experts |
| Imitation targets | Built only from held-out predictions for outer-training points | Training labels score OOF predictions, as intended supervised targets |
| No-correct-expert rows | Zero loss weight; uniform placeholder | Masked fraction and all target arrays saved |
| Gate fitting and stopping | Fixed initialization, epochs, lr; no test input | Only one schedule tested |
| Single expert selection | OOF accuracy determines deployable baseline | Tie uses first kernel; no tuned C |
| Test-best / Perfect / Oracle | Explicitly isolated post-prediction diagnostics | Test labels enter these hindsight metrics; they are not deployable methods |
| Nested predictability | Each outer probe fold rebuilds training targets via inner CV inside its fit subset | Conditional target-valid evaluation and small-sample probes; 68 MLP convergence warnings retained |

Tests change held-fold labels without changing teacher predictions/fitted transforms/theta. Another test flips all outer-test labels and checks identical targets, gate states, losses, and deployable predictions. Nested-fold tests verify index isolation. These are substantive regression checks, not proof that every possible leakage pathway is absent.

Historical kernel-aware code remains available but uses in-sample training diagnostics. Its documentation's assertion that each point's label did not affect its SVM prediction was false. Historical probe CV also shared preprocessing and generated targets across its second CV. The new nested probe is a different, valid protocol; its numbers must not silently replace the old table as if they were the same run.

## Statistics and multiple comparisons

- The primary family is exactly three variants versus original joint alignment. Paired t and Wilcoxon are both two-sided and each family is Holm-adjusted. Nominal bootstrap intervals are paired across seed rows and are not simultaneous family intervals.
- The two matched fixed-bank comparators are secondary families, with all-nine Holm t values additionally reported. Decoder/normalization interventions form an 18-comparison family. The nested probe tests form a six-comparison family. There is no global significance claim assembled by choosing a favorable family.
- Standard deviations use ddof=1. Differences are rounded before Wilcoxon ranks; the previous stored p-values were affected by floating-point tie splitting and implementation-dependent defaults. Historical values remain unchanged and can be inspected alongside the corrected calculation.
- Wine splits are not independent observations. Nominal paired p-values and seed-bootstrap CIs are not sufficient population-level confirmation. A corrected-resampled-t variance-inflation sensitivity is provided for the actual capped test/train ratio 20/40; it is approximate for this design. It is not a substitute for genuinely new evaluation data.
- Optimization restarts share the same data. Their within-setting ranges and means are descriptive; restarts were not pooled into large-n dataset-level hypothesis tests. No best test restart was selected.
- A failed significance test does not establish equivalence or rule out a bottleneck. No power/equivalence claim is made for these small samples.

## Historical claims requiring correction

| Prior claim | Stored evidence / corrected interpretation |
|---|---|
| Wine is independently confirmed | Original 25 splits: +0.046, nominal t p=.00852. Separate 25 splits: +0.014, p=.295. Pooled discovery+replication significance does not restore independent confirmation. New seed values also do not create a new Wine dataset. |
| All other PSD datasets were not rerun | Five additional 25-seed PSD files are present. Breast-cancer +0.034 has nominal t p=.0473; circles -0.008 has nominal t p=.0429. These do not establish corrected joint-test benefits. |
| Diversity rho=-.372094 is uniquely exact | That value is reproduced using raw floating mean differences. Equal decimal accuracy gains acquired artificial rank distinctions. Rounding mean gains before ranking gives rho=-.398940, p=.140745; neither analysis is significant. |
| Noise increases the advantage convincingly | The historical gap changes from .046 to .064. The paired difference-in-differences is .018 with nominal t p=.185 and 95% bootstrap CI [-.006,.046]; an increasing mean curve is not evidence of a confirmed interaction. |
| Four bottlenecks ruled out | Nonsignificant ablations do not rule out information, objective, capacity or overfitting explanations. Uniform router weights also do not mean the complete model is untrained: theta and the SVM still train. |
| 77–82% gate precision is far above 33% chance | Several experts may be correct. The new Wine uniform-random chance baseline conditional on Oracle success is 76.2%, versus alignment gate precision 81.3%. |
| Perfect Router is an achievable upper bound | It is one hindsight weighting heuristic. Wine hard/soft/margin hindsight gates yield .948/.940/.944; changing training-gate construction changes these values again. No maximization theorem or optimal gate search is present. |
| Oracle bounds the combined classifier | Oracle bounds choosing among these fixed expert predictions. A new combined Gram SVM can classify samples correctly even if all those experts fail. A monotone decomposition is not guaranteed. |
| Stable Wine restart accuracy rules out optimization generally | The original Wine raw restart results are absent. New XOR/synth_hard checks show substantial variation even with a fixed bank. Accuracy variation alone still does not prove poor local optima. |

The 15-dataset screens mix seed counts and often have incomplete configuration provenance. Several files contain duplicate subsets/arrays; they are not independent evidence. Numbers from five-seed Q4 tables, ten-seed exploratory tables and 25-seed Wine tables must not be combined as though they came from a single paired experiment.

## Kernel and implementation audit

The sample-level construction is PSD for valid base kernels because each term is `D_m K_m D_m`. The legacy pair router remains explicitly invalid and is tested as a historical known failure; it is not used by new results.

For noiseless fidelity, an input-independent terminal unitary cancels. Therefore Kernel 2's trailing star CNOT block does not cause noiseless kernel diversity. Kernel 3's final RY/RZ columns and trailing ring also cancel; only its earlier parameter block affects noiseless overlaps. New backend tests check both nonzero active gradients and zero trailing-column gradients. Circuits are preserved for comparability rather than silently redesigned.

The new optional exact-statevector backend matches original overlap values and gradients within 2e-6 in tests. It is a classical-simulator optimization, not evidence of a quantum speedup. Four-qubit kernels are classically tractable here.

Uniform sample gates yield `sum(K)/9`, whereas a uniform global mixture yields `sum(K)/3`; at fixed C this changes effective regularization. Diagonal normalization is investigated as an exploratory intervention, not assumed harmless. It does not provide a statistically established rescue in the current comparison family.

The noisy overlap construction is not covered by the ideal PSD proof. A fresh 12-point audit finds nonzero asymmetry for depolarizing and bit-flip channels when the matrix is evaluated without forced mirroring. Symmetrized eigenvalues were positive in this check, but that is not a general proof. No hardware-robustness claim is supported.

Reproducibility repairs include objective-name validation, an exact-equivalent faster backend, fold-local preprocessing, atomic new-run checkpoints, explicit configuration/environment/source guards, explicit seed and fold IDs, finite JSON handling, corrected paired statistics, and a legacy chunk-runner guard against appending a different configuration. Legacy report SD-band-overlap heuristics were replaced by paired comparisons, averaging repeated noise levels within seed.

## Remaining limitations and next justified work

This is a small-data simulation study with 40 training and 20 test observations per split, not a publish-ready general advantage claim. Imitation freezes an alignment-trained bank; the experiment cannot determine whether a different jointly trained bank would help. Expert margins are not calibrated. OOF teacher targets change with preprocessing/fold size, and all-wrong points are excluded from imitation supervision. Probe failures are not information-theoretic impossibility results, particularly with unresolved MLP convergence warnings.

The next experiment should directly measure teacher/target reliability and use inner-validation selection for any normalization/C changes, or optimize a differentiable surrogate matched to the actual combined classifier. Any confirmatory claim needs a frozen independent evaluation design. Additional datasets alone will not resolve the current mechanism question.

Method references: [sklearn preprocessing leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html), [SciPy Wilcoxon tie handling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html), [sklearn statistical comparison example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html).
