# Research continuation record

## Milestone 1 — repository and scientific audit (2026-09-18)

Inspected every source module, both narratives, tests, and all 38 input JSON result files (inventory and checksums in `input_manifest.json`; machine audit will record every result leaf). The archive contains no Expert-Imitation code/results, no dependency manifest, and no raw results for the documented Perfect-Router, nonlinear probe, or Wine optimization-stability tables. Those historical tables cannot be independently reanalyzed from the archive.

Stored Wine PSD results support a descriptive +0.046 accuracy difference on 25 splits; the separate 25-split replication has +0.014 and is nonsignificant. Pooling discovery and replication does not turn this into independent confirmation. Splits reuse Wine observations; nominal paired tests and seed bootstrap assume independence they do not have at the population level. New analysis adds a corrected-resampled-t sensitivity analysis and retains this limitation explicitly.

Claims changed:
- Nonsignificance does not rule out information, capacity, overfitting, objective, or mechanism limitations. No equivalence tests or adequate power justification were supplied.
- Best single was often the maximum **test** accuracy. A validation-selection helper exists but is not used by the main historical scripts. New results distinguish OOF-selected single (deployable selection) and test-best single (hindsight diagnostic).
- Perfect Router is one label-informed heuristic, not an optimized mathematical ceiling. Any-expert-correct Oracle bounds selection among fixed experts, not an SVM trained on a new combined kernel; the combined SVM can succeed when every expert fails.
- Chance gate precision is not 1/3 when several experts can be correct. Use the fraction of correct experts per point as the uniform-random baseline.
- Kernel-aware training diagnostics are in-sample predictions; their own labels influenced the fitted SVM. Test labels are not used, but the claimed training-point label exclusion is false.
- The old predictability probe fits preprocessing before inner CV and reuses generated targets across another CV; this is not genuinely nested evaluation. It must not be used as a leakage-free proof of an information limit.
- Kernel 2's trailing input-independent star entanglement cancels in noiseless fidelity; Kernel 3's last two parameter columns plus trailing ring also cancel. The earlier gradient test checked only the whole tensor. Preserve the circuits for comparison, but stop attributing noiseless diversity to those canceled blocks.
- Historical raw JSON arrays often lack explicit seed IDs, source version, router type, or environment. Duplicate copies are not new evidence. Some stored PSD reruns postdate contradictory roadmap entries.
- Uniform sample gates produce sum(K)/9, while legacy uniform averaging produces sum(K)/3; at fixed SVM C these are different regularization scales. New mechanism diagnostics include diagonal normalization.
- The noisy overlap circuit is not automatically a PSD kernel. Symmetric=True merely mirrors entries. Historical noise figures will be labeled legacy diagnostics, not hardware or general PSD evidence.

Next justified experiment: the frozen protocol in `configs/expert_imitation_wine.json`, saved before viewing any new outcomes. Five inner folds, fold-local scaler/PCA/angle scaling, fold-local alignment theta learning, fold-local expert SVMs; OOF predictions only for targets. Hard/soft/margin-weighted targets; all-wrong rows excluded from target loss. Shared outer-trained bank frozen during 100-step gate training; original joint alignment plus matched fixed-bank alignment and classification controls retained. 25 paired outer seeds. No downstream test labels in targets, fitting, stopping, or model selection. Test-informed diagnostics are computed only after every deployable prediction is frozen.

Practical threshold is fixed at +2 percentage points (an exploratory design choice, not a validated domain utility threshold). Expansion to XOR/synth_hard requires the full recorded decision rule. Failure triggers mechanism investigation instead of dataset expansion.

## Milestone 2 — leakage-free Wine Expert-Imitation, 25 seeds complete

Exact configuration: `configs/expert_imitation_wine.json`; resolved environment and source digest: `results/continuation/wine/manifest.json`. All 25 seed files contain fold fit/held indices, fold-specific preprocessing/theta, OOF margins, all targets/weights, learned gate states, training losses/gradient norms, test predictions, and kernel matrices. The statevector backend agrees with overlap values and parameter gradients to the tested tolerance (2e-6); historical joint-alignment mean is reproduced (0.898).

| Objective | Mean accuracy | Difference vs original joint alignment | Holm t p | Holm Wilcoxon p | Nominal paired-bootstrap 95% CI on difference | Cohen dz |
|---|---:|---:|---:|---:|---|---:|
| Joint alignment (15 epochs) | 0.898 | — | — | — | — | — |
| Matched fixed-bank alignment (100 epochs) | 0.912 | — | — | — | — | — |
| Matched fixed-bank classification surrogate (100 epochs) | 0.916 | — | — | — | — | — |
| Hard imitation | 0.878 | -0.020 | 0.2724 | 0.3441 | [-0.054, 0.010] | -0.238 |
| Soft imitation | 0.852 | -0.046 | 0.0412 | 0.0518 | [-0.078, -0.012] | -0.532 |
| Margin-weighted imitation | 0.872 | -0.026 | 0.2724 | 0.3441 | [-0.060, 0.004] | -0.308 |

No variant improves. Soft imitation's negative difference passes the nominal Holm t family but narrowly fails the Holm Wilcoxon family; do not call this jointly confirmed harm. Repeated-split dependence further weakens inferential claims. Secondary matched controls are reported separately, with an additional nine-comparison Holm sensitivity family. No hyperparameters were changed after observing the 25-seed outcomes.

Mean diagnostic references: OOF-selected single 0.834; test-best single 0.852; margin-weighted hindsight Perfect Router 0.944; fixed-expert Oracle 0.962. Relative to test-best single, original alignment captures 50.0% of this heuristic Perfect-Router gap; hard 28.3%, soft approximately 0%, margin-weighted 21.7%. Relative to OOF-selected single, the corresponding ratios are 58.2%, 40.0%, 16.4%, and 34.5%. Both definitions, paired bootstrap intervals, and denominator warnings are stored; they are diagnostic ratios, not mathematical efficiency bounds.

Claim changed: direct correctness imitation is **not supported as an improvement under this protocol**. This negative finding does not eliminate all supervised routing objectives, teacher schedules, or target formulations. Targets inherit limited fold-teacher accuracy, margin scale differences, and a distribution shift between 32-point fold teachers and the 40-point final bank.

Next justified experiments: no imitation expansion to XOR/synth_hard. Analyze same-gate Gram normalization, hard expert selection, and soft voting on Wine; contrast several hindsight target heuristics without treating any as a proven ceiling. Run predeclared optimization checks on XOR and synth_hard, separating gate-only restarts with a shared frozen bank from joint router/theta restarts, with several fixed splits and longer training budgets. These diagnostics are exploratory; do not choose a deployable model by test performance.

## Milestone 3 — mechanism interventions and optimization stability

Mechanism configuration: same stored 25 Wine seeds, same expert bank and same learned gates; compare original Gram SVM against unit-diagonal-normalized Gram SVM (C=1), top-1 expert prediction, and gate-weighted expert-sign vote. All 18 contrasts (six objectives x three interventions) form one exploratory Holm family. No decoder is selected for deployment. Exact paired tests: `results/continuation/mechanism_analysis.json`.

| Objective | Original Gram | Normalized Gram | Top-1 expert | Soft vote |
|---|---:|---:|---:|---:|
| Joint alignment | 0.898 | 0.894 | 0.782 | 0.778 |
| Fixed-bank alignment | 0.912 | 0.912 | 0.784 | 0.784 |
| Fixed-bank classification | 0.916 | 0.918 | 0.816 | 0.814 |
| Hard imitation | 0.878 | 0.876 | 0.844 | 0.848 |
| Soft imitation | 0.852 | 0.876 | 0.800 | 0.808 |
| Margin imitation | 0.872 | 0.890 | 0.842 | 0.852 |

Normalization's +0.024 for soft imitation and +0.018 for margin imitation do not survive Holm over 18 contrasts (adjusted t p=0.1046 and 0.3530; adjusted Wilcoxon p=0.1052 and 0.3347). Normalization changes scale/geometry and effective SVM regularization; these contrasts do not isolate one causal mechanism. Direct expert decoding does not rescue imitation.

Imitation improves top-1 expert correctness in some variants while reducing combined-SVM accuracy. Hard imitation fits its OOF target identities at 98.7% training accuracy but reaches 84.4% top-1 expert test accuracy. The targets/metrics are different across train/test; this is a diagnostic discrepancy, not a formal overfitting estimate. Gate precision conditional on Oracle success: alignment 81.3%, hard 87.7%, margin 87.6%; uniform-random selection baseline **76.2%, not 33.3%**. The alignment gate's 81.3% should not be sold as a huge chance-adjusted routing gain.

Hindsight gate construction matters: Wine test accuracies are 0.948 (hard), 0.940 (soft), 0.944 (margin); replacing their in-sample training gates with OOF gates gives 0.924/0.930/0.936. This dependence on a heuristic rules out treating a single Perfect-Router value as an optimal mathematical ceiling.

Optimization configuration: `configs/optimization_stability.json`; 2 datasets x 3 fixed data seeds x 8 training seeds x 3 learning rates (0.01/0.05/0.1) x 2 budgets (15/75) x 2 modes = **576 runs**. Joint mode varies router and theta initialization. Gate-only mode uses one training-only, alignment-trained bank per fixed split for every restart; only gate initialization/lr/budget vary. Every prediction, loss curve, final loss, entropy, theta, and data hash is saved. Restarts are described within fixed splits and are not treated as independent dataset replications.

| Dataset / mode | Mean within-setting restart range, 15 epochs | Mean range, 75 epochs | Largest range across tested settings |
|---|---:|---:|---:|
| XOR / joint | 0.139 | 0.144 | 0.250 |
| XOR / gate only | 0.150 | 0.161 | 0.300 |
| synth_hard / joint | 0.244 | 0.161 | 0.450 |
| synth_hard / gate only | 0.172 | 0.172 | 0.350 |

These are range statistics over eight restarts, not uncertainty intervals. With 20 test points, accuracy changes in 0.05 increments. Sensitivity remains with a fixed kernel bank; longer training sometimes helps and sometimes hurts. Test accuracy variation is evidence of training/generalization sensitivity, not proof of bad local minima. Do not select the best test restart or transfer the previous Wine-only stability statement to these datasets.

Noise validity check: Wine seed 0, first 12 training points, frozen learned theta, full unsymmetrized overlap matrices, levels 0/.05/.1 and depolarizing/bit-flip/phase-flip. Maximum asymmetry is 5.77e-5 for depolarizing, 2.78e-4 for bit flip, and numerical zero for phase flip. Symmetrized eigenvalues were positive in this small check, which is not a PSD theorem. Legacy noise curves are retained with this qualification.

Claims changed: a mismatch between expert imitation and combined-SVM training remains plausible; neither normalization nor alternative decoding establishes a fix. Optimization sensitivity is present on the two newly checked datasets. Next justified experiment: a properly nested validation-selected regularization/normalization study or a differentiable objective matched to the actual combined classifier, with teacher-target stability measured before optimizing it. Do not claim the current mechanism is proven incapable.

## Milestone 4 — nested probes, figure exports and provenance corrections

Exact configuration: `configs/mechanistic_diagnostics.json`; 10 seeds per dataset, five outer probe folds, four target-generating inner folds per outer fit subset. All transformations and quantum teachers are local to their fit subsets. Linear probe: LogisticRegression(max_iter=2000). MLP: (16,16), max_iter=1000, fixed seed. Majority baseline is refitted on each outer fit subset. Evaluate only held points with a correct expert; this is conditional expert-identity accuracy, not task accuracy. The MLP emitted 68 convergence warnings across the 150 outer probe fits; warnings are stored, and no extra training was chosen after viewing test results.

| Dataset | Linear | MLP | Majority |
|---|---:|---:|---:|
| Wine | 0.5341 | 0.5348 | 0.6038 |
| XOR | 0.4258 | 0.4620 | 0.3475 |
| synth_hard | 0.3769 | 0.3615 | 0.3915 |

Across six probe comparisons, XOR MLP's improvement has Holm t p=.0321 but Holm Wilcoxon p=.0686; no positive probe comparison passes both. Wine MLP is worse than majority under both nominal corrected families, but repeated-split dependence and convergence warnings preclude a strong mechanistic impossibility claim. The corrected-resampled sensitivity is also nonsignificant. No imitation is fitted on XOR/synth_hard in this diagnostic.

New saved alignment-only decompositions reproduce the historical 10-seed means exactly:
Wine .840 → .890 → .945 → .960; XOR .795 → .775 → .815 → .905; synth_hard .675 → .670 → .820 → .865. These arrows label quantities, not an ordered-bound theorem. The new raw predictions replace missing provenance for this specific diagnostic without inventing the previous raw files.

The complete historical audit covers **38 JSON files**. The previously reported diversity Spearman value -.372094 is reproducible when floating-point near-ties are ranked separately; rounding mean gains before ranking gives **-.398940 (p=.140745)**. Both are stored with their tie policies and neither supports a predictive law. Noise's apparent increase in gap is not a significant paired difference-in-differences (p=.185).

Generated **11 publication figure designs**, each exported as 300-dpi PNG, vector PDF and vector SVG. Every quantitative panel reads saved experimental data; plotted values and 636 source hashes are recorded in `figures/figure_manifest.json`. The system architecture is explicitly a schematic. Confidence intervals are labeled descriptive/pointwise; optimization ranges are labeled min/max rather than CIs. No missing historical result was fabricated.

Next justified action: finish test and integrity gates, then preserve the complete repository with its negative findings, audit, fixed protocols, raw checkpoints, and figures.

## Milestone 5 — final reproducibility and scientific gates

Validation completed:
- **19 tests passed** (`results/continuation/tests_final.log`): original PSD/symmetry tests, statevector/overlap values and gradients, fold-local raw preprocessing equivalence, target definitions and masking, held-label isolation, outer-test-label isolation, nested-fold isolation, Holm/zero/tie/ratio edge cases.
- `python -m experiments.verify_results` passed: all 38 historical result files unchanged; all 25 Wine seeds complete; all 150 stored gate/kernel classifier predictions replayed; exact historical Wine baseline per-seed match; all 576 stability runs and 30 nested-diagnostic runs checked; source/config manifests match; statistical analysis recalculated; figure source hashes checked.
- `python -m pip check`: no broken requirements. Python sources compiled successfully. Direct dependency and transitive-environment pins are saved.
- Figure layouts visually reviewed; labels distinguish hindsight, pointwise uncertainty, pooled visualizations, and restart ranges. Figures contain measured data, not invented examples.

Final claim: this protocol does not support Expert-Imitation as an improvement on Wine. No additional imitation datasets were run. The mechanism investigation finds an objective/decoder mismatch worth studying but no statistically established rescue. Optimization sensitivity is present on XOR/synth_hard. Historical stronger claims about eliminated bottlenecks, independent confirmation, chance routing, and theoretical ceilings are withdrawn or qualified.

Unresolved limitations are explicit in `SCIENTIFIC_AUDIT.md`: finite-data split dependence, small evaluation sets, one fixed imitation schedule, uncalibrated margins, teacher/final-bank shift, MLP convergence warnings, incomplete old-run provenance, and no hardware or quantum-advantage evidence. The continuation is reproducible; unavailable historical code versions cannot be reconstructed honestly from the archive.

Next justified experiment remains a training-only validation study of teacher/target stability and a classifier-matched objective or normalization/C choice, followed by a frozen independent evaluation. Further unplanned test-split tuning is not justified by these results.
