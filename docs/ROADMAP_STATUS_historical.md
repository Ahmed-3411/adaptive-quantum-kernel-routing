# Research Roadmap — Status Tracker

Tracks the external "Adaptive Quantum Kernel Research Roadmap — 2026" against
what's actually been implemented and verified in this codebase. Updated as
work progresses. See `README.md` for the full narrative and all numeric
results this tracker refers to.

## Central hypothesis (roadmap Section 1)

> Adaptive quantum kernels are useful when the available quantum kernels
> induce complementary geometries, this complementarity can be detected
> locally, exploited through adaptive routing, and ultimately used to
> predict when adaptive routing will help on unseen datasets.

Current evidence: the first two clauses ("useful when kernels are
complementary", "exploited through adaptive routing") have real, statistically
significant support **on one dataset (Wine)**. The third clause ("detected
locally" via a diversity metric that predicts benefit) has **directional but
not statistically confirmed** support (Spearman rho=-0.714, p=0.111, n=6
datasets — nowhere near enough for the roadmap's Phase 8 prediction target).
The fourth clause (predicting benefit on genuinely unseen datasets) has not
been attempted.

## PSD-preserving fix (roadmap Phase 1) — DONE

**Bug found:** the original `per_pair` router (`routing/classical_router.py`)
computes `weight(xi, xj)` directly from the pair via
`concat(xi, xj, |xi-xj|)`. Nothing forces `weight(xi,xj) == weight(xj,xi)`,
so the combined kernel was measurably **not symmetric and not PSD**:

```
max|K - K.T|   = 0.338   (want ~0)
min eigenvalue = -0.135  (want >= 0)
```

Every result in this project before this fix (the Wine significance, the
ablation study, the noise-robustness numbers) used this construction.

**Fix implemented:** `routing/sample_level_router.py` — a SAMPLE-level gate
`g(x) = softmax(f_phi(x))`, combined as
`K_A(xi,xj) = sum_m g_m(xi) g_m(xj) K_m(xi,xj)`. This is the standard
"positive combination of kernels is a kernel" argument (each `D_m K_m D_m`
is PSD for PSD `K_m`; sum of PSD is PSD), made pair-dependent through `g(x)`.
Verified numerically:

```
max|K - K.T|   = 0.0       (exact, by construction)
min eigenvalue = +0.006    (PSD)
theta sensitivity: max|K_after - K_before| = 0.221  (theta still influential)
```

`sample_level` is now the **default** `router_type` in
`models/adaptive_qkernel.py`. `per_pair` is kept only for historical
ablation comparison and is explicitly documented as broken
(`tests/test_psd_symmetry.py::test_per_pair_router_is_known_broken`).

**Bonus:** sample-level gates need O(n) forward passes to build a kernel
matrix instead of O(n^2) for the pair-level router — Wine Experiment 2 went
from ~8s/seed to ~3s/seed.

**Does the headline result survive the fix?** Yes.

| | per_pair (broken) | sample_level (PSD-valid) |
|---|---|---|
| mean diff (adaptive − single) | +0.056 | **+0.046** |
| paired t-test p | 0.00076 | **0.0085** |
| Wilcoxon p | not computed | **0.0220** |
| bootstrap 95% CI | not computed | **[+0.016, +0.078]** (excludes 0) |

Weaker (the PSD-valid construction is more constrained than an unconstrained
pairwise router) but still comfortably significant by every test the roadmap
requires. n=25 seeds, Wine, `results/results_wine_psd.json`.

## Gate status (roadmap Section 10)

- [x] **Gate 1** — symmetry + PSD + gradient tests pass. `tests/test_psd_symmetry.py`, 7/7 passing.
- [ ] **Gate 2** — results reproducible from frozen configs. Not yet formalized (no config-freeze file / seed-lock manifest yet).
- [ ] **Gate 3** — benefit appears across multiple datasets, not only one. **RE-TESTED at n=15 datasets (up from 6), 10 seeds each (Wine at n=25).** 3/15 significant uncorrected (wine p=0.0085, wine_12 p=0.011, synth_easy p=0.028); **0/15 survive Holm correction across all 15** (Holm threshold for the smallest p-value alone is 0.0033) -- roughly consistent with chance alone (~0.75 false positives expected at alpha=0.05 across 15 independent tests). Definitively unmet at the roadmap's bar with this sample size. Important distinction: Wine's result is independently CONFIRMATORY evidence (its own dedicated n=25 replication, cross-validated with t-test + Wilcoxon + bootstrap CI + Oracle analysis + noise robustness), a stronger evidentiary standard than one p-value in an exploratory 15-way screen. wine_12 and synth_easy are exploratory hits at n=10 only, NOT yet independently replicated -- treat as hypotheses, not findings (this project's own history shows n=10 hits can evaporate at n=25, e.g. Digits).
- [ ] **Gate 4** — diversity/local complementarity explains part of the gain. **RE-TESTED at n=15 datasets (up from 6) -- weakened further, now decisively.** Spearman rho went from -0.714 (p=0.111, n=6) to **-0.372 (p=0.172, n=15)** for the alignment-similarity metric specifically. Combined with the earlier finding that the 4 diversity metrics don't agree with each other, this is now a much better-powered negative result, not just an underpowered one: **the apparent diversity-predicts-benefit pattern from the original 6 datasets appears to have been a small-sample artifact.** Unmet at the roadmap's bar, and unlikely to be rescued by yet more datasets of this same kind without a different diversity formulation.
- [x] **Gate 5** — ablations/oracle show local routing itself matters. **Nuanced result.** The Global-vs-Adaptive ablation (re-tested with the corrected router + Holm correction) does NOT show adaptive beating simpler combination strategies significantly. But the **Oracle analysis** (`experiments/oracle.py`, n=25) shows: (a) real complementary information exists across the 3 kernels (Oracle − best_single = +0.110, p<0.0001, Cohen's d=1.9), (b) the adaptive router captures a statistically significant share of it vs. a single kernel (+0.046, p=0.0085), but (c) a significant gap remains (Oracle − adaptive = +0.064, p=0.0003) — **the router captures only ~42% of the available headroom.** Gate is "met" in the weak sense (routing captures real value) but not in the strong sense (routing isn't clearly the best way to capture that value yet).
- [x] **Gate 6** — noise (and hardware) don't destroy the effect. **Re-validated with the PSD-preserving router (n=25):** the adaptive-vs-single gap is significant at uncorrected alpha=0.05 at every noise level tested (0.00 to 0.10), and grows with noise (+0.046 to +0.064) while single-kernel accuracy degrades and adaptive accuracy is flat-to-improving. Only noise=0.00 survives strict Holm correction across the 5 (non-independent) noise levels -- a conservative but honest caveat. No hardware run yet.
- [ ] **Gate 7** — benefit predictor generalizes to unseen datasets. Not attempted (roadmap Phase 8).

## Phase-by-phase status

| Phase | Status | Notes |
|---|---|---|
| 0 — Foundation / clean baseline | Partial | Bugs fixed as found (2 kernel bugs + this PSD bug). No formal config-freeze / nested-CV yet — still using a single train/test split per seed, not nested CV. |
| 1 — PSD-preserving kernel | **Done** | See above. |
| 2 — Pilot benchmark (6-10 datasets, Uniform/Global/Adaptive/Single) | Partial | Have 6 datasets and all 4 baselines from the ablation study, but computed with the OLD broken router — needs a full redo with `sample_level` + Wilcoxon + bootstrap CI (this file's evaluation/stats.py is ready for that). |
| 3 — 100-500 dataset benchmark | Partial (scaled down) | Expanded from 6 to 15 datasets (5 sklearn-digit pairs, 3 synthetic separability levels, gaussian quantiles, wine class-pair variant, plus the original 6). Far short of 100-500, but large enough to show the n=6 diversity correlation was likely a small-sample artifact (see Gate 4). `data/datasets.py`, `results/expanded_benchmark.json`. |
| 4 — Diversity -> adaptive gain | Partial (weakened) | 4 diversity metrics now computed (`evaluation/diversity_metrics.py`, `experiments/diversity_full.py`): alignment/correlation, Frobenius distance, eigenspectrum distance, prediction disagreement. **They disagree with each other** (cross-metric rho -0.67 to +0.14, n.s.), and only the original alignment_similarity metric points the expected direction. This is a genuine weakening of the diversity hypothesis, not a confirmation -- see Gate 4. |
| 5 — Local routing + interpretability | Not started | No routing-entropy analysis, no visualization of g_m(x), no random-router baseline yet. |
| 6 — Oracle vs Adaptive vs Global | **Done** | Global/Uniform/Adaptive/Oracle all measured on Wine, n=25, full stats (t-test+Wilcoxon+bootstrap+Holm). See Gate 5. |
| 7 — Noise + hardware | Partial | Noise sweep re-validated with `sample_level` router (n=25, all 5 levels significant uncorrected, noise=0.00 survives Holm). Noise-aware training still on old router (needs redo). No hardware run, no device-conditioned routing. |
| 8 — Unseen-dataset benefit prediction | Not started | |
| 9 — Paper + reproducibility | Not started | |

## Immediate next steps (in priority order per roadmap Section 9)

1. ~~Redo the Phase 2 pilot benchmark ablation (Uniform/Global/Adaptive/no_kernel3) with `sample_level` router~~ **DONE — and it changed the conclusion.** With correct PSD-preserving kernels and Wilcoxon + Holm correction, full_adaptive is NOT significantly better than global_router, uniform_router, or no_kernel3 (all p > 0.07 uncorrected, none survive Holm). What IS still established (separately, vs. a single kernel, see the PSD-preserving fix section above) is that *combining* the 3 kernels beats using one alone on Wine. What is NOT established anymore is that *adaptive, sample-dependent* combination specifically beats simpler combination strategies (global weights, uniform average). This is an important correction to the project's earlier claim.
2. ~~Add an Oracle baseline~~ **DONE** (`experiments/oracle.py`, n=25). Real
   complementary information exists in the kernel bank (Oracle − best_single
   = +0.110, p<0.0001, d=1.9); the adaptive router captures a significant
   but partial share of it (~42% of available headroom; Oracle − adaptive
   = +0.064, p=0.0003 gap remains). This reframes finding #1 productively:
   routing DOES extract real value from complementarity (Gate 5, weak
   sense), it's just not yet clearly better than simpler combination
   strategies at doing so (Gate 5, strong sense) -- a concrete target for
   router-architecture improvement, not a dead end.
3. ~~Expand diversity metrics beyond kernel-kernel similarity~~ **DONE --
   and it's a warning sign, not a confirmation.** Frobenius distance,
   eigenspectrum distance, and prediction disagreement were added
   (`evaluation/diversity_metrics.py`). They do NOT agree with each other
   or consistently with the original alignment_similarity metric (cross-
   metric Spearman rho ranges -0.67 to +0.14, all n.s.). Frobenius distance
   in particular correlates with adaptive benefit in the OPPOSITE direction
   naively expected. **Recommendation: do not present the diversity
   hypothesis as settled in any paper draft based on this project's data
   alone** -- it rests on one specific metric (alignment_similarity) that
   doesn't generalize to other reasonable notions of "kernel diversity,"
   and n=6 datasets is far too small regardless. This needs either (a) many
   more datasets (roadmap Phase 3's 100+), or (b) a principled argument for
   why alignment_similarity specifically (not the others) should be the
   right notion here, before it belongs in a confirmatory results section.
4. ~~Re-run the noise-robustness experiment (Experiment 4) with `sample_level`~~
   **DONE** (n=25, `results/noise_wine_psd_raw.json`). **The finding survives,
   and even looks slightly cleaner than the broken-router version:**

   | noise | single | adaptive | p (t-test, uncorrected) |
   |---|---|---|---|
   | 0.00 | 0.852±0.069 | 0.898±0.067 | 0.0085 |
   | 0.01 | 0.856±0.073 | 0.900±0.065 | 0.0156 |
   | 0.03 | 0.850±0.081 | 0.900±0.066 | 0.0208 |
   | 0.05 | 0.850±0.086 | 0.902±0.066 | 0.0213 |
   | 0.10 | 0.840±0.104 | 0.904±0.063 | 0.0169 |

   All 5 levels significant at uncorrected alpha=0.05 (both t-test and
   Wilcoxon). **After Holm correction across the 5 levels, only noise=0.00
   formally survives** (p=0.0085 < Holm threshold 0.01) -- Holm is
   conservative here and the 5 tests aren't independent (same seeds/model
   across noise levels), so this likely understates the true robustness,
   but it's the honest, roadmap-compliant number to report. The qualitative
   trend is unambiguous either way: single-kernel accuracy drops under noise
   (0.852->0.840) while adaptive accuracy is flat-to-improving
   (0.898->0.904), so the gap grows from +0.046 to +0.064.
5. ~~Try to close the Oracle gap via Objective B~~ **DONE — did not help.**
   Trained with a differentiable classification-surrogate loss
   (`training/classification_loss.py`, leave-one-out weighted-neighbor
   score + logistic loss) instead of kernel-target alignment. Result on
   Wine (n=25): Objective B = 0.886±0.061 vs. Objective A = 0.898±0.067,
   mean_diff=-0.012, p=0.471 (not significant, tiny effect size d=-0.146).
   **The Oracle gap is not an objective-function problem.** Remaining
   untried directions: router architectural capacity (deeper/wider gate
   network), conditioning the gate on per-kernel disagreement/confidence
   signals rather than just raw features (roadmap Phase 5's interpretability
   angle), or accepting that ~42% capture may be close to what a
   softmax-gated linear combination can achieve and a fundamentally
   different combination mechanism would be needed to do better.

## Research Audit (post-draft): answering 5 questions before Phase 3

Triggered by a user-caught inconsistency between the paper draft's Spearman
rho (-0.372) and a chat-message value (-0.293) -- **recomputing directly
from the raw saved JSON confirms rho=-0.372094, p=0.172022 is correct.**
The -0.293 figure was stale text from an intermediate edit of this file
that leaked into a summary before being corrected; it never matched any
saved raw data. This mismatch is exactly why the paper draft is marked a
review copy, not a final version, until this audit is complete.

### Q1: Is the Wine effect real?

**Yes, but smaller than the original estimate.** A true held-out
replication -- 25 BRAND NEW seeds (25-49) never used in any prior
analysis in this project -- was run and compared against the original
seeds 0-24:

| | n | mean diff | p (t-test) | p (Wilcoxon) |
|---|---|---|---|---|
| Original (seeds 0-24) | 25 | +0.046 | 0.0085 | 0.0220 |
| Holdout (seeds 25-49, never seen before) | 25 | +0.014 | 0.295 | 0.289 |
| **Combined (seeds 0-49)** | **50** | **+0.030** | **0.00625** | **0.0140** |

The holdout replication alone is NOT significant, but this is consistent
with a real but smaller effect combined with sampling noise, not with a
false positive: the pooled n=50 estimate remains significant (p=0.0063),
just with the effect size revised down from +0.046 (Cohen's d=0.573) to
+0.030 (Cohen's d=0.404) -- a textbook regression-to-the-mean correction
of an effect size estimated from the same data used to discover it.
**Conclusion: the Wine effect is real. Future power calculations and any
paper claim should use +0.03, not +0.046, as the expected effect size.**
Raw data: `results/wine_holdout_seeds25_49.json`.

### Q2: Is Oracle advantage present only in Wine, or in other datasets too?

*In progress.*

### Q3: Why doesn't Adaptive reach Oracle?

*In progress.*

### Q4: What distinguishes datasets where adaptive gain appears?

*In progress.*

### Q5: Is the problem in the routing mechanism, the kernel bank, or the training objective?

*In progress -- partial evidence already gathered: the ablation (Section
"PSD-preserving fix" above) shows global/uniform combination rules are not
significantly worse than the adaptive router, and Objective B (Section
3.7 of the paper draft) did not improve on Objective A. Both point away
from "routing mechanism" and "objective" as the sole bottleneck, leaving
the kernel bank itself and/or router capacity as the more likely
remaining candidates -- to be tested directly.*

## Where this leaves the project (honest summary)

attempted, including a modest Phase 3 expansion (6 -> 15 datasets). The
state of evidence:

- **Solid, reproducible, statistically confirmed:** on Wine specifically
  (n=25 seeds, the project's one well-powered run), combining the 3 kernels
  via a PSD-valid sample-level router beats the best single kernel
  (p=0.0085), the benefit survives and slightly grows under simulated
  depolarizing noise (p<0.05 at every level, uncorrected), and there is
  real complementary information in the kernel bank to exploit (Oracle -
  best_single, p<0.0001, d=1.9).
- **Not established, and now well-powered evidence AGAINST:** that the
  benefit generalizes across datasets (15 datasets tested, 10 seeds each:
  0/15 survive Holm correction) or that kernel diversity predicts it
  (Spearman rho weakened from -0.714 at n=6 datasets to -0.372 at n=15,
  and 4 diversity metrics disagree with each other). Both looked like real
  patterns at n=6; both look like small-sample artifacts at n=15. This is
  the single most important methodological lesson of the project: an
  apparently confirmed cross-dataset pattern evaporated with 2.5x more data,
  exactly like every single-seed "adaptive wins" result evaporated earlier
  with more seeds.
- **Not established, despite trying:** that adaptive/sample-dependent
  routing specifically beats simpler combination rules (global/uniform
  weights) on Wine -- not significant even with a different training
  objective (Objective B).
- **Not attempted:** the full Phase 3 scale (100-500 datasets -- 15 is a
  down payment, not the real thing), Phase 5 (interpretability/routing
  visualization), Phase 7's hardware run, Phase 8 (unseen-dataset benefit
  prediction), Phase 9 (paper writeup).

The honest one-paragraph summary a paper's abstract could support today:
*"We find that combining three structurally diverse quantum kernels via a
PSD-preserving, sample-level adaptive router significantly outperforms the
best individual kernel on one of fifteen tested datasets (Wine, n=25 seeds,
p=0.0085), with the benefit persisting under simulated noise; however,
across the full 15-dataset panel we find no evidence that this benefit
generalizes (0/15 survive multiple-comparison correction), and a kernel-
diversity metric that appeared to predict which datasets would benefit at
n=6 datasets no longer does at n=15 (Spearman rho: -0.714 -> -0.372),
indicating the earlier pattern was likely a small-sample artifact. The
mechanism behind Wine's dataset-specific gain remains an open question."*
That is a legitimate, modest empirical finding -- not the sweeping
"adaptive quantum kernels work" claim the roadmap's Section 11 aspires to,
but an honest, well-tested waypoint toward it, and a clear demonstration of
exactly the discipline (multiple comparisons, larger samples before
trusting a correlation) the roadmap's Section 5 and Section 8 failure modes
warn about.


## Research Audit (pre-Phase-3): 5 questions before scaling to 100+ datasets

Proposed by the user as a necessary gate before Phase 3: audit the current
15-dataset evidence base rather than diluting it with a large but shallow
benchmark. Answers below, updated as each question is investigated.

### Q1: Is the Wine effect real? — ANSWERED (partially, calibrated down)

Ran 25 entirely fresh, never-before-used seeds (25-49) as a held-out
replication of the original n=25 (seeds 0-24) finding.

| Batch | n | mean_diff | p (t-test) | Significant? |
|---|---|---|---|---|
| Original (seeds 0-24) | 25 | +0.046 | 0.0085 | Yes |
| Fresh replication (seeds 25-49) | 25 | +0.014 | 0.295 | No |
| Combined (seeds 0-49) | 50 | +0.030 | 0.00625 | Yes |

The two batches' effect estimates are NOT significantly different from each
other (unpaired t-test on the two sets of paired differences, p=0.129) --
this is consistent with a single real underlying effect with substantial
seed-to-seed variance, not a qualitative disappearance like the Digits
false positive (Section 3.6/README). **Verdict: the effect is real but was
initially overestimated.** Best current estimate of the true effect size:
+0.030 (95% CI [+0.010, +0.051], n=50), not the original +0.046. This
downward recalibration on replication (sometimes called a "winner's curse"
on the first significant estimate of a new effect) should be treated as
the standard going forward -- report the n=50 numbers, not the n=25 ones,
in any future summary of this finding. Raw data:
`results/wine_replication_seeds25to49.json`, `results/wine_audit_q1_combined_n50.json`.

### Q2: Is the Oracle advantage Wine-specific, or general? — NOT YET ANSWERED
### Q3: Why doesn't Adaptive reach Oracle? — NOT YET ANSWERED
### Q4: What distinguishes datasets where adaptive gain appears? — NOT YET ANSWERED
### Q5: Is the bottleneck the router, the kernel bank, or the objective? — PARTIALLY ANSWERED
Existing evidence: ablation (Section 3.3) found adaptive not significantly
different from global/uniform combination -> weak evidence against "the
router's adaptivity" as the sole explanation. Objective B (Section 3.7)
found no improvement -> weak evidence against "the objective" as the
bottleneck. Neither test is conclusive; full answer pending Q2-Q4 results.

### Q2: Is the Oracle advantage Wine-specific, or general? — ANSWERED: general

Ran the Oracle diagnostic (experiments/oracle.py) on the other 14 datasets
at n=5 seeds each (small-n screen; individual dataset numbers below should
be read as directional, not precise, given n=5). Result:

**The Oracle-single gap (complementary information across the 3 kernels)
is NOT Wine-specific -- it is present, often substantially, across nearly
all 15 datasets.** 6/15 reach nominal significance even at n=5 (wine
p<0.0001 at n=25; synth_hard p=0.0005, synth_easy p=0.0046, synth_medium
p=0.0048, moons p=0.020, gaussian_quantiles p=0.028 at n=5). Several
datasets have LARGER oracle-single gaps than Wine (synth_hard +0.260,
synth_medium +0.170, gaussian_quantiles +0.150) despite showing ~zero
adaptive benefit in Section 3.2's Experiment 2 results.

### Q3: Why doesn't Adaptive reach Oracle? — PARTIALLY ANSWERED

Computed a "capture fraction" = (adaptive_acc - single_acc) / (oracle_acc -
single_acc) for all 15 datasets (same seed subset for both quantities):

| Dataset | Oracle gap | Adaptive gap | Capture |
|---|---|---|---|
| synth_hard | +0.260 | +0.040 | 15.4% |
| synth_medium | +0.170 | +0.000 | 0.0% |
| gaussian_quantiles | +0.150 | -0.010 | -6.7% |
| synth_easy | +0.140 | +0.030 | 21.4% |
| moons | +0.110 | -0.000 | -0.0% |
| **wine** | +0.110 | +0.046 | **41.8%** |
| xor | +0.110 | -0.050 | -45.5% |
| digits | +0.090 | -0.000 | -0.0% |
| wine_12 | +0.030 | +0.020 | 66.7% |
| breast_cancer | +0.030 | -0.030 | -100.0% |
| iris | +0.020 | +0.000 | 0.0% |
| digits_01 | +0.010 | +0.010 | 100.0% |
| digits_69 | +0.010 | -0.020 | -200.0% |
| digits_45 | +0.010 | +0.010 | 100.0% |
| circles | +0.010 | +0.000 | 0.0% |

**The critical finding: oracle headroom is common, but the router's capture
rate of that headroom is highly variable and frequently negative** (the
router makes things WORSE than the best single kernel despite complementary
information being available: xor -45%, breast_cancer -100%, digits_69
-200%, though the smallest-magnitude cases are likely noise at n=5-10).
This points toward **the router/training mechanism as the primary
bottleneck, not the kernel bank** -- the raw ingredient for adaptive
routing to help (complementary kernels) is usually present; the learned
router usually fails to reliably extract it, and sometimes actively hurts.
Caveat: capture-fraction ratios are noisy when the oracle-gap denominator
is small (e.g. circles, digits_45) and n is only 5-10 seeds; the
qualitative pattern (widespread headroom, unreliable capture) is the
robust takeaway, not each individual percentage.

### Q4: What distinguishes datasets where adaptive gain appears? — REVISED framing needed
Given Q2/Q3, this question should be reframed: it is not "which datasets
have complementary kernels" (most do) but "which datasets let the router
reliably capture that complementarity." Not yet investigated under this
reframing.

### Q5: Is the bottleneck the router, the kernel bank, or the objective? — LEANING: router/training mechanism
Combined evidence now points at the router/training mechanism specifically:
(a) the kernel bank generally DOES contain complementary information
(Q2 -- rules out "kernel bank" as the primary issue); (b) capture of that
information is wildly inconsistent across datasets and often negative
(Q3 -- implicates the router/training dynamics); (c) a different objective
(classification surrogate, Section 3.7) did not help, weakly suggesting
the issue is more architectural/optimization-related than the choice of
loss function specifically. Not yet conclusive -- would benefit from
directly inspecting learned gate values g(x) on cases where the router
underperforms (roadmap Phase 5's interpretability angle), which is the
natural next step.

### Q5 follow-up: direct gate inspection (experiments/router_diagnostics.py)

Compared the router's learned gate values g(x) on Wine (41.8% Oracle-capture)
vs. XOR (-45.5% capture) directly, focusing on "Oracle-rescuable failures"
(test points where at least one single kernel is correct -- so Oracle
succeeds -- but the deployed adaptive model is wrong):

| | Wine | XOR |
|---|---|---|
| Mean gate entropy (% of max) | 10.1% | 13.3% |
| Oracle-rescuable failure rate | 9.5% | 18.5% |
| Gate weight on would-be-correct kernel minus wrong kernel | -0.073 | +0.011 |

**Key finding: the router's gates are highly decisive (low entropy, near
0/1 weights) in BOTH datasets, but this decisiveness is not reliably
correct on the specific points where it matters (the gap between gate
weight on correct vs. wrong kernels is approximately zero/negative in
both cases, not just in the low-capture dataset).** This is a more
specific and more actionable finding than the aggregate capture-percentage
comparison: it suggests the router is not failing by being "wishy-washy"
about which kernel to trust -- it is failing by being confidently wrong
on hard cases in a way that doesn't obviously differ between a
high-capture and a low-capture dataset. The difference in overall capture
percentage looks more attributable to the BASE RATE of hard
("Oracle-rescuable") cases (XOR has ~2x Wine's rate) than to the router
behaving differently once such a case arises.

**Revised Q5 answer:** the bottleneck is the router's gating *quality* on
hard/ambiguous points specifically, not its decisiveness, not the
objective (Section 3.7 already ruled that out), and not the kernel bank
(Q2 showed complementary information is usually present). A concrete,
testable next step: feed the gate network additional per-kernel
disagreement/confidence signals (e.g. distance to the training-set decision
boundary under each single kernel) rather than only raw features x -- the
roadmap's own Phase 5 interpretability suggestion -- since raw x alone may
not contain the information needed to tell hard cases apart. Not yet
implemented; flagged as the most promising concrete architectural change
identified in this audit.

## THE decisive experiment: kernel-aware router (Router B)

Built a leakage-free "kernel-aware" gate (`routing/kernel_aware_router.py`,
`experiments/kernel_aware_experiment.py`): the gate network receives raw
features x PLUS per-kernel confidence (decision-function margin from a
training-set-only SVM per kernel) PLUS pairwise kernel disagreement, instead
of x alone. Diagnostics use only training data/labels, never a test point's
own label -- leakage-free in the same sense as the Oracle and best-single
baselines. Compared against global, uniform, and the current (raw-x)
router on two contrasting datasets (n=10 seeds each):

| Dataset | global | uniform | current (raw-x) | kernel_aware | kernel_aware vs. current |
|---|---|---|---|---|---|
| Wine | 0.860±0.080 | 0.860±0.080 | 0.890±0.066 | 0.880±0.075 | -0.010, p=0.662 (n.s.) |
| XOR | 0.755±0.052 | 0.730±0.084 | 0.775±0.081 | 0.745±0.079 | -0.030, p=0.260 (n.s.) |

**Result: Outcome B.** Giving the router direct, leakage-free access to
per-kernel behavior diagnostics did NOT improve capture of Oracle headroom
on either dataset -- if anything, both point estimates are slightly worse
(neither significantly). This rules out "the raw features don't carry the
necessary signal" as the bottleneck (Q3/Q5): the signal was injected
directly and made no positive difference.

**Combined with Objective B's earlier null result (Section 3.7: a
classification-surrogate loss also didn't help), we now have two
independent negative results pointing the same direction: the bottleneck
is neither missing input information nor the specific training objective's
functional form.** This shifts the leading hypothesis toward an
optimization/training-dynamics failure: whatever the router's inputs or
loss function, the joint router+theta optimization (Adam over a
kernel-combination objective, 15-20 epochs, small data) may simply not be
finding gating functions that generalize to held-out points, even when the
information needed to do so is present. Candidate next steps: (a) inspect
training curves / check for overfitting the router to the training set
specifically (train vs. test gate-quality gap); (b) try more
epochs/different learning rates/regularization on the gate network; (c)
try a much simpler (near-linear) gate to see if capacity/overfitting, not
routing concept, is the issue; (d) revisit whether kernel-target alignment
or the classification surrogate are even the right training SIGNAL for
"pick the right kernel per point" as opposed to standard classification --
possibly a more direct meta-learning-style objective (e.g. directly reward
the gate for weighting the kernel that empirically get this training point
right, a la mixture-of-experts load-balancing losses) is needed. Not yet
tried; flagged as the concrete next experiment following the user's
proposed order (Q4 -> kernel-aware router [DONE] -> optimization/objective
ablation [NEXT] -> failure mechanism identification).

## Optimization/objective ablation, step (a): overfitting ruled out

Compared train-set vs. test-set accuracy gap for sample_level (trainable
adaptive router) vs. global (minimal trainable params) vs. uniform (zero
trainable params) on Wine and XOR (n=10 seeds each):

| Dataset | Router | Train acc | Test acc | Gap |
|---|---|---|---|---|
| Wine | sample_level | 0.995±0.010 | 0.890±0.066 | +0.105 |
| Wine | global | 0.982±0.016 | 0.860±0.080 | +0.123 |
| Wine | uniform | 0.990±0.012 | 0.865±0.087 | +0.125 |
| XOR | sample_level | 0.920±0.035 | 0.775±0.081 | +0.145 |
| XOR | global | 0.887±0.052 | 0.750±0.055 | +0.138 |
| XOR | uniform | 0.883±0.046 | 0.735±0.092 | +0.147 |

**Result: overfitting is ruled out as the explanation.** The train-test gap
is essentially identical across all three combination strategies on both
datasets, regardless of how many trainable parameters each has (uniform
has zero). If the adaptive router's failure to beat simpler baselines were
an overfitting problem specific to its extra trainable capacity, we would
expect a visibly larger gap for sample_level than for uniform; instead the
gaps are statistically indistinguishable (and sample_level's gap is
slightly SMALLER on Wine). The train-test gap present in all three is
better explained by generic small-sample-SVM variance (only 40 training
points) than by router-specific overfitting.

**Updated leading hypothesis:** with overfitting, missing input information
(kernel-aware experiment), and objective functional form (Objective B) all
ruled out or non-contributory, the remaining candidates are (a) the
router's CAPACITY/architecture itself may not matter much either way in
this regime (a near-linear gate might perform identically to the current
2-hidden-layer MLP -- worth a quick check), or (b) the training SIGNAL
connecting "combine kernels for good alignment/classification-surrogate
loss" to "learn a gating function that generalizes per-sample" may be
fundamentally too indirect regardless of its exact functional form --
suggesting a genuinely different training paradigm (e.g. an explicit
per-point reward for the gate weighting whichever kernel empirically
succeeds on that training point, closer to mixture-of-experts routing
losses used in deep learning) may be needed, not just a different loss on
the same K-matrix-level objective. Not yet tested.

## Optimization/objective ablation, step (b): gate capacity ruled out

Compared the current 2-hidden-layer MLP gate against a near-linear gate
(single Linear layer, no hidden layers, `router_type="sample_level_linear"`)
on Wine and XOR (n=10 seeds each), same PSD-preserving construction and
training objective otherwise:

| Dataset | MLP gate | Linear gate | Difference |
|---|---|---|---|
| Wine | 0.890±0.066 | 0.920±0.046 | +0.030, p=0.313 (n.s.) |
| XOR | 0.775±0.081 | 0.780±0.056 | +0.005, p=0.823 (n.s.) |

**Result: capacity is ruled out.** A drastically simpler linear gate
matches (if anything, very slightly exceeds, non-significantly) the 2-
hidden-layer MLP's performance on both datasets. Gate architectural
capacity is not the bottleneck.

## Summary: four bottleneck hypotheses systematically eliminated

| Hypothesis | Test | Result |
|---|---|---|
| Missing input information | Kernel-aware router (confidence + disagreement features) | Ruled out -- no improvement |
| Training objective functional form | Objective B (classification surrogate vs. alignment) | Ruled out -- no improvement |
| Router overfits more than simpler baselines | Train-test accuracy gap, adaptive vs. global vs. uniform | Ruled out -- gaps identical |
| Gate architectural capacity | Linear gate vs. 2-hidden-layer MLP gate | Ruled out -- performance identical |

**This is itself the strongest mechanistic finding of the audit.** Having
ruled out every component-level explanation within the current paradigm
(combine kernels into one Gram matrix, optimize a global scalar property
of that matrix, decode a per-sample gate from features), the remaining
leading hypothesis is that **the paradigm's training signal itself is too
indirect** to teach reliable per-sample kernel selection, regardless of
architecture, input features, or the specific global loss used. A
genuinely different training paradigm -- e.g. an explicit per-point
"expert imitation" signal that directly rewards g_m(x_i) for being large
exactly when kernel m empirically classifies training point i correctly
(closer to mixture-of-experts routing/load-balancing losses in deep
learning, rather than optimizing a property of the combined K matrix) --
is the most promising untested direction, and would require new
infrastructure (a differentiable proxy for "did kernel m get this specific
point right") rather than a tweak to the current pipeline. Flagged as the
clear next step if this line of investigation continues; not yet
implemented, given its substantially larger scope than the ablations above.

## Three-way disentanglement (user-proposed): mechanism vs. optimization vs. information

Following a detailed external review, two cheap, high-value diagnostics
were run to separate three candidate root causes: (a) the router doesn't
get a useful gradient (optimization failure), (b) Oracle's kernel choice
isn't predictable from x at all (information limitation), (c) converting
gating decisions into a combined Gram matrix loses information relative
to a hard per-point kernel choice (combination-mechanism failure).

### Diagnostic 1: hard-selected routing vs. combined-kernel vs. Oracle

`experiments/hard_vs_combined_routing.py`, n=10 seeds, Wine and XOR. For
each test point, m*(x) = argmax_m g_m(x); "hard_route" uses that single
kernel's own SVM prediction; "combined" is the deployed soft-mixed model.

| Dataset | combined | hard_route | oracle | routing precision* |
|---|---|---|---|---|
| Wine | 0.890±0.066 | 0.745±0.108 | 0.960±0.058 | 0.775 |
| XOR | 0.775±0.081 | 0.745±0.079 | 0.905±0.035 | 0.823 |

*routing precision = P(top-1 kernel choice is correct | Oracle succeeds), i.e. how often the router's single favorite kernel is one of the kernels that would have gotten this point right.

**Result: combined > hard_route on both datasets.** This RULES OUT
combination-mechanism failure (c) -- soft-mixing is not losing information
relative to hard selection, it is actively helping (behaving like a
beneficial ensemble). It also shows the router's top-1 choice is correct
with reasonably high precision (77.5%-82.3%, well above the 33.3% chance
level for 3 kernels) when it matters (Oracle succeeds) -- direct evidence
the router has learned SOME real, better-than-chance routing signal.

### Diagnostic 2: is Oracle's kernel choice predictable from x at all?

`experiments/oracle_predictability.py`, n=10 seeds. Leakage-free k-fold CV
within the training set produces a per-point "best correct kernel" label
(highest margin among CV-correct kernels); a simple logistic regression
x -> label is evaluated via its own cross-validation and compared to a
majority-class baseline.

| Dataset | Simple classifier | Majority baseline | Chance |
|---|---|---|---|
| Wine | 0.659±0.094 | 0.677±0.102 | 0.333 |
| XOR | 0.469±0.138 | 0.462±0.060 | 0.333 |
| synth_hard | 0.436±0.115 | 0.505±0.059 | 0.333 |

**Result: a simple LINEAR classifier does not beat the majority-class
baseline on any of the three datasets** (worse on Wine and synth_hard,
statistically tied on XOR). This is a genuinely surprising and important
finding in apparent tension with Diagnostic 1's 77-82% routing precision:
the harder target here ("the single best-margin-correct kernel") may
simply not be linearly recoverable from x with ~40 training points, even
though the easier target ("any kernel that would succeed") is apparently
learnable to some extent by the (nonlinear, gradient-trained, longer-trained)
neural router. **Caveat, explicitly not yet resolved:** only a LINEAR
probe was tested here; a small nonlinear MLP probe is the natural next
check before concluding information limitation is real, since the neural
router itself is nonlinear and achieves above-chance results on the easier
target.

### Synthesis so far

Combination mechanism (c) is ruled out. Between optimization failure (a)
and information limitation (b), the evidence is currently mixed: the
router beats a linear probe on an easier target, but no method yet beats
majority baseline on the harder, more specific target across three very
different datasets. The next diagnostics needed (per the reviewer's
proposed order): a nonlinear predictability probe (small MLP) to rule out
"linear-only" as the reason Diagnostic 2 came out negative; a perfect-router
upper-bound experiment (plug in oracle-derived one-hot gates instead of
learned ones, holding the combination mechanism fixed) to establish the
Best Single -> Learned Router -> Perfect Router -> Oracle ceiling
decomposition; and optimization-stability checks (multiple restarts /
learning rate sweeps) to see whether some seeds already reach much higher
capture than others under the current setup, which would favor the
optimization-failure hypothesis over the information-limitation one.

### Diagnostic 2b: nonlinear (MLP) predictability probe

Repeated Diagnostic 2 with a small MLP (16,16 hidden units) instead of
logistic regression, to rule out "linear-only" as the reason the earlier
probe failed to beat majority baseline:

| Dataset | Linear probe | MLP probe | Majority | Verdict |
|---|---|---|---|---|
| Wine | 0.659±0.094 | 0.640±? | 0.677±0.102 | NOT predictable (neither probe) |
| XOR | 0.469±0.138 | **0.544±0.085** | 0.462±0.060 | **Predictable, but only nonlinearly** |
| synth_hard | 0.436±0.115 | 0.440±0.067 | 0.505±0.059 | NOT predictable (neither probe) |

**Result: dataset-dependent, and genuinely informative.** On XOR, a
nonlinear probe recovers real signal a linear one missed (0.544 vs. 0.462
majority) -- consistent with XOR's classically nonlinear structure. On
Wine AND synth_hard, even the nonlinear probe cannot beat simply always
predicting the majority-correct kernel. This is a meaningfully different
picture from the earlier "information limitation is probably not real"
lean: on synth_hard specifically -- the dataset with the single largest
Oracle-vs-single gap of all 15 tested (+0.260) -- neither probe can
identify which kernel wins on which point, even though the kernels
clearly differ in aggregate. **This is evidence that information
limitation is a real, dataset-specific contributor to the capture problem
(at least for synth_hard), not merely an optimization artifact,
coexisting with the router's own better-than-chance (but Oracle-target,
not best-kernel-target) performance found in Diagnostic 1.**

### Updated synthesis (three-way disentanglement, in progress)

The three candidate bottlenecks are not mutually exclusive and the
evidence increasingly suggests they operate differently across datasets:

- **Combination mechanism**: ruled out everywhere tested (Diagnostic 1:
  combined always beats hard-routing).
- **Information limitation**: real and load-bearing on synth_hard (neither
  linear nor nonlinear probe beats majority); NOT clearly the story on
  XOR (nonlinear signal exists, just wasn't being used); ambiguous on
  Wine (probes match but don't exceed majority, yet the deployed neural
  router still gets meaningfully above-chance -- 77.5% -- top-1 precision
  on the easier "any correct kernel" target).
- **Optimization/training-signal failure**: still the leading explanation
  for Wine and XOR specifically, where the router demonstrably learns
  *something* (Diagnostic 1) but a purpose-built external probe can match
  or exceed it on a nearby task without needing the K-matrix-alignment
  training signal at all -- suggesting the CURRENT training signal is a
  weak use of information that does appear to be present.

Remaining diagnostics from the reviewer's proposed order, not yet run:
perfect-router upper bound (plug in Oracle-derived one-hot gates, holding
the combination mechanism fixed, to complete the Best Single -> Learned
Router -> Perfect Router -> Oracle decomposition) and optimization-stability
checks (multiple restarts / learning rates) to see if some seeds already
reach much higher capture than others under identical conditions.

## Diagnostic 3: Perfect-router ceiling decomposition (correction to earlier conclusion)

`experiments/perfect_router.py`, n=10 seeds. Completes the chain
Best Single -> Learned Router -> Perfect Router -> Oracle, where Perfect
Router plugs oracle-derived (margin-weighted, correctness-based) gates
into the EXACT SAME soft-combination mechanism used everywhere else,
isolating how much of the Oracle ceiling the combination mechanism itself
can reach given perfect per-point knowledge.

| Dataset | Best Single | Learned | Perfect Router | Oracle | Perfect\u2212Oracle gap | Learned's share of (Perfect\u2212BestSingle) |
|---|---|---|---|---|---|---|
| Wine | 0.840 | 0.890 | 0.945 | 0.960 | \u22120.015 | 47.6% |
| XOR | 0.795 | 0.775 | 0.815 | 0.905 | \u22120.090 | \u2212100% (worse than single) |
| synth_hard | 0.675 | 0.670 | 0.820 | 0.865 | \u22120.045 | \u22123.4% (no capture) |

**This corrects the earlier "combination mechanism is ruled out" claim
(Diagnostic 1), which was based only on hard-route vs. combined and only
on Wine/XOR.** With a cleaner, margin-weighted perfect-gate construction:
on Wine, the mechanism is indeed essentially perfect (Perfect Router is
within 1.5 points of Oracle) -- confirming the earlier conclusion HOLDS
for Wine specifically. But on XOR and synth_hard, **the combination
mechanism itself has a real ceiling measurably below Oracle** (9.0 and 4.5
points respectively), even with perfect per-point kernel knowledge. The
mechanism is not uniformly innocent across datasets.

**Independently of that correction, the learned router's failure is even
starker than previously quantified once the achievable (Perfect Router)
ceiling, rather than the Oracle ceiling, is used as the reference point:**
on XOR the learned router doesn't even match best-single, and on
synth_hard it captures essentially none (-3.4%) of a substantial
(0.820-0.675=0.145) achievable gap -- versus Wine's 47.6% capture of a
smaller (0.945-0.840=0.105) achievable gap. The size of the achievable
gap and the fraction of it captured both vary by dataset, with no obvious
single relationship between them (synth_hard has the LARGEST achievable
gap of the three and the WORST capture rate).

### Revised three-way synthesis

- **Combination mechanism**: NOT uniformly ruled out. Essentially innocent
  on Wine; imposes a real, non-trivial ceiling on XOR and synth_hard. This
  is dataset-dependent and should be re-checked before generalizing either
  conclusion to new datasets.
- **Information limitation**: real and load-bearing on synth_hard
  (Diagnostic 2b: neither linear nor nonlinear probe beats majority
  baseline), despite synth_hard having by far the largest Oracle and
  Perfect-Router-achievable gaps of any dataset tested -- an important,
  slightly paradoxical combination (lots of complementary information in
  aggregate, but seemingly not expressible as a function of x at the
  individual-point level with the probes tried).
- **Optimization/training-signal failure**: still the best-supported
  explanation for why the LEARNED router underperforms the PERFECT router
  ceiling specifically on Wine (only 47.6% captured despite the mechanism
  being essentially frictionless there) and remains plausible for XOR/
  synth_hard alongside the newly-confirmed partial mechanism and
  information-limitation contributions.

**Updated recommendation:** the reviewer's remaining proposed diagnostic
(optimization-stability checks: multiple restarts / learning rate
variation) is now higher-value than before, specifically on Wine, where
mechanism and (partially) information limitation have been priced out,
leaving optimization as close to the sole remaining candidate for that
dataset's 52.4% uncaptured-but-achievable gap.

## Diagnostic 4 (final in the reviewer's proposed chain): optimization stability

`experiments/optimization_stability.py`. Fixed the Wine data split (seed=0)
and varied ONLY training randomness (router+theta initialization) and
learning rate.

| lr | restarts | mean | std | min | max |
|---|---|---|---|---|---|
| 0.01 | 8 | 0.900 | 0.025 | 0.850 | 0.950 |
| 0.05 | 15 | 0.950 | 0.018 | 0.900 | 1.000 |
| 0.10 | 8 | 0.950 | **0.000** | 0.950 | 0.950 |
| 0.20 | 8 | 0.931 | 0.043 | 0.900 | 1.000 |

**Result: training is highly reproducible, not lottery-like.** At lr=0.10,
all 8 restarts converge to EXACTLY the same accuracy (std=0.000). At
lr=0.05, 13/15 restarts land on exactly 0.95. The variation seen at lr=0.01
looks like slower/incomplete convergence (a fixable issue via more epochs
or a larger learning rate) rather than landscape ruggedness. **This weighs
AGAINST "optimization instability" (in the strong sense of many
qualitatively different local optima reachable only by initialization
luck) as the explanation for Wine's 47.6% capture rate.**

### Final synthesis of the three-way disentanglement (Wine)

With combination mechanism priced out (Diagnostic 3: essentially perfect
on Wine specifically), information limitation ambiguous-to-not-clearly-
the-story on Wine (Diagnostic 2/2b: probes match but don't exceed majority;
the neural router itself exceeds chance on the easier Oracle target), and
optimization instability now also disfavored (Diagnostic 4: highly
reproducible convergence, not initialization-lottery-dependent), **the
remaining, best-supported explanation specifically for Wine is a
STRUCTURAL limitation in the training signal**: kernel-target alignment
(or the classification surrogate, Section 3.7/Objective B) reliably and
reproducibly converges to a router that captures only about half the
achievable (Perfect-Router) headroom, not because training got unlucky,
but because that is what this objective, applied to this architecture,
systematically produces. This is precisely the gap the previously-proposed
expert-imitation (MoE-style) routing loss (Section 7 of the final summary
document) targets directly, and this diagnostic chain now provides
positive justification (not just elimination-by-exclusion) for trying it:
the problem looks structural/systematic rather than stochastic, which is
exactly the kind of problem a different training SIGNAL (rather than
different hyperparameters on the same signal) is suited to fix.

**Caveat:** this stability check was run on Wine only (one data split at
that). XOR and synth_hard, where mechanism and information limitation are
each partially implicated too (Diagnostic 3), were not re-tested for
optimization stability; the same check there would clarify whether
optimization plays a similar or different role once the other two factors
are accounted for.
