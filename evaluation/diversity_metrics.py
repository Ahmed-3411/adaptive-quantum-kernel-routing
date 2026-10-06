"""
evaluation/diversity_metrics.py

Roadmap Phase 4: "Define multiple diversity metrics: alignment/correlation,
Frobenius distance, eigenspectrum distance, disagreement." Relying on a
single diversity notion (kernel-kernel alignment, used in
evaluation/kernel_analysis.py / experiments/kernel_diversity.py) risks
building a hypothesis around one metric's idiosyncrasies. This module adds
three more, each capturing a different notion of "how different are two
kernels":

    kernel_alignment_similarity(Ka, Kb)  -- already in kernel_analysis.py;
        cosine-like similarity of the two Gram matrices as vectors. High
        value = redundant kernels. (Included here as a thin re-export so
        callers get all four metrics from one place.)

    frobenius_distance(Ka, Kb) -- ||Ka - Kb||_F, normalized by the number of
        entries so it's comparable across dataset sizes. Unlike the
        alignment (which is scale-invariant), this is sensitive to
        overall magnitude differences between the two kernels, not just
        their "shape".

    eigenspectrum_distance(Ka, Kb) -- L2 distance between the two kernels'
        eigenvalue spectra, each normalized to sum to 1 (so it's a
        distribution-shape comparison, not a magnitude comparison). Two
        kernels can be very different pointwise (high Frobenius distance)
        but induce similar "effective dimensionality" (low eigenspectrum
        distance), or vice versa -- this metric is not redundant with
        Frobenius distance.

    prediction_disagreement(Ka, Kb, X_train, y_train, X_test) -- trains an
        SVC(kernel="precomputed") on each kernel independently (train-only,
        no leakage) and measures the fraction of TEST points where their
        predictions disagree. This is the most directly relevant notion for
        an Oracle-style analysis: two kernels that agree on every point
        offer no complementary information no matter how different their
        Gram matrices look numerically.
"""

import numpy as np
from sklearn.svm import SVC


def kernel_alignment_similarity(Ka, Kb):
    num = np.sum(Ka * Kb)
    denom = np.linalg.norm(Ka) * np.linalg.norm(Kb) + 1e-12
    return float(num / denom)


def frobenius_distance(Ka, Kb):
    diff = Ka - Kb
    n = Ka.shape[0] * Ka.shape[1]
    return float(np.linalg.norm(diff) / np.sqrt(n))


def eigenspectrum_distance(Ka, Kb):
    eig_a = np.linalg.eigvalsh((Ka + Ka.T) / 2)
    eig_b = np.linalg.eigvalsh((Kb + Kb.T) / 2)
    eig_a = np.clip(eig_a, 0, None)
    eig_b = np.clip(eig_b, 0, None)
    eig_a = eig_a / (eig_a.sum() + 1e-12)
    eig_b = eig_b / (eig_b.sum() + 1e-12)
    # both already sorted ascending by eigvalsh with matching length (same n)
    return float(np.linalg.norm(eig_a - eig_b))


def prediction_disagreement(Ka_train, Ka_test, Kb_train, Kb_test, y_train):
    clf_a = SVC(kernel="precomputed").fit(Ka_train, y_train)
    clf_b = SVC(kernel="precomputed").fit(Kb_train, y_train)
    pred_a = clf_a.predict(Ka_test)
    pred_b = clf_b.predict(Kb_test)
    return float(np.mean(pred_a != pred_b))


def all_pairwise_diversity(kernel_mats_train, kernel_mats_test, y_train, kernel_names):
    """
    kernel_mats_train/test: dict name -> (n,n) / (n_test,n) numpy arrays.
    Returns dict "nameA_vs_nameB" -> {alignment_similarity, frobenius_distance,
    eigenspectrum_distance, prediction_disagreement}.
    """
    out = {}
    for i in range(len(kernel_names)):
        for j in range(i + 1, len(kernel_names)):
            a, b = kernel_names[i], kernel_names[j]
            Ka_tr, Kb_tr = kernel_mats_train[a], kernel_mats_train[b]
            Ka_te, Kb_te = kernel_mats_test[a], kernel_mats_test[b]
            out[f"{a}_vs_{b}"] = {
                "alignment_similarity": kernel_alignment_similarity(Ka_tr, Kb_tr),
                "frobenius_distance": frobenius_distance(Ka_tr, Kb_tr),
                "eigenspectrum_distance": eigenspectrum_distance(Ka_tr, Kb_tr),
                "prediction_disagreement": prediction_disagreement(Ka_tr, Ka_te, Kb_tr, Kb_te, y_train),
            }
    return out
