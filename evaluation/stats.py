"""Paired descriptive inference; repeated-split population inference needs caution.

Wilcoxon uses rounded differences (12 decimals) and the explicit asymptotic
method with Wilcox zero handling, to avoid platform-dependent exact/tie rules.
Bootstrap resamples seed pairs, NOT individual test predictions. This does not
remove dependence from reusing a finite dataset across splits.
"""
import numpy as np
from scipy import stats


def paired_comparison(a, b, n_boot=10000, seed=0, alpha=0.05, test_train_ratio=None):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.ndim != 1 or b.shape != a.shape or len(a) < 2 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('Need finite, aligned one-dimensional arrays with at least two pairs')
    diffs = np.round(a - b, 12)
    n = len(diffs)
    mean, sd = float(diffs.mean()), float(diffs.std(ddof=1))
    p_t = float(stats.ttest_1samp(diffs, 0).pvalue) if sd > 1e-15 else (1.0 if abs(mean) < 1e-15 else 0.0)
    nz = diffs[diffs != 0]
    p_w = float(stats.wilcoxon(diffs, zero_method='wilcox', method='asymptotic').pvalue) if len(nz) else 1.0
    ranks = stats.rankdata(abs(nz))
    rbc = float(np.sum(ranks * np.sign(nz)) / ranks.sum()) if len(nz) else 0.0
    rng = np.random.default_rng(seed)
    boot = diffs[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha/2, 1-alpha/2])
    d = mean/sd if sd > 1e-15 else (0.0 if abs(mean) < 1e-15 else None)
    r = {'n':n, 'mean_a':float(a.mean()), 'sd_a':float(a.std(ddof=1)),
         'mean_b':float(b.mean()), 'sd_b':float(b.std(ddof=1)),
         'mean_diff':mean, 'sd_diff':sd, 'p_ttest':p_t, 'p_wilcoxon':p_w,
         'ci95_lo':float(lo), 'ci95_hi':float(hi), 'ci95_excludes_zero':bool(lo > 0 or hi < 0),
         'cohens_d':d, 'cohens_dz':d, 'hedges_gz':None if d is None else d*(1-3/(4*n-5)),
         'rank_biserial':rbc, 'significant_ttest':bool(p_t<=alpha), 'significant_wilcoxon':bool(p_w<=alpha),
         'n_zero_differences':int(n-len(nz)), 'bootstrap_draws':n_boot, 'bootstrap_seed':seed,
         'wilcoxon_method':'asymptotic; zero_method=wilcox; differences rounded to 12 decimals',
         'inference_warning':'Nominal paired tests and seed bootstrap; reused observations across splits are dependent.'}
    if test_train_ratio is not None:
        if test_train_ratio <= 0:
            raise ValueError('Positive test/train ratio required')
        se = np.sqrt((1/n + test_train_ratio) * sd**2)
        pc = float(2*stats.t.sf(abs(mean/se), n-1)) if se > 1e-15 else (1.0 if abs(mean)<1e-15 else 0.0)
        radius = float(stats.t.ppf(1-alpha/2,n-1)*se)
        r.update(p_corrected_resampled_t=pc, corrected_ci_lo=mean-radius, corrected_ci_hi=mean+radius,
                 corrected_test_train_ratio=test_train_ratio,
                 corrected_test_note='Nadeau-Bengio variance-inflation sensitivity; approximate for capped repeated holdouts')
    return r


def holm_correction(p_values, alpha=0.05):
    names = list(p_values) if isinstance(p_values, dict) else list(range(len(p_values)))
    p = np.array([p_values[k] for k in names],float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Holm requires finite p-values in [0,1]')
    order = np.argsort(p, kind='stable'); out={}; previous=0.0
    for rank, idx in enumerate(order):
        adj = min(1.,max(previous,(len(p)-rank)*p[idx]));previous=adj
        out[names[idx]]={'p_value':float(p[idx]), 'p_adjusted':float(adj),
                         'holm_threshold':float(alpha/(len(p)-rank)), 'reject_null':bool(adj<=alpha)}
    return out


def capture_rate(method, single, perfect, n_boot=20000, seed=0):
    method,single,perfect=map(lambda x:np.asarray(x,float),(method,single,perfect))
    if method.ndim!=1 or method.shape!=single.shape or method.shape!=perfect.shape or len(method)<2:
        raise ValueError('Aligned capture arrays required')
    gain, gap = method-single, perfect-single
    denominator=float(gap.mean())
    rng=np.random.default_rng(seed);idx=rng.integers(len(gap),size=(n_boot,len(gap)))
    den=gap[idx].mean(axis=1);valid=den>1e-12
    ratios=gain[idx].mean(axis=1)[valid]/den[valid]
    # Do not silently condition away unstable denominator draws.
    stable=bool(valid.mean()>=.975 and denominator>1e-12)
    ci=np.quantile(ratios,[.025,.975]).tolist() if stable else [None,None]
    return {'estimate':float(gain.mean()/denominator) if denominator>1e-12 else None,
            'mean_gain':float(gain.mean()),'mean_diagnostic_headroom':denominator,
            'ci95_lo':ci[0],'ci95_hi':ci[1],'bootstrap_nonpositive_denominator_fraction':float(1-valid.mean()),
            'interval_reliable':stable,'definition':'ratio of mean differences, not mean of per-seed ratios',
            'warning':'Perfect Router is a hindsight heuristic; ratio is not a bounded efficiency or causal quantity.'}


def format_result(name, r):
    return f"{name}: n={r['n']} diff={r['mean_diff']:+.4f}, p(t)={r['p_ttest']:.5g}, p(W)={r['p_wilcoxon']:.5g}, CI=[{r['ci95_lo']:+.4f},{r['ci95_hi']:+.4f}], dz={r['cohens_d']}"
