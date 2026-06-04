"""Statistical tests that make the ENSO->rainfall composite defensible.

Without significance testing a "composite" is just a difference of means that
could easily be noise — especially with the modest number of El Nino events in
a ~55-year record (~10-15). This module provides:

* Welch's t-test (unequal variance) and the non-parametric Mann-Whitney U test
  comparing El-Nino-month vs neutral-month rainfall.
* Cohen's d effect size (standardised, so it is comparable across regions).
* A bootstrap confidence interval on the composite mean anomaly.
* Lagged cross-correlation of ONI vs rainfall (ENSO teleconnections to Sri Lanka
  act with a seasonal lag), with the analytic significance of the peak.

All p-values are two-sided. We treat p < 0.10 as "suggestive" and p < 0.05 as
"significant" given the inherently small event sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

SIG = 0.05
SUGGESTIVE = 0.10


@dataclass
class CompositeTest:
    n_elnino: int
    n_other: int
    mean_diff: float        # El Nino mean minus other-months mean (same units)
    cohens_d: float
    t_p: float              # Welch's t-test p-value
    mw_p: float             # Mann-Whitney U p-value
    ci_low: float           # bootstrap 95% CI on the El-Nino-month mean anomaly
    ci_high: float

    @property
    def significant(self) -> bool:
        return min(self.t_p, self.mw_p) < SIG

    @property
    def suggestive(self) -> bool:
        return min(self.t_p, self.mw_p) < SUGGESTIVE

    @property
    def confidence(self) -> float:
        """0-1 weight from the better p-value (1 = highly significant)."""
        p = min(self.t_p, self.mw_p)
        if np.isnan(p):
            return 0.0
        return float(max(0.0, min(1.0, 1.0 - p / SUGGESTIVE)))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def _bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 7) -> tuple[float, float]:
    if len(x) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def composite_test(elnino_vals: np.ndarray, other_vals: np.ndarray,
                   anomaly_vals: np.ndarray | None = None) -> CompositeTest:
    """Test whether El-Nino-month values differ from the rest.

    ``elnino_vals`` / ``other_vals`` are raw (e.g. precip) for the two groups.
    ``anomaly_vals`` (optional) are the El-Nino-month *anomalies* used for the
    bootstrap CI (so the CI is expressed relative to climatology).
    """
    a = np.asarray(elnino_vals, float)
    a = a[~np.isnan(a)]
    b = np.asarray(other_vals, float)
    b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return CompositeTest(len(a), len(b), float("nan"), float("nan"),
                             float("nan"), float("nan"), float("nan"), float("nan"))

    t_p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    try:
        mw_p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except ValueError:
        mw_p = float("nan")
    d = _cohens_d(a, b)

    boot_src = anomaly_vals if anomaly_vals is not None else a
    boot_src = np.asarray(boot_src, float)
    boot_src = boot_src[~np.isnan(boot_src)]
    ci_low, ci_high = _bootstrap_ci(boot_src)

    return CompositeTest(len(a), len(b), float(a.mean() - b.mean()), d,
                         t_p, mw_p, ci_low, ci_high)


@dataclass
class LagResult:
    lags: list[int]
    corr: list[float]
    pvals: list[float]
    best_lag: int
    best_r: float
    best_p: float


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjusted p-values (q-values).

    Controls the expected proportion of false positives across the family of
    tests — essential here because we run ~100 district x season composites and
    would otherwise expect several "significant" results by chance alone. NaNs
    are passed through (tests that could not be run).
    """
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    mask = ~np.isnan(p)
    pm = p[mask]
    n = pm.size
    if n == 0:
        return out
    order = np.argsort(pm)
    ranked = pm[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]      # enforce monotonicity
    q = np.clip(q, 0.0, 1.0)
    res = np.empty(n)
    res[order] = q
    out[mask] = res
    return out


def lag_correlation(oni: np.ndarray, rain_anom: np.ndarray,
                    max_lag: int = 6) -> LagResult:
    """Cross-correlate ONI (leading) against rainfall anomaly at lags 0..max_lag.

    A positive lag means ONI *leads* rainfall by that many months. Returns the
    Pearson r and analytic p at each lag plus the strongest (by |r|).
    """
    oni = np.asarray(oni, float)
    rain = np.asarray(rain_anom, float)
    lags, corr, pvals = [], [], []
    for k in range(max_lag + 1):
        if k == 0:
            x, y = oni, rain
        else:
            x, y = oni[:-k], rain[k:]
        m = ~np.isnan(x) & ~np.isnan(y)
        if m.sum() < 8:
            lags.append(k); corr.append(float("nan")); pvals.append(float("nan"))
            continue
        r, p = stats.pearsonr(x[m], y[m])
        lags.append(k); corr.append(float(r)); pvals.append(float(p))

    valid = [(c, k, p) for k, c, p in zip(lags, corr, pvals) if not np.isnan(c)]
    if valid:
        best_r, best_lag, best_p = max(valid, key=lambda t: abs(t[0]))
    else:
        best_r, best_lag, best_p = float("nan"), 0, float("nan")
    return LagResult(lags, corr, pvals, int(best_lag), float(best_r), float(best_p))
