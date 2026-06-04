"""Core ENSO<->rainfall analysis: anomalies, composites and correlations.

Statistically-defensible method
-------------------------------
1. Build a monthly climatology per region and calendar month, and express each
   month as an anomaly (absolute / percent / z-score) for the visual phase plots.
2. For significance testing, aggregate rainfall to **seasonal totals per
   season-year** (one value per monsoon season per year) — so the samples are
   ENSO *events/years*, not autocorrelated months. (~12 El Nino years vs ~18
   neutral years, instead of ~56 non-independent months.)
3. **Detrend** each region x season seasonal series (remove the linear
   1981-present trend) so a long-term trend that happens to coincide with the
   timing of El Nino events cannot masquerade as an ENSO signal.
4. Classify each season-year by its mean ONI and compare El Nino years against
   neutral years with Welch's t-test, Mann-Whitney U, Cohen's d and a bootstrap
   CI.
5. **Benjamini-Hochberg FDR** correction is applied across all region x season
   tests; "significant" means the FDR q-value < 0.05.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as st
from .config import MONTH_TO_SEASON, SEASONS
from .enso import classify

# How complete a season must be (fraction of its months) to count that year.
_SEASON_MINFRAC = 0.6


def build_panel(weather_monthly: pd.DataFrame, oni_monthly: pd.Series) -> pd.DataFrame:
    """Join weather with ONI and add anomalies, ENSO tags and a season-year.

    The *season-year* groups the north-east monsoon's December with the
    following Jan-Feb (so Dec 1997 + Jan/Feb 1998 form one NEM season).
    """
    df = weather_monthly.copy()
    df["season"] = df["month"].map(MONTH_TO_SEASON)
    df["season_year"] = np.where((df["season"] == "NEM") & (df["month"] == 12),
                                 df["year"] + 1, df["year"])

    # Climatology per region x calendar-month (for the monthly phase plots).
    clim = (
        df.groupby(["region", "month"])
        .agg(precip_clim=("precip", "mean"),
             precip_std=("precip", "std"),
             temp_clim=("temp", "mean"))
        .reset_index()
    )
    df = df.merge(clim, on=["region", "month"], how="left")

    df["precip_anom"] = df["precip"] - df["precip_clim"]
    df["precip_pct"] = np.where(df["precip_clim"] > 1e-6,
                                100.0 * df["precip_anom"] / df["precip_clim"], np.nan)
    df["precip_z"] = np.where(df["precip_std"] > 1e-6,
                              df["precip_anom"] / df["precip_std"], np.nan)
    df["temp_anom"] = df["temp"] - df["temp_clim"]

    oni = oni_monthly.rename("oni")
    df = df.merge(oni, left_on="date", right_index=True, how="left")
    df["enso"] = df["oni"].apply(lambda v: classify(v) if pd.notna(v) else np.nan)
    return df


def _detrend(y: np.ndarray) -> np.ndarray:
    """Remove a linear trend from a 1-D series, preserving its mean."""
    n = y.size
    if n < 4 or np.all(np.isnan(y)):
        return y
    x = np.arange(n, dtype=float)
    m = ~np.isnan(y)
    if m.sum() < 4:
        return y
    slope, intercept = np.polyfit(x[m], y[m], 1)
    return y - (slope * x + intercept) + np.nanmean(y)


def seasonal_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Seasonal totals per region x season x season-year, detrended, ENSO-tagged.

    Columns: region, season, season_year, precip_sum (detrended), precip_pct,
    precip_z, temp_mean, oni, phase, n_months.
    """
    need = {s: max(1, int(round(len(m) * _SEASON_MINFRAC)))
            for s, m in SEASONS.items()}
    g = (panel.dropna(subset=["season"])
         .groupby(["region", "season", "season_year"])
         .agg(precip_sum=("precip", "sum"), n_months=("precip", "size"),
              temp_mean=("temp", "mean"), oni=("oni", "mean"))
         .reset_index())
    g = g[g.apply(lambda r: r["n_months"] >= need.get(r["season"], 1), axis=1)]

    rows = []
    for (region, season), sub in g.groupby(["region", "season"]):
        sub = sub.sort_values("season_year").copy()
        sub["precip_sum"] = _detrend(sub["precip_sum"].to_numpy())
        clim = sub["precip_sum"].mean()
        std = sub["precip_sum"].std(ddof=1)
        sub["precip_pct"] = 100.0 * (sub["precip_sum"] - clim) / clim if clim > 1e-6 else np.nan
        sub["precip_z"] = (sub["precip_sum"] - clim) / std if std and std > 1e-6 else np.nan
        sub["temp_mean"] = sub["temp_mean"] - sub["temp_mean"].mean()
        sub["phase"] = sub["oni"].apply(lambda v: classify(v) if pd.notna(v) else np.nan)
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def elnino_composite(panel: pd.DataFrame, min_years: int = 4) -> pd.DataFrame:
    """Event/year-based El Nino composite per region x season, FDR-corrected.

    El Nino *season-years* are compared with neutral season-years (independent
    samples), then Benjamini-Hochberg FDR is applied across all tests. Returns
    one row per (region, season).
    """
    seas = seasonal_table(panel)
    out = []
    for (region, season), sub in seas.groupby(["region", "season"]):
        en = sub[sub["phase"] == "El Nino"]
        neu = sub[sub["phase"] == "Neutral"]
        corr = _safe_corr(sub["oni"], sub["precip_z"])
        base = dict(region=region, season=season, n_elnino=int(len(en)),
                    n_neutral=int(len(neu)), oni_rain_corr=corr)

        if len(en) < min_years or len(neu) < min_years:
            out.append({**base, "precip_pct": np.nan, "precip_z": np.nan,
                        "temp_anom": np.nan, "t_p": np.nan, "mw_p": np.nan,
                        "cohens_d": np.nan, "ci_low": np.nan, "ci_high": np.nan})
            continue

        test = st.composite_test(en["precip_sum"].to_numpy(),
                                 neu["precip_sum"].to_numpy(),
                                 anomaly_vals=en["precip_pct"].to_numpy())
        out.append({
            **base,
            "precip_pct": float(en["precip_pct"].mean()),
            "precip_z": float(en["precip_z"].mean()),
            "temp_anom": float(en["temp_mean"].mean()),
            "t_p": test.t_p, "mw_p": test.mw_p, "cohens_d": test.cohens_d,
            "ci_low": test.ci_low, "ci_high": test.ci_high,
        })

    df = pd.DataFrame(out)
    # Family-wise FDR control across every district x season test.
    df["q_value"] = st.benjamini_hochberg(df["t_p"].to_numpy())
    df["significant"] = df["q_value"] < 0.05
    df["suggestive"] = df["q_value"] < 0.10
    # Confidence (0-1) for the impact model is driven by the FDR q-value, so a
    # signal that does not survive multiple-comparison control contributes ~0.
    df["confidence"] = (1.0 - df["q_value"] / 0.10).clip(lower=0.0, upper=1.0)
    df["confidence"] = df["confidence"].fillna(0.0)
    return df


def lag_analysis(panel: pd.DataFrame, max_lag: int = 6) -> pd.DataFrame:
    """Best ONI->rainfall lead/lag per region (months ONI leads rainfall)."""
    rows = []
    for region, g in panel.groupby("region"):
        g = g.sort_values("date")
        res = st.lag_correlation(g["oni"].to_numpy(), g["precip_z"].to_numpy(), max_lag)
        rows.append(dict(region=region, best_lag=res.best_lag, best_r=res.best_r,
                         best_p=res.best_p,
                         lags=res.lags, corr=res.corr, pvals=res.pvals))
    return pd.DataFrame(rows)


def region_season_matrix(composite: pd.DataFrame, metric: str = "precip_pct") -> pd.DataFrame:
    """Pivot composite into a region x season matrix for heatmaps."""
    order = list(SEASONS.keys())
    m = composite.pivot(index="region", columns="season", values=metric)
    return m.reindex(columns=[s for s in order if s in m.columns])


def phase_composite_by_month(panel: pd.DataFrame, region: str) -> pd.DataFrame:
    """For one region, mean precip per calendar month split by ENSO phase."""
    g = panel[(panel["region"] == region)].dropna(subset=["enso"])
    pivot = (
        g.groupby(["month", "enso"])["precip"].mean().unstack("enso")
    )
    return pivot.reindex(range(1, 13))


def _safe_corr(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    if m.sum() < 5:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])
