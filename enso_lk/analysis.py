"""Core ENSO<->rainfall analysis: anomalies, composites and correlations.

Method (fully data-driven)
--------------------------
1. For every region, build a monthly climatology (mean precip / temp per
   calendar month over the full record).
2. Express each month as an anomaly vs that climatology (absolute, percent and
   standardised z-score).
3. Tag each month with the *concurrent* ONI value and ENSO phase.
4. Composite: average the anomalies across all El-Nino months and compare with
   the long-term mean -> the typical El-Nino signal, per region and per season.
5. Correlate monthly ONI against the monthly rainfall anomaly to measure how
   tightly each region/season tracks ENSO.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import stats as st
from .config import MONTH_TO_SEASON, SEASONS
from .enso import classify


def build_panel(weather_monthly: pd.DataFrame, oni_monthly: pd.Series) -> pd.DataFrame:
    """Join weather with ONI and add anomalies + ENSO tags.

    ``weather_monthly`` is the concatenated per-region frame from
    :func:`weather.fetch_all_regions`; ``oni_monthly`` is the month-indexed ONI.
    """
    df = weather_monthly.copy()
    df["season"] = df["month"].map(MONTH_TO_SEASON)

    # Climatology per region x calendar-month.
    clim = (
        df.groupby(["region", "month"])
        .agg(precip_clim=("precip", "mean"),
             precip_std=("precip", "std"),
             temp_clim=("temp", "mean"))
        .reset_index()
    )
    df = df.merge(clim, on=["region", "month"], how="left")

    df["precip_anom"] = df["precip"] - df["precip_clim"]
    df["precip_pct"] = np.where(
        df["precip_clim"] > 1e-6,
        100.0 * df["precip_anom"] / df["precip_clim"],
        np.nan,
    )
    df["precip_z"] = np.where(
        df["precip_std"] > 1e-6,
        df["precip_anom"] / df["precip_std"],
        np.nan,
    )
    df["temp_anom"] = df["temp"] - df["temp_clim"]

    # Concurrent ONI / ENSO phase.
    oni = oni_monthly.rename("oni")
    df = df.merge(oni, left_on="date", right_index=True, how="left")
    df["enso"] = df["oni"].apply(lambda v: classify(v) if pd.notna(v) else np.nan)
    return df


def elnino_composite(panel: pd.DataFrame) -> pd.DataFrame:
    """Mean El-Nino rainfall/temperature signal per region x season, with stats.

    For each (region, season) the El-Nino-month rainfall is compared against the
    neutral-month rainfall using Welch's t-test and Mann-Whitney U, with Cohen's
    d effect size and a bootstrap CI on the percent anomaly. Returns one row per
    (region, season).
    """
    out = []
    for (region, season), g in panel.dropna(subset=["enso"]).groupby(["region", "season"]):
        en = g[g["enso"] == "El Nino"]
        neu = g[g["enso"] == "Neutral"]
        corr = _safe_corr(g["oni"], g["precip_z"])

        base = dict(region=region, season=season, n_elnino=int(len(en)),
                    n_neutral=int(len(neu)), oni_rain_corr=corr)

        if len(en) < 3:
            out.append({**base, "precip_pct": np.nan, "precip_z": np.nan,
                        "temp_anom": np.nan, "t_p": np.nan, "mw_p": np.nan,
                        "cohens_d": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                        "confidence": 0.0, "significant": False})
            continue

        test = st.composite_test(
            en["precip"].to_numpy(),
            neu["precip"].to_numpy() if len(neu) else g["precip"].to_numpy(),
            anomaly_vals=en["precip_pct"].to_numpy(),
        )
        out.append({
            **base,
            "precip_pct": float(en["precip_pct"].mean()),
            "precip_z": float(en["precip_z"].mean()),
            "temp_anom": float(en["temp_anom"].mean()),
            "t_p": test.t_p, "mw_p": test.mw_p, "cohens_d": test.cohens_d,
            "ci_low": test.ci_low, "ci_high": test.ci_high,
            "confidence": test.confidence, "significant": bool(test.significant),
        })
    return pd.DataFrame(out)


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
