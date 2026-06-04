"""Standardized Precipitation Index (SPI) from the CHIRPS district series.

SPI (McKee et al., 1993) is the WMO-recommended meteorological drought index.
For an accumulation window of ``k`` months it answers: *how unusual is the last
k months' rainfall, in standard-deviation units, for this location and time of
year?* SPI <= -1 is moderate drought, <= -1.5 severe, <= -2 extreme; symmetric
on the wet side.

Method (per district, per calendar month):
1. Rolling k-month precipitation total.
2. Fit a two-parameter **gamma** distribution to the positive totals for that
   calendar month, with a point mass for zero months (mixed distribution):
   ``H(x) = q + (1-q) * G(x)``  where q = P(total = 0).
3. Transform the cumulative probability to the standard normal: ``SPI = Φ⁻¹(H)``.

This is the standard fitting approach; it standardises away each district's mean
and seasonal cycle so SPI is comparable across the whole island.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DROUGHT_CLASSES = [
    (-np.inf, -2.0, "Extreme drought"),
    (-2.0, -1.5, "Severe drought"),
    (-1.5, -1.0, "Moderate drought"),
    (-1.0, -0.5, "Mild dry"),
    (-0.5, 0.5, "Near normal"),
    (0.5, 1.0, "Mild wet"),
    (1.0, 1.5, "Moderately wet"),
    (1.5, 2.0, "Very wet"),
    (2.0, np.inf, "Extremely wet"),
]


def classify_spi(v: float) -> str:
    if pd.isna(v):
        return "n/a"
    for lo, hi, label in DROUGHT_CLASSES:
        if lo <= v < hi:
            return label
    return "n/a"


def _fit_spi(accum: np.ndarray) -> np.ndarray:
    """Convert one calendar-month series of k-month totals to SPI values."""
    out = np.full(accum.shape, np.nan)
    valid = ~np.isnan(accum)
    x = accum[valid]
    if x.size < 10:
        return out
    zero = x == 0
    q = zero.mean()
    pos = x[~zero]
    if pos.size < 8 or np.all(pos == pos[0]):
        return out
    try:
        a, loc, scale = stats.gamma.fit(pos, floc=0)
    except Exception:  # noqa: BLE001
        return out
    if not np.isfinite(a) or scale <= 0:
        return out
    cdf = q + (1 - q) * stats.gamma.cdf(x, a, loc=0, scale=scale)
    cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
    out[valid] = stats.norm.ppf(cdf)
    return out


def spi_series(precip: pd.Series, scale: int) -> pd.Series:
    """SPI for a single district's monthly precip series (date-indexed).

    ``precip`` must be a monthly Series indexed by month-start timestamps.
    """
    s = precip.sort_index()
    accum = s.rolling(scale, min_periods=scale).sum()
    df = pd.DataFrame({"accum": accum})
    df["month"] = df.index.month
    spi = pd.Series(np.nan, index=s.index)
    for m, grp in df.groupby("month"):
        spi.loc[grp.index] = _fit_spi(grp["accum"].to_numpy())
    return spi


def district_spi(monthly: pd.DataFrame, scales=(3, 6, 12)) -> pd.DataFrame:
    """Long SPI frame for every district: region, date, scale, spi.

    ``monthly`` columns: region, date, precip (the CHIRPS district frame).
    """
    rows = []
    for region, g in monthly.groupby("region"):
        ser = g.set_index("date")["precip"]
        for k in scales:
            spi = spi_series(ser, k)
            rows.append(pd.DataFrame({"region": region, "date": spi.index,
                                      "scale": k, "spi": spi.values}))
    return pd.concat(rows, ignore_index=True)


def current_spi(spi_long: pd.DataFrame) -> pd.DataFrame:
    """Latest available SPI per district & scale -> wide table.

    Columns: region, SPI3, SPI6, SPI12, date, status (from SPI-6).
    """
    rows = []
    for region, g in spi_long.dropna(subset=["spi"]).groupby("region"):
        rec = {"region": region}
        last_date = g["date"].max()
        for k in (3, 6, 12):
            gk = g[g["scale"] == k].sort_values("date")
            rec[f"SPI{k}"] = float(gk["spi"].iloc[-1]) if len(gk) else np.nan
            if len(gk):
                last_date = max(last_date, gk["date"].iloc[-1])
        rec["date"] = last_date
        rec["status"] = classify_spi(rec.get("SPI6", np.nan))
        rows.append(rec)
    return pd.DataFrame(rows)


def elnino_spi(spi_long: pd.DataFrame, oni_monthly: pd.Series, scale: int = 3) -> pd.Series:
    """Mean SPI during El-Nino months per district (default SPI-3)."""
    from .enso import classify
    g = spi_long[spi_long["scale"] == scale].copy()
    g = g.merge(oni_monthly.rename("oni"), left_on="date", right_index=True, how="left")
    g["enso"] = g["oni"].apply(lambda v: classify(v) if pd.notna(v) else None)
    en = g[(g["enso"] == "El Nino")].dropna(subset=["spi"])
    return en.groupby("region")["spi"].mean()
