"""Real-time statistical ENSO forecast from the live ONI series.

Everything here is computed at run time from the observed Oceanic Niño Index —
no values are hard-coded and nothing is pre-baked. The method is **damped
persistence**, the standard skill benchmark for ENSO: the forecast at lead *k*
is the current ONI scaled by the historical lag-*k* autocorrelation, which
naturally relaxes toward climatology as the lead grows. Forecast uncertainty at
each lead is estimated by **back-testing** that same rule over the whole record,
and the El Niño / Neutral / La Niña probabilities follow from a normal
distribution about the forecast.

This is a transparent statistical model, **not** a dynamical (coupled
ocean–atmosphere) simulation; for the official multi-model outlook see NOAA CPC /
IRI. Damped persistence is nonetheless competitive with dynamical models at short
leads and is the honest, reproducible choice for an in-app real-time forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import ONI_ELNINO, ONI_LANINA

_SEAS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
         "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]


def enso_forecast(oni_monthly: pd.Series, max_lead: int = 9) -> pd.DataFrame:
    """Damped-persistence ONI forecast with calibrated probabilities.

    Returns one row per lead (1..max_lead): date, seas, oni (central forecast),
    sd, lower/upper (95 %), p_elnino, p_neutral, p_lanina.
    """
    s = oni_monthly.dropna().sort_index()
    x = s.to_numpy(dtype=float)
    last_val = float(x[-1])
    last_date = s.index[-1]

    rows = []
    for k in range(1, max_lead + 1):
        a, b = x[:-k], x[k:]
        if len(a) < 12:
            continue
        r = float(np.corrcoef(a, b)[0, 1])          # lag-k autocorrelation
        resid = b - r * a                            # back-tested errors
        sd = float(np.std(resid, ddof=1))
        fc = r * last_val                            # damped-persistence forecast
        p_el = float(stats.norm.sf((ONI_ELNINO - fc) / sd)) if sd > 0 else float(fc >= ONI_ELNINO)
        p_la = float(stats.norm.cdf((ONI_LANINA - fc) / sd)) if sd > 0 else float(fc <= ONI_LANINA)
        p_neu = max(0.0, 1.0 - p_el - p_la)
        d = last_date + pd.DateOffset(months=k)
        rows.append(dict(
            lead=k, date=d, seas=_SEAS[(d.month - 1) % 12],
            oni=fc, sd=sd, lower=fc - 1.96 * sd, upper=fc + 1.96 * sd,
            p_elnino=p_el, p_neutral=p_neu, p_lanina=p_la,
        ))
    return pd.DataFrame(rows)


def headline(fc: pd.DataFrame) -> str:
    """One-line plain-language summary of the next ~season."""
    if fc.empty:
        return "Insufficient data for a forecast."
    near = fc.iloc[min(2, len(fc) - 1)]              # ~3 months out
    probs = {"El Niño": near["p_elnino"], "Neutral": near["p_neutral"],
             "La Niña": near["p_lanina"]}
    lead_phase = max(probs, key=probs.get)
    return (f"Statistical outlook for {near['seas']} {near['date']:%Y}: "
            f"{lead_phase} most likely ({probs[lead_phase] * 100:.0f}% — "
            f"El Niño {probs['El Niño'] * 100:.0f}% / Neutral "
            f"{probs['Neutral'] * 100:.0f}% / La Niña {probs['La Niña'] * 100:.0f}%).")
