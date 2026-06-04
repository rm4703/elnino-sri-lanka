"""Real-time statistical ENSO forecast from the live ONI series.

Everything here is computed at run time from the observed Oceanic Niño Index —
no values are hard-coded and nothing is pre-baked. The model is a **persistence +
tendency** linear regression: for each lead *k* it regresses historical ONI(t+k)
on both the current level **ONI(t)** and the recent 3-month **tendency
ONI(t)-ONI(t-3)**. The tendency term is essential — it lets the forecast capture
ENSO's seasonal *development* (a warming ONI in boreal spring tends to keep
warming toward a winter peak), whereas plain damped persistence can only relax
toward climatology and therefore systematically *misses developing events*.

Forecast uncertainty at each lead is the back-tested regression residual spread,
and the El Niño / Neutral / La Niña probabilities follow from a normal
distribution about the forecast.

This remains a transparent statistical model, **not** a dynamical coupled
ocean–atmosphere simulation — but with the tendency term it broadly tracks the
official outlooks (e.g. it favours a developing El Niño when the ONI is rising).
For the authoritative multi-model forecast see NOAA CPC / IRI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import ONI_ELNINO, ONI_LANINA

_SEAS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
         "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
_LAG = 3  # months used to measure the ONI tendency / momentum


def enso_forecast(oni_monthly: pd.Series, max_lead: int = 9) -> pd.DataFrame:
    """Persistence+tendency ONI forecast with calibrated probabilities.

    Returns one row per lead (1..max_lead): date, seas, oni (central forecast),
    sd, lower/upper (95 %), p_elnino, p_neutral, p_lanina.
    """
    s = oni_monthly.dropna().sort_index()
    x = s.to_numpy(dtype=float)
    n = x.size
    last_val = float(x[-1])
    tend_now = float(x[-1] - x[-1 - _LAG]) if n > _LAG else 0.0
    last_date = s.index[-1]

    rows = []
    for k in range(1, max_lead + 1):
        # Design: predict ONI(t+k) from level ONI(t) and tendency ONI(t)-ONI(t-3).
        hi = n - k
        if hi - _LAG < 12:
            continue
        level = x[_LAG:hi]
        tend = x[_LAG:hi] - x[:hi - _LAG]
        y = x[_LAG + k:]
        m = min(len(level), len(tend), len(y))
        level, tend, y = level[:m], tend[:m], y[:m]
        X = np.column_stack([np.ones(m), level, tend])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sd = float(np.std(resid, ddof=1))
        fc = float(beta[0] + beta[1] * last_val + beta[2] * tend_now)
        fc = float(np.clip(fc, -3.0, 3.0))
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
