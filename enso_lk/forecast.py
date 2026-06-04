"""Real-time statistical ENSO forecast from the live ONI series.

Everything here is computed at run time from the observed Oceanic Niño Index —
no values are hard-coded. The model is a **month-conditioned persistence +
tendency** regression: to forecast the target calendar month *M* from the current
month *C*, it regresses historical ONI(year, M) on the same year's ONI(year, C)
and its 3-month tendency. Conditioning on the *target month* is what makes the
forecast respect ENSO's strong seasonal cycle — in particular its **phase-locking
to a December peak**: the learned May→December relationship already encodes that
spring El Niños keep growing into winter, so a rising spring ONI is projected to
*strengthen toward a winter peak* rather than mean-revert. Plain (month-blind)
persistence cannot do this and systematically under-forecasts the winter peak.

Forecast uncertainty at each lead is the back-tested regression residual spread,
and the El Niño / Neutral / La Niña probabilities follow from a normal
distribution about the forecast. This is still a transparent **statistical**
model, not a dynamical coupled-model simulation; for the authoritative
multi-model outlook see NOAA CPC / IRI (shown alongside for comparison).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import ONI_ELNINO, ONI_LANINA

_SEAS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ",
         "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
_LAG = 3  # months used to measure the ONI tendency / momentum


def _shift_ym(year: int, month: int, k: int) -> tuple[int, int]:
    idx = (year * 12 + (month - 1)) + k
    return idx // 12, idx % 12 + 1


def enso_forecast(oni_monthly: pd.Series, max_lead: int = 9) -> pd.DataFrame:
    """Month-conditioned ONI forecast with calibrated probabilities.

    Returns one row per lead (1..max_lead): date, seas, oni (central forecast),
    sd, lower/upper (95 %), p_elnino, p_neutral, p_lanina.
    """
    s = oni_monthly.dropna().sort_index()
    om = {(d.year, d.month): float(v) for d, v in s.items()}
    last_date = s.index[-1]
    C, LY = last_date.month, last_date.year
    last_val = om[(LY, C)]
    ym3 = _shift_ym(LY, C, -_LAG)
    tend_now = last_val - om[ym3] if ym3 in om else 0.0

    rows = []
    for k in range(1, max_lead + 1):
        ty, M = _shift_ym(LY, C, k)
        # Build (level, tendency) -> target pairs across all years, fixing the
        # predictor month C and target month M (so the seasonal cycle is built in).
        L, T, Y = [], [], []
        for yr in range(min(y for y, _ in om), LY + 1):
            p3 = _shift_ym(yr, C, -_LAG)
            tgt = _shift_ym(yr, C, k)
            if (yr, C) in om and p3 in om and tgt in om:
                L.append(om[(yr, C)])
                T.append(om[(yr, C)] - om[p3])
                Y.append(om[tgt])
        if len(Y) < 12:
            continue
        L, T, Y = np.array(L), np.array(T), np.array(Y)
        X = np.column_stack([np.ones(len(L)), L, T])
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        sd = float(np.std(resid, ddof=1))
        fc = float(np.clip(beta[0] + beta[1] * last_val + beta[2] * tend_now, -3.0, 3.0))
        p_el = float(stats.norm.sf((ONI_ELNINO - fc) / sd)) if sd > 0 else float(fc >= ONI_ELNINO)
        p_la = float(stats.norm.cdf((ONI_LANINA - fc) / sd)) if sd > 0 else float(fc <= ONI_LANINA)
        p_neu = max(0.0, 1.0 - p_el - p_la)
        d = pd.Timestamp(ty, M, 1)
        rows.append(dict(
            lead=k, date=d, seas=_SEAS[(M - 1) % 12],
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
