"""Indian Ocean Dipole (IOD) — and a symmetric El Nino / La Nina view.

For Sri Lanka the IOD rivals ENSO as a driver of the October–December rains: a
**positive IOD** (warm west / cool east Indian Ocean) tends to *enhance* the
second inter-monsoon, and the two oceans often act together. This module:

* fetches the **Dipole Mode Index (DMI)** — the HadISST-based IOD index from
  NOAA PSL — and classifies positive / neutral / negative IOD seasons;
* composites Sri Lanka's national Oct–Nov rainfall by IOD phase and, symmetric
  to the El Nino work, by ENSO phase (so **La Nina** gets the same treatment);
* tabulates the **ENSO x IOD interaction** for the Oct–Nov season.

DMI event threshold: +/- 0.4 deg C (Bureau of Meteorology convention).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from . import analysis, cache, enso, stats as st

DMI_URL = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
IOD_THRESH = 0.4


def fetch_dmi(max_age_hours: float = 24.0) -> pd.DataFrame:
    """Monthly Dipole Mode Index as columns ``date`` and ``dmi``."""
    cached = cache.get("dmi", {"u": DMI_URL}, max_age_hours)
    if cached is not None:
        text = cached["text"]
    else:
        try:
            r = requests.get(DMI_URL, timeout=30)
            r.raise_for_status()
            text = r.text
            cache.put("dmi", {"u": DMI_URL}, {"text": text})
        except requests.RequestException:
            return pd.DataFrame(columns=["date", "dmi"])
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) != 13 or not (p[0].isdigit() and len(p[0]) == 4):
            continue
        year = int(p[0])
        for m in range(1, 13):
            v = float(p[m])
            if abs(v) > 10:        # missing-value sentinel
                continue
            rows.append(dict(date=pd.Timestamp(year, m, 1), dmi=v))
    return pd.DataFrame(rows)


def classify_iod(v: float) -> str:
    if pd.isna(v):
        return "n/a"
    if v >= IOD_THRESH:
        return "Positive IOD"
    if v <= -IOD_THRESH:
        return "Negative IOD"
    return "Neutral IOD"


def _national_sim(panel: pd.DataFrame, dmi: pd.DataFrame) -> pd.DataFrame:
    """National Oct–Nov (SIM) seasonal rainfall per year, with ONI & SON-DMI tags."""
    panel = analysis.ensure_season_cols(panel)
    nat = (panel.groupby(["date", "year", "month", "season", "season_year"],
                         as_index=False)
           .agg(precip=("precip", "mean"), temp=("temp", "mean"),
                oni=("oni", "mean")))
    nat["region"] = "Sri Lanka"
    seas = analysis.seasonal_table(nat)
    sim = seas[seas["season"] == "SIM"].copy()

    # IOD index for each year = mean DMI over its Sep–Nov peak.
    if not dmi.empty:
        d = dmi.copy()
        d["year"] = d["date"].dt.year
        son = (d[d["date"].dt.month.isin([9, 10, 11])]
               .groupby("year")["dmi"].mean())
        sim["dmi"] = sim["season_year"].map(son)
    else:
        sim["dmi"] = np.nan

    sim["enso_phase"] = sim["oni"].apply(
        lambda v: enso.classify(v) if pd.notna(v) else "n/a")
    sim["iod_phase"] = sim["dmi"].apply(classify_iod)
    return sim


def _phase_composite(sim: pd.DataFrame, col: str, phases: list[str],
                     baseline: str) -> pd.DataFrame:
    """Mean national SIM rainfall anomaly per phase vs a baseline phase."""
    base = sim[sim[col] == baseline]["precip_sum"].to_numpy()
    rows = []
    for ph in phases:
        grp = sim[sim[col] == ph]
        p = np.nan
        if len(grp) >= 3 and len(base) >= 3 and ph != baseline:
            t = st.composite_test(grp["precip_sum"].to_numpy(), base)
            p = float(min(t.t_p, t.mw_p))
        rows.append(dict(phase=ph, n=int(len(grp)),
                         mean_pct=float(grp["precip_pct"].mean()) if len(grp) else np.nan,
                         p=p))
    return pd.DataFrame(rows)


def partial_regression(sim: pd.DataFrame) -> dict:
    """Separate the *independent* ENSO vs IOD effect on Oct–Nov rainfall.

    Because the ONI and DMI are themselves correlated, a simple composite cannot
    say which ocean drives the rainfall. A multiple regression of the
    (standardised) seasonal rainfall on **both** standardised indices yields each
    driver's partial effect, holding the other fixed — the statistically correct
    way to attribute a signal between two collinear predictors.
    """
    from scipy import stats as sps
    d = sim.dropna(subset=["precip_z", "oni", "dmi"])
    n = len(d)
    if n < 8:
        return dict(n=n, enso_beta=np.nan, enso_p=np.nan,
                    iod_beta=np.nan, iod_p=np.nan, r2=np.nan, oni_dmi_corr=np.nan)
    y = (d["precip_z"] - d["precip_z"].mean()).to_numpy()
    oni = ((d["oni"] - d["oni"].mean()) / d["oni"].std(ddof=1)).to_numpy()
    dmi = ((d["dmi"] - d["dmi"].mean()) / d["dmi"].std(ddof=1)).to_numpy()
    X = np.column_stack([np.ones(n), oni, dmi])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - X.shape[1]
    mse = (resid @ resid) / dof
    cov = mse * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2 * sps.t.sf(np.abs(tvals), dof)
    ss_tot = float((y ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    return dict(n=int(n), enso_beta=float(beta[1]), enso_p=float(pvals[1]),
                iod_beta=float(beta[2]), iod_p=float(pvals[2]), r2=float(r2),
                oni_dmi_corr=float(np.corrcoef(oni, dmi)[0, 1]))


def analyze(panel: pd.DataFrame, dmi: pd.DataFrame) -> dict:
    """Bundle the IOD + ENSO-phase Oct–Nov analysis for the dashboard."""
    sim = _national_sim(panel, dmi)
    enso_comp = _phase_composite(sim, "enso_phase",
                                 ["El Nino", "Neutral", "La Nina"], "Neutral")
    iod_comp = _phase_composite(sim, "iod_phase",
                                ["Positive IOD", "Neutral IOD", "Negative IOD"],
                                "Neutral IOD")

    # ENSO x IOD interaction (mean Oct–Nov anomaly, El Nino/La Nina x +/- IOD).
    joint = []
    for e in ["El Nino", "La Nina"]:
        for i in ["Positive IOD", "Negative IOD"]:
            g = sim[(sim["enso_phase"] == e) & (sim["iod_phase"] == i)]
            joint.append(dict(enso=e, iod=i, n=int(len(g)),
                              mean_pct=float(g["precip_pct"].mean()) if len(g) else np.nan))
    return dict(sim=sim, enso_comp=enso_comp, iod_comp=iod_comp,
                joint=pd.DataFrame(joint), partial=partial_regression(sim))
