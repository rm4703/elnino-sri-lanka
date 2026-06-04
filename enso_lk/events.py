"""Canonical ENSO *event* framework: developing-year composites + EP/CP flavour.

The district composites tag each month by its *concurrent* ONI. That is fine for
a snapshot, but the textbook ENSO teleconnection is organised around the **life
cycle of a discrete event**: an El Nino develops in boreal summer–autumn of
year 0, peaks in DJF(0/1), and decays through year 1. This module:

1. Detects discrete El Nino events from the ONI (NOAA's 5-overlapping-season
   rule) and locates each event's developing year (Y0) and decay year (Y1).
2. Composites Sri Lanka's *national* rainfall in event-relative seasons
   (SW monsoon Y0, 2nd inter-monsoon Y0, NE monsoon peak, 1st inter-monsoon Y1).
3. Classifies each event as **Eastern-Pacific (canonical)** or
   **Central-Pacific (Modoki)** from the Niño-3 vs Niño-4 anomaly at the peak,
   and contrasts the rainfall response of the two flavours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from . import analysis, cache, stats as st
from .config import ONI_ELNINO

NINO_URL = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"


def fetch_nino_indices(max_age_hours: float = 24.0) -> pd.DataFrame:
    """Monthly Niño-3 / Niño-4 / Niño-3.4 SST anomalies (date-indexed)."""
    cached = cache.get("nino", {"u": NINO_URL}, max_age_hours)
    if cached is not None:
        text = cached["text"]
    else:
        try:
            r = requests.get(NINO_URL, timeout=30)
            r.raise_for_status()
            text = r.text
            cache.put("nino", {"u": NINO_URL}, {"text": text})
        except requests.RequestException:
            return pd.DataFrame(columns=["date", "nino3", "nino4", "nino34"])
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) < 10 or not p[0].isdigit():
            continue
        try:
            rows.append(dict(date=pd.Timestamp(int(p[0]), int(p[1]), 1),
                             nino3=float(p[5]), nino4=float(p[7]), nino34=float(p[9])))
        except ValueError:
            continue
    return pd.DataFrame(rows)


def detect_events(oni_monthly: pd.Series, min_run: int = 5) -> pd.DataFrame:
    """Discrete El Nino events from the monthly ONI series.

    Returns one row per event: peak_date, peak_oni, y0 (developing year),
    y1 (decay year).
    """
    s = oni_monthly.sort_index()
    above = s >= ONI_ELNINO
    events = []
    run_start = None
    for date, hot in above.items():
        if hot and run_start is None:
            run_start = date
        elif not hot and run_start is not None:
            run = s[(s.index >= run_start) & (s.index < date)]
            if len(run) >= min_run:
                events.append(run)
            run_start = None
    if run_start is not None:
        run = s[s.index >= run_start]
        if len(run) >= min_run:
            events.append(run)

    rows = []
    for run in events:
        peak_date = run.idxmax()
        # Developing (winter) year: peaks in DJF, so Y0 is the year of the
        # December of the peak winter.
        y0 = peak_date.year if peak_date.month >= 7 else peak_date.year - 1
        rows.append(dict(peak_date=peak_date, peak_oni=float(run.max()),
                         y0=int(y0), y1=int(y0 + 1)))
    return pd.DataFrame(rows)


def classify_flavour(events: pd.DataFrame, nino: pd.DataFrame) -> pd.DataFrame:
    """Tag each event Eastern-Pacific (EP) or Central-Pacific (CP / Modoki).

    Uses the DJF-peak Niño-3 vs Niño-4 anomaly: EP if Niño-3 dominates, CP if
    Niño-4 dominates.
    """
    if events.empty or nino.empty:
        ev = events.copy()
        ev["nino3"] = ev["nino4"] = np.nan
        ev["flavour"] = "n/a"
        return ev
    nser = nino.set_index("date")
    out = events.copy()
    n3, n4, flav = [], [], []
    for _, e in events.iterrows():
        win = [pd.Timestamp(e["y0"], 12, 1), pd.Timestamp(e["y1"], 1, 1),
               pd.Timestamp(e["y1"], 2, 1)]
        sub = nser.reindex(win)
        a3, a4 = sub["nino3"].mean(), sub["nino4"].mean()
        n3.append(a3)
        n4.append(a4)
        flav.append("Eastern-Pacific" if (pd.notna(a3) and pd.notna(a4) and a3 >= a4)
                    else "Central-Pacific" if pd.notna(a4) else "n/a")
    out["nino3"], out["nino4"], out["flavour"] = n3, n4, flav
    return out


def _national_seasonal(panel: pd.DataFrame) -> pd.DataFrame:
    """Country-mean seasonal rainfall table (reuses the district pipeline)."""
    panel = analysis.ensure_season_cols(panel)
    nat = (panel.groupby(["date", "year", "month", "season", "season_year"],
                         as_index=False)
           .agg(precip=("precip", "mean"), temp=("temp", "mean"),
                oni=("oni", "mean")))
    nat["region"] = "Sri Lanka"
    return analysis.seasonal_table(nat)


# Event-relative seasons: (label, actual monsoon season, year offset from Y0).
REL_SEASONS = [
    ("SW monsoon / Yala (develop. yr)", "SWM", 0),
    ("2nd inter-monsoon, Oct–Nov (develop. yr)", "SIM", 0),
    ("NE monsoon / Maha (peak winter)", "NEM", 1),
    ("1st inter-monsoon, Mar–Apr (decay yr)", "FIM", 1),
]


def developing_composite(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """National rainfall composite for each event-relative season, with stats."""
    seas = _national_seasonal(panel)
    out = []
    for label, season, off in REL_SEASONS:
        sub = seas[seas["season"] == season]
        ev_years = {int(e["y0"] + off) for _, e in events.iterrows()}
        ev = sub[sub["season_year"].isin(ev_years)]
        base = sub[~sub["season_year"].isin(ev_years)]
        if len(ev) < 3 or len(base) < 3:
            out.append(dict(rel_season=label, season=season, n=len(ev),
                            mean_pct=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan))
            continue
        test = st.composite_test(ev["precip_sum"].to_numpy(),
                                 base["precip_sum"].to_numpy(),
                                 anomaly_vals=ev["precip_pct"].to_numpy())
        out.append(dict(rel_season=label, season=season, n=int(len(ev)),
                        mean_pct=float(ev["precip_pct"].mean()),
                        ci_low=test.ci_low, ci_high=test.ci_high,
                        p=float(min(test.t_p, test.mw_p))))
    return pd.DataFrame(out)


def flavour_composite(panel: pd.DataFrame, events: pd.DataFrame,
                      season: str = "SIM", off: int = 0) -> pd.DataFrame:
    """Mean national rainfall anomaly in a season, split by EP vs CP flavour."""
    seas = _national_seasonal(panel)
    sub = seas[seas["season"] == season]
    rows = []
    for flav in ["Eastern-Pacific", "Central-Pacific"]:
        yrs = {int(e["y0"] + off) for _, e in events.iterrows() if e["flavour"] == flav}
        vals = sub[sub["season_year"].isin(yrs)]["precip_pct"]
        rows.append(dict(flavour=flav, n=int(len(vals)),
                         mean_pct=float(vals.mean()) if len(vals) else np.nan))
    return pd.DataFrame(rows)
