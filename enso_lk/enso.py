"""Fetch and interpret the ENSO state from NOAA's Oceanic Nino Index (ONI)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

from . import cache
from .config import ONI_ELNINO, ONI_LANINA

ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

# The 3-letter ONI "season" code is centred on its middle month. This maps each
# rolling season to the calendar month it is centred on, giving a clean monthly
# ONI time series (DJF -> Jan, JFM -> Feb, ... NDJ -> Dec).
SEAS_CENTER_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


def fetch_oni(max_age_hours: float = 24.0) -> pd.DataFrame:
    """Return the full ONI history as a DataFrame indexed by month.

    Columns: ``date`` (month-centred timestamp), ``oni`` (anomaly, deg C),
    ``sst`` (total SST), ``seas``, ``year``.
    """
    cached = cache.get("oni", {"url": ONI_URL}, max_age_hours)
    if cached is not None:
        text = cached["text"]
    else:
        resp = requests.get(ONI_URL, timeout=30)
        resp.raise_for_status()
        text = resp.text
        cache.put("oni", {"url": ONI_URL}, {"text": text})

    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 4 or parts[0] == "SEAS":
            continue
        seas, yr, total, anom = parts
        if seas not in SEAS_CENTER_MONTH:
            continue
        month = SEAS_CENTER_MONTH[seas]
        try:
            rows.append(
                dict(
                    seas=seas,
                    year=int(yr),
                    date=pd.Timestamp(year=int(yr), month=month, day=1),
                    sst=float(total),
                    oni=float(anom),
                )
            )
        except ValueError:
            continue

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def classify(oni: float) -> str:
    if oni >= ONI_ELNINO:
        return "El Nino"
    if oni <= ONI_LANINA:
        return "La Nina"
    return "Neutral"


def monthly_oni(df: pd.DataFrame) -> pd.Series:
    """ONI value indexed by month start timestamp (for joining with weather)."""
    return df.set_index("date")["oni"]


@dataclass
class EnsoStatus:
    latest_date: pd.Timestamp
    latest_oni: float
    phase: str
    trend_per_season: float        # slope of ONI over the last ~year
    projection: list[tuple[str, float]]   # (label, value) recent + linear proj
    headline: str
    developing_elnino: bool


def assess_status(df: pd.DataFrame, lookback: int = 4, project: int = 3) -> EnsoStatus:
    """Summarise the current state and whether an El Nino is *developing*."""
    recent = df.tail(lookback).copy()
    latest = df.iloc[-1]
    # Linear trend (deg C per season) over the lookback window.
    x = np.arange(len(recent))
    slope, intercept = np.polyfit(x, recent["oni"].to_numpy(), 1)

    # Simple linear projection of the next few seasons.
    proj: list[tuple[str, float]] = []
    last_idx = len(recent) - 1
    months = list(SEAS_CENTER_MONTH.keys())
    last_seas = latest["seas"]
    cur = months.index(last_seas)
    for k in range(1, project + 1):
        val = float(slope * (last_idx + k) + intercept)
        cur = (cur + 1) % 12
        proj.append((months[cur], val))

    proj_peak = max([latest["oni"]] + [v for _, v in proj])
    developing = (
        slope > 0.05
        and proj_peak >= ONI_ELNINO
        and latest["oni"] > ONI_LANINA
    )

    phase = classify(latest["oni"])
    if developing and phase != "El Nino":
        headline = (
            f"Developing El Niño — the ONI is {latest['oni']:+.2f} °C and rising "
            f"(~{slope:+.2f} °C per season) and is projected to cross the "
            f"+{ONI_ELNINO:.1f} °C El Niño threshold."
        )
    elif phase == "El Nino":
        headline = f"Active El Niño — ONI {latest['oni']:+.2f} °C."
    elif phase == "La Nina":
        headline = f"La Niña conditions — ONI {latest['oni']:+.2f} °C."
    else:
        headline = f"ENSO-neutral — ONI {latest['oni']:+.2f} °C."

    return EnsoStatus(
        latest_date=latest["date"],
        latest_oni=float(latest["oni"]),
        phase=phase,
        trend_per_season=float(slope),
        projection=proj,
        headline=headline,
        developing_elnino=bool(developing),
    )


def elnino_event_years(df: pd.DataFrame) -> list[int]:
    """Calendar years that contain a sustained El Nino (>=3 El-Nino months)."""
    d = df.copy()
    d["phase"] = d["oni"].apply(classify)
    counts = d[d["phase"] == "El Nino"].groupby("year").size()
    return sorted(counts[counts >= 3].index.tolist())


def fetch_soi(max_age_hours: float = 24.0) -> pd.DataFrame:
    """Fetch the Southern Oscillation Index (SOI) from NOAA CPC."""
    url = "https://www.cpc.ncep.noaa.gov/data/indices/soi"
    cached = cache.get("soi", {"url": url}, max_age_hours)
    if cached is not None:
        text = cached["text"]
    else:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        text = resp.text
        cache.put("soi", {"url": url}, {"text": text})

    rows = []
    # Skip header until YEAR is found
    lines = text.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith("YEAR"):
            start_idx = i + 1
            break
            
    for line in lines[start_idx:]:
        parts = line.split()
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        year = int(parts[0])
        for m, val_str in enumerate(parts[1:], start=1):
            try:
                val = float(val_str)
                if val <= -99.9: continue # Missing value
                rows.append({
                    "date": pd.Timestamp(year=year, month=m, day=1),
                    "soi": val
                })
            except ValueError:
                continue
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def fetch_nino34(max_age_hours: float = 24.0) -> pd.DataFrame:
    """Fetch raw monthly Nino 3.4 SST anomalies from NOAA CPC."""
    url = "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices"
    cached = cache.get("nino34", {"url": url}, max_age_hours)
    if cached is not None:
        text = cached["text"]
    else:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        text = resp.text
        cache.put("nino34", {"url": url}, {"text": text})

    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 10 and parts[0].isdigit():
            # YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM NINO3.4 ANOM
            try:
                yr, mon = int(parts[0]), int(parts[1])
                nino34_anom = float(parts[9])
                rows.append({
                    "date": pd.Timestamp(year=yr, month=mon, day=1),
                    "nino34": nino34_anom
                })
            except ValueError:
                continue
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
