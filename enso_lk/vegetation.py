"""MODIS satellite vegetation health (NDVI -> VCI) from the ORNL DAAC REST API.

NASA MODIS MOD13Q1 gives 250 m, 16-day NDVI from 2000-present. The keyless ORNL
"fixed subset" API caps each call at ~10 time-steps, so a district's full record
is fetched in chunks and cached to disk. We aggregate to monthly NDVI and derive
the **Vegetation Condition Index**:

    VCI = 100 * (NDVI - NDVI_min) / (NDVI_max - NDVI_min)

per calendar month (Kogan, 1995). VCI < 35 indicates vegetation stress / drought;
it is the vegetation half of the Vegetation Health Index and a standard satellite
agricultural-drought proxy. Because fetching is heavy, this runs **on demand for
one district at a time**.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://modis.ornl.gov/rst/api/v1"
PRODUCT = "MOD13Q1"
BAND = "250m_16_days_NDVI"
SCALE = 0.0001
CHUNK = 10               # max time-steps the API allows per request
CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


def _dates(lat: float, lon: float) -> list[dict]:
    r = requests.get(f"{BASE}/{PRODUCT}/dates",
                     params=dict(latitude=lat, longitude=lon),
                     headers={"Accept": "application/json"}, timeout=40)
    r.raise_for_status()
    return r.json()["dates"]


def _subset(lat: float, lon: float, d0: str, d1: str, km: int = 3) -> list[dict]:
    r = requests.get(
        f"{BASE}/{PRODUCT}/subset",
        params=dict(latitude=lat, longitude=lon, band=BAND,
                    startDate=d0, endDate=d1, kmAboveBelow=km, kmLeftRight=km),
        headers={"Accept": "application/json"}, timeout=90)
    r.raise_for_status()
    return r.json()["subset"]


def fetch_ndvi(name: str, lat: float, lon: float, max_age_days: float = 60) -> pd.DataFrame:
    """Monthly NDVI for one district (cached). Columns: date, ndvi."""
    cache = CACHE_DIR / f"ndvi_{name.replace(' ', '_')}.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 86400 <= max_age_days:
        raw = json.loads(cache.read_text())
    else:
        modis_dates = [d["modis_date"] for d in _dates(lat, lon)]
        records = []
        for i in range(0, len(modis_dates), CHUNK):
            block = modis_dates[i:i + CHUNK]
            for attempt in range(3):
                try:
                    sub = _subset(lat, lon, block[0], block[-1])
                    break
                except requests.RequestException:
                    if attempt == 2:
                        sub = []
                    time.sleep(2 * (attempt + 1))
            for step in sub:
                vals = [v for v in step["data"] if v is not None and v > -2000]
                if vals:
                    records.append((step["calendar_date"],
                                    float(np.mean(vals)) * SCALE))
            time.sleep(0.25)  # be polite to the API
        raw = records
        cache.write_text(json.dumps(raw))

    if not raw:
        return pd.DataFrame(columns=["date", "ndvi"])
    df = pd.DataFrame(raw, columns=["date", "ndvi"])
    df["date"] = pd.to_datetime(df["date"])
    # 16-day -> monthly mean.
    monthly = (df.set_index("date")["ndvi"].resample("MS").mean()
               .dropna().reset_index())
    return monthly


def add_vci(monthly: pd.DataFrame) -> pd.DataFrame:
    """Add Vegetation Condition Index (per calendar month) to a monthly NDVI frame."""
    df = monthly.copy()
    df["month"] = df["date"].dt.month
    lo = df.groupby("month")["ndvi"].transform("min")
    hi = df.groupby("month")["ndvi"].transform("max")
    df["vci"] = np.where(hi > lo, 100 * (df["ndvi"] - lo) / (hi - lo), np.nan)
    return df


def enso_composite(monthly_vci: pd.DataFrame, oni_monthly: pd.Series) -> dict:
    """Mean VCI and NDVI anomaly by ENSO phase for one district."""
    from .enso import classify
    df = monthly_vci.merge(oni_monthly.rename("oni"), left_on="date",
                           right_index=True, how="left")
    df["enso"] = df["oni"].apply(lambda v: classify(v) if pd.notna(v) else None)
    out = {}
    for phase in ["El Nino", "Neutral", "La Nina"]:
        sub = df[df["enso"] == phase]
        out[phase] = dict(vci=float(sub["vci"].mean()) if len(sub) else np.nan,
                          n=int(len(sub)))
    corr = np.nan
    d = df.dropna(subset=["oni", "vci"])
    if len(d) > 10:
        corr = float(np.corrcoef(d["oni"], d["vci"])[0, 1])
    out["oni_vci_corr"] = corr
    return out
