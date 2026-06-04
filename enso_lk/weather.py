"""Fetch historical daily weather for each region from the Open-Meteo archive.

Open-Meteo's archive API (ERA5 reanalysis) is free and key-less. We pull daily
precipitation and mean temperature, then aggregate to a monthly frame that can
be joined against the monthly ONI series.
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pandas as pd
import requests

from . import cache
from .config import ARCHIVE_LAG_DAYS, HIST_START, REGIONS

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Open-Meteo's free archive endpoint rate-limits bursts. Be a polite client:
# a short gap between live calls plus exponential backoff on HTTP 429/5xx.
_INTER_REQUEST_SLEEP = 1.2
_MAX_RETRIES = 5


def _archive_end() -> str:
    return (date.today() - timedelta(days=ARCHIVE_LAG_DAYS)).isoformat()


def _get_with_backoff(params: dict) -> dict:
    delay = 2.0
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        resp = requests.get(ARCHIVE_URL, params=params, timeout=90)
        if resp.status_code == 200:
            return resp.json()["daily"]
        if resp.status_code in (429, 500, 502, 503, 504):
            last_exc = requests.HTTPError(f"{resp.status_code} on attempt {attempt + 1}")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Open-Meteo archive unavailable after retries: {last_exc}")


def fetch_region_monthly(name: str, max_age_hours: float = 24.0) -> pd.DataFrame:
    """Monthly precip (sum) and temperature (mean) for one region.

    Returns columns: ``date`` (month start), ``precip`` (mm/month),
    ``temp`` (deg C), ``year``, ``month``.
    """
    reg = REGIONS[name]
    end = _archive_end()
    params = dict(
        latitude=reg["lat"],
        longitude=reg["lon"],
        start_date=HIST_START,
        end_date=end,
        daily="precipitation_sum,temperature_2m_mean",
        timezone="Asia/Colombo",
    )

    cached = cache.get("archive", params, max_age_hours)
    if cached is not None:
        daily = cached
    else:
        daily = _get_with_backoff(params)
        cache.put("archive", params, daily)
        time.sleep(_INTER_REQUEST_SLEEP)  # throttle only live (uncached) calls

    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    monthly = pd.DataFrame(
        {
            "precip": df["precipitation_sum"].resample("MS").sum(min_count=20),
            "temp": df["temperature_2m_mean"].resample("MS").mean(),
        }
    ).dropna()
    monthly = monthly.reset_index().rename(columns={"time": "date"})
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month
    monthly["region"] = name
    return monthly


def fetch_all_regions(max_age_hours: float = 24.0) -> pd.DataFrame:
    """Concatenated monthly frame for every configured region."""
    frames = [fetch_region_monthly(name, max_age_hours) for name in REGIONS]
    return pd.concat(frames, ignore_index=True)
