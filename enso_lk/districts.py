"""District-level spatial analysis from CHIRPS satellite rainfall.

Pipeline
--------
1. Download the 25 Sri Lanka district polygons (geoBoundaries ADM2, GeoJSON).
2. Download CHIRPS v2.0 monthly satellite rainfall (~5 km) clipped to a Sri
   Lanka bounding box from the IRI Data Library as NetCDF (1981-present).
3. For every district, take the **zonal mean** of all CHIRPS grid cells whose
   centre falls inside the district polygon -> a district monthly rainfall series.
4. Feed that into the same anomaly / composite / significance / impact engine
   used for the point analysis, then render it as a choropleth.

Only ``shapely`` (point-in-polygon) and ``xarray``/``netCDF4`` (read the grid)
are needed — no heavy GIS stack.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from shapely.geometry import shape
from shapely.prepared import prep

from datetime import date

from .config import (ADM2_URL, CHIRPS_BBOX, DISTRICT_ZONE, district_meta)

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
GEOJSON_PATH = CACHE_DIR / "lka_adm2.geojson"

# CHIRPS starts in 1981. The IRI server times out on a single 45-year pull, so we
# fetch in time chunks and stitch them — each chunk is small and quick.
CHIRPS_START_YEAR = 1981
_CHUNK_TMPL = (
    "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/"
    ".precipitation/X/{x0}/{x1}/RANGEEDGES/Y/{y0}/{y1}/RANGEEDGES/"
    "T/({m0})/({m1})/RANGEEDGES/data.nc"
)
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# --------------------------------------------------------------------------- #
# Downloads (cached to disk)
# --------------------------------------------------------------------------- #
def load_boundaries(max_age_days: float = 90) -> dict:
    """Return the district GeoJSON, downloading + caching if needed."""
    if not GEOJSON_PATH.exists() or _age_days(GEOJSON_PATH) > max_age_days:
        resp = requests.get(ADM2_URL, timeout=60, allow_redirects=True)
        resp.raise_for_status()
        GEOJSON_PATH.write_bytes(resp.content)
    return json.loads(GEOJSON_PATH.read_text())


def _fetch_chunk(y0: int, y1: int, dest: Path, retries: int = 4) -> None:
    url = _CHUNK_TMPL.format(m0=f"Jan%20{y0}", m1=f"Dec%20{y1}", **CHIRPS_BBOX)
    delay = 3.0
    for attempt in range(retries):
        try:
            with requests.get(url, timeout=180, stream=True) as resp:
                if resp.status_code in (500, 502, 503, 504, 429):
                    raise requests.HTTPError(f"{resp.status_code}")
                resp.raise_for_status()
                tmp = dest.with_suffix(".tmp")
                with open(tmp, "wb") as fh:
                    for c in resp.iter_content(chunk_size=1 << 16):
                        fh.write(c)
                # Validate it is a real NetCDF before committing.
                if tmp.stat().st_size < 200 or tmp.read_bytes()[:3] != b"CDF":
                    raise requests.HTTPError("non-NetCDF response")
                tmp.replace(dest)
                return
        except (requests.RequestException, OSError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(f"CHIRPS chunk {y0}-{y1} failed: {exc}") from exc
            time.sleep(delay)
            delay = min(delay * 2, 40.0)


def download_chirps(max_age_days: float = 25) -> list[Path]:
    """Fetch CHIRPS in decade chunks (cached); return the list of NetCDF paths."""
    end_year = date.today().year
    bounds = list(range(CHIRPS_START_YEAR, end_year + 1, 10)) + [end_year + 1]
    paths: list[Path] = []
    for i in range(len(bounds) - 1):
        y0, y1 = bounds[i], min(bounds[i + 1] - 1, end_year)
        dest = CACHE_DIR / f"chirps_lk_{y0}_{y1}.nc"
        # Past chunks are immutable; only the chunk containing the current year
        # is refreshed once it goes stale.
        is_current = y1 >= end_year
        if not dest.exists() or (is_current and _age_days(dest) > max_age_days):
            _fetch_chunk(y0, y1, dest)
        paths.append(dest)
    return paths


def _age_days(p: Path) -> float:
    return (time.time() - p.stat().st_mtime) / 86400.0


# --------------------------------------------------------------------------- #
# Zonal statistics
# --------------------------------------------------------------------------- #
def _decode_time(t_vals: np.ndarray, units: str) -> pd.DatetimeIndex:
    # CHIRPS via IRI uses "months since 1960-01-01"; T is mid-month (x.5).
    base = pd.Timestamp(units.split("since")[-1].strip().split()[0])
    months = np.floor(t_vals).astype(int)
    return pd.DatetimeIndex([base + pd.DateOffset(months=int(m)) for m in months])


def district_monthly(max_age_days: float = 30) -> tuple[pd.DataFrame, dict, dict]:
    """Return (monthly_frame, geojson, meta) for all districts.

    ``monthly_frame`` columns: region (= shapeName), date, precip, temp(NaN),
    year, month. ``meta`` maps region -> impact-model metadata dict.
    """
    geojson = load_boundaries()
    paths = download_chirps(max_age_days)

    cubes, all_dates = [], []
    xs = ys = None
    for p in paths:
        ds = xr.open_dataset(p, decode_times=False)
        var = "precipitation" if "precipitation" in ds.data_vars else list(ds.data_vars)[0]
        if xs is None:
            xs, ys = ds["X"].values, ds["Y"].values
        d = _decode_time(ds["T"].values,
                         ds["T"].attrs.get("units", "months since 1960-01-01"))
        cubes.append(ds[var].transpose("T", "Y", "X").values)
        all_dates.append(d)
        ds.close()
    cube = np.concatenate(cubes, axis=0)            # (time, y, x)
    dates = pd.DatetimeIndex(np.concatenate([d.values for d in all_dates]))
    # De-duplicate any overlap between chunks, keep chronological order.
    order = np.argsort(dates.values)
    dates, cube = dates[order], cube[order]
    _, uniq = np.unique(dates.values, return_index=True)
    dates, cube = dates[uniq], cube[uniq]

    # Grid of cell-centre coordinates.
    gx, gy = np.meshgrid(xs, ys)            # (Y, X)
    flat_x, flat_y = gx.ravel(), gy.ravel()

    rows = []
    meta: dict[str, dict] = {}
    for feat in geojson["features"]:
        name = feat["properties"]["shapeName"]
        if name not in DISTRICT_ZONE:
            continue
        geom = shape(feat["geometry"])
        inside = _mask_inside(geom, flat_x, flat_y)
        if inside.sum() == 0:                # tiny district: fall back to centroid cell
            c = geom.centroid
            idx = int(np.argmin((flat_x - c.x) ** 2 + (flat_y - c.y) ** 2))
            inside = np.zeros(flat_x.size, bool)
            inside[idx] = True

        mask2d = inside.reshape(gx.shape)    # (Y, X)
        # Zonal mean per month, ignoring CHIRPS missing (<0) values.
        sub = cube[:, mask2d]                # (time, n_cells)
        sub = np.where(sub < 0, np.nan, sub)
        series = np.nanmean(sub, axis=1)

        df = pd.DataFrame({"region": name, "date": dates, "precip": series})
        df["temp"] = np.nan
        df["n_cells"] = int(mask2d.sum())
        rows.append(df)

        c = geom.centroid
        meta[name] = district_meta(name, DISTRICT_ZONE[name], float(c.y), float(c.x))

    out = pd.concat(rows, ignore_index=True).dropna(subset=["precip"])
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    return out, geojson, meta


def _mask_inside(geom, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Vectorised-ish point-in-polygon over cell centres for one geometry."""
    from shapely.geometry import Point
    pgeom = prep(geom)
    minx, miny, maxx, maxy = geom.bounds
    cand = (xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)
    out = np.zeros(xs.size, bool)
    idxs = np.nonzero(cand)[0]
    for i in idxs:
        if pgeom.contains(Point(xs[i], ys[i])):
            out[i] = True
    return out
