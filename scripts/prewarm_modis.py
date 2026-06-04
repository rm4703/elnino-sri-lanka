"""Pre-fetch MODIS NDVI for all 25 districts so the dashboard explorer is instant.

Each district is cached to .cache/ndvi_<district>.json; already-cached districts
are skipped automatically by vegetation.fetch_ndvi.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from shapely.geometry import shape

from enso_lk import districts, vegetation

WORKERS = 4   # districts fetched in parallel (each district is internally serial)


def _one(item):
    name, lat, lon = item
    t = time.time()
    try:
        df = vegetation.fetch_ndvi(name, lat, lon)
        return name, len(df), time.time() - t, None
    except Exception as exc:  # noqa: BLE001
        return name, 0, time.time() - t, str(exc)


def main() -> None:
    g = districts.load_boundaries()
    items = []
    for f in g["features"]:
        name = f["properties"]["shapeName"]
        if name not in districts.DISTRICT_ZONE:
            continue
        c = shape(f["geometry"]).centroid
        items.append((name, float(c.y), float(c.x)))
    items.sort()

    print(f"Pre-warming MODIS NDVI for {len(items)} districts "
          f"({WORKERS} parallel)…", flush=True)
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_one, it): it[0] for it in items}
        for fut in as_completed(futs):
            name, n, dt, err = fut.result()
            done += 1
            tag = f"FAILED: {err}" if err else f"{n:4d} months"
            print(f"[{done:2d}/{len(items)}] {name:<26} {tag} ({dt:4.0f}s)", flush=True)
    print(f"DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
