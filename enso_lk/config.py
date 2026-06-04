"""Static configuration: Sri Lanka regions, monsoon seasons, ENSO thresholds.

Domain background
-----------------
Sri Lanka's rainfall is governed by four monsoon periods:

* FIM  (First Inter-monsoon, Mar-Apr)   - convective rain, island-wide.
* SWM  (South-West Monsoon, May-Sep)    - wets the SW "wet zone"; drives the
                                          *Yala* cultivation season and feeds the
                                          central-highland hydropower catchments.
* SIM  (Second Inter-monsoon, Oct-Nov)  - heaviest, most island-wide rain;
                                          main flood / landslide window.
* NEM  (North-East Monsoon, Dec-Feb)    - wets the N & E "dry zone"; drives the
                                          *Maha* cultivation season.

Published ENSO research for Sri Lanka (Zubair & Ropelewski, 2006; Zubair 2002)
finds El Nino is generally associated with ENHANCED Oct-Dec rainfall and
SUPPRESSED South-West-monsoon (May-Aug) rainfall. This module only encodes the
*structure* (regions, seasons, thresholds); the actual anomalies are computed
empirically from observed data in ``analysis.py`` so conclusions stay data-driven.
"""

from __future__ import annotations

# --- ENSO / ONI thresholds (NOAA CPC definition) ---------------------------
ONI_ELNINO = 0.5      # ONI >= +0.5 deg C  -> El Nino conditions
ONI_LANINA = -0.5     # ONI <= -0.5 deg C  -> La Nina conditions

# --- Monsoon seasons: calendar month -> season code ------------------------
SEASONS = {
    "FIM": [3, 4],            # First inter-monsoon
    "SWM": [5, 6, 7, 8, 9],   # South-west monsoon (Yala)
    "SIM": [10, 11],          # Second inter-monsoon (flood window)
    "NEM": [12, 1, 2],        # North-east monsoon (Maha)
}

SEASON_LABELS = {
    "FIM": "First Inter-monsoon (Mar-Apr)",
    "SWM": "South-West Monsoon / Yala (May-Sep)",
    "SIM": "Second Inter-monsoon (Oct-Nov)",
    "NEM": "North-East Monsoon / Maha (Dec-Feb)",
}

MONTH_TO_SEASON = {m: s for s, months in SEASONS.items() for m in months}

# --- Representative analysis points across Sri Lanka's climatic zones -------
# Each point carries the climatic zone plus which impact dimensions it is most
# relevant to. Coordinates are the town/area centroids.
REGIONS = {
    "Colombo": dict(
        lat=6.93, lon=79.85, zone="Wet zone (SW lowland)",
        district="Colombo",
        relevance=["flood", "rainfall"],
        note="Dense urban lowland; flash-flood prone in SIM.",
    ),
    "Galle": dict(
        lat=6.05, lon=80.22, zone="Wet zone (S coast)",
        district="Galle",
        relevance=["flood", "rainfall", "agriculture"],
        note="Southern wet zone, SWM-dependent.",
    ),
    "Ratnapura": dict(
        lat=6.68, lon=80.40, zone="Wet zone (SW hills)",
        district="Ratnapura",
        relevance=["flood", "rainfall"],
        note="Highest landslide/flood risk in the country.",
    ),
    "Nuwara Eliya": dict(
        lat=6.97, lon=80.77, zone="Central highlands",
        district="Nuwara Eliya",
        relevance=["agriculture", "hydropower"],
        note="Tea heartland; Kotmale/Victoria hydropower catchment.",
    ),
    "Kandy": dict(
        lat=7.29, lon=80.64, zone="Central highlands (mid)",
        district="Kandy",
        relevance=["agriculture", "hydropower"],
        note="Mahaweli upper basin; tea + mixed crops.",
    ),
    "Anuradhapura": dict(
        lat=8.31, lon=80.41, zone="Dry zone (N-central)",
        district="Anuradhapura",
        relevance=["agriculture", "rainfall", "hydropower"],
        note="Tank-irrigated Maha paddy; drought-sensitive.",
    ),
    "Batticaloa": dict(
        lat=7.71, lon=81.70, zone="Dry zone (East)",
        district="Batticaloa",
        relevance=["agriculture", "flood"],
        note="Eastern Maha paddy; NEM flood exposure.",
    ),
    "Jaffna": dict(
        lat=9.66, lon=80.02, zone="Dry zone (North)",
        district="Jaffna",
        relevance=["agriculture", "rainfall"],
        note="Northern dry zone, almost entirely NEM-fed.",
    ),
    "Hambantota": dict(
        lat=6.12, lon=81.12, zone="Dry zone (SE)",
        district="Hambantota",
        relevance=["rainfall", "agriculture"],
        note="Driest SE corner; chronic drought risk.",
    ),
    "Trincomalee": dict(
        lat=8.57, lon=81.23, zone="Dry zone (NE coast)",
        district="Trincomalee",
        relevance=["agriculture", "flood"],
        note="NE coast; Mahaweli delta + Maha paddy.",
    ),
}

# Historical weather window for building the climatology / ENSO composites.
HIST_START = "1970-01-01"

# How many days the ERA5 archive typically lags real-time.
ARCHIVE_LAG_DAYS = 6


# --------------------------------------------------------------------------- #
# District-level (satellite CHIRPS) configuration
# --------------------------------------------------------------------------- #
# Climatic zone for each of the 25 administrative districts (keyed by the
# geoBoundaries ADM2 ``shapeName``). Zones drive the impact model exactly as the
# point analysis does: wet (SW quarter), highland (central massif), dry (N/E/SE).
DISTRICT_ZONE = {
    # Wet zone (south-west quarter)
    "Colombo District": "wet", "Gampaha District": "wet", "Kalutara District": "wet",
    "Galle District": "wet", "Matara District": "wet", "Ratnapura District": "wet",
    "Kegalle District": "wet",
    # Central highlands
    "Nuwara Eliya District": "highland", "Kandy District": "highland",
    "Badulla District": "highland",
    # Dry / intermediate zone (north, east, south-east, north-central)
    "Jaffna District": "dry", "Kilinochchi District": "dry", "Mannar District": "dry",
    "Mullaitivu District": "dry", "Vavuniya District": "dry",
    "Anuradhapura District": "dry", "Polonnaruwa District": "dry",
    "Trincomalee District": "dry", "Batticaloa District": "dry",
    "Ampara District": "dry", "Puttalam District": "dry",
    "Kurunegala District": "dry", "Matale District": "dry",
    "Monaragala District": "dry", "Hambantota District": "dry",
}

# Districts with notable flood/landslide exposure and hydropower catchments.
FLOOD_DISTRICTS = {
    "Colombo District", "Gampaha District", "Kalutara District", "Galle District",
    "Matara District", "Ratnapura District", "Kegalle District", "Kandy District",
    "Nuwara Eliya District", "Batticaloa District", "Ampara District",
    "Trincomalee District", "Polonnaruwa District",
}
HYDRO_DISTRICTS = {
    "Nuwara Eliya District", "Kandy District", "Badulla District",
    "Kegalle District", "Ratnapura District", "Polonnaruwa District",
}


def district_meta(name: str, zone: str, lat: float, lon: float) -> dict:
    """Build the impact-model metadata dict for one district."""
    zlabel = {"wet": "Wet zone", "highland": "Central highlands", "dry": "Dry zone"}[zone]
    relevance = ["rainfall", "agriculture"]
    if name in FLOOD_DISTRICTS:
        relevance.append("flood")
    if name in HYDRO_DISTRICTS:
        relevance.append("hydropower")
    short = name.replace(" District", "")
    return dict(lat=lat, lon=lon, zone=zlabel, district=short,
                relevance=relevance, note=f"{short} ({zlabel}) — CHIRPS satellite.")


# CHIRPS satellite rainfall (UCSB Climate Hazards Center), served clipped to a
# Sri Lanka bounding box by the IRI Data Library. ~0.05 deg (~5 km), 1981->now.
CHIRPS_BBOX = dict(x0=79.5, x1=82.1, y0=5.7, y1=10.0)
CHIRPS_URL = (
    "http://iridl.ldeo.columbia.edu/SOURCES/.UCSB/.CHIRPS/.v2p0/.monthly/.global/"
    ".precipitation/X/{x0}/{x1}/RANGEEDGES/Y/{y0}/{y1}/RANGEEDGES/data.nc"
).format(**CHIRPS_BBOX)

# geoBoundaries ADM2 (district) polygons for Sri Lanka.
ADM2_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/"
    "LKA/ADM2/geoBoundaries-LKA-ADM2_simplified.geojson"
)
