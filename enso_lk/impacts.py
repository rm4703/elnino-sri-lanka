"""Translate the empirical, significance-tested ENSO composite into sector risk.

Design principles (so the output is scientifically defensible)
--------------------------------------------------------------
* **Effect size, not raw anomaly.** Magnitudes are driven by Cohen's *d* of the
  El-Nino vs neutral rainfall comparison, which standardises for each region's
  natural variability.
* **Significance gating.** Each seasonal signal is multiplied by its confidence
  (derived from the test p-value). A signal that fails significance testing
  contributes ~nothing, so noise cannot manufacture risk.
* **Worst-case aggregation for risk.** A sector's risk is taken from the season
  that *physically drives* it (the strongest relevant signal), not an average
  across seasons — averaging a strong wet-season signal against a dry one hides
  both. Direction, by contrast, uses a signed weighted mean.
* **Physically-causal season mapping.** Drought from the recharge seasons a zone
  depends on; flood from the second inter-monsoon (+ NE monsoon in the east);
  agriculture from the cultivation seasons (Yala=SWM, Maha=NEM) plus the
  pre-season inter-monsoon; hydropower from highland inflow seasons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import REGIONS

# Hydrological importance (0-1) of each season to a zone's water supply. Used as
# a multiplier on the (confidence-damped) effect size when picking the dominant
# risk signal. 1.0 = the season the zone most depends on.
ZONE_SEASON_IMPORTANCE = {
    "wet":      {"FIM": 0.70, "SWM": 1.00, "SIM": 0.50, "NEM": 0.30},
    "highland": {"FIM": 0.70, "SWM": 1.00, "SIM": 0.60, "NEM": 0.30},
    "dry":      {"FIM": 0.70, "SWM": 0.20, "SIM": 0.70, "NEM": 1.00},
}
# Signed weighting for the overall wet/dry *direction* of a season.
ZONE_SEASON_WEIGHTS = {
    "wet":      {"FIM": 0.20, "SWM": 0.45, "SIM": 0.25, "NEM": 0.10},
    "highland": {"FIM": 0.20, "SWM": 0.42, "SIM": 0.28, "NEM": 0.10},
    "dry":      {"FIM": 0.20, "SWM": 0.05, "SIM": 0.30, "NEM": 0.45},
}
SEASONS4 = ["FIM", "SWM", "SIM", "NEM"]
SCALE = 70.0  # maps a confidence-damped |effect size| to the 0-100 risk range


def _zone_class(zone_str: str) -> str:
    z = zone_str.lower()
    if "highland" in z:
        return "highland"
    if "wet" in z:
        return "wet"
    return "dry"


def _mag(x: float, scale: float = SCALE) -> float:
    if pd.isna(x):
        return float("nan")
    return float(min(100.0, abs(x) * scale))


def _wmean(eff: dict[str, float], weights: dict[str, float]) -> float:
    num = den = 0.0
    for s, w in weights.items():
        v = eff.get(s)
        if v is not None and not pd.isna(v):
            num += w * v
            den += w
    return num / den if den else float("nan")


def region_impacts(composite: pd.DataFrame, current_spi: pd.DataFrame | None = None, meta: dict | None = None) -> pd.DataFrame:
    """One row per region with significance-aware sector risk scores.

    ``meta`` maps region name -> dict(lat, lon, zone, district, relevance, note).
    Defaults to the point-based :data:`config.REGIONS`; pass a district metadata
    dict to score the CHIRPS district analysis with the identical engine.

    Columns: region, lat, lon, zone, district, supply_z, drought, flood,
    agriculture, hydropower, overall, direction, confidence, note.
    """
    meta = REGIONS if meta is None else meta
    d_mat = composite.pivot(index="region", columns="season", values="cohens_d")
    z_mat = composite.pivot(index="region", columns="season", values="precip_z")
    t_mat = composite.pivot(index="region", columns="season", values="temp_anom")
    c_mat = composite.pivot(index="region", columns="season", values="confidence")

    rows = []
    for region, meta_r in meta.items():
        if region not in d_mat.index:
            continue
        zc = _zone_class(meta_r["zone"])
        imp_w = ZONE_SEASON_IMPORTANCE[zc]
        dir_w = ZONE_SEASON_WEIGHTS[zc]

        def cell(mat, s):
            return mat.loc[region].get(s, np.nan) if region in mat.index else np.nan

        conf = {s: (0.0 if pd.isna(cell(c_mat, s)) else float(cell(c_mat, s)))
                for s in SEASONS4}
        # Confidence-damped, signed effect size per season (+ wetter / - drier).
        eff = {s: (0.0 if pd.isna(cell(d_mat, s)) else float(cell(d_mat, s))) * conf[s]
               for s in SEASONS4}
        temp = {s: cell(t_mat, s) for s in SEASONS4}

        # --- Drought / water deficit: worst significant drying in a season the
        #     zone depends on (importance-weighted). ---------------------------
        dry_contrib = [(-min(eff[s], 0.0)) * imp_w[s] for s in SEASONS4]
        drought = _mag(max(dry_contrib))

        # --- Flood & landslide: wettest signal in the flood window. ----------
        flood_contrib = [max(eff["SIM"], 0.0) * 1.0]
        if zc == "dry":
            flood_contrib.append(max(eff["NEM"], 0.0) * 0.6)
        flood = _mag(max(flood_contrib))
        if "flood" in meta_r["relevance"]:
            flood = min(100.0, flood * 1.2)  # known exposure (Ratnapura/Colombo/East)

        # --- Agriculture: cultivation-season water stress + tea heat. --------
        crop_season = "NEM" if zc == "dry" else "SWM"   # Maha vs Yala
        crop_stress = max(
            (-min(eff[crop_season], 0.0)) * 1.0,
            (-min(eff["FIM"], 0.0)) * 0.7,               # pre-season dryness
        )
        agri = _mag(crop_stress)
        # Excess rain at planting/harvest (waterlogging, lodging) for paddy zones.
        if eff["SIM"] > 0.5 and "agriculture" in meta_r["relevance"]:
            agri = max(agri, _mag(eff["SIM"] - 0.5, scale=90.0))
        # Tea estates: warm + dry SW monsoon stresses yield/quality.
        if zc == "highland" and "agriculture" in meta_r["relevance"]:
            heat = temp["SWM"]
            heat_term = 0.0 if pd.isna(heat) else min(100.0, max(heat, 0.0) * 60.0)
            agri = float(np.nanmax([agri, _mag(max(-eff["SWM"], 0.0)) * 0.6 + heat_term]))

        # --- Water & hydropower: highland inflow seasons (net). --------------
        hydro = np.nan
        if "hydropower" in meta_r["relevance"]:
            inflow = _wmean({"FIM": eff["FIM"], "SWM": eff["SWM"], "SIM": eff["SIM"]},
                            {"FIM": 0.25, "SWM": 0.45, "SIM": 0.30})
            hydro = _mag(min(inflow, 0.0)) if not pd.isna(inflow) else np.nan

        # --- Dynamic Vulnerability Multiplier based on real-time ground state ---
        if current_spi is not None and "region" in current_spi.columns:
            spi_df = current_spi.set_index("region")
            if region in spi_df.index:
                spi6 = float(spi_df.loc[region, "SPI6"])
                if not pd.isna(spi6) and spi6 < 0:
                    # Ground is already dry. Amplify risk (max +30% per standard deviation of drought)
                    multiplier = 1.0 + (abs(spi6) * 0.3)
                    drought = min(100.0, drought * multiplier)
                    agri = min(100.0, agri * multiplier)
                elif not pd.isna(spi6) and spi6 > 1.0:
                    # Ground is saturated. Amplify flood risk.
                    flood_mult = 1.0 + (abs(spi6) * 0.2)
                    flood = min(100.0, flood * flood_mult)

        # --- Direction & overall --------------------------------------------
        supply_z = _wmean(eff, dir_w)   # signed net seasonal effect
        direction = ("wetter" if supply_z > 0.08
                     else "drier" if supply_z < -0.08 else "near-normal")
        dims = [d for d in [drought, flood, agri, hydro] if not pd.isna(d)]
        overall = float(np.mean(dims)) if dims else np.nan
        reg_conf = _wmean(conf, imp_w)

        rows.append(dict(
            region=region, lat=meta_r["lat"], lon=meta_r["lon"], zone=meta_r["zone"],
            district=meta_r["district"], supply_z=supply_z, drought=drought,
            flood=flood, agriculture=agri, hydropower=hydro, overall=overall,
            direction=direction, confidence=reg_conf, note=meta_r["note"],
        ))
    return pd.DataFrame(rows)


def national_summary(impacts: pd.DataFrame) -> dict:
    """Aggregate the regional picture into national headline numbers."""
    def m(col):
        v = impacts[col].dropna()
        return float(v.mean()) if len(v) else float("nan")

    drier = int((impacts["direction"] == "drier").sum())
    wetter = int((impacts["direction"] == "wetter").sum())
    worst = impacts.sort_values("overall", ascending=False).head(3)["region"].tolist()
    return dict(
        drought=m("drought"), flood=m("flood"), agriculture=m("agriculture"),
        hydropower=m("hydropower"), overall=m("overall"),
        regions_drier=drier, regions_wetter=wetter, top_risk_regions=worst,
    )


def risk_label(score: float) -> str:
    if pd.isna(score):
        return "n/a"
    if score >= 60:
        return "High"
    if score >= 35:
        return "Moderate"
    if score >= 15:
        return "Low"
    return "Minimal"
