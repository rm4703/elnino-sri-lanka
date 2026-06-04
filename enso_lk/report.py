"""Build downloadable district reports (CSV + PDF) from the analysis tables."""

from __future__ import annotations

import io
from datetime import date

import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

SECTORS = ["drought", "flood", "agriculture", "hydropower", "overall"]
SEASON_COLS = ["FIM %", "SWM %", "SIM %", "NEM %"]


def assemble_table(dimp: pd.DataFrame, dcomp: pd.DataFrame,
                   spi_cur: pd.DataFrame | None = None) -> pd.DataFrame:
    """One tidy row per district: scores + seasonal CHIRPS anomalies + current SPI."""
    t = dimp.copy()
    t["district"] = t["region"].str.replace(" District", "", regex=False)
    for s in ["FIM", "SWM", "SIM", "NEM"]:
        anom = dcomp[dcomp["season"] == s].set_index("region")["precip_pct"]
        sig = dcomp[dcomp["season"] == s].set_index("region")["significant"]
        t[f"{s} %"] = t["region"].map(anom)
        t[f"{s} sig"] = t["region"].map(sig)
    if spi_cur is not None:
        for col in ["SPI3", "SPI6", "SPI12", "status"]:
            if col in spi_cur.columns:
                t[col] = t["region"].map(spi_cur.set_index("region")[col])
    cols = (["district", "zone", "direction", "confidence", *SECTORS,
             "FIM %", "SWM %", "SIM %", "NEM %"]
            + [c for c in ["SPI3", "SPI6", "SPI12", "status"] if c in t.columns])
    return t[cols].sort_values("overall", ascending=False).reset_index(drop=True)


def to_csv(table: pd.DataFrame) -> bytes:
    return table.to_csv(index=False).encode()


def to_pdf(table: pd.DataFrame, status_headline: str, summary: dict) -> bytes:
    """Render a multi-page PDF summary report and return the bytes."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # --- Page 1: title + national summary ---------------------------------
        fig = plt.figure(figsize=(11.7, 8.3))   # A4 landscape
        fig.text(0.5, 0.92, "El Niño – Sri Lanka Impact Report",
                 ha="center", fontsize=22, weight="bold")
        fig.text(0.5, 0.87, f"Generated {date.today():%d %b %Y} · "
                 "CHIRPS satellite + NOAA ONI", ha="center", fontsize=11,
                 color="#555")
        fig.text(0.06, 0.78, "ENSO status", fontsize=14, weight="bold")
        fig.text(0.06, 0.74, status_headline, fontsize=11, wrap=True)

        fig.text(0.06, 0.64, "National sector risk (mean across 25 districts, 0–100)",
                 fontsize=14, weight="bold")
        bars = ["drought", "flood", "agriculture", "hydropower", "overall"]
        ax = fig.add_axes([0.08, 0.30, 0.5, 0.28])
        vals = [summary.get(b, 0) or 0 for b in bars]
        ax.barh(bars[::-1], vals[::-1], color="#d9772b")
        ax.set_xlim(0, 100)
        ax.set_xlabel("risk score")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

        lines = [
            f"Regions trending drier: {summary.get('regions_drier', '-')} / "
            f"{len(table)}",
            f"Regions trending wetter: {summary.get('regions_wetter', '-')}",
            "Highest overall exposure: " + ", ".join(summary.get("top_risk_regions", [])),
        ]
        fig.text(0.62, 0.55, "\n\n".join(lines), fontsize=11, va="top")
        fig.text(0.06, 0.06, "Analytical aid — not an official forecast. "
                 "Sources: NOAA CPC; UCSB CHIRPS; NASA MODIS. "
                 "Method after Zubair & Ropelewski (2006); McKee et al. (1993).",
                 fontsize=8, color="#777")
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: district table -------------------------------------------
        disp = table.copy()
        for c in disp.columns:
            if disp[c].dtype.kind in "fc":
                disp[c] = disp[c].map(lambda v: "" if pd.isna(v) else f"{v:.0f}"
                                      if abs(v) >= 1 or v == 0 else f"{v:.1f}")
        fig2 = plt.figure(figsize=(11.7, 8.3))
        fig2.text(0.5, 0.95, "District-level scores & satellite rainfall signal",
                  ha="center", fontsize=15, weight="bold")
        ax2 = fig2.add_axes([0.02, 0.02, 0.96, 0.88])
        ax2.axis("off")
        tbl = ax2.table(cellText=disp.values, colLabels=disp.columns,
                        loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6.5)
        tbl.scale(1, 1.25)
        for (r, _), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor("#33485f")
                cell.set_text_props(color="white", weight="bold")
        pdf.savefig(fig2)
        plt.close(fig2)

    return buf.getvalue()
