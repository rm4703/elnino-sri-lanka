"""El Nino -> Sri Lanka impact analysis dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from streamlit_option_menu import option_menu

# Consistent modern Plotly look across every figure.
pio.templates["enso"] = go.layout.Template(
    layout=dict(
        font=dict(family="Inter, sans-serif", color="#0f172a", size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        colorway=["#0ea5e9", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6", "#14b8a6"],
        xaxis=dict(gridcolor="#eef2f7", zerolinecolor="#e2e8f0"),
        yaxis=dict(gridcolor="#eef2f7", zerolinecolor="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=30, r=10, l=10, b=10),
    )
)
pio.templates.default = "plotly_white+enso"

from enso_lk import (analysis, districts, enso, events, forecast, impacts, iod,
                     official, report, spi, vegetation, weather)
from enso_lk.config import ONI_ELNINO, ONI_LANINA, SEASON_LABELS, SEASONS

st.set_page_config(page_title="El Niño · Sri Lanka", layout="wide", page_icon="🌊",
                   initial_sidebar_state="expanded")

PHASE_COLORS = {"El Nino": "#ef4444", "La Nina": "#3b82f6", "Neutral": "#94a3b8"}
RISK_SCALE = [(0, "#2ca02c"), (0.35, "#fee08b"), (0.6, "#fc8d59"), (1.0, "#d73027")]

# Phase -> (accent colour, soft background, emoji) for the hero/status styling.
PHASE_THEME = {
    "El Nino":  ("#dc2626", "#fef2f2", "🔴"),
    "La Nina":  ("#2563eb", "#eff6ff", "🔵"),
    "Neutral":  ("#d97706", "#fffbeb", "🟡"),
}
# Pretty (accented) display names; the internal keys stay ASCII for lookups.
PRETTY = {"El Nino": "El Niño", "La Nina": "La Niña", "Neutral": "Neutral"}

# --- Modern, responsive dashboard styling --------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 4.5rem; max-width: 1400px; }
#MainMenu, [data-testid="stDecoration"] { visibility: hidden; }
footer { visibility: hidden; }
/* Hide the Streamlit "Deploy" button + toolbar actions */
[data-testid="stAppDeployButton"], [data-testid="stDeployButton"] { display: none !important; }
[data-testid="stToolbarActions"] { display: none !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px;
    padding: 14px 18px; box-shadow: 0 1px 3px rgba(15,23,42,.06);
    transition: box-shadow .2s ease, transform .2s ease;
}
[data-testid="stMetric"]:hover { box-shadow: 0 6px 18px rgba(15,23,42,.10); transform: translateY(-1px); }
[data-testid="stMetricLabel"] p { font-size: .78rem; color: #64748b; font-weight: 600;
    text-transform: uppercase; letter-spacing: .04em; }
[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; color: #0f172a; }

/* Tabs (scroll horizontally instead of wrapping on small screens) */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 6px; border-bottom: none; overflow-x: auto; flex-wrap: nowrap;
    scrollbar-width: thin;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background:#f1f5f9; border-radius: 10px 10px 0 0; padding: 8px 16px;
    font-weight: 600; font-size:.9rem; color:#475569; white-space: nowrap;
}
[data-testid="stTabs"] [aria-selected="true"] { background:#0ea5e9; color:#fff; }

/* Headings & misc */
h1,h2,h3 { letter-spacing:-.01em; }
h3 { font-weight:700; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow:hidden; border:1px solid #e2e8f0; }
section[data-testid="stSidebar"] { background:#0f172a; }
section[data-testid="stSidebar"] * { color:#e2e8f0; }
section[data-testid="stSidebar"] [data-testid="stMetric"] { background:#1e293b; border-color:#334155; }
section[data-testid="stSidebar"] [data-testid="stMetricValue"] { color:#f8fafc; }
.stDownloadButton button, .stButton button { border-radius:10px; font-weight:600; }

/* Hero */
.enso-hero { background:linear-gradient(120deg,#0c4a6e 0%,#0e7490 55%,#0891b2 100%);
    border-radius:18px; padding:24px 28px; margin-bottom:16px; color:#fff;
    box-shadow:0 8px 28px rgba(8,74,110,.28); }
.enso-hero-row { display:flex; justify-content:space-between; align-items:center;
    flex-wrap:wrap; gap:12px; }
.enso-hero-title { font-size:1.65rem; font-weight:800; letter-spacing:-.02em; line-height:1.15; }
.enso-hero-sub { opacity:.85; font-size:.9rem; margin-top:3px; }
.enso-pill { padding:8px 18px; border-radius:999px; font-weight:700; font-size:.92rem;
    white-space:nowrap; box-shadow:0 2px 8px rgba(0,0,0,.2); }
.enso-headline { margin-top:14px; font-size:1.0rem; font-weight:500;
    background:rgba(255,255,255,.12); padding:10px 16px; border-radius:12px; }

/* Fixed footer */
.enso-footer { position:fixed; left:0; bottom:0; width:100%; z-index:1000;
    background:rgba(15,23,42,.97); backdrop-filter:blur(8px); color:#cbd5e1;
    border-top:1px solid #1e293b; padding:8px 20px; font-size:.8rem;
    display:flex; justify-content:space-between; align-items:center; gap:10px; }
.enso-footer a { color:#38bdf8; font-weight:700; text-decoration:none; }
.enso-footer .muted { color:#64748b; }

/* ---- Mobile / responsive ---- */
@media (max-width: 680px) {
    .block-container { padding-left:.7rem; padding-right:.7rem; padding-bottom:5rem; }
    .enso-hero { padding:18px 18px; border-radius:14px; }
    .enso-hero-title { font-size:1.2rem; }
    .enso-hero-sub { font-size:.8rem; }
    .enso-headline { font-size:.88rem; }
    .enso-pill { font-size:.8rem; padding:6px 14px; }
    [data-testid="stMetricValue"] { font-size:1.2rem; }
    [data-testid="stMetricLabel"] p { font-size:.66rem; }
    [data-testid="stTabs"] [data-baseweb="tab"] { padding:6px 11px; font-size:.78rem; }
    .enso-footer { font-size:.7rem; padding:7px 12px; }
    .enso-footer .hide-sm { display:none; }
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data loading (cached for the session)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def load_enso():
    oni = enso.fetch_oni()
    return oni, enso.assess_status(oni), official.fetch_cpc_outlook()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_core():
    """The unified 25-district CHIRPS analysis used across every tab."""
    onim = enso.monthly_oni(enso.fetch_oni())
    dm, geojson, dmeta = districts.district_monthly()
    panel = analysis.build_panel(dm.drop(columns=["n_cells"]), onim)
    comp = analysis.elnino_composite(panel)
    lag = analysis.lag_analysis(panel)
    imp = impacts.region_impacts(comp, meta=dmeta)
    spil = spi.district_spi(dm[["region", "date", "precip"]])
    spi_cur = spi.current_spi(spil)
    en_spi3 = spi.elnino_spi(spil, onim, scale=3)
    table = report.assemble_table(imp, comp, spi_cur)
    # Canonical ENSO event framework (developing year + EP/CP flavour).
    ev = events.classify_flavour(events.detect_events(onim), events.fetch_nino_indices())
    dev = events.developing_composite(panel, ev)
    flav = events.flavour_composite(panel, ev, "SIM", 0)
    # Indian Ocean Dipole + symmetric ENSO-phase (La Niña) analysis.
    dmi = iod.fetch_dmi()
    iod_an = iod.analyze(panel, dmi)
    return dict(
        geojson=geojson, meta=dmeta, panel=panel, comp=comp, lag=lag, imp=imp,
        cells=dm.groupby("region")["n_cells"].first(),
        span=(dm["date"].min(), dm["date"].max()), spi_cur=spi_cur,
        en_spi3=en_spi3, table=table, summary=impacts.national_summary(imp),
        events=ev, dev=dev, flavour=flav, dmi=dmi, iod=iod_an,
    )


@st.cache_data(ttl=14 * 24 * 3600, show_spinner=False)
def load_vegetation(name, lat, lon):
    ndvi = vegetation.fetch_ndvi(name, lat, lon)
    if ndvi.empty:
        return None, None
    vci = vegetation.add_vci(ndvi)
    comp = vegetation.enso_composite(vci, enso.monthly_oni(enso.fetch_oni()))
    return vci, comp


@st.cache_data(show_spinner=False)
def _report_bytes(table, headline, summary):
    return report.to_csv(table), report.to_pdf(table, headline, summary)


def _short(name: str) -> str:
    return name.replace(" District", "")


try:
    oni_df, status, cpc_outlook = load_enso()
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load ENSO data (NOAA): {exc}")
    st.stop()


# --------------------------------------------------------------------------- #
# Hero header
# --------------------------------------------------------------------------- #
accent, soft, dot = PHASE_THEME[status.phase]
pill = "DEVELOPING EL NIÑO" if status.developing_elnino and status.phase != "El Nino" \
    else status.phase.upper()
st.markdown(f"""
<div class="enso-hero">
  <div class="enso-hero-row">
    <div>
      <div class="enso-hero-title">🌊 El Niño × Sri Lanka — Impact Intelligence</div>
      <div class="enso-hero-sub">Live ENSO monitoring · CHIRPS &amp; MODIS satellite
         analysis · 25-district risk</div>
    </div>
    <div class="enso-pill" style="background:{accent};">{dot}&nbsp; {pill}</div>
  </div>
  <div class="enso-headline">{status.headline}</div>
</div>
""", unsafe_allow_html=True)

# --- Unified 25-district analysis (scoped status spinner) ------------------- #
with st.status("Building 25-district satellite analysis…", expanded=False) as _s:
    try:
        _s.write("Reading CHIRPS satellite grids & computing district zonal stats…")
        D = load_core()
        _s.write("Significance testing, SPI drought indices & impact scoring…")
        _s.update(label=f"✓ 25 districts analysed · CHIRPS "
                  f"{D['span'][0]:%Y}–{D['span'][1]:%Y}", state="complete")
    except Exception as exc:  # noqa: BLE001
        _s.update(label="District analysis failed", state="error")
        st.error(f"Could not build the district analysis: {exc}")
        st.stop()

panel, composite, lag_df = D["panel"], D["comp"], D["lag"]
imp, summary = D["imp"], D["summary"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current ONI", f"{status.latest_oni:+.2f} °C", help="Oceanic Niño Index anomaly")
c2.metric("Trend", f"{status.trend_per_season:+.2f} /season",
          delta="warming" if status.trend_per_season > 0 else "cooling")
c3.metric("Phase", PRETTY[status.phase])
c4.metric("National impact", f"{summary['overall']:.0f}/100",
          help="Mean adverse-risk score across 25 districts & 4 sectors")

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🌊 Control panel")
    st.metric("ENSO phase", PRETTY[status.phase], help=status.headline)
    st.metric("Latest ONI", f"{status.latest_oni:+.2f} °C",
              f"{status.trend_per_season:+.2f}/season")
    in_drought = int((D["spi_cur"]["SPI6"] < -1.0).sum())
    st.metric("Districts in drought now", f"{in_drought} / 25",
              help="SPI-6 < −1 (CHIRPS satellite)")
    st.markdown("---")
    st.markdown("**Top exposure**")
    for r in summary["top_risk_regions"]:
        st.markdown(f"&nbsp;&nbsp;🔸 {_short(r)}")
    st.markdown("---")
    _dmi_end = D["dmi"]["date"].max() if len(D["dmi"]) else None
    st.caption(f"**Live data** · ONI {status.latest_date:%b %Y} · "
               f"CHIRPS → {D['span'][1]:%b %Y}"
               + (f" · DMI → {_dmi_end:%b %Y}" if _dmi_end is not None else ""))
    st.caption("Everything on every tab is computed in real time from these feeds — "
               "no values are pre-baked.")
    st.caption("Sources: NOAA CPC (ONI/Niño) · NOAA PSL (DMI) · UCSB CHIRPS · "
               "NASA MODIS · geoBoundaries")
    if st.button("🔄 Refresh live data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Analytical aid — not an official forecast.")

PAGES = ["ENSO Status", "ENSO Events", "Ocean Drivers", "Spatial Impact",
         "Districts", "Region Details", "Sector Impacts", "Methods"]
page = option_menu(
    None, PAGES,
    icons=["graph-up-arrow", "diagram-3", "water", "map", "grid-3x3-gap-fill",
           "geo-alt", "bar-chart-line-fill", "journal-text"],
    orientation="horizontal", default_index=0,
    styles={
        "container": {"padding": "6px", "background-color": "#f1f5f9",
                      "border-radius": "14px", "margin-bottom": "10px"},
        "icon": {"color": "#0ea5e9", "font-size": "15px"},
        "nav-link": {"font-size": "14px", "font-weight": "600", "color": "#475569",
                     "border-radius": "10px", "padding": "9px 14px",
                     "--hover-color": "#e2e8f0", "white-space": "nowrap"},
        "nav-link-selected": {"background-color": "#0ea5e9", "color": "#ffffff"},
    },
)
# Test/deep-link hook: honour an explicit page override if one is set.
page = st.session_state.get("nav_override", page)


# --------------------------------------------------------------------------- #
# Tab 1 — ENSO status & projection
# --------------------------------------------------------------------------- #
if page == "ENSO Status":
    st.subheader("Oceanic Niño Index (ONI) — history & near-term projection")
    yrs = st.slider("Years of history", 5, int((oni_df['date'].max() - oni_df['date'].min()).days / 365), 25)
    cut = oni_df["date"].max() - pd.DateOffset(years=yrs)
    h = oni_df[oni_df["date"] >= cut]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["date"], y=h["oni"], mode="lines", name="ONI",
                             line=dict(color="#222", width=2)))
    fig.add_hrect(y0=ONI_ELNINO, y1=4, fillcolor="#d62728", opacity=0.08, line_width=0)
    fig.add_hrect(y0=-4, y1=ONI_LANINA, fillcolor="#1f77b4", opacity=0.08, line_width=0)
    fig.add_hline(y=ONI_ELNINO, line_dash="dot", line_color="#d62728",
                  annotation_text="El Niño +0.5 °C")
    fig.add_hline(y=ONI_LANINA, line_dash="dot", line_color="#1f77b4",
                  annotation_text="La Niña −0.5 °C")
    # Real-time statistical forecast (damped persistence), computed live from ONI.
    fc = forecast.enso_forecast(enso.monthly_oni(oni_df))
    last = oni_df.iloc[-1]
    if not fc.empty:
        fx = [last["date"]] + list(fc["date"])
        fig.add_trace(go.Scatter(
            x=fx + fx[::-1],
            y=[last["oni"]] + list(fc["upper"]) + ([last["oni"]] + list(fc["lower"]))[::-1],
            fill="toself", fillcolor="rgba(214,39,40,0.12)", line=dict(width=0),
            hoverinfo="skip", name="95% range", showlegend=True))
        fig.add_trace(go.Scatter(x=[last["date"]] + list(fc["date"]),
                                 y=[last["oni"]] + list(fc["oni"]),
                                 mode="lines+markers", name="statistical forecast",
                                 line=dict(color="#d62728", dash="dash")))
    fig.update_layout(height=420, yaxis_title="ONI anomaly (°C)", margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    # Official CPC outlook (dynamical-model consensus) for comparison.
    if cpc_outlook.get("available"):
        prob_txt = " · ".join(f"**{p}%** El Niño in {per}" for p, per in cpc_outlook["probs"][:2])
        st.success(
            f"**Official outlook — NOAA CPC{(' (' + cpc_outlook['status'] + ')') if cpc_outlook.get('status') else ''}.** "
            + (cpc_outlook.get("synopsis") or "")
            + (f"\n\nHeadline probabilities: {prob_txt}." if prob_txt else ""),
            icon="🛰️")
        st.caption("↑ The authoritative multi-model consensus (live from NOAA CPC). "
                   "Compare it with the in-house statistical forecast below — they "
                   "should broadly agree on direction; the statistical model is "
                   "typically a little more conservative on the winter peak.")

    if not fc.empty:
        st.markdown("#### In-house statistical ENSO outlook — probabilities by season")
        st.caption(forecast.headline(fc))
        prob = fc.melt(id_vars=["seas", "date"],
                       value_vars=["p_elnino", "p_neutral", "p_lanina"],
                       var_name="phase", value_name="prob")
        prob["phase"] = prob["phase"].map({"p_elnino": "El Niño", "p_neutral": "Neutral",
                                           "p_lanina": "La Niña"})
        prob["label"] = prob["seas"] + " " + prob["date"].dt.strftime("%Y")
        figp = px.bar(prob, x="label", y="prob", color="phase", height=320,
                      color_discrete_map={"El Niño": "#ef4444", "Neutral": "#94a3b8",
                                          "La Niña": "#3b82f6"},
                      labels=dict(prob="probability", label="", phase=""))
        figp.update_layout(barmode="stack", yaxis_tickformat=".0%")
        st.plotly_chart(figp, use_container_width=True)

    st.info(
        f"**Interpretation.** {status.headline}\n\n"
        "The ONI is the 3-month running sea-surface-temperature anomaly in the "
        "Niño-3.4 region; values at or above **+0.5 °C** define El Niño. In the "
        "CHIRPS satellite record for Sri Lanka, El Niño is most strongly associated "
        "with a **sharp reduction in first inter-monsoon (March–April) rainfall** "
        "and **wetter second-inter-monsoon (October–November) conditions**, raising "
        "flood and landslide risk; the south-west-monsoon signal is weak."
    )
    st.caption("⚠️ The forecast is a **statistical persistence-plus-tendency model** "
               "computed live from the latest ONI — it uses the current level *and* "
               "its recent momentum, so it captures ENSO *development* (it favours a "
               "growing El Niño while the ONI is rising, broadly tracking the "
               "official outlooks). It is **not** a dynamical coupled-model "
               "simulation, and skill still drops across the boreal-spring "
               "predictability barrier. For the authoritative multi-model forecast "
               "see the [NOAA CPC / IRI ENSO outlook](https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/).")

    en_years = enso.elnino_event_years(oni_df)
    st.caption("El Niño years in the record used to build the impact composite: "
               + ", ".join(str(y) for y in en_years) + ".")


# --------------------------------------------------------------------------- #
# ENSO Events — canonical developing-year framework + EP/CP flavour
# --------------------------------------------------------------------------- #
if page == "ENSO Events":
    st.subheader("ENSO events — developing-year rainfall response")
    st.caption("The textbook teleconnection follows an event's life cycle: an "
               "El Niño develops in year 0, peaks in December–February, and "
               "decays through year 1. Here Sri Lanka's *national* CHIRPS rainfall "
               "is composited in **event-relative** seasons across all events "
               "since 1981 — a stricter, autocorrelation-aware view than tagging "
               "individual months.")

    dev = D["dev"].dropna(subset=["mean_pct"]).copy()
    dev["err_low"] = dev["mean_pct"] - dev["ci_low"]
    dev["err_high"] = dev["ci_high"] - dev["mean_pct"]
    dev["sig"] = dev["p"].apply(lambda p: "significant (p<0.05)" if p < 0.05
                                else "suggestive (p<0.10)" if p < 0.10 else "n.s.")
    figd = px.bar(dev, x="mean_pct", y="rel_season", orientation="h",
                  color="sig", height=360,
                  color_discrete_map={"significant (p<0.05)": "#0ea5e9",
                                      "suggestive (p<0.10)": "#f59e0b", "n.s.": "#cbd5e1"},
                  labels=dict(mean_pct="national rainfall anomaly (%)", rel_season="",
                              sig="significance"),
                  error_x="err_high", error_x_minus="err_low")
    figd.add_vline(x=0, line_color="#475569")
    st.plotly_chart(figd, use_container_width=True)
    st.caption("Bars = mean national rainfall anomaly vs non-event years; whiskers "
               "= bootstrap 95 % CI. The robust signal is a **wetter second "
               "inter-monsoon (Oct–Nov) in the developing year**, with drier "
               "first-inter-monsoon conditions in the following (decay) year.")

    cE1, cE2 = st.columns([3, 2])
    with cE1:
        st.markdown("#### Detected El Niño events & Pacific flavour")
        et = D["events"].copy()
        et = et[et["y1"] >= D["span"][0].year]   # events with CHIRPS coverage
        et["event"] = et["y0"].astype(str) + "–" + et["y1"].astype(str).str[-2:]
        show = et[["event", "peak_oni", "nino3", "nino4", "flavour"]].rename(
            columns={"peak_oni": "peak ONI", "nino3": "Niño-3", "nino4": "Niño-4",
                     "flavour": "type"})
        st.dataframe(show.round(2), use_container_width=True, hide_index=True, height=360)
    with cE2:
        st.markdown("#### Eastern- vs Central-Pacific")
        fl = D["flavour"].dropna(subset=["mean_pct"])
        figf = px.bar(fl, x="flavour", y="mean_pct", color="flavour", height=300,
                      color_discrete_map={"Eastern-Pacific": "#ef4444",
                                          "Central-Pacific": "#8b5cf6"},
                      labels=dict(mean_pct="Oct–Nov rainfall anomaly (%)", flavour=""))
        figf.update_layout(showlegend=False)
        st.plotly_chart(figf, use_container_width=True)
        st.caption("Oct–Nov response by event type (Niño-3 vs Niño-4 at the peak). "
                   "Both flavours wet Sri Lanka similarly in this record.")
    st.info("**Why this differs from the district map.** Compositing by *event* "
            "(≈13 independent El Niños) instead of by month, and aligning to the "
            "developing year, recovers the canonical **significant Oct–Nov "
            "enhancement** — a signal that the stricter month-level, "
            "FDR-corrected district test reduces to *suggestive*. Both views are "
            "shown deliberately so you can see how the conclusion depends on the "
            "statistical framing.")


# --------------------------------------------------------------------------- #
# Ocean Drivers — Indian Ocean Dipole + symmetric La Niña
# --------------------------------------------------------------------------- #
if page == "Ocean Drivers":
    st.subheader("Ocean drivers of the Oct–Nov rains — ENSO, La Niña & the IOD")
    st.caption("For Sri Lanka the **Indian Ocean Dipole (IOD)** rivals ENSO as a "
               "driver of the second inter-monsoon (Oct–Nov). Because the two "
               "oceans are correlated, we disentangle their *independent* effects "
               "with a multiple regression — not just composites.")

    A = D["iod"]
    dmi = D["dmi"]

    # DMI timeline
    if len(dmi):
        dd = dmi[dmi["date"] >= dmi["date"].max() - pd.DateOffset(years=40)]
        figm = go.Figure()
        figm.add_trace(go.Scatter(x=dd["date"], y=dd["dmi"], mode="lines",
                                  line=dict(color="#0e7490", width=1.6), name="DMI"))
        figm.add_hrect(y0=0.4, y1=3, fillcolor="#ef4444", opacity=0.07, line_width=0)
        figm.add_hrect(y0=-3, y1=-0.4, fillcolor="#3b82f6", opacity=0.07, line_width=0)
        figm.add_hline(y=0.4, line_dash="dot", line_color="#ef4444",
                       annotation_text="positive IOD +0.4 °C")
        figm.add_hline(y=-0.4, line_dash="dot", line_color="#3b82f6",
                       annotation_text="negative IOD −0.4 °C")
        figm.update_layout(height=300, yaxis_title="Dipole Mode Index (°C)",
                           margin=dict(t=10))
        st.plotly_chart(figm, use_container_width=True)

    cL, cR = st.columns(2)
    enso_c = A["enso_comp"].copy()
    enso_c["label"] = enso_c["phase"].map({"El Nino": "El Niño", "Neutral": "Neutral",
                                           "La Nina": "La Niña"})
    figE = px.bar(enso_c, x="label", y="mean_pct", color="label", height=320,
                  color_discrete_map={"El Niño": "#ef4444", "Neutral": "#94a3b8",
                                      "La Niña": "#3b82f6"},
                  labels=dict(mean_pct="Oct–Nov anomaly (%)", label=""))
    figE.update_layout(showlegend=False, title="By ENSO phase")
    cL.plotly_chart(figE, use_container_width=True)

    figI = px.bar(A["iod_comp"], x="phase", y="mean_pct", color="phase", height=320,
                  color_discrete_map={"Positive IOD": "#ef4444", "Neutral IOD": "#94a3b8",
                                      "Negative IOD": "#3b82f6"},
                  labels=dict(mean_pct="Oct–Nov anomaly (%)", phase=""))
    figI.update_layout(showlegend=False, title="By IOD phase")
    cR.plotly_chart(figI, use_container_width=True)
    st.caption("Mean national Oct–Nov rainfall anomaly vs neutral years. **La Niña** "
               "shows the symmetric opposite of El Niño; **positive IOD** looks like "
               "the strongest wet signal — but see the attribution below.")

    # Partial-regression attribution (the rigorous bit)
    p = A["partial"]
    st.markdown("#### Disentangling ENSO vs IOD — multiple regression")
    m = st.columns(4)
    m[0].metric("ENSO independent effect",
                "—" if pd.isna(p["enso_beta"]) else f"β {p['enso_beta']:+.2f}",
                ("n/a" if pd.isna(p["enso_p"]) else
                 f"p={p['enso_p']:.3f}" + (" ✓" if p["enso_p"] < 0.05 else "")))
    m[1].metric("IOD independent effect",
                "—" if pd.isna(p["iod_beta"]) else f"β {p['iod_beta']:+.2f}",
                ("n/a" if pd.isna(p["iod_p"]) else
                 f"p={p['iod_p']:.3f}" + (" ✓" if p["iod_p"] < 0.05 else "")))
    m[2].metric("ONI ↔ DMI correlation",
                "—" if pd.isna(p["oni_dmi_corr"]) else f"{p['oni_dmi_corr']:+.2f}",
                help="The two drivers are correlated, so composites overlap.")
    m[3].metric("Model R²", "—" if pd.isna(p["r2"]) else f"{p['r2']:.2f}",
                help=f"n = {p['n']} seasons")
    enso_sig = (not pd.isna(p["enso_p"])) and p["enso_p"] < 0.05
    iod_sig = (not pd.isna(p["iod_p"])) and p["iod_p"] < 0.05
    st.info(
        f"**Attribution.** The ONI and DMI are correlated (r = "
        f"{p['oni_dmi_corr']:+.2f}), so their simple composites partly overlap. "
        "Regressing Oct–Nov rainfall on **both** standardised indices isolates "
        "each driver's *independent* contribution: "
        + ("**ENSO is the significant independent driver** "
           f"(p = {p['enso_p']:.3f}), " if enso_sig else
           f"ENSO's independent effect is not significant (p = {p['enso_p']:.3f}), ")
        + ("**IOD also contributes independently**."
           if iod_sig else
           "while the **IOD's apparent effect is largely shared with ENSO and is "
           "not independently significant** in this record. In other words, much "
           "of the strong positive-IOD composite above is really the ENSO signal.")
    )

    st.markdown("#### ENSO × IOD interaction (Oct–Nov)")
    j = A["joint"].copy()
    j["enso"] = j["enso"].map({"El Nino": "El Niño", "La Nina": "La Niña"})
    j = j.rename(columns={"enso": "ENSO", "iod": "IOD", "n": "n (seasons)",
                          "mean_pct": "Oct–Nov anomaly %"})
    st.dataframe(j.round(1), use_container_width=True, hide_index=True)
    st.caption("⚠️ These combined cells have **very small samples** (n shown) — "
               "the El Niño + positive-IOD 'double whammy' and the La Niña + "
               "negative-IOD dry case are indicative only, not significance-tested. "
               "Data: HadISST DMI (NOAA PSL) + CHIRPS rainfall.")


# --------------------------------------------------------------------------- #
# Tab 2 — Spatial impact map
# --------------------------------------------------------------------------- #
if page == "Spatial Impact":
    st.subheader("How the El Niño impact varies across Sri Lanka")
    st.warning("**Read me:** the *rainfall* layers are data-driven and "
               "significance-tested (FDR-corrected). The *sector-risk* layers "
               "(drought / flood / agriculture / hydropower) are a **transparent "
               "but un-validated heuristic** — they weight the significant rainfall "
               "signals by sector relevance, and have **not** been calibrated "
               "against observed droughts, yields, reservoir levels or floods. "
               "Treat them as indicative, not as measured risk.", icon="⚠️")
    colA, colB = st.columns([1, 2])
    metric = colA.selectbox(
        "Map layer",
        ["overall", "drought", "flood", "agriculture", "hydropower"],
        format_func=lambda s: {
            "overall": "Overall adverse risk",
            "drought": "Drought / water-deficit risk",
            "flood": "Flood & landslide risk",
            "agriculture": "Agriculture (paddy / tea) risk",
            "hydropower": "Water & hydropower risk",
        }[s],
    )
    colA.markdown(
        "Colour (and bubble size) = the sector risk score (**0–100**) for the "
        "selected impact. This shows *where* a developing El Niño bites hardest: "
        "the wet south-west and central highlands carry the highest overall "
        "exposure — driven by the spring rainfall deficit and the wetter, "
        "flood-prone second inter-monsoon — while the northern dry zone shows the "
        "largest drought (SPI) response."
    )

    use_choropleth = colA.toggle("Filled district map", value=True,
                                 help="Choropleth (filled districts) vs bubble map")
    mdf = imp.dropna(subset=[metric]).copy()
    mdf["risk"] = mdf[metric]
    mdf["name"] = mdf["region"].map(_short)
    mdf["confidence"] = mdf["confidence"].round(2)
    if use_choropleth:
        fig_map = px.choropleth_map(
            mdf, geojson=D["geojson"], locations="region",
            featureidkey="properties.shapeName", color="risk",
            color_continuous_scale="YlOrRd", range_color=(0, 100),
            map_style="carto-positron", zoom=6.3, center=dict(lat=7.85, lon=80.7),
            opacity=0.8, hover_name="name", height=620,
            hover_data={"zone": True, "direction": True, "risk": ":.0f",
                        "region": False}, labels={"risk": "risk 0–100"})
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        colB.plotly_chart(fig_map, use_container_width=True)
        mdf = None  # rendered
    else:
        fig_map = px.scatter_map(
            mdf, lat="lat", lon="lon", size="risk", color="risk",
            color_continuous_scale="YlOrRd", size_max=34, zoom=6.3,
            center=dict(lat=7.85, lon=80.7), hover_name="name",
            hover_data={"zone": True, "direction": True, "risk": ":.0f",
                        "lat": False, "lon": False, "name": False},
            range_color=(0, 100), height=620,
        )
        fig_map.update_traces(customdata=mdf[["zone", "direction", "confidence"]].values,
                              hovertemplate="<b>%{hovertext}</b><br>%{customdata[0]}<br>"
                              "trend: %{customdata[1]}<br>risk: %{marker.size:.0f}/100<br>"
                              "signal confidence: %{customdata[2]}<extra></extra>")
        fig_map.update_layout(map_style="carto-positron", margin=dict(l=0, r=0, t=0, b=0))
        colB.plotly_chart(fig_map, use_container_width=True)

    st.markdown("#### District × season rainfall signal during El Niño (% vs normal)")
    pct = analysis.region_season_matrix(composite, "precip_pct")
    # Annotate each cell with the % anomaly plus an FDR-significance marker:
    #   **  q < 0.05 (significant)    *  q < 0.10 (suggestive)
    qm = analysis.region_season_matrix(composite, "q_value")
    text = pct.copy().astype(object)
    for r in pct.index:
        for c in pct.columns:
            v = pct.loc[r, c]
            if pd.isna(v):
                text.loc[r, c] = ""
                continue
            q = qm.loc[r, c]
            mark = "**" if q < 0.05 else "*" if q < 0.10 else ""
            text.loc[r, c] = f"{v:+.0f}{mark}"
    pct_lbl = pct.rename(columns=SEASON_LABELS)
    pct_lbl.index = [_short(r) for r in pct_lbl.index]
    fig_hm = px.imshow(pct_lbl, color_continuous_scale="RdBu", zmin=-60, zmax=60,
                       aspect="auto", labels=dict(color="% vs normal"))
    fig_hm.update_traces(text=text.values, texttemplate="%{text}")
    fig_hm.update_layout(height=max(440, 24 * len(pct_lbl) + 90), margin=dict(t=10))
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption("Blue = wetter than normal during El Niño, red = drier. "
               "**  = significant,  *  = suggestive — after **Benjamini–Hochberg "
               "FDR correction** across all 100 district × season tests, using "
               "**event-based** (season-year, detrended) samples. Only the robust "
               "island-wide **March–April drying** survives FDR; the October–"
               "November wetting is suggestive here (it is significant in the "
               "event-aligned national composite — see the **ENSO Events** tab).")

    st.markdown("#### ENSO lead/lag — how far ONI precedes the rainfall response")
    lag_show = lag_df[["region", "best_lag", "best_r", "best_p"]].copy()
    lag_show["region"] = lag_show["region"].map(_short)
    lag_show.columns = ["District", "Best lag (months)", "Peak corr (r)", "p-value"]
    lagp = lag_df.copy()
    lagp["name"] = lagp["region"].map(_short)
    figl = px.bar(lagp.sort_values("best_r"), x="name", y="best_r", color="best_lag",
                  color_continuous_scale="Viridis", height=380,
                  labels=dict(best_r="peak ONI↔rain corr", best_lag="lag (mo)", name=""))
    cL, cR = st.columns([2, 1])
    cL.plotly_chart(figl, use_container_width=True)
    cR.dataframe(lag_show.round(3), use_container_width=True, hide_index=True, height=380)
    st.caption("A positive lag means the ONI leads the rainfall response by that "
               "many months. Negative correlations reflect El Niño's drying of the "
               "dominant rainy seasons; a low |r| is expected, because monthly "
               "tropical rainfall is noisy and the El Niño signal is concentrated "
               "in just two seasons.")


# --------------------------------------------------------------------------- #
# Tab 3 — District-level satellite (CHIRPS) choropleth
# --------------------------------------------------------------------------- #
if page == "Districts":
    st.subheader("District-level analysis from CHIRPS satellite rainfall (~5 km)")
    st.caption("Gridded satellite-and-station rainfall (UCSB CHIRPS v2.0, "
               "1981–present) is zonally averaged over each of Sri Lanka's 25 "
               "districts and run through the significance-tested composite engine. "
               "Switch the map layer to compare sector risk, per-season rainfall "
               "anomalies, and the SPI drought index.")

    if D:  # unified 25-district analysis is loaded once at startup
        d_geojson = D["geojson"]
        d_comp, d_imp, d_cells = D["comp"], D["imp"], D["cells"]
        d_span, d_spi, d_enspi = D["span"], D["spi_cur"], D["en_spi3"]
        d_meta, d_table = D["meta"], D["table"]

        if d_geojson is not None:
            cc = st.columns([1, 2])
            view = cc[0].radio(
                "Map layer",
                ["overall", "drought", "flood", "agriculture", "hydropower",
                 "FIM", "SWM", "SIM", "NEM", "SPI_now", "SPI_en"],
                format_func=lambda s: {
                    "overall": "Overall adverse risk", "drought": "Drought risk",
                    "flood": "Flood/landslide risk", "agriculture": "Agriculture risk",
                    "hydropower": "Hydropower risk",
                    "FIM": "Rain anomaly — Mar–Apr (FIM)",
                    "SWM": "Rain anomaly — Yala/SW (May–Sep)",
                    "SIM": "Rain anomaly — Oct–Nov (SIM)",
                    "NEM": "Rain anomaly — Maha/NE (Dec–Feb)",
                    "SPI_now": "SPI-6 — current drought state",
                    "SPI_en": "SPI-3 — mean in El Niño months"}[s],
            )
            is_rain = view in ("FIM", "SWM", "SIM", "NEM")
            is_spi = view in ("SPI_now", "SPI_en")
            if is_rain:
                sub = d_comp[d_comp["season"] == view]
                val = sub.set_index("region")["precip_pct"]
                sig = sub.set_index("region")["significant"]
                cmap, rng, clabel = "RdBu", (-50, 50), "% vs normal"
            elif view == "SPI_now":
                val, sig = d_spi.set_index("region")["SPI6"], None
                cmap, rng, clabel = "RdBu", (-2, 2), "SPI-6 (now)"
            elif view == "SPI_en":
                val, sig = d_enspi, None
                cmap, rng, clabel = "RdBu", (-0.5, 0.5), "mean SPI-3 (El Niño)"
            else:
                val, sig = d_imp.set_index("region")[view], None
                cmap, rng, clabel = "YlOrRd", (0, 100), "risk 0–100"

            mdf = pd.DataFrame({"region": val.index, "value": val.values})
            mdf["name"] = mdf["region"].str.replace(" District", "", regex=False)
            if sig is not None:
                mdf["sig"] = mdf["region"].map(sig).map({True: "yes", False: "no"})

            fig_ch = px.choropleth_map(
                mdf, geojson=d_geojson, locations="region",
                featureidkey="properties.shapeName", color="value",
                color_continuous_scale=cmap, range_color=rng,
                map_style="carto-positron", zoom=6.3,
                center=dict(lat=7.85, lon=80.7), opacity=0.78,
                hover_name="name", height=640,
                hover_data={"value": ":.2f", "region": False,
                            **({"sig": True} if sig is not None else {})},
                labels={"value": clabel},
            )
            fig_ch.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            cc[1].plotly_chart(fig_ch, use_container_width=True)

            if is_rain:
                nsig = int(d_comp[(d_comp.season == view)]["significant"].sum())
                meananom = d_comp[d_comp.season == view]["precip_pct"].mean()
                cc[0].metric(f"{view}: mean anomaly", f"{meananom:+.0f}%")
                cc[0].metric("Districts significant (p<0.05)", f"{nsig} / 25")
                cc[0].caption("Hover shows statistical significance. "
                              "Blue = wetter, red = drier.")
            elif is_spi:
                in_drought = int((d_spi["SPI6"] < -1.0).sum())
                cc[0].metric("Districts in drought now (SPI-6 < −1)", f"{in_drought} / 25")
                cc[0].metric("Driest district now",
                             d_spi.loc[d_spi['SPI6'].idxmin(), 'region'].replace(' District', ''),
                             f"SPI-6 {d_spi['SPI6'].min():+.2f}")
                cc[0].caption("SPI (McKee 1993) in σ-units: < −1 drought, > +1 wet. "
                              "Red = dry. ‘El Niño’ layer = average SPI-3 across all "
                              "historical El Niño months.")
            else:
                cc[0].metric("Districts analysed", "25")
                cc[0].metric("Avg grid cells / district", f"{d_cells.mean():.0f}")
                cc[0].caption(f"CHIRPS coverage {d_span[0]:%Y}–{d_span[1]:%Y}. "
                              "Darker = higher adverse risk.")

            # --- Download report ---------------------------------------------
            csv_bytes, pdf_bytes = _report_bytes(
                d_table, status.headline, impacts.national_summary(d_imp))
            dl = cc[0].columns(2)
            dl[0].download_button("⬇️ CSV", csv_bytes, "srilanka_elnino_districts.csv",
                                  "text/csv", use_container_width=True)
            dl[1].download_button("⬇️ PDF report", pdf_bytes,
                                  "srilanka_elnino_report.pdf", "application/pdf",
                                  use_container_width=True)

            st.markdown("#### District scores, satellite rainfall signal & current SPI")
            showcols = ["district", "zone", "direction", "drought", "flood",
                        "agriculture", "hydropower", "overall",
                        "FIM %", "SWM %", "SIM %", "NEM %", "SPI6", "status"]
            showcols = [c for c in showcols if c in d_table.columns]
            grad = [c for c in ["drought", "flood", "agriculture", "hydropower",
                                "overall"] if c in d_table.columns]
            st.dataframe(
                d_table[showcols].style
                .background_gradient(cmap="YlOrRd", vmin=0, vmax=100, subset=grad)
                .background_gradient(cmap="RdBu", vmin=-50, vmax=50,
                                     subset=["FIM %", "SWM %", "SIM %", "NEM %"])
                .background_gradient(cmap="RdBu", vmin=-2, vmax=2, subset=["SPI6"])
                .format({**{c: "{:.0f}" for c in grad + ["FIM %", "SWM %", "SIM %", "NEM %"]},
                         "SPI6": "{:+.2f}"}),
                use_container_width=True, height=480, hide_index=True,
            )
            st.caption("Across all 25 districts the satellite record shows the same "
                       "pattern: the **March–April (first inter-monsoon) drying is "
                       "statistically significant island-wide**, while the wetter "
                       "**October–November (second inter-monsoon)** response — the "
                       "main flood window — concentrates in the south-west wet zone "
                       "and central highlands.")

            # --- MODIS vegetation explorer (on demand, per district) ----------
            st.divider()
            st.markdown("#### 🛰️ MODIS satellite vegetation health (NDVI → VCI)")
            st.caption("NASA MODIS MOD13Q1 250 m NDVI (2000–present) for one district. "
                       "First load fetches ~60 chunked tiles (~1–2 min, cached after).")
            shorts = sorted(d_imp["region"].str.replace(" District", "", regex=False))
            vc = st.columns([2, 1])
            pick = vc[0].selectbox("District", shorts)
            if vc[1].button(f"🛰️ Load MODIS NDVI — {pick}", use_container_width=True):
                st.session_state["veg_district"] = pick + " District"
                st.rerun()

            vd = st.session_state.get("veg_district")
            if vd and vd in d_meta:
                m = d_meta[vd]
                with st.status(f"Fetching MODIS NDVI for {_short(vd)} "
                               "(cached after first load)…", expanded=False) as _vs:
                    vci, vcomp = load_vegetation(vd, m["lat"], m["lon"])
                    _vs.update(label=f"✓ MODIS NDVI loaded — {_short(vd)}",
                               state="complete")
                if vci is None:
                    st.warning(f"MODIS NDVI unavailable for {vd.replace(' District','')}.")
                else:
                    short = vd.replace(" District", "")
                    oni_m = enso.monthly_oni(oni_df)
                    vv = vci.merge(oni_m.rename("oni"), left_on="date",
                                   right_index=True, how="left")
                    vv["phase"] = vv["oni"].apply(
                        lambda v: PRETTY.get(enso.classify(v), "n/a")
                        if pd.notna(v) else "n/a")
                    mcols = st.columns(4)
                    mcols[0].metric(f"{short}: VCI in El Niño",
                                    f"{vcomp['El Nino']['vci']:.0f}",
                                    help="<35 = vegetation stress")
                    mcols[1].metric("VCI in Neutral", f"{vcomp['Neutral']['vci']:.0f}")
                    mcols[2].metric("VCI in La Niña", f"{vcomp['La Nina']['vci']:.0f}")
                    cr = vcomp["oni_vci_corr"]
                    mcols[3].metric("ONI ↔ VCI corr",
                                    "n/a" if pd.isna(cr) else f"{cr:+.2f}")
                    figv = px.scatter(
                        vv, x="date", y="ndvi", color="phase",
                        color_discrete_map={PRETTY[k]: v for k, v in PHASE_COLORS.items()},
                        height=340,
                        labels=dict(ndvi="monthly NDVI", phase="ENSO phase"))
                    figv.add_scatter(x=vv["date"], y=vv["ndvi"].rolling(6).mean(),
                                     mode="lines", line=dict(color="#333", width=1),
                                     name="6-mo mean", showlegend=False)
                    st.plotly_chart(figv, use_container_width=True)
                    diff = vcomp["El Nino"]["vci"] - vcomp["Neutral"]["vci"]
                    st.caption(
                        f"During El Niño, {short}'s vegetation condition runs "
                        f"**{abs(diff):.0f} VCI points {'lower' if diff < 0 else 'higher'}** "
                        f"than neutral years (El Niño n={vcomp['El Nino']['n']} months). "
                        "Heavily irrigated districts show muted NDVI response.")


# --------------------------------------------------------------------------- #
# Tab 4 — Region detail
# --------------------------------------------------------------------------- #
if page == "Region Details":
    region = st.selectbox("District", list(imp["region"]), format_func=_short)
    row = imp[imp["region"] == region].iloc[0]
    st.markdown(f"### {_short(region)} District &nbsp;·&nbsp; {row['zone']}")
    st.caption(row["note"])

    m = st.columns(5)
    for col, key, name in zip(
        m,
        ["drought", "flood", "agriculture", "hydropower", "overall"],
        ["Drought", "Flood", "Agriculture", "Hydropower", "Overall"],
    ):
        val = row[key]
        col.metric(name, "n/a" if pd.isna(val) else f"{val:.0f}",
                   help=impacts.risk_label(val))

    st.markdown("#### Monthly rainfall by ENSO phase")
    pc = analysis.phase_composite_by_month(panel, region)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    figp = go.Figure()
    for phase in ["El Nino", "Neutral", "La Nina"]:
        if phase in pc.columns:
            figp.add_trace(go.Bar(x=months, y=pc[phase], name=PRETTY[phase],
                                  marker_color=PHASE_COLORS[phase]))
    figp.update_layout(barmode="group", height=380,
                       yaxis_title="mean rainfall (mm/month)", margin=dict(t=10))
    st.plotly_chart(figp, use_container_width=True)
    st.caption("Compare the red (El Niño) bars against the grey (neutral) bars to "
               "read the month-by-month El Niño rainfall signal for this district. "
               "Data: CHIRPS satellite, 1981–present.")

    st.markdown("#### Seasonal El Niño signal & statistical test")
    cs = composite[composite["region"] == region].copy()
    cs["season"] = cs["season"].map(SEASON_LABELS)
    cs["95% CI (%)"] = cs.apply(
        lambda r: "n/a" if pd.isna(r["ci_low"]) else f"[{r['ci_low']:+.0f}, {r['ci_high']:+.0f}]",
        axis=1)
    cs["verdict"] = cs["q_value"].apply(
        lambda q: "significant" if q < 0.05 else "suggestive" if q < 0.10 else "not sig.")
    table = cs[["season", "n_elnino", "n_neutral", "precip_pct", "cohens_d",
                "q_value", "95% CI (%)", "verdict"]].rename(
        columns={"season": "Season", "n_elnino": "n El Niño yrs",
                 "n_neutral": "n neutral yrs", "precip_pct": "rain anom %",
                 "cohens_d": "Cohen's d", "q_value": "q (FDR)"})
    st.dataframe(table.round(3), use_container_width=True, hide_index=True)
    st.caption("Samples are **El Niño vs neutral *years*** (detrended seasonal "
               "totals), not months — so they are statistically independent. "
               "**q** is the Benjamini–Hochberg FDR-adjusted p-value across all "
               "district × season tests; Cohen's _d_ ≈ 0.2 small / 0.5 medium / "
               "0.8 large; the CI is a 2,000-sample bootstrap on the anomaly.")


# --------------------------------------------------------------------------- #
# Tab 5 — Sector impacts table + bars
# --------------------------------------------------------------------------- #
if page == "Sector Impacts":
    st.subheader("Sectoral impact scores — all 25 districts")
    st.warning("These 0–100 scores are a **transparent heuristic** that weights "
               "the significance-tested rainfall signals by each zone's "
               "sector relevance. They are **not** calibrated against observed "
               "drought, crop-yield, reservoir or flood records — use them to "
               "compare districts, not as absolute risk.", icon="⚠️")
    show = imp[["region", "zone", "direction", "confidence", "drought", "flood",
                "agriculture", "hydropower", "overall"]].copy()
    show["region"] = show["region"].map(_short)
    show = show.rename(columns={"region": "district"}).sort_values(
        "overall", ascending=False)
    st.dataframe(
        show.style.background_gradient(
            cmap="YlOrRd", subset=["drought", "flood", "agriculture", "hydropower", "overall"],
            vmin=0, vmax=100,
        ).format({**{c: "{:.0f}" for c in ["drought", "flood", "agriculture",
                                           "hydropower", "overall"]},
                  "confidence": "{:.2f}"}),
        use_container_width=True, height=500, hide_index=True,
    )
    st.caption("Risk scores 0–100 (higher = more adverse). *confidence* (0–1) is "
               "the statistical confidence of the district's dominant seasonal "
               "signal — low values mean the estimate is weakly constrained.")

    s = summary
    st.markdown(
        f"**National read-out.** During a typical El Niño, "
        f"{s['regions_drier']} of {len(imp)} districts trend **drier** and "
        f"{s['regions_wetter']} trend **wetter**. Mean sector risk — "
        f"drought {s['drought']:.0f}, flood {s['flood']:.0f}, "
        f"agriculture {s['agriculture']:.0f}, hydropower {s['hydropower']:.0f} (/100). "
        f"Highest overall exposure: **{', '.join(_short(r) for r in s['top_risk_regions'])}**."
    )

    long = imp.copy()
    long["name"] = long["region"].map(_short)
    long = long.melt(id_vars="name",
                     value_vars=["drought", "flood", "agriculture", "hydropower"],
                     var_name="sector", value_name="score").dropna()
    figb = px.bar(long.sort_values("name"), x="name", y="score", color="sector",
                  barmode="group", height=460, labels=dict(score="risk (0-100)", name=""))
    st.plotly_chart(figb, use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab 6 — Methodology
# --------------------------------------------------------------------------- #
if page == "Methods":
    st.markdown("### How this system works")
    st.markdown(
        """
**1. ENSO state (NOAA CPC).** The Oceanic Niño Index (ONI) — the three-month
running mean sea-surface-temperature anomaly in the Niño-3.4 region — is fetched
live. An ONI at or above **+0.5 °C** sustained over five overlapping seasons
defines El Niño. The ENSO Status tab adds an in-house **month-conditioned
statistical forecast** (persistence + tendency that respects ENSO's phase-locking
to a December peak, computed live from the ONI) and shows it next to the
**official NOAA CPC dynamical-model outlook** — fetched live — for direct
comparison.

**2. Satellite rainfall (UCSB CHIRPS).** Gridded satellite-and-station rainfall
from **CHIRPS v2.0** (~5 km, 1981–present) is clipped to Sri Lanka and zonally
averaged over all 25 district polygons (geoBoundaries ADM2). CHIRPS blends
thermal-infrared satellite estimates with gauge data, so it captures spatial
detail that sparse station networks miss.

**3. Event-based, detrended composites.** Rainfall is aggregated to **seasonal
totals per season-year** so the samples are ENSO *years* (~12 El Niño vs ~18
neutral), **not** autocorrelated months — this respects statistical
independence. Each district × season series is **linearly detrended** before
compositing, so a long-term trend cannot be mistaken for an ENSO signal. Each
season-year is then classified by its mean ONI.

**4. Significance testing with multiple-comparison control.** El Niño years are
compared with neutral years using **Welch's t-test** *and* the **Mann–Whitney U**
test, with **Cohen's _d_** and a **2,000-sample bootstrap 95 % CI**. Crucially,
**Benjamini–Hochberg FDR correction** is applied across all 100 district × season
tests — so "significant" (q < 0.05) accounts for the many tests run. A lagged
cross-correlation (ONI leading rainfall 0–6 months) locates the response delay.

**5. Canonical event framework (ENSO Events tab).** Discrete El Niño events are
detected (NOAA's five-overlapping-season rule) and aligned to their **developing
year (Y0) → decay year (Y1)** life cycle, then Sri Lanka's national rainfall is
composited in event-relative seasons. Events are also split into
**Eastern-Pacific (canonical)** vs **Central-Pacific (Modoki)** flavours from the
Niño-3 vs Niño-4 anomaly at the peak. This recovers the literature's significant
**Oct–Nov developing-year enhancement**, complementing the stricter district view.

**5b. Ocean drivers — IOD, La Niña & collinearity (Ocean Drivers tab).** The
**Indian Ocean Dipole** (HadISST Dipole Mode Index, NOAA PSL) is composited
alongside a symmetric **La Niña** analysis. Because the ONI and DMI are
themselves correlated (r ≈ 0.65), a simple composite cannot attribute the Oct–Nov
signal to one ocean — so we fit a **multiple regression** of seasonal rainfall on
*both* standardised indices, reporting each driver's **independent partial effect
and p-value**. In this record ENSO emerges as the significant independent driver
and the IOD's apparent effect is largely shared with ENSO — a conclusion only the
regression (not the composites) can support.

**6. Sector impact model — a transparent heuristic.** Risk magnitudes are driven
by the **FDR-confidence-damped effect size** of the physically causal season, so
a signal that fails multiple-comparison control contributes almost nothing. Each
sector takes its strongest relevant season (worst-case); direction uses a signed
weighted mean. **These 0–100 scores are *not* calibrated against observed
drought, yield, reservoir or flood records** — they are an interpretable
synthesis for comparing districts, not validated risk estimates.

**7. Spatial view.** Scores are mapped across Sri Lanka so the geographic
pattern is visible — highest overall exposure in the wet south-west and central
highlands, the largest drought (SPI) response in the northern dry zone.

**8. Drought index (SPI).** From the CHIRPS series we compute the WMO-recommended
**Standardized Precipitation Index** (SPI-3 / 6 / 12) by fitting a gamma
distribution per district and calendar month (McKee et al., 1993), giving both
the *current* drought state and the *El Niño* composite SPI per district.

**9. Satellite vegetation (MODIS).** On demand for a chosen district, NASA
**MODIS MOD13Q1** 250 m NDVI (2000–present) is retrieved from the ORNL DAAC and
converted to the **Vegetation Condition Index** (VCI; Kogan, 1995), then
composited by ENSO phase as a satellite agricultural-drought cross-check.

**10. Reporting.** The district table (impact scores, seasonal satellite anomalies
and current SPI) exports to **CSV** and a multi-page **PDF** report.

#### Key findings
- **First inter-monsoon (March–April) drying** survives FDR correction in
  **21 / 25 districts** (q ≈ 0.0002) — the most robust El Niño rainfall signal,
  appearing mainly in the event **decay year**.
- **Second inter-monsoon (October–November) enhancement** (+18 % nationally,
  p ≈ 0.003) is significant in the **event-aligned developing-year composite**,
  but only *suggestive* in the stricter per-district FDR test — both views are
  shown so the dependence on statistical framing is explicit.
- The **south-west and north-east monsoon signals are weak / not significant**;
  Eastern- and Central-Pacific events affect Sri Lanka similarly in this record.

#### Caveats & honest limitations
- The **sector impact scores are an un-validated heuristic** (not calibrated to
  observed droughts, yields, reservoirs or floods) — comparative, not absolute.
- CHIRPS (~5 km) smooths very local extremes and is weaker over the orographic
  central highlands; NDVI is confounded by irrigation and land-use.
- Districts are spatially correlated, so the count of "significant" districts is
  not 25 independent pieces of evidence.
- The linear ONI projection is a heuristic, **not** a dynamical forecast — for
  operational outlooks consult NOAA CPC and Sri Lanka's Department of Meteorology.
- Only ~12 El Niño events constrain the composite, and event "flavour" /
  multi-year persistence add residual uncertainty.

*Sources: NOAA Climate Prediction Center (ONI); UCSB Climate Hazards Center
(CHIRPS); NASA MODIS / ORNL DAAC (NDVI); geoBoundaries (district polygons).
Method informed by Zubair & Ropelewski (2006), McKee et al. (1993) and
Kogan (1995).*
        """
    )

# --------------------------------------------------------------------------- #
# Fixed footer (always visible, on every tab)
# --------------------------------------------------------------------------- #
st.markdown(f"""
<div class="enso-footer">
  <span><span class="hide-sm">🌊 El Niño × Sri Lanka · </span>
     <span class="muted">analytical aid — not a forecast ·
     CHIRPS {panel['date'].min():%Y}–{panel['date'].max():%Y}</span></span>
  <span>Developed by
     <a href="https://dineshmadhushanka.vercel.app/" target="_blank">Dinesh Madhushanka ↗</a></span>
</div>
""", unsafe_allow_html=True)
