# El Niño → Sri Lanka Impact Analysis System

An interactive, **data-driven** dashboard that detects the current/upcoming
El Niño state and quantifies — with statistical significance testing — how it is
likely to affect Sri Lanka **across regions and sectors**.

## What it does

1. **Detects ENSO state** from NOAA CPC's live Oceanic Niño Index (ONI), and
   flags a *developing* El Niño from the recent trend before it crosses +0.5 °C.
2. **Builds empirical composites** of rainfall/temperature for 10 locations
   spanning Sri Lanka's climatic zones, using Open-Meteo ERA5 data since 1970.
3. **Significance-tests** every region × monsoon-season signal (Welch t-test,
   Mann–Whitney U, Cohen's *d* effect size, bootstrap 95% CI) and runs a lagged
   ONI→rainfall cross-correlation.
4. **Scores four sectors** — drought/water-deficit, floods & landslides,
   agriculture (paddy/tea), and water/hydropower — driven by the *significant*
   signals, so noise can't manufacture risk.
5. **Maps it spatially** — point map + 25-district choropleth (CHIRPS satellite).
6. **Drought index (SPI-3/6/12)** computed per district from CHIRPS, giving the
   current drought state and the El Niño composite.
7. **MODIS satellite vegetation** (NDVI → VCI) per district, on demand.
8. **Exports** a CSV + multi-page PDF district report.

## Headline empirical finding

For Sri Lanka, El Niño most robustly produces a **drier First Inter-monsoon
(Mar–Apr)** island-wide (large, highly significant effect) and a **wetter Second
Inter-monsoon (Oct–Nov)** in the south-west and central highlands (flood window).
The south-west and north-east monsoon ENSO signals are weak/insignificant at
these grid points — the dashboard reflects this honestly.

## Setup

A dedicated conda environment is used:

```bash
conda create -y -n elnino-lk python=3.12
conda activate elnino-lk
pip install -r requirements.txt
```

## Run

```bash
./run.sh          # or:  streamlit run app.py
```

Then open <http://localhost:8501>. First load fetches live data (cached to
`.cache/` for 6 h); later loads are instant.

## Layout

| File | Purpose |
|------|---------|
| `app.py` | Streamlit dashboard (5 tabs incl. spatial map) |
| `enso_lk/enso.py` | ONI fetch/parse, phase classification, developing-El-Niño detection |
| `enso_lk/weather.py` | Open-Meteo ERA5 fetch (with backoff/throttle) → monthly frame |
| `enso_lk/analysis.py` | Climatology, anomalies, El-Niño composites, lag analysis |
| `enso_lk/stats.py` | Significance tests, effect size, bootstrap CI, lag correlation |
| `enso_lk/impacts.py` | Significance-aware sector impact model |
| `enso_lk/districts.py` | CHIRPS satellite download + district zonal statistics |
| `enso_lk/spi.py` | Standardized Precipitation Index (drought) |
| `enso_lk/vegetation.py` | MODIS NDVI → VCI (satellite vegetation health) |
| `enso_lk/report.py` | CSV + PDF report generation |
| `enso_lk/config.py` | Regions, districts, monsoon seasons, ENSO thresholds |

## Caveats

ERA5 grid-point rainfall smooths local extremes; the ONI projection is a simple
linear heuristic, **not** a dynamical forecast. For operational outlooks consult
NOAA CPC and Sri Lanka's Department of Meteorology. This is an analytical aid.

*Sources: NOAA Climate Prediction Center; Open-Meteo (ERA5); ENSO–Sri Lanka
rainfall relationships after Zubair & Ropelewski (2006).*
