# GTFS Disruption Early Warning Dashboard

Production-ready **Streamlit** conversion of the React GTFS disruption dashboard.  
Provides a **30-minute prediction horizon** for transit disruptions with a dark-themed command-center UI.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the dashboard
streamlit run app.py
```

Opens at **http://localhost:8501** by default.

---

## Pages

| Page | Description |
|------|-------------|
| **Overview** | KPI summary (Service Delivered, On-Time %, Active Disruptions), route status table, severity distribution bar chart |
| **Live Map** | Plotly Scattermapbox with routes colour-coded by predicted severity; filter by Normal/Disrupted |
| **Alerts** | Real-time alert cards sorted by severity & confidence; filter by severity level |
| **Predictions** | Severity pie/bar chart, confidence histogram, top-N route bar chart, detail table |
| **Analytics** | Time-series trend lines, bunching vs delay scatter, on-time % by route, fleet utilisation histogram, summary stats |
| **Settings** | Data source, ML model type, confidence threshold, auto-refresh interval, severity filters |

---

## Connecting a Real GTFS Feed

1. Go to **Settings → Data Source → GTFS Feed**
2. Enter your agency's GTFS Realtime URL, e.g.:
   - `https://api.agency.com/gtfs-realtime/vehicle_positions.pb`
3. Extend `generate_route_data()` in `app.py` with a real fetch + protobuf decode call.

Public directories: [Transitland](https://www.transit.land/), [OpenMobilityData](https://openmobilitydata.org/)

---

## Severity Levels

| Level | Bunching Index | Colour |
|-------|---------------|--------|
| NORMAL | < 0.30 | 🟢 Green |
| MINOR | 0.30–0.60 | 🔵 Blue |
| MODERATE | 0.60–0.80 | 🟠 Orange |
| SEVERE | > 0.80 | 🔴 Red |

---

## Architecture

```
app.py                  ← Single-file Streamlit app
├── Data layer          generate_route_data(), compute_predictions(), generate_alerts()
├── KPI layer           compute_kpis()
├── Component helpers   metric_card(), severity_badge(), section_header()
├── Pages               page_overview/live_map/alerts/predictions/analytics/settings()
└── Router              st.session_state["page"]
```

All data is generated deterministically via a `seed` integer that increments on each auto-refresh, giving realistic live-data behaviour without an external dependency.

---

## Customisation

| Want to… | Edit… |
|----------|-------|
| Change refresh interval defaults | `init_state()` → `refresh_interval` |
| Add a new KPI card | `compute_kpis()` + `page_overview()` |
| Swap demo data for real feed | Replace body of `generate_route_data()` |
| Add a new page | Define `page_mypage()` and add to sidebar + router |
| Adjust severity thresholds | `compute_predictions()` → `severity()` inner function |
