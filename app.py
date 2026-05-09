"""
GTFS Disruption Dashboard — 30-Minute Early Warning System
Production-ready Streamlit conversion of the React prototype.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json
from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Optional

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GTFS Disruption Dashboard",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME / CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  html, body, [data-testid="stApp"] {
    background-color: #0a1628;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
  }
  [data-testid="stSidebar"] {
    background-color: #06101e;
    border-right: 1px solid #1e3a5f;
  }
  [data-testid="stSidebar"] * { color: #94a3b8 !important; }
  [data-testid="stSidebar"] .sidebar-route-active { color: #00d9ff !important; }

  /* ── Metric cards ── */
  .metric-card {
    background: #0d1f35;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
  }
  .metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
  }
  .metric-card.cyan::before  { background: linear-gradient(90deg, #00d9ff, #0080ff); }
  .metric-card.green::before { background: linear-gradient(90deg, #2ed573, #00b894); }
  .metric-card.red::before   { background: linear-gradient(90deg, #ff4757, #c0392b); }
  .metric-card.orange::before{ background: linear-gradient(90deg, #ffa502, #e17055); }
  .metric-card.blue::before  { background: linear-gradient(90deg, #1e90ff, #0052cc); }

  .metric-label { font-size: 11px; font-family: 'IBM Plex Mono', monospace;
                  text-transform: uppercase; letter-spacing: .1em; color: #64748b; }
  .metric-value { font-size: 32px; font-weight: 700; font-family: 'IBM Plex Mono', monospace;
                  line-height: 1.1; margin: 6px 0 4px; }
  .metric-delta { font-size: 12px; font-family: monospace; }
  .metric-delta.pos { color: #2ed573; }
  .metric-delta.neg { color: #ff4757; }

  .cyan-text   { color: #00d9ff !important; }
  .green-text  { color: #2ed573 !important; }
  .red-text    { color: #ff4757 !important; }
  .orange-text { color: #ffa502 !important; }
  .blue-text   { color: #1e90ff !important; }

  /* ── Severity badges ── */
  .badge { display:inline-block; padding:3px 10px; border-radius:9999px;
           font-size:11px; font-weight:700; font-family:monospace; letter-spacing:.05em; }
  .badge-normal   { background:#2ed57322; color:#2ed573; border:1px solid #2ed573; }
  .badge-minor    { background:#1e90ff22; color:#1e90ff; border:1px solid #1e90ff; }
  .badge-moderate { background:#ffa50222; color:#ffa502; border:1px solid #ffa502; }
  .badge-severe   { background:#ff475722; color:#ff4757; border:1px solid #ff4757; }

  /* ── Alert cards ── */
  .alert-card {
    background:#0d1f35; border:1px solid #1e3a5f; border-radius:8px;
    padding:14px 18px; margin-bottom:10px;
  }
  .alert-card.severe   { border-left:4px solid #ff4757; }
  .alert-card.moderate { border-left:4px solid #ffa502; }
  .alert-card.minor    { border-left:4px solid #1e90ff; }

  /* ── Section headers ── */
  .section-header {
    font-size:18px; font-weight:700; font-family:'IBM Plex Mono', monospace;
    color:#00d9ff; margin-bottom:12px; margin-top:4px;
  }

  /* ── Tables ── */
  .data-table { width:100%; border-collapse:collapse; font-size:13px; }
  .data-table th { background:#0d1f35; color:#00d9ff; font-family:monospace;
                   padding:10px 14px; text-align:left; border-bottom:1px solid #1e3a5f; }
  .data-table td { padding:9px 14px; border-bottom:1px solid #1e3a5f; color:#cbd5e1; }
  .data-table tr:hover td { background:#1e3a5f33; }
  .data-table .mono { font-family:monospace; color:#00d9ff; }

  /* ── Divider ── */
  hr { border-color:#1e3a5f !important; }

  /* ── Streamlit overrides ── */
  div[data-testid="stHorizontalBlock"] > div { gap:16px; }
  .stSelectbox > label, .stSlider > label, .stNumberInput > label,
  .stTextInput > label, .stMultiSelect > label { color:#64748b !important;
    font-size:11px; font-family:monospace; text-transform:uppercase; letter-spacing:.08em; }
  div[data-baseweb="select"] > div { background:#0d1f35 !important; border-color:#1e3a5f !important; }
  div[data-baseweb="select"] span { color:#e2e8f0 !important; }
  .stSlider > div > div > div { background: #00d9ff !important; }

  /* ── Sidebar active nav ── */
  .nav-item { padding:8px 12px; border-radius:6px; font-size:13px; cursor:pointer;
              font-family:monospace; transition:all .2s; }
  .nav-item:hover { background:#1e3a5f44; color:#00d9ff !important; }
  .nav-item.active { background:#00d9ff18; color:#00d9ff !important; border-left:2px solid #00d9ff; }

  /* ── Pulse animation for live indicator ── */
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .live-dot { display:inline-block; width:8px; height:8px; border-radius:50%;
              background:#2ed573; animation:pulse 1.5s infinite; margin-right:6px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────
class SeverityLevel(IntEnum):
    NORMAL   = 0
    MINOR    = 1
    MODERATE = 2
    SEVERE   = 3

SEVERITY_CONFIG = {
    SeverityLevel.NORMAL:   {"label": "NORMAL",   "color": "#2ed573", "cls": "badge-normal"},
    SeverityLevel.MINOR:    {"label": "MINOR",     "color": "#1e90ff", "cls": "badge-minor"},
    SeverityLevel.MODERATE: {"label": "MODERATE",  "color": "#ffa502", "cls": "badge-moderate"},
    SeverityLevel.SEVERE:   {"label": "SEVERE",    "color": "#ff4757", "cls": "badge-severe"},
}

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0a1628",
    font=dict(color="#94a3b8", family="IBM Plex Mono, monospace", size=11),
    xaxis=dict(gridcolor="#1e3a5f", linecolor="#1e3a5f", tickfont=dict(color="#64748b")),
    yaxis=dict(gridcolor="#1e3a5f", linecolor="#1e3a5f", tickfont=dict(color="#64748b")),
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
)


# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────
@st.cache_data(ttl=30)
def generate_route_data(n_routes: int = 25, seed: int = None) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    now = datetime.utcnow()
    rows = []
    for i in range(1, n_routes + 1):
        bi = rng.random()
        base_delay = rng.random() * 30
        rows.append({
            "route_id": f"Route_{i:03d}",
            "timestamp": now,
            "speed_mean": 20 + rng.random() * 30,
            "speed_std": 2 + rng.random() * 13,
            "delay_mean_5m": rng.random() * 120,
            "delay_mean_15m": base_delay + rng.random() * 180,
            "delay_mean_30m": base_delay + rng.random() * 240,
            "bunching_index": bi,
            "on_time_pct": 0.6 + rng.random() * 0.4,
            "headway_variance": 5 + rng.random() * 55,
            "alert_nlp_score": rng.random() * 0.5,
            "alert_count": int(rng.random() * 5),
            "fleet_utilization": 0.7 + rng.random() * 0.3,
            "lat": 52.0 + rng.random() * 0.5,
            "lon": 4.5 + rng.random() * 1.5,
            "disruption_type": "DISRUPTION" if bi > 0.6 else "ON_TIME",
            "prediction_30min": rng.random() * 3,
        })
    return pd.DataFrame(rows)


def compute_predictions(df: pd.DataFrame, confidence_threshold: float = 0.5) -> pd.DataFrame:
    def severity(bi):
        if bi < 0.3: return SeverityLevel.NORMAL
        if bi < 0.6: return SeverityLevel.MINOR
        if bi < 0.8: return SeverityLevel.MODERATE
        return SeverityLevel.SEVERE

    rng = np.random.RandomState(42)
    preds = df.copy()
    preds["severity_class"] = preds["bunching_index"].apply(severity)
    preds["confidence"] = 0.75 + rng.random(len(preds)) * 0.2
    preds["predicted_delay_min"] = preds["delay_mean_15m"] * (1 + rng.random(len(preds)) * 0.3)
    preds["lead_time_min"] = 10 + rng.random(len(preds)) * 20
    preds["severity_label"] = preds["severity_class"].map(lambda s: SEVERITY_CONFIG[s]["label"])
    preds["severity_color"] = preds["severity_class"].map(lambda s: SEVERITY_CONFIG[s]["color"])
    return preds


def generate_alerts(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    alert_id = 1
    for _, p in preds[preds["severity_class"] > SeverityLevel.NORMAL].iterrows():
        sev = SeverityLevel(p["severity_class"])
        rows.append({
            "id": f"ALT-{alert_id:04d}",
            "route_id": p["route_id"],
            "severity": sev,
            "severity_label": SEVERITY_CONFIG[sev]["label"],
            "severity_color": SEVERITY_CONFIG[sev]["color"],
            "severity_cls": SEVERITY_CONFIG[sev]["cls"].replace("badge-", ""),
            "confidence": p["confidence"],
            "predicted_delay_min": p["predicted_delay_min"],
            "lead_time_min": p["lead_time_min"],
            "message": f"Disruption detected on {p['route_id']} — {SEVERITY_CONFIG[sev]['label']} severity",
            "timestamp": datetime.utcnow() - timedelta(minutes=int(p["lead_time_min"])),
            "resolved": False,
        })
        alert_id += 1
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not df.empty:
        df = df.sort_values(["severity", "confidence"], ascending=[False, False]).reset_index(drop=True)
    return df


def compute_kpis(routes: pd.DataFrame, preds: pd.DataFrame) -> dict:
    avg_delay = routes["delay_mean_15m"].mean()
    active_disruptions = int((preds["severity_class"] > SeverityLevel.NORMAL).sum())
    avg_on_time = routes["on_time_pct"].mean() * 100
    avg_confidence = preds["confidence"].mean() * 100
    return {
        "service_delivered": round(96.5 + (np.random.random() - 0.5) * 3, 1),
        "on_time_performance": round(avg_on_time, 1),
        "data_quality_score": round(94.0 + (np.random.random() - 0.5) * 2, 1),
        "active_disruptions": active_disruptions,
        "throughput": int(np.random.random() * 5000 + 1000),
        "incident_response_time": round(avg_delay, 0),
        "travel_time_index": round(1.3 + (np.random.random() - 0.5) * 0.4, 2),
        "congestion_level": round(avg_delay / 10, 1),
        "route_efficiency": round(85 + np.random.random() * 10, 1),
        "prediction_confidence": round(avg_confidence, 1),
        "avg_lead_time": round(15 + np.random.random() * 15, 1),
        "total_routes": len(routes),
        "normal_count": int((preds["severity_class"] == SeverityLevel.NORMAL).sum()),
    }


def generate_timeseries(metric: str = "delay_mean_15m", hours: int = 72, seed=7) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    now = datetime.utcnow()
    ts = [now - timedelta(hours=h) for h in range(hours, -1, -1)]
    if metric == "delay_mean_15m":
        vals = 10 + rng.random(len(ts)) * 25
    elif metric == "bunching_index":
        vals = 0.2 + rng.random(len(ts)) * 0.6
    else:
        vals = 70 + rng.random(len(ts)) * 25
    return pd.DataFrame({"timestamp": ts, "value": vals})


# ─────────────────────────────────────────────
# COMPONENT HELPERS
# ─────────────────────────────────────────────
def metric_card(label: str, value, unit: str = "", delta: str = "", color: str = "cyan"):
    delta_cls = "pos" if delta.startswith("+") else "neg" if delta.startswith("-") else ""
    delta_html = f'<div class="metric-delta {delta_cls}">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card {color}">
      <div class="metric-label">{label}</div>
      <div class="metric-value {color}-text">{value}{f'<span style="font-size:16px;color:#64748b"> {unit}</span>' if unit else ''}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


def severity_badge(sev: SeverityLevel) -> str:
    cfg = SEVERITY_CONFIG[sev]
    return f'<span class="badge {cfg["cls"]}">{cfg["label"]}</span>'


def section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "page": "Overview",
        "confidence_threshold": 0.5,
        "model_type": "RandomForest",
        "data_source": "Demo Data",
        "n_routes": 25,
        "auto_refresh": True,
        "refresh_interval": 30,
        "severity_filter": ["MINOR", "MODERATE", "SEVERE"],
        "last_refresh": time.time(),
        "seed": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─────────────────────────────────────────────
# AUTO-REFRESH
# ─────────────────────────────────────────────
if st.session_state["auto_refresh"]:
    elapsed = time.time() - st.session_state["last_refresh"]
    if elapsed > st.session_state["refresh_interval"]:
        st.session_state["seed"] += 1
        st.session_state["last_refresh"] = time.time()


# ─────────────────────────────────────────────
# LOAD DATA (cached per seed)
# ─────────────────────────────────────────────
routes_df  = generate_route_data(st.session_state["n_routes"], seed=st.session_state["seed"])
preds_df   = compute_predictions(routes_df, st.session_state["confidence_threshold"])
alerts_df  = generate_alerts(preds_df)
kpis       = compute_kpis(routes_df, preds_df)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 20px">
      <div style="font-size:20px;font-weight:800;font-family:'IBM Plex Mono',monospace;color:#00d9ff;">
        🚌 GTFS Watch
      </div>
      <div style="font-size:11px;color:#334155;margin-top:2px;">Early Warning Dashboard</div>
    </div>""", unsafe_allow_html=True)

    pages = ["Overview", "Live Map", "Alerts", "Predictions", "Analytics", "Settings"]
    page_icons = {"Overview":"📊","Live Map":"🗺️","Alerts":"🚨","Predictions":"🔮","Analytics":"📈","Settings":"⚙️"}

    for p in pages:
        active = "active" if st.session_state["page"] == p else ""
        if st.button(f"{page_icons[p]}  {p}", key=f"nav_{p}",
                     use_container_width=True,
                     type="primary" if active else "secondary"):
            st.session_state["page"] = p
            st.rerun()

    st.markdown("---")

    # Live status
    n_disruptions = kpis["active_disruptions"]
    status_color = "#ff4757" if n_disruptions > 5 else "#ffa502" if n_disruptions > 2 else "#2ed573"
    st.markdown(f"""
    <div style="padding:10px 0">
      <div style="font-size:10px;color:#334155;font-family:monospace;text-transform:uppercase;margin-bottom:6px;">
        System Status
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span class="live-dot" style="background:{status_color};"></span>
        <span style="font-size:13px;color:{status_color};font-family:monospace;font-weight:600;">
          {'ALERT' if n_disruptions > 5 else 'WARNING' if n_disruptions > 2 else 'NORMAL'}
        </span>
      </div>
      <div style="margin-top:8px;font-size:11px;color:#475569;">
        {n_disruptions} active disruption{'s' if n_disruptions!=1 else ''}
      </div>
      <div style="font-size:10px;color:#334155;font-family:monospace;margin-top:4px;">
        {datetime.utcnow().strftime('%H:%M:%S UTC')}
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.session_state["seed"] += 1
        st.session_state["last_refresh"] = time.time()
        st.rerun()


# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────
def page_overview():
    st.markdown('<h1 style="font-size:32px;font-weight:800;font-family:IBM Plex Mono,monospace;color:#00d9ff;margin-bottom:4px;">System Overview</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569;margin-bottom:24px;">Real-time transit disruption monitoring · 30-min early warning</p>', unsafe_allow_html=True)

    section_header("Digital Boardroom — KPI Overview")
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Service Delivered", f"{kpis['service_delivered']}", "%", "-1.5%", "cyan")
    with c2: metric_card("On-Time Performance", f"{kpis['on_time_performance']}", "%", "-1.8%", "blue")
    with c3: metric_card("Data Quality Score", f"{kpis['data_quality_score']}", "%", "+0.5%", "green")
    with c4: metric_card("Active Disruptions", kpis["active_disruptions"], "", f"+{np.random.randint(1,4)} new", "red")

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Traffic Management KPIs")
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: metric_card("Throughput", f"{kpis['throughput']:,}", "veh/hr", "", "cyan")
    with c2: metric_card("Incident Response", f"{kpis['incident_response_time']:.0f}", "min", "", "orange")
    with c3: metric_card("Travel Time Index", f"{kpis['travel_time_index']:.2f}", "", "", "blue")
    with c4: metric_card("Congestion Level", f"{kpis['congestion_level']:.1f}", "min", "", "orange")
    with c5: metric_card("Route Efficiency", f"{kpis['route_efficiency']:.1f}", "%", "", "green")

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Current Status")
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Total Routes", kpis["total_routes"], "", "", "cyan")
    with c2: metric_card("Normal Operations", kpis["normal_count"], "", "", "green")
    with c3: metric_card("Active Disruptions", kpis["active_disruptions"], "", "", "red")
    with c4: metric_card("Avg Confidence", f"{kpis['prediction_confidence']:.0f}", "%", "", "blue")

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Route Status — Top 15")

    display = preds_df[["route_id","delay_mean_15m","bunching_index","severity_class","confidence","on_time_pct"]].head(15).copy()
    badges = display["severity_class"].apply(lambda s: severity_badge(SeverityLevel(int(s))))
    table_rows = ""
    for _, r in display.iterrows():
        sev = SeverityLevel(int(r["severity_class"]))
        badge = severity_badge(sev)
        table_rows += f"""
        <tr>
          <td class="mono">{r['route_id']}</td>
          <td>{r['delay_mean_15m']:.0f}s</td>
          <td>{r['bunching_index']:.2f}</td>
          <td>{badge}</td>
          <td>{r['confidence']*100:.0f}%</td>
          <td>{r['on_time_pct']*100:.0f}%</td>
        </tr>"""

    st.markdown(f"""
    <table class="data-table">
      <thead><tr>
        <th>Route ID</th><th>Delay (15m)</th><th>Bunching Index</th>
        <th>Severity</th><th>Confidence</th><th>On-Time %</th>
      </tr></thead>
      <tbody>{table_rows}</tbody>
    </table>""", unsafe_allow_html=True)

    # Severity mini bar chart
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Severity Distribution")
    counts = preds_df["severity_class"].value_counts().sort_index()
    labels = [SEVERITY_CONFIG[SeverityLevel(i)]["label"] for i in counts.index]
    colors = [SEVERITY_CONFIG[SeverityLevel(i)]["color"] for i in counts.index]
    fig = go.Figure(go.Bar(x=labels, y=counts.values, marker_color=colors,
                           text=counts.values, textposition="outside",
                           textfont=dict(color="#e2e8f0")))
    fig.update_layout(**PLOTLY_TEMPLATE, height=220, title=None)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: LIVE MAP
# ─────────────────────────────────────────────
def page_live_map():
    st.markdown('<h1 style="font-size:32px;font-weight:800;font-family:IBM Plex Mono,monospace;color:#00d9ff;margin-bottom:4px;">Live Map</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569;margin-bottom:20px;">Real-time route positions coloured by predicted severity</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        show_normal = st.checkbox("Show Normal", value=True)
    with c2:
        show_disrupted = st.checkbox("Show Disrupted", value=True)
    with c3:
        map_style = st.selectbox("Map style", ["carto-darkmatter", "open-street-map", "stamen-terrain"])

    filtered = preds_df.copy()
    if not show_normal:
        filtered = filtered[filtered["severity_class"] > SeverityLevel.NORMAL]
    if not show_disrupted:
        filtered = filtered[filtered["severity_class"] == SeverityLevel.NORMAL]

    fig = go.Figure()

    for sev in [SeverityLevel.SEVERE, SeverityLevel.MODERATE, SeverityLevel.MINOR, SeverityLevel.NORMAL]:
        sub = filtered[filtered["severity_class"] == int(sev)]
        if sub.empty:
            continue
        cfg = SEVERITY_CONFIG[sev]
        hover = (
            "<b>" + sub["route_id"] + "</b><br>" +
            "Severity: " + sub["severity_label"] + "<br>" +
            "Delay 15m: " + sub["delay_mean_15m"].round(0).astype(str) + "s<br>" +
            "Bunching: " + sub["bunching_index"].round(2).astype(str) + "<br>" +
            "Confidence: " + (sub["confidence"]*100).round(0).astype(str) + "%"
        )
        fig.add_trace(go.Scattermapbox(
            lat=sub["lat"], lon=sub["lon"],
            mode="markers",
            marker=dict(size=12 + int(sev)*4, color=cfg["color"], opacity=0.85),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            name=cfg["label"],
        ))

    fig.update_layout(
        mapbox=dict(style=map_style, center=dict(lat=52.1326, lon=5.2913), zoom=9),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=540,
        legend=dict(bgcolor="#0d1f35", font=dict(color="#94a3b8"), bordercolor="#1e3a5f", borderwidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:8px;">
      {''.join([f'<span><span class="badge {SEVERITY_CONFIG[s]["cls"]}">{SEVERITY_CONFIG[s]["label"]}</span> {int((preds_df["severity_class"]==int(s)).sum())} routes</span>' for s in SeverityLevel])}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: ALERTS
# ─────────────────────────────────────────────
def page_alerts():
    st.markdown('<h1 style="font-size:32px;font-weight:800;font-family:IBM Plex Mono,monospace;color:#00d9ff;margin-bottom:4px;">Active Alerts</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569;margin-bottom:20px;">Real-time disruption alerts sorted by severity and confidence</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        sort_by = st.selectbox("Sort By", ["Severity", "Confidence", "Route", "Lead Time"])
    with c2:
        sev_filter = st.selectbox("Filter Severity", ["All", "MINOR", "MODERATE", "SEVERE"])
    with c3:
        metric_card("Active Alerts", len(alerts_df) if not alerts_df.empty else 0, "", "", "red")

    if alerts_df.empty:
        st.markdown("""
        <div style="background:#0d1f35;border:1px solid #1e3a5f;border-radius:10px;
                    padding:48px;text-align:center;margin-top:16px;">
          <div style="font-size:48px;">✅</div>
          <div style="color:#2ed573;font-size:18px;font-weight:700;font-family:monospace;margin-top:12px;">
            No Active Alerts
          </div>
          <div style="color:#475569;font-size:13px;margin-top:6px;">All systems operating normally</div>
        </div>""", unsafe_allow_html=True)
        return

    df = alerts_df.copy()
    if sev_filter != "All":
        df = df[df["severity_label"] == sev_filter]
    if sort_by == "Confidence":
        df = df.sort_values("confidence", ascending=False)
    elif sort_by == "Route":
        df = df.sort_values("route_id")
    elif sort_by == "Lead Time":
        df = df.sort_values("lead_time_min")

    st.markdown(f"<br><p style='color:#64748b;font-size:12px;font-family:monospace;'>{len(df)} alert{'s' if len(df)!=1 else ''} shown</p>", unsafe_allow_html=True)

    for _, row in df.iterrows():
        sev_cls = row["severity_cls"]
        ts_str = row["timestamp"].strftime("%H:%M:%S UTC")
        st.markdown(f"""
        <div class="alert-card {sev_cls}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;">
            <div>
              <span class="badge badge-{sev_cls}">{row['severity_label']}</span>
              <span style="font-family:monospace;color:#00d9ff;font-size:13px;margin-left:10px;font-weight:600;">
                {row['route_id']}
              </span>
              <span style="font-size:11px;color:#475569;margin-left:8px;">{row['id']}</span>
            </div>
            <div style="text-align:right;">
              <div style="font-size:11px;color:#475569;font-family:monospace;">🕐 {ts_str}</div>
            </div>
          </div>
          <div style="margin-top:10px;color:#cbd5e1;font-size:13px;">{row['message']}</div>
          <div style="margin-top:10px;display:flex;gap:24px;">
            <div>
              <div style="font-size:10px;color:#475569;font-family:monospace;text-transform:uppercase;">Confidence</div>
              <div style="font-size:15px;font-weight:700;font-family:monospace;color:#00d9ff;">{row['confidence']*100:.0f}%</div>
            </div>
            <div>
              <div style="font-size:10px;color:#475569;font-family:monospace;text-transform:uppercase;">Predicted Delay</div>
              <div style="font-size:15px;font-weight:700;font-family:monospace;color:{row['severity_color']};">{row['predicted_delay_min']:.0f} min</div>
            </div>
            <div>
              <div style="font-size:10px;color:#475569;font-family:monospace;text-transform:uppercase;">Lead Time</div>
              <div style="font-size:15px;font-weight:700;font-family:monospace;color:#ffa502;">{row['lead_time_min']:.0f} min</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: PREDICTIONS
# ─────────────────────────────────────────────
def page_predictions():
    st.markdown('<h1 style="font-size:32px;font-weight:800;font-family:IBM Plex Mono,monospace;color:#00d9ff;margin-bottom:4px;">Prediction Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569;margin-bottom:20px;">ML model predictions and confidence metrics — 30-min horizon</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        chart_type = st.selectbox("Chart Type", ["Histogram", "Bar", "Scatter"])
    with c2:
        top_n = st.slider("Top N Routes", 5, 50, 20, step=5)
    with c3:
        conf_thresh = st.slider("Confidence Threshold", 0, 100,
                                int(st.session_state["confidence_threshold"]*100), step=5)
        st.session_state["confidence_threshold"] = conf_thresh / 100

    # Summary KPIs
    total_disrupt = int((preds_df["severity_class"] > SeverityLevel.NORMAL).sum())
    est_impact = int(np.random.random() * 40000 + 10000)
    avg_lead = preds_df["lead_time_min"].mean()
    avg_conf = preds_df["confidence"].mean() * 100

    cols = st.columns(4)
    with cols[0]: metric_card("Avg Lead Time", f"{avg_lead:.1f}", "min", "", "cyan")
    with cols[1]: metric_card("Avg Confidence", f"{avg_conf:.0f}", "%", "", "blue")
    with cols[2]: metric_card("Total Disruptions", total_disrupt, "", "", "red")
    with cols[3]: metric_card("Est. Passengers Affected", f"{est_impact:,}", "", "", "orange")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    # Severity distribution
    with col_l:
        section_header("Severity Distribution")
        counts = preds_df["severity_class"].value_counts().sort_index()
        labels = [SEVERITY_CONFIG[SeverityLevel(i)]["label"] for i in counts.index]
        colors = [SEVERITY_CONFIG[SeverityLevel(i)]["color"] for i in counts.index]
        if chart_type == "Bar":
            fig = go.Figure(go.Bar(x=labels, y=counts.values, marker_color=colors,
                                   text=counts.values, textposition="outside"))
        else:
            fig = go.Figure(go.Pie(labels=labels, values=counts.values, hole=0.45,
                                   marker_colors=colors,
                                   textinfo="label+percent",
                                   textfont=dict(color="#e2e8f0", size=11)))
        fig.update_layout(**PLOTLY_TEMPLATE, height=280)
        st.plotly_chart(fig, use_container_width=True)

    # Confidence histogram
    with col_r:
        section_header("Confidence Distribution")
        filtered = preds_df[preds_df["confidence"] >= conf_thresh/100]
        fig = go.Figure(go.Histogram(
            x=filtered["confidence"] * 100,
            nbinsx=20,
            marker_color="#00d9ff",
            opacity=0.75,
        ))
        fig.add_vline(x=conf_thresh, line_dash="dash", line_color="#ffa502",
                      annotation_text=f"Threshold: {conf_thresh}%",
                      annotation_font_color="#ffa502")
        fig.update_layout(**PLOTLY_TEMPLATE, height=280,
                          xaxis_title="Confidence %", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

    # Prediction table
    st.markdown("<br>", unsafe_allow_html=True)
    section_header(f"Top {top_n} Route Predictions")
    top = preds_df.nlargest(top_n, "bunching_index")[
        ["route_id","severity_label","confidence","predicted_delay_min","lead_time_min","bunching_index"]
    ].copy()

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=top["route_id"], y=top["predicted_delay_min"],
        marker_color=top["severity_label"].map({
            "NORMAL":"#2ed573","MINOR":"#1e90ff","MODERATE":"#ffa502","SEVERE":"#ff4757"
        }),
        text=top["predicted_delay_min"].round(0).astype(int).astype(str) + "m",
        textposition="outside", textfont=dict(color="#e2e8f0"),
        name="Predicted Delay",
    ))
    fig2.update_layout(**PLOTLY_TEMPLATE, height=240,
                       xaxis_title="Route", yaxis_title="Predicted Delay (min)")
    st.plotly_chart(fig2, use_container_width=True)

    # Detail table
    rows_html = ""
    for _, r in top.iterrows():
        sev = SEVERITY_CONFIG[[s for s in SeverityLevel if SEVERITY_CONFIG[s]["label"]==r["severity_label"]][0]]
        rows_html += f"""<tr>
          <td class="mono">{r['route_id']}</td>
          <td><span class="badge {sev['cls']}">{r['severity_label']}</span></td>
          <td>{r['confidence']*100:.0f}%</td>
          <td>{r['predicted_delay_min']:.0f} min</td>
          <td>{r['lead_time_min']:.0f} min</td>
          <td>{r['bunching_index']:.2f}</td>
        </tr>"""

    st.markdown(f"""
    <table class="data-table">
      <thead><tr>
        <th>Route</th><th>Severity</th><th>Confidence</th>
        <th>Predicted Delay</th><th>Lead Time</th><th>Bunching Index</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: ANALYTICS
# ─────────────────────────────────────────────
def page_analytics():
    st.markdown('<h1 style="font-size:32px;font-weight:800;font-family:IBM Plex Mono,monospace;color:#00d9ff;margin-bottom:4px;">Performance Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569;margin-bottom:20px;">Historical trends and performance metrics</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        metric = st.selectbox("Metric", ["delay_mean_15m", "bunching_index", "on_time_pct"],
                              format_func=lambda x: {"delay_mean_15m":"Delay (15m)","bunching_index":"Bunching Index","on_time_pct":"On-Time %"}[x])
    with c2:
        agg = st.selectbox("Aggregation", ["Mean", "Median", "Max"])
    with c3:
        time_range = st.selectbox("Time Range", ["7 days", "30 days", "90 days"])

    hours = {"7 days": 168, "30 days": 720, "90 days": 2160}[time_range]
    ts_df = generate_timeseries(metric, hours=hours)

    col_l, col_r = st.columns(2)

    with col_l:
        section_header(f"{metric.replace('_',' ').title()} Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts_df["timestamp"], y=ts_df["value"],
            mode="lines", line=dict(color="#00d9ff", width=1.5),
            fill="tozeroy", fillcolor="rgba(0,217,255,0.08)",
            name=metric,
        ))
        if hours <= 168:
            # Rolling average
            roll = ts_df["value"].rolling(6).mean()
            fig.add_trace(go.Scatter(
                x=ts_df["timestamp"], y=roll,
                mode="lines", line=dict(color="#ffa502", width=2, dash="dash"),
                name="6h avg",
            ))
        fig.update_layout(**PLOTLY_TEMPLATE, height=280)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        section_header("Delay vs Bunching Index (Scatter)")
        fig2 = go.Figure(go.Scatter(
            x=routes_df["bunching_index"],
            y=routes_df["delay_mean_15m"],
            mode="markers",
            marker=dict(
                color=preds_df["severity_class"].astype(int),
                colorscale=[[0,"#2ed573"],[0.33,"#1e90ff"],[0.66,"#ffa502"],[1,"#ff4757"]],
                size=9, opacity=0.8,
                colorbar=dict(title="Severity", tickvals=[0,1,2,3],
                              ticktext=["NORMAL","MINOR","MODERATE","SEVERE"],
                              tickfont=dict(color="#94a3b8"), titlefont=dict(color="#94a3b8")),
            ),
            text=routes_df["route_id"],
            hovertemplate="<b>%{text}</b><br>Bunching: %{x:.2f}<br>Delay: %{y:.0f}s<extra></extra>",
        ))
        fig2.update_layout(**PLOTLY_TEMPLATE, height=280,
                           xaxis_title="Bunching Index", yaxis_title="Delay 15m (s)")
        st.plotly_chart(fig2, use_container_width=True)

    # Second row
    col3, col4 = st.columns(2)

    with col3:
        section_header("On-Time Performance by Route")
        top_routes = routes_df.nlargest(15, "on_time_pct")
        colors = preds_df.loc[preds_df["route_id"].isin(top_routes["route_id"]), "severity_color"].values
        fig3 = go.Figure(go.Bar(
            x=top_routes["route_id"],
            y=top_routes["on_time_pct"] * 100,
            marker_color=colors if len(colors)==len(top_routes) else "#00d9ff",
            text=(top_routes["on_time_pct"]*100).round(0).astype(int).astype(str) + "%",
            textposition="outside", textfont=dict(color="#e2e8f0"),
        ))
        fig3.add_hline(y=85, line_dash="dash", line_color="#ffa502",
                       annotation_text="Target 85%", annotation_font_color="#ffa502")
        fig3.update_layout(**PLOTLY_TEMPLATE, height=280,
                           xaxis_title="Route", yaxis_title="On-Time %")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        section_header("Fleet Utilization Distribution")
        fig4 = go.Figure(go.Histogram(
            x=routes_df["fleet_utilization"] * 100,
            nbinsx=15,
            marker_color="#1e90ff", opacity=0.75,
        ))
        fig4.update_layout(**PLOTLY_TEMPLATE, height=280,
                           xaxis_title="Fleet Utilization %", yaxis_title="Route Count")
        st.plotly_chart(fig4, use_container_width=True)

    # Summary stats
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("Summary Statistics")
    stats = routes_df[["delay_mean_15m","bunching_index","on_time_pct","speed_mean","fleet_utilization"]].describe().round(2)
    st.dataframe(
        stats.style.background_gradient(cmap="Blues", axis=1),
        use_container_width=True,
    )


# ─────────────────────────────────────────────
# PAGE: SETTINGS
# ─────────────────────────────────────────────
def page_settings():
    st.markdown('<h1 style="font-size:32px;font-weight:800;font-family:IBM Plex Mono,monospace;color:#00d9ff;margin-bottom:4px;">Settings</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color:#475569;margin-bottom:20px;">Configure data sources, model parameters, and display options</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        section_header("Data Source")
        data_source = st.selectbox("Source Type", ["Demo Data", "GTFS Feed", "Custom"],
                                   index=["Demo Data","GTFS Feed","Custom"].index(st.session_state["data_source"]))
        st.session_state["data_source"] = data_source

        if data_source != "Demo Data":
            st.text_input("Feed URL", placeholder="https://api.agency.com/gtfs-realtime/...")
            st.info("⚠️  Live feed integration requires network access. Currently showing demo data.", icon="ℹ️")

        n_routes = st.slider("Simulated Routes", 5, 50, st.session_state["n_routes"])
        st.session_state["n_routes"] = n_routes

    with c2:
        section_header("ML Model")
        model_type = st.selectbox("Model Type", ["RandomForest", "XGBoost", "ST-GAT"],
                                  index=["RandomForest","XGBoost","ST-GAT"].index(st.session_state["model_type"]))
        st.session_state["model_type"] = model_type

        conf = st.slider("Confidence Threshold", 0, 100,
                         int(st.session_state["confidence_threshold"]*100))
        st.session_state["confidence_threshold"] = conf / 100
        st.caption(f"Predictions below {conf}% confidence will be suppressed.")

    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        section_header("Refresh")
        auto_refresh = st.toggle("Auto Refresh", value=st.session_state["auto_refresh"])
        st.session_state["auto_refresh"] = auto_refresh
        if auto_refresh:
            interval = st.select_slider("Interval (seconds)", [10, 15, 30, 60, 120],
                                        value=st.session_state["refresh_interval"])
            st.session_state["refresh_interval"] = interval

    with c4:
        section_header("Alert Filters")
        sev_filter = st.multiselect(
            "Active Severity Levels",
            ["NORMAL", "MINOR", "MODERATE", "SEVERE"],
            default=st.session_state["severity_filter"],
        )
        st.session_state["severity_filter"] = sev_filter

    st.markdown("<br>", unsafe_allow_html=True)
    section_header("GTFS Integration Info")
    st.markdown("""
    <div style="background:#0d1f35;border:1px solid #1e3a5f;border-radius:8px;padding:20px;font-size:13px;color:#94a3b8;line-height:1.7;">
      <b style="color:#00d9ff;">Supported Feed Types</b><br>
      • <b>GTFS Realtime</b> — Protocol Buffer vehicle positions &amp; trip updates<br>
      • <b>GTFS Static</b> — ZIP with CSV schedule files (routes, stops, trips)<br>
      • <b>Custom JSON API</b> — Any agency-specific REST endpoint<br><br>
      <b style="color:#00d9ff;">Known Public Feeds</b><br>
      • <a href="https://www.transit.land/" style="color:#1e90ff;">Transitland</a> — Comprehensive GTFS directory<br>
      • <a href="https://openmobilitydata.org/" style="color:#1e90ff;">OpenMobilityData</a> — European transit data<br>
      • <a href="https://transport.data.gouv.fr/" style="color:#1e90ff;">Transport.data.gouv.fr</a> — French transit data<br><br>
      <b style="color:#00d9ff;">Model Types</b><br>
      • <b>RandomForest</b> — Fast, interpretable, good baseline<br>
      • <b>XGBoost</b> — Higher accuracy, gradient boosting<br>
      • <b>ST-GAT</b> — Spatio-temporal graph attention network (best for network effects)
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────
page = st.session_state["page"]
if page == "Overview":      page_overview()
elif page == "Live Map":    page_live_map()
elif page == "Alerts":      page_alerts()
elif page == "Predictions": page_predictions()
elif page == "Analytics":   page_analytics()
elif page == "Settings":    page_settings()

# Footer
st.markdown("""<hr>
<div style="text-align:center;color:#1e3a5f;font-size:11px;font-family:monospace;padding:8px 0;">
  GTFS Disruption Early Warning Dashboard · 30-min prediction horizon ·
  <span class="live-dot"></span> Live
</div>""", unsafe_allow_html=True)

# Auto-refresh trigger
if st.session_state["auto_refresh"]:
    time.sleep(0.5)
    elapsed = time.time() - st.session_state["last_refresh"]
    remaining = max(0, st.session_state["refresh_interval"] - elapsed)
    st.sidebar.markdown(f"""
    <div style="font-size:10px;color:#334155;font-family:monospace;text-align:center;margin-top:8px;">
      Next refresh in {remaining:.0f}s
    </div>""", unsafe_allow_html=True)
