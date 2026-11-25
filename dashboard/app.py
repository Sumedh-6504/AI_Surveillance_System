"""Streamlit dashboard for AI Surveillance monitoring."""
import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import math

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.models import init_db, get_events, get_statistics, cleanup_old_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = str(PROJECT_ROOT / "data" / "surveillance.db")


def resolve_db_path(db_path: str) -> str:
    """Resolve custom relative paths from the repository root."""
    path = Path(db_path).expanduser()
    return str((path if path.is_absolute() else PROJECT_ROOT / path).resolve())



st.set_page_config(
    page_title="AI Surveillance Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# Custom CSS — dark theme, glassmorphism, animations
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-card: rgba(15, 23, 42, 0.6);
    --bg-card-hover: rgba(30, 41, 59, 0.8);
    --border-glass: rgba(99, 102, 241, 0.15);
    --accent-blue: #6366f1;
    --accent-cyan: #22d3ee;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --glow-blue: 0 0 20px rgba(99, 102, 241, 0.3);
    --glow-cyan: 0 0 20px rgba(34, 211, 238, 0.3);
    --glow-rose: 0 0 20px rgba(244, 63, 94, 0.3);
}

.stApp {
    background: var(--bg-primary);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.08), transparent),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(34, 211, 238, 0.05), transparent);
    font-family: 'Inter', -apple-system, sans-serif;
}

/* hide default streamlit header/footer */
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { display: none; }
div[data-testid="stToolbar"] { display: none; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 3px; }

/* ---------- TOP BAR ---------- */
.top-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 0 24px 0;
    border-bottom: 1px solid var(--border-glass);
    margin-bottom: 28px;
}
.top-bar-left {
    display: flex;
    align-items: center;
    gap: 14px;
}
.top-bar-logo {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    box-shadow: var(--glow-blue);
}
.top-bar-title {
    font-size: 22px; font-weight: 800; color: var(--text-primary);
    letter-spacing: -0.5px;
}
.top-bar-subtitle {
    font-size: 12px; color: var(--text-muted);
    letter-spacing: 2px; text-transform: uppercase; margin-top: 2px;
}
.top-bar-right {
    display: flex; align-items: center; gap: 20px;
}
.live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 20px;
    font-size: 12px; font-weight: 600; color: var(--accent-emerald);
    letter-spacing: 0.5px;
}
.live-dot {
    width: 7px; height: 7px;
    background: var(--accent-emerald);
    border-radius: 50%;
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }
    50% { opacity: 0.6; box-shadow: 0 0 0 6px rgba(16,185,129,0); }
}
.offline-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 14px;
    background: rgba(244, 63, 94, 0.12);
    border: 1px solid rgba(244, 63, 94, 0.3);
    border-radius: 20px;
    font-size: 12px; font-weight: 600; color: var(--accent-rose);
}
.clock-display {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px; color: var(--text-secondary);
    letter-spacing: 1px;
}

/* ---------- METRIC CARDS ---------- */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
@media (max-width: 900px) {
    .metric-grid { grid-template-columns: repeat(2, 1fr); }
}
.metric-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 22px 24px;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    background: var(--bg-card-hover);
    transform: translateY(-2px);
    border-color: rgba(99,102,241,0.3);
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.metric-card.blue::before { background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); }
.metric-card.rose::before { background: linear-gradient(90deg, var(--accent-rose), #fb7185); }
.metric-card.amber::before { background: linear-gradient(90deg, var(--accent-amber), #fbbf24); }
.metric-card.emerald::before { background: linear-gradient(90deg, var(--accent-emerald), #34d399); }

.metric-icon {
    width: 40px; height: 40px;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; margin-bottom: 14px;
}
.metric-icon.blue { background: rgba(99,102,241,0.15); }
.metric-icon.rose { background: rgba(244,63,94,0.15); }
.metric-icon.amber { background: rgba(245,158,11,0.15); }
.metric-icon.emerald { background: rgba(16,185,129,0.15); }

.metric-label {
    font-size: 12px; font-weight: 500; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px;
}
.metric-value {
    font-size: 32px; font-weight: 800; color: var(--text-primary);
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.metric-change {
    font-size: 11px; margin-top: 8px; font-weight: 500;
}
.metric-change.up { color: var(--accent-rose); }
.metric-change.down { color: var(--accent-emerald); }
.metric-change.neutral { color: var(--text-muted); }

/* ---------- THREAT LEVEL ---------- */
.threat-container {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    text-align: center;
}
.threat-title {
    font-size: 13px; font-weight: 600; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 18px;
}
.threat-gauge {
    position: relative;
    width: 180px; height: 100px;
    margin: 0 auto 12px auto;
}
.threat-label {
    font-size: 22px; font-weight: 800; letter-spacing: 1px; margin-top: 4px;
}
.threat-label.low { color: var(--accent-emerald); }
.threat-label.medium { color: var(--accent-amber); }
.threat-label.high { color: #f97316; }
.threat-label.critical { color: var(--accent-rose); text-shadow: var(--glow-rose); }

/* ---------- SECTION HEADER ---------- */
.section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 28px 0 16px 0;
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-blue);
    box-shadow: var(--glow-blue);
}
.section-title {
    font-size: 16px; font-weight: 700; color: var(--text-primary);
    letter-spacing: -0.3px;
}
.section-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border-glass), transparent);
}

/* ---------- EVENT FEED ---------- */
.event-feed {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.event-row {
    display: grid;
    grid-template-columns: 50px 1fr 100px 90px 60px;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    transition: all 0.25s ease;
}
.event-row:hover {
    background: var(--bg-card-hover);
    border-color: rgba(99,102,241,0.25);
    transform: translateX(4px);
}
.event-sev-badge {
    padding: 4px 0;
    border-radius: 6px;
    font-size: 10px; font-weight: 700;
    text-align: center; text-transform: uppercase;
    letter-spacing: 0.5px;
}
.sev-critical {
    background: rgba(244,63,94,0.15); color: var(--accent-rose);
    border: 1px solid rgba(244,63,94,0.3);
    animation: flash-critical 2s ease-in-out infinite;
}
@keyframes flash-critical {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}
.sev-high {
    background: rgba(249,115,22,0.15); color: #f97316;
    border: 1px solid rgba(249,115,22,0.3);
}
.sev-medium {
    background: rgba(245,158,11,0.15); color: var(--accent-amber);
    border: 1px solid rgba(245,158,11,0.3);
}
.sev-low {
    background: rgba(16,185,129,0.15); color: var(--accent-emerald);
    border: 1px solid rgba(16,185,129,0.3);
}
.event-rule {
    font-size: 13px; font-weight: 600; color: var(--text-primary);
}
.event-type {
    font-size: 11px; color: var(--text-muted); margin-top: 2px;
}
.event-camera {
    font-size: 12px; color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
}
.event-time {
    font-size: 11px; color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    text-align: right;
}
.event-count {
    font-size: 13px; font-weight: 700; color: var(--accent-cyan);
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
}

/* ---------- CHART PANELS ---------- */
.chart-panel {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 24px;
}
.chart-panel-title {
    font-size: 13px; font-weight: 600; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 16px;
}

/* ---------- EMPTY STATE ---------- */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    background: var(--bg-card);
    border: 1px dashed rgba(99,102,241,0.2);
    border-radius: 16px;
}
.empty-icon {
    font-size: 48px; margin-bottom: 16px; opacity: 0.5;
}
.empty-title {
    font-size: 18px; font-weight: 700; color: var(--text-primary); margin-bottom: 8px;
}
.empty-desc {
    font-size: 13px; color: var(--text-muted); max-width: 400px; margin: 0 auto;
    line-height: 1.6;
}
.empty-cmd {
    display: inline-block; margin-top: 16px;
    padding: 8px 16px;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: var(--accent-blue);
}

/* ---------- SIDEBAR OVERRIDES ---------- */
section[data-testid="stSidebar"] {
    background: rgba(10, 14, 23, 0.95);
    border-right: 1px solid var(--border-glass);
}
section[data-testid="stSidebar"] .stMarkdown { color: var(--text-secondary); }

/* plotly chart background */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* remove streamlit element gaps */
.block-container { padding-top: 24px; }
div[data-testid="stVerticalBlock"] > div { gap: 0.4rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper functions (DB layer untouched)
# ---------------------------------------------------------------------------

def format_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def format_time_short(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def time_ago(ts: float) -> str:
    diff = time.time() - ts
    if diff < 60:
        return f"{int(diff)}s ago"
    if diff < 3600:
        return f"{int(diff // 60)}m ago"
    return f"{int(diff // 3600)}h ago"


@st.cache_data(ttl=5)
def fetch_events(db_path: str, limit: int = 100) -> list:
    try:
        init_db(db_path)
        return get_events(db_path, limit=limit)
    except Exception as e:
        st.error(f"Failed to fetch events: {e}")
        return []


@st.cache_data(ttl=10)
def fetch_statistics(db_path: str) -> dict:
    try:
        init_db(db_path)
        now = time.time()
        hour_ago = now - 3600
        return get_statistics(db_path)
    except Exception as e:
        st.error(f"Failed to fetch statistics: {e}")
        return {}



def compute_threat_level(critical: int, high: int, medium: int, low: int) -> tuple:
    score = critical * 40 + high * 20 + medium * 5 + low * 1
    if score == 0:
        return ("CLEAR", "low", 0)
    if score <= 10:
        return ("LOW", "low", min(score / 10 * 25, 25))
    if score <= 40:
        return ("MODERATE", "medium", 25 + (score - 10) / 30 * 25)
    if score <= 100:
        return ("HIGH", "high", 50 + (score - 40) / 60 * 25)
    return ("CRITICAL", "critical", min(75 + (score - 100) / 100 * 25, 100))


def make_gauge_svg(pct: float, level: str) -> str:
    colors = {
        "low": "#10b981",
        "medium": "#f59e0b",
        "high": "#f97316",
        "critical": "#f43f5e",
    }
    color = colors.get(level, "#6366f1")
    angle = 180 * (pct / 100)
    rad = math.radians(180 - angle)
    end_x = 90 + 70 * math.cos(rad)
    end_y = 90 - 70 * math.sin(rad)
    large = 1 if angle > 180 else 0

    return f"""
    <svg viewBox="0 0 180 100" width="180" height="100">
        <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none"
              stroke="rgba(100,116,139,0.15)" stroke-width="10" stroke-linecap="round"/>
        <path d="M 20 90 A 70 70 0 {large} 1 {end_x:.1f} {end_y:.1f}" fill="none"
              stroke="{color}" stroke-width="10" stroke-linecap="round"
              style="filter: drop-shadow(0 0 6px {color}40);"/>
        <text x="90" y="80" text-anchor="middle"
              font-family="JetBrains Mono, monospace" font-size="24" font-weight="800"
              fill="{color}">{int(pct)}%</text>
    </svg>"""


def build_severity_donut(by_severity: dict) -> None:
    """Render a plotly donut chart for severity distribution."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.bar_chart(pd.DataFrame(list(by_severity.items()), columns=["Severity", "Count"]).set_index("Severity"))
        return

    order = ["critical", "high", "medium", "low"]
    colors_map = {"critical": "#f43f5e", "high": "#f97316", "medium": "#f59e0b", "low": "#10b981"}
    labels, values, colors = [], [], []
    for s in order:
        v = by_severity.get(s, 0)
        if v > 0:
            labels.append(s.upper())
            values.append(v)
            colors.append(colors_map[s])

    if not values:
        st.markdown('<p style="text-align:center;color:#64748b;padding:30px 0;">No data</p>', unsafe_allow_html=True)
        return

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color='#0a0e17', width=2)),
        textinfo='label+value',
        textfont=dict(size=11, family='Inter', color='#f1f5f9'),
        hovertemplate='%{label}: %{value} events<extra></extra>',
    )])
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=10, l=10, r=10),
        height=240,
        showlegend=False,
        font=dict(family='Inter', color='#94a3b8'),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def build_type_bars(by_type: dict) -> None:
    """Render a plotly horizontal bar chart for event types."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.bar_chart(pd.DataFrame(list(by_type.items()), columns=["Type", "Count"]).set_index("Type"))
        return

    if not by_type:
        st.markdown('<p style="text-align:center;color:#64748b;padding:30px 0;">No data</p>', unsafe_allow_html=True)
        return

    sorted_items = sorted(by_type.items(), key=lambda x: x[1], reverse=True)
    labels = [i[0] for i in sorted_items]
    values = [i[1] for i in sorted_items]

    fig = go.Figure(data=[go.Bar(
        x=values, y=labels, orientation='h',
        marker=dict(
            color=values,
            colorscale=[[0, '#6366f1'], [0.5, '#22d3ee'], [1, '#10b981']],
            line=dict(width=0),
            cornerradius=4,
        ),
        text=values, textposition='outside',
        textfont=dict(size=12, family='JetBrains Mono', color='#94a3b8'),
        hovertemplate='%{y}: %{x} events<extra></extra>',
    )])
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=10, l=10, r=60),
        height=240,
        xaxis=dict(visible=False),
        yaxis=dict(
            tickfont=dict(size=12, family='Inter', color='#cbd5e1'),
            autorange='reversed',
        ),
        font=dict(family='Inter', color='#94a3b8'),
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def build_timeline(events: list) -> None:
    """Render a plotly timeline showing event density over time."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    if not events:
        return

    sev_colors = {"critical": "#f43f5e", "high": "#f97316", "medium": "#f59e0b", "low": "#10b981"}

    timestamps = [datetime.fromtimestamp(e.get("timestamp", 0)) for e in events]
    severities = [e.get("severity", "low") for e in events]
    colors = [sev_colors.get(s, "#6366f1") for s in severities]
    rules = [e.get("rule_name", "event") or "event" for e in events]
    hover = [f"{r}<br>{s.upper()}<br>{t.strftime('%H:%M:%S')}" for r, s, t in zip(rules, severities, timestamps)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=[sev_colors.get(s, 0) for s in severities],
        mode='markers',
        marker=dict(size=10, color=colors, opacity=0.8,
                    line=dict(width=1, color='rgba(255,255,255,0.1)')),
        text=hover, hoverinfo='text',
    ))

    sev_order = ["low", "medium", "high", "critical"]
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=30, l=60, r=10),
        height=180,
        xaxis=dict(
            gridcolor='rgba(100,116,139,0.08)',
            tickfont=dict(size=10, family='JetBrains Mono', color='#64748b'),
        ),
        yaxis=dict(
            tickvals=[sev_colors[s] for s in sev_order],
            ticktext=[s.upper() for s in sev_order],
            tickfont=dict(size=10, family='Inter', color='#64748b'),
            gridcolor='rgba(100,116,139,0.08)',
        ),
        font=dict(family='Inter', color='#94a3b8'),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    db_path = DEFAULT_DB_PATH
    init_db(db_path)


    # --- Sidebar (collapsed by default) ---
    with st.sidebar:
        st.markdown("### Settings")
        db_path = resolve_db_path(st.text_input("Database Path", value=DEFAULT_DB_PATH))
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (sec)", 5, 60, 10)
        st.markdown("---")
        if st.button("Refresh Now"):
            st.cache_data.clear()
        days_to_keep = st.number_input("Cleanup older than (days)", min_value=1, max_value=365, value=7)
        if st.button("Cleanup Old Data"):
            from src.database.models import get_connection
            with get_connection(db_path) as conn:
                cursor = conn.cursor()
                cutoff = time.time() - (days_to_keep * 24 * 60 * 60)
                cursor.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
                deleted = cursor.rowcount
                conn.commit()
            st.success(f"Deleted {deleted} old events")

    # --- Fetch data ---
    events = fetch_events(db_path)
    stats = fetch_statistics(db_path)
    total_events = stats.get("total_events", len(events))
    by_severity = stats.get("by_severity", {})
    by_type = stats.get("by_type", {})
    critical_count = by_severity.get("critical", 0)
    high_count = by_severity.get("high", 0)
    medium_count = by_severity.get("medium", 0)
    low_count = by_severity.get("low", 0)

    # --- Check system status ---
    system_active = False
    try:
        from src.database.models import get_connection
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cutoff = time.time() - 300
            cursor.execute("SELECT COUNT(*) FROM events WHERE timestamp >= ?", (cutoff,))
            system_active = cursor.fetchone()[0] > 0
    except Exception:
        pass

    # === TOP BAR ===
    now_str = datetime.now().strftime("%H:%M:%S  ·  %b %d, %Y")
    status_html = (
        '<span class="live-badge"><span class="live-dot"></span>SYSTEM LIVE</span>'
        if system_active else
        '<span class="offline-badge">OFFLINE</span>'
    )
    st.markdown(f"""
    <div class="top-bar">
        <div class="top-bar-left">
            <div class="top-bar-logo">🛡️</div>
            <div>
                <div class="top-bar-title">Command Center</div>
                <div class="top-bar-subtitle">AI Surveillance System</div>
            </div>
        </div>
        <div class="top-bar-right">
            {status_html}
            <span class="clock-display">{now_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # === METRIC CARDS ===
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-card blue">
            <div class="metric-icon blue">📡</div>
            <div class="metric-label">Total Events (1h)</div>
            <div class="metric-value">{total_events}</div>
            <div class="metric-change neutral">last 60 minutes</div>
        </div>
        <div class="metric-card rose">
            <div class="metric-icon rose">🚨</div>
            <div class="metric-label">Critical Alerts</div>
            <div class="metric-value">{critical_count}</div>
            <div class="metric-change {'up' if critical_count > 0 else 'neutral'}">
                {'immediate attention' if critical_count > 0 else 'all clear'}
            </div>
        </div>
        <div class="metric-card amber">
            <div class="metric-icon amber">⚠️</div>
            <div class="metric-label">High Severity</div>
            <div class="metric-value">{high_count}</div>
            <div class="metric-change {'up' if high_count > 0 else 'neutral'}">
                {'requires review' if high_count > 0 else 'none pending'}
            </div>
        </div>
        <div class="metric-card emerald">
            <div class="metric-icon emerald">📊</div>
            <div class="metric-label">Avg Detections</div>
            <div class="metric-value">{stats.get('avg_detections', 0):.1f}</div>
            <div class="metric-change neutral">per event</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # === THREAT LEVEL + CHARTS ROW ===
    col_threat, col_donut, col_bars = st.columns([1, 1, 1.4])

    with col_threat:
        threat_label, threat_class, threat_pct = compute_threat_level(
            critical_count, high_count, medium_count, low_count
        )
        gauge_svg = make_gauge_svg(threat_pct, threat_class)
        st.markdown(f"""
        <div class="threat-container">
            <div class="threat-title">Threat Level</div>
            {gauge_svg}
            <div class="threat-label {threat_class}">{threat_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_donut:
        st.markdown("""
        <div class="chart-panel">
            <div class="chart-panel-title">Severity Breakdown</div>
        </div>
        """, unsafe_allow_html=True)
        build_severity_donut(by_severity)

    with col_bars:
        st.markdown("""
        <div class="chart-panel">
            <div class="chart-panel-title">Events by Rule</div>
        </div>
        """, unsafe_allow_html=True)
        build_type_bars(by_type)

    # === EVENT TIMELINE ===
    if events:
        st.markdown("""
        <div class="section-header">
            <div class="section-dot"></div>
            <div class="section-title">Event Timeline</div>
            <div class="section-line"></div>
        </div>
        """, unsafe_allow_html=True)
        build_timeline(events)

    # === LIVE EVENT FEED ===
    st.markdown(f"""
    <div class="section-header">
        <div class="section-dot" style="background: var(--accent-cyan); box-shadow: var(--glow-cyan);"></div>
        <div class="section-title">Live Event Feed</div>
        <div class="section-line"></div>
        <span style="font-size:12px; color:#64748b; font-family:'JetBrains Mono',monospace;">
            {len(events)} events
        </span>
    </div>
    """, unsafe_allow_html=True)

    if events:
        feed_html = '<div class="event-feed">'
        for event in events[:30]:
            sev = event.get("severity", "low").lower()
            rule = event.get("rule_name", "—") or "—"
            etype = event.get("event_type", "")
            camera = event.get("camera_id", "—")
            ts = event.get("timestamp", 0)
            det_count = event.get("detection_count", 0)

            feed_html += f"""
            <div class="event-row">
                <div><span class="event-sev-badge sev-{sev}">{sev[:4]}</span></div>
                <div>
                    <div class="event-rule">{rule}</div>
                    <div class="event-type">{etype}</div>
                </div>
                <div class="event-camera">{camera}</div>
                <div class="event-count">{det_count}</div>
                <div class="event-time">{time_ago(ts)}</div>
            </div>"""
        feed_html += '</div>'
        st.markdown(feed_html, unsafe_allow_html=True)

        with st.expander("Inspect Event Details"):
            selected = st.selectbox(
                "Select Event",
                options=[e["id"] for e in events[:20]],
                format_func=lambda x: f"Event #{x}"
            )
            event = next((e for e in events if e["id"] == selected), None)
            if event:
                st.json(event)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📡</div>
            <div class="empty-title">No Events Detected</div>
            <div class="empty-desc">
                The surveillance system hasn't recorded any events yet.
                Start the detection pipeline to begin monitoring.
            </div>
            <div class="empty-cmd">python main.py --config configs/default.yaml</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Auto refresh ---
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
