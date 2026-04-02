"""Streamlit dashboard for AI Surveillance monitoring."""
import streamlit as st
import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.models import get_events, get_statistics, cleanup_old_data


st.set_page_config(
    page_title="AI Surveillance Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


def format_timestamp(ts: float) -> str:
    """Format Unix timestamp to readable string."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def severity_color(severity: str) -> str:
    """Get color for severity level."""
    colors = {
        "low": "🟢",
        "medium": "🟡",
        "high": "🟠",
        "critical": "🔴"
    }
    return colors.get(severity.lower(), "⚪")


@st.cache_data(ttl=5)
def fetch_events(db_path: str, limit: int = 100) -> list:
    """Fetch recent events from database."""
    try:
        return get_events(db_path, limit=limit)
    except Exception as e:
        st.error(f"Failed to fetch events: {e}")
        return []


@st.cache_data(ttl=10)
def fetch_statistics(db_path: str) -> dict:
    """Fetch statistics from database."""
    try:
        now = time.time()
        hour_ago = now - 3600
        return get_statistics(db_path, start_time=hour_ago)
    except Exception as e:
        st.error(f"Failed to fetch statistics: {e}")
        return {}


def main():
    st.title("🎯 AI Surveillance Dashboard")
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Settings")

    db_path = st.sidebar.text_input(
        "Database Path",
        value="data/surveillance.db"
    )

    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 5, 60, 10)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Actions")

    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()

    days_to_keep = st.sidebar.number_input(
        "Cleanup old data (days)",
        min_value=1,
        max_value=365,
        value=7
    )

    if st.sidebar.button("🗑️ Cleanup Old Data"):
        from src.database.models import get_connection
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cutoff = time.time() - (days_to_keep * 24 * 60 * 60)
            cursor.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
            deleted = cursor.rowcount
            conn.commit()
        st.sidebar.success(f"Deleted {deleted} old events")

    # Main content
    col1, col2, col3, col4 = st.columns(4)

    # Fetch data
    events = fetch_events(db_path)
    stats = fetch_statistics(db_path)

    # Metrics
    total_events = stats.get("total_events", len(events))
    by_severity = stats.get("by_severity", {})

    critical_count = by_severity.get("critical", 0)
    high_count = by_severity.get("high", 0)
    medium_count = by_severity.get("medium", 0)
    low_count = by_severity.get("low", 0)

    col1.metric("Total Events (1h)", total_events)
    col2.metric("🔴 Critical", critical_count)
    col3.metric("🟠 High", high_count)
    col4.metric("🟡 Medium", medium_count)

    st.markdown("---")

    # Charts row
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Events by Type")
        by_type = stats.get("by_type", {})
        if by_type:
            df_type = pd.DataFrame(
                list(by_type.items()),
                columns=["Type", "Count"]
            )
            st.bar_chart(df_type.set_index("Type"))
        else:
            st.info("No data available")

    with chart_col2:
        st.subheader("Events by Severity")
        if by_severity:
            df_sev = pd.DataFrame(
                list(by_severity.items()),
                columns=["Severity", "Count"]
            )
            st.bar_chart(df_sev.set_index("Severity"))
        else:
            st.info("No data available")

    st.markdown("---")

    # Events table
    st.subheader("Recent Events")

    if events:
        # Create DataFrame
        event_data = []
        for event in events[:50]:  # Show last 50
            event_data.append({
                "Time": format_timestamp(event.get("timestamp", 0)),
                "Severity": severity_color(event.get("severity", "")) + " " + event.get("severity", "").upper(),
                "Type": event.get("event_type", ""),
                "Rule": event.get("rule_name", ""),
                "Camera": event.get("camera_id", ""),
                "Detections": event.get("detection_count", 0)
            })

        df_events = pd.DataFrame(event_data)
        st.dataframe(df_events, use_container_width=True, hide_index=True)

        # Event details expander
        if events:
            with st.expander("View Event Details"):
                selected = st.selectbox(
                    "Select Event",
                    options=[e["id"] for e in events[:20]],
                    format_func=lambda x: f"Event #{x}"
                )

                event = next((e for e in events if e["id"] == selected), None)
                if event:
                    st.json(event)
    else:
        st.info("No events recorded yet. Start the surveillance system to begin collecting data.")

    # Live status
    st.markdown("---")
    st.subheader("System Status")

    # Check if surveillance system is running (by checking recent detections)
    try:
        from src.database.models import get_connection
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            # Check for any activity in last 5 minutes
            cutoff = time.time() - 300
            cursor.execute("SELECT COUNT(*) FROM events WHERE timestamp >= ?", (cutoff,))
            recent_count = cursor.fetchone()[0]

            if recent_count > 0:
                st.success("✅ Surveillance system is ACTIVE")
            else:
                st.warning("⚠️ No recent events - surveillance system may not be running")
                st.info("Run `python main.py --config configs/default.yaml` in another terminal")
    except Exception as e:
        st.warning(f"Could not check status: {e}")

    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
