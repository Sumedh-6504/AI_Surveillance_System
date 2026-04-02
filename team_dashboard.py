"""Team progress dashboard for tracking development tasks."""
import streamlit as st
import pandas as pd
from datetime import datetime
import json
from pathlib import Path

st.set_page_config(
    page_title="Team Progress Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Surveillance - Team Progress Dashboard")
st.markdown("---")

# Team members
TEAM_MEMBERS = {
    "Shivam": {
        "role": "Core Development",
        "tasks": [
            {"name": "YOLOv8 Detection Module", "status": "done", "progress": 100},
            {"name": "DeepSORT Tracking", "status": "done", "progress": 100},
            {"name": "Behavior Analysis", "status": "done", "progress": 100},
            {"name": "Rule Engine", "status": "done", "progress": 100},
            {"name": "Main Pipeline Integration", "status": "done", "progress": 100},
        ]
    },
    "Sumedh": {
        "role": "Dashboard & UI",
        "tasks": [
            {"name": "Streamlit Monitoring Dashboard", "status": "done", "progress": 100},
            {"name": "Team Progress Dashboard", "status": "done", "progress": 100},
            {"name": "Real-time Alert Display", "status": "done", "progress": 100},
            {"name": "Statistics & Charts", "status": "done", "progress": 100},
        ]
    },
    "Snehasich": {
        "role": "Testing & QA",
        "tasks": [
            {"name": "Unit Tests - Detection", "status": "pending", "progress": 0},
            {"name": "Unit Tests - Tracking", "status": "pending", "progress": 0},
            {"name": "Integration Tests", "status": "pending", "progress": 0},
            {"name": "Performance Benchmarks", "status": "pending", "progress": 0},
        ]
    },
    "Suhas": {
        "role": "Documentation & DevOps",
        "tasks": [
            {"name": "README Documentation", "status": "done", "progress": 100},
            {"name": "API Documentation", "status": "in_progress", "progress": 50},
            {"name": "Docker Setup", "status": "pending", "progress": 0},
            {"name": "CI/CD Pipeline", "status": "pending", "progress": 0},
        ]
    }
}

# Sidebar
st.sidebar.header("Filters")
selected_member = st.sidebar.selectbox(
    "Team Member",
    options=["All"] + list(TEAM_MEMBERS.keys())
)

st.sidebar.markdown("---")
st.sidebar.subheader("Project Stats")

# Calculate overall progress
total_tasks = 0
completed_tasks = 0
in_progress_tasks = 0

for member, data in TEAM_MEMBERS.items():
    for task in data["tasks"]:
        total_tasks += 1
        if task["status"] == "done":
            completed_tasks += 1
        elif task["status"] == "in_progress":
            in_progress_tasks += 1

overall_progress = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0

st.sidebar.metric("Total Tasks", total_tasks)
st.sidebar.metric("Completed", completed_tasks)
st.sidebar.metric("In Progress", in_progress_tasks)
st.sidebar.metric("Pending", total_tasks - completed_tasks - in_progress_tasks)

st.sidebar.markdown("---")
st.sidebar.progress(overall_progress / 100)
st.sidebar.write(f"**Overall Progress: {overall_progress:.1f}%**")

# Main content
if selected_member == "All":
    # Show all team members
    for member, data in TEAM_MEMBERS.items():
        st.subheader(f"{member} - {data['role']}")

        # Progress bar
        member_tasks = data["tasks"]
        member_completed = sum(1 for t in member_tasks if t["status"] == "done")
        member_progress = member_completed / len(member_tasks) * 100 if member_tasks else 0

        st.progress(member_progress / 100)
        st.write(f"**Progress: {member_progress:.1f}%** ({member_completed}/{len(member_tasks)} tasks)")

        # Task table
        task_data = []
        for task in member_tasks:
            status_icon = {
                "done": "✅",
                "in_progress": "🔄",
                "pending": "⏳"
            }.get(task["status"], "❓")

            task_data.append({
                "Task": task["name"],
                "Status": status_icon,
                "Progress": f"{task['progress']}%"
            })

        df = pd.DataFrame(task_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("---")
else:
    # Show selected member
    data = TEAM_MEMBERS[selected_member]
    st.subheader(f"{selected_member} - {data['role']}")

    member_tasks = data["tasks"]
    member_completed = sum(1 for t in member_tasks if t["status"] == "done")
    member_progress = member_completed / len(member_tasks) * 100 if member_tasks else 0

    st.progress(member_progress / 100)
    st.write(f"**Progress: {member_progress:.1f}%** ({member_completed}/{len(member_tasks)} tasks)")

    task_data = []
    for task in member_tasks:
        status_icon = {
            "done": "✅",
            "in_progress": "🔄",
            "pending": "⏳"
        }.get(task["status"], "❓")

        task_data.append({
            "Task": task["name"],
            "Status": status_icon,
            "Progress": f"{task['progress']}%"
        })

    df = pd.DataFrame(task_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

# Timeline section
st.markdown("---")
st.subheader("📅 Project Timeline")

timeline_data = [
    {"Date": "2024-01-15", "Event": "Project Kickoff", "Status": "done"},
    {"Date": "2024-01-20", "Event": "Core Detection Module", "Status": "done"},
    {"Date": "2024-01-25", "Event": "Tracking Implementation", "Status": "done"},
    {"Date": "2024-01-28", "Event": "Dashboard Development", "Status": "done"},
    {"Date": "2024-02-01", "Event": "Testing Phase", "Status": "in_progress"},
    {"Date": "2024-02-05", "Event": "Documentation Complete", "Status": "pending"},
    {"Date": "2024-02-10", "Event": "Final Release", "Status": "pending"},
]

df_timeline = pd.DataFrame(timeline_data)
st.dataframe(df_timeline, use_container_width=True, hide_index=True)

# Notes section
st.markdown("---")
st.subheader("📝 Team Notes")

notes = st.text_area("Add a note", height=100)
if st.button("Add Note"):
    st.success("Note added! (This would be saved to a file in production)")

st.info("💡 Tip: Run with `streamlit run team_dashboard.py`")
