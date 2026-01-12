"""
Streamlit dashboard for machine health monitoring
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .critical { color: #ff4444; font-weight: bold; }
    .warning { color: #ffaa00; font-weight: bold; }
    .healthy { color: #44aa44; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏭 Predictive Maintenance Dashboard")
st.markdown("Real-time machine health monitoring and failure prediction")

# Sidebar
st.sidebar.header("Configuration")
n_machines = st.sidebar.slider("Number of Machines", 5, 100, 20)
update_frequency = st.sidebar.select_slider("Refresh Rate (seconds)", [5, 10, 30, 60], 10)

# Generate sample data for demo
np.random.seed(42)


def generate_machine_data(n_machines: int) -> pd.DataFrame:
    """Generate synthetic machine data"""
    data = []

    for machine_id in range(1, n_machines + 1):
        # Degradation factor
        degradation = np.random.uniform(0, 1)

        # Machine metrics
        health_score = max(0, 1.0 - degradation)
        failure_probability = degradation
        rul_days = max(1, int((1 - degradation) * 365))
        current_cycle = np.random.randint(100, 1000)

        # Status determination
        if failure_probability > 0.7:
            status = "🔴 Critical"
            color = "red"
        elif failure_probability > 0.4:
            status = "🟡 Warning"
            color = "orange"
        else:
            status = "🟢 Healthy"
            color = "green"

        data.append(
            {
                "Machine ID": machine_id,
                "Health Score": health_score,
                "Failure Probability": failure_probability,
                "RUL (Days)": rul_days,
                "Cycles": current_cycle,
                "Status": status,
                "Color": color,
                "Last Update": datetime.now() - timedelta(seconds=np.random.randint(0, 300)),
            }
        )

    return pd.DataFrame(data)


# Main dashboard
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Machine Details", "Alerts", "Analytics"])

# TAB 1: Overview
with tab1:
    df_machines = generate_machine_data(n_machines)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        healthy = len(df_machines[df_machines["Status"] == "🟢 Healthy"])
        st.metric("Healthy Machines", healthy, delta=f"+{healthy}")

    with col2:
        warning = len(df_machines[df_machines["Status"] == "🟡 Warning"])
        st.metric("Warning", warning, delta=f"-{warning}")

    with col3:
        critical = len(df_machines[df_machines["Status"] == "🔴 Critical"])
        st.metric("Critical", critical, delta=f"-{critical}")

    with col4:
        avg_health = df_machines["Health Score"].mean()
        st.metric("Avg Health Score", f"{avg_health:.2%}")

    st.divider()

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Health scores
        fig = px.bar(
            df_machines.sort_values("Health Score", ascending=False),
            x="Machine ID",
            y="Health Score",
            color="Status",
            color_discrete_map={"🟢 Healthy": "green", "🟡 Warning": "orange", "🔴 Critical": "red"},
            title="Machine Health Scores",
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Failure probability
        fig = px.scatter(
            df_machines,
            x="RUL (Days)",
            y="Failure Probability",
            size="Cycles",
            color="Status",
            color_discrete_map={"🟢 Healthy": "green", "🟡 Warning": "orange", "🔴 Critical": "red"},
            title="RUL vs Failure Probability",
            hover_data=["Machine ID"],
        )
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: Machine Details
with tab2:
    st.subheader("Detailed Machine Status")

    df_machines = generate_machine_data(n_machines)
    machine_options = df_machines["Machine ID"].tolist()
    selected_machine = st.selectbox("Select Machine", machine_options)

    machine_data = df_machines[df_machines["Machine ID"] == selected_machine].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Health Score",
            f"{machine_data['Health Score']:.1%}",
            delta=f"{np.random.randint(-5, 5)}%",
        )

    with col2:
        st.metric(
            "Failure Risk",
            f"{machine_data['Failure Probability']:.1%}",
            delta=f"{np.random.randint(-5, 5)}%",
        )

    with col3:
        st.metric(
            "RUL",
            f"{machine_data['RUL (Days)']} days",
            delta=f"-{np.random.randint(1, 10)} days",
        )

    with col4:
        st.metric(
            "Current Cycle",
            f"{machine_data['Cycles']:,}",
            delta=f"+{np.random.randint(1, 100)}",
        )

    st.divider()

    # Time series
    st.subheader("Sensor Trends")

    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    sensor_data = pd.DataFrame(
        {
            "Date": dates,
            "Sensor_1": np.cumsum(np.random.randn(100)) + 100,
            "Sensor_2": np.cumsum(np.random.randn(100)) + 100,
            "Sensor_3": np.cumsum(np.random.randn(100)) + 100,
        }
    )

    fig = go.Figure()

    for col in ["Sensor_1", "Sensor_2", "Sensor_3"]:
        fig.add_trace(go.Scatter(x=sensor_data["Date"], y=sensor_data[col], mode="lines", name=col))

    fig.update_layout(
        title="Sensor Readings Over Time",
        xaxis_title="Date",
        yaxis_title="Sensor Value",
        hovermode="x unified",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

# TAB 3: Alerts
with tab3:
    st.subheader("Active Alerts")

    alerts_data = {
        "Machine ID": [5, 12, 18, 3],
        "Alert Type": ["Critical Degradation", "High Temperature", "Vibration Spike", "Pressure Anomaly"],
        "Severity": ["🔴 Critical", "🟡 Warning", "🟡 Warning", "🟢 Info"],
        "Time": [
            "2 minutes ago",
            "15 minutes ago",
            "1 hour ago",
            "3 hours ago",
        ],
    }

    df_alerts = pd.DataFrame(alerts_data)
    st.dataframe(df_alerts, use_container_width=True, hide_index=True)

    st.subheader("Recommended Actions")
    st.info(
        """
        **Machine 5**: Schedule immediate maintenance. Degradation rate exceeds threshold.
        **Machine 12**: Monitor temperature readings. Consider proactive inspection.
        """
    )

# TAB 4: Analytics
with tab4:
    st.subheader("Fleet Analytics")

    df_machines = generate_machine_data(n_machines)

    col1, col2 = st.columns(2)

    with col1:
        # Status distribution
        status_counts = df_machines["Status"].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Fleet Status Distribution",
            color_discrete_map={"🟢 Healthy": "green", "🟡 Warning": "orange", "🔴 Critical": "red"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # RUL distribution
        fig = px.histogram(
            df_machines,
            x="RUL (Days)",
            nbins=20,
            title="RUL Distribution",
            labels={"RUL (Days)": "Remaining Useful Life (Days)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Correlation heatmap
    st.subheader("Feature Correlations")
    metrics_corr = df_machines[
        ["Health Score", "Failure Probability", "RUL (Days)", "Cycles"]
    ].corr()

    fig = px.imshow(
        metrics_corr,
        labels=dict(x="Metric", y="Metric", color="Correlation"),
        title="Metric Correlations",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

with col2:
    st.caption(f"Machines monitored: {n_machines}")

with col3:
    st.caption("Predictive Maintenance Platform v1.0")
