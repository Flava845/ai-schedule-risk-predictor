"""
AI Schedule Risk Predictor — Streamlit dashboard.

MVP prototype: takes task schedule/workload inputs and classifies each
task as Low / Medium / High delay risk using a hybrid rule-based +
machine-learning engine, with mitigation suggestions to support earlier
intervention (per the project's SMART objectives).
"""

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model.predictor import predict_batch, predict_task

HERE = os.path.dirname(__file__)
SAMPLE_DATA_PATH = os.path.join(HERE, "data", "tasks_synthetic.csv")

RISK_COLORS = {"Low": "#2E7D32", "Medium": "#F9A825", "High": "#C62828"}

st.set_page_config(page_title="AI Schedule Risk Predictor", page_icon="⏱️", layout="wide")

st.title("⏱️ AI Schedule Risk Predictor")
st.caption(
    "Identify tasks likely to be delayed, based on progress, dependencies, "
    "resources and deadlines — with simple mitigation suggestions."
)

tab_single, tab_batch, tab_about = st.tabs(["🔍 Assess a Task", "📊 Project Portfolio", "ℹ️ About"])


# ---------------------------------------------------------------- single task
with tab_single:
    st.subheader("Enter task details")

    col1, col2, col3 = st.columns(3)
    with col1:
        task_name = st.text_input("Task name", value="Develop risk analysis module")
        days_until_deadline = st.number_input("Days until deadline", value=5, step=1)
        percent_complete = st.slider("Current % complete", 0, 100, 40)
        expected_percent_complete = st.slider("Expected % complete by now", 0, 100, 70)
    with col2:
        num_dependencies = st.number_input("Number of dependencies", value=3, min_value=0, step=1)
        num_dependencies_incomplete = st.number_input(
            "Dependencies still incomplete", value=1, min_value=0,
            max_value=int(num_dependencies) if num_dependencies else 0, step=1,
        )
        resource_availability_pct = st.slider("Resource availability (%)", 0, 100, 60)
    with col3:
        team_workload_score = st.slider("Team workload (0=idle, 10=overloaded)", 0.0, 10.0, 5.0, 0.5)
        past_delays_count = st.number_input("Times this task type has slipped before", value=0, min_value=0, step=1)

    if st.button("Assess risk", type="primary"):
        task = dict(
            days_until_deadline=days_until_deadline,
            percent_complete=percent_complete,
            expected_percent_complete=expected_percent_complete,
            num_dependencies=num_dependencies,
            num_dependencies_incomplete=num_dependencies_incomplete,
            resource_availability_pct=resource_availability_pct,
            team_workload_score=team_workload_score,
            past_delays_count=past_delays_count,
        )
        result = predict_task(task)
        level = result["final_level"]
        color = RISK_COLORS[level]

        st.markdown("---")
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(
                f"""
                <div style="background-color:{color}20;border:2px solid {color};
                border-radius:10px;padding:24px;text-align:center;">
                    <div style="font-size:15px;opacity:0.7;">Predicted Risk for</div>
                    <div style="font-size:16px;font-weight:600;margin-bottom:8px;opacity:0.95;">{task_name}</div>
                    <div style="font-size:40px;font-weight:800;color:{color};">{level}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if result["escalated"]:
                st.warning(
                    f"Escalated from ML prediction ({result['ml_level']}) by rule guardrails."
                )
            st.metric("ML model confidence", f"{max(result['ml_probabilities'].values()):.0%}")

        with r2:
            proba_df = pd.DataFrame(
                {"Risk level": list(result["ml_probabilities"].keys()),
                 "Probability": list(result["ml_probabilities"].values())}
            )
            fig = px.bar(
                proba_df, x="Risk level", y="Probability", color="Risk level",
                color_discrete_map=RISK_COLORS, range_y=[0, 1], text_auto=".0%",
                title="ML model probability by risk level",
            )
            fig.update_layout(showlegend=False, height=280, margin=dict(t=40, b=10))
            st.plotly_chart(fig, width='stretch')

        st.markdown("#### Why this risk level was triggered")
        if result["triggered_reasons"]:
            for reason, weight in sorted(result["triggered_reasons"], key=lambda t: -t[1]):
                st.markdown(f"- {reason}")
        else:
            st.markdown("- No major risk indicators detected.")

        st.markdown("#### Suggested mitigation actions")
        for s in result["mitigation_suggestions"]:
            st.markdown(f"- {s}")


# ---------------------------------------------------------------- batch / portfolio
with tab_batch:
    st.subheader("Score a whole project's task list")
    st.caption(
        "Upload a CSV with the required columns, or use the bundled synthetic "
        "sample dataset to see the tool score an entire project portfolio."
    )

    required_cols = [
        "task_id", "task_name", "days_until_deadline", "percent_complete",
        "expected_percent_complete", "num_dependencies", "num_dependencies_incomplete",
        "resource_availability_pct", "team_workload_score", "past_delays_count",
    ]

    uploaded = st.file_uploader("Upload task CSV", type=["csv"])
    use_sample = st.checkbox("Use bundled sample dataset instead", value=uploaded is None)

    df = None
    if uploaded is not None and not use_sample:
        df = pd.read_csv(uploaded)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            df = None
    elif use_sample:
        df = pd.read_csv(SAMPLE_DATA_PATH)

    if df is not None:
        with st.spinner("Scoring tasks..."):
            scored = predict_batch(df.head(300))

        c1, c2, c3, c4 = st.columns(4)
        counts = scored["final_level"].value_counts()
        c1.metric("Total tasks", len(scored))
        c2.metric("🟢 Low risk", int(counts.get("Low", 0)))
        c3.metric("🟠 Medium risk", int(counts.get("Medium", 0)))
        c4.metric("🔴 High risk", int(counts.get("High", 0)))

        pie = px.pie(
            counts.reset_index(), names="final_level", values="count",
            color="final_level", color_discrete_map=RISK_COLORS,
            title="Portfolio risk breakdown", hole=0.45,
        )
        st.plotly_chart(pie, width='stretch')

        level_filter = st.multiselect(
            "Filter by risk level", options=["Low", "Medium", "High"],
            default=["Medium", "High"],
        )
        display_cols = [
            "task_id", "task_name", "days_until_deadline", "percent_complete",
            "expected_percent_complete", "final_level", "escalated", "mitigation_suggestions",
        ]
        filtered = scored[scored["final_level"].isin(level_filter)] if level_filter else scored
        st.dataframe(
            filtered[[c for c in display_cols if c in filtered.columns]],
            width='stretch', hide_index=True,
        )

        st.download_button(
            "Download scored results as CSV",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="scored_tasks.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------- about
with tab_about:
    st.subheader("About this prototype")
    st.markdown(
        """
This is the MVP prototype for the **AI Schedule Risk Predictor** project.

**How risk is calculated (hybrid approach):**
1. A **rule-based engine** scores each task on progress gap, incomplete
   dependencies, resource availability, team workload, days to deadline,
   and past delay history — and can *escalate* a task to a higher risk
   level when it hits an unambiguous danger condition (e.g. deadline
   already passed).
2. A **machine-learning classifier** (Random Forest, trained on a
   synthetic dataset since real student/assignment data wasn't available)
   predicts Low / Medium / High risk from the same features.
3. The **final risk level** is the ML prediction, unless the rule
   engine's guardrails require escalating it further.

**Scope note:** this MVP uses synthetic data and a simplified model for
demonstration; it does not include live data integration or full-scale
deployment.
        """
    )
    metrics_path = os.path.join(HERE, "model", "metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
        st.metric("Model test accuracy", f"{metrics['accuracy']:.1%}")
        st.caption("Target per project Assumptions: 70-80% acceptable for MVP.")
