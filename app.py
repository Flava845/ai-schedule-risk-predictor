"""
AI Schedule Risk Predictor — Streamlit dashboard.

MVP prototype: takes task schedule/workload inputs and classifies each
task as Low / Medium / High delay risk using a hybrid rule-based +
machine-learning engine, with mitigation suggestions to support earlier
intervention (per the project's SMART objectives).
"""

import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from model.predictor import predict_batch, predict_task

HERE = os.path.dirname(__file__)
SAMPLE_DATA_PATH = os.path.join(HERE, "data", "tasks_synthetic.csv")

# Fixed status palette (good / warning / critical) — chosen for contrast and
# colorblind-safe separation rather than arbitrary hues, since risk level is
# a state, not a category.
RISK_META = {
    "Low": {"color": "#0ca30c", "icon": "✅", "chip": "Low risk"},
    "Medium": {"color": "#fab219", "icon": "⚠️", "chip": "Medium risk"},
    "High": {"color": "#d03b3b", "icon": "🚨", "chip": "High risk"},
}
RISK_COLORS = {k: v["color"] for k, v in RISK_META.items()}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"

st.set_page_config(page_title="AI Schedule Risk Predictor", page_icon="⏱️", layout="wide")

CUSTOM_CSS = """
<style>
.block-container { padding-top: 2.5rem; padding-bottom: 3rem; }

.hero-title { font-size: 2.1rem; font-weight: 800; color: #0b0b0b; margin-bottom: 0.15rem; }
.hero-sub { font-size: 1rem; color: #52514e; margin-bottom: 1.6rem; }

.card {
    background: #fcfcfb;
    border: 1px solid rgba(11,11,11,0.10);
    border-radius: 14px;
    padding: 22px 24px;
}

.stat-tile {
    background: #fcfcfb;
    border: 1px solid rgba(11,11,11,0.10);
    border-radius: 12px;
    padding: 18px 20px;
    height: 100%;
}
.stat-tile .stat-label {
    font-size: 12.5px; font-weight: 700; letter-spacing: .03em; text-transform: uppercase;
    color: #52514e; display: flex; align-items: center; gap: 6px;
}
.stat-tile .stat-value { font-size: 30px; font-weight: 800; color: #0b0b0b; margin-top: 6px; }

.risk-badge { border-radius: 16px; padding: 30px 22px; text-align: center; }
.risk-badge .eyebrow { font-size: 13px; font-weight: 600; letter-spacing: .02em; text-transform: uppercase; color: #52514e; }
.risk-badge .task-name { font-size: 16px; font-weight: 700; color: #0b0b0b; margin: 6px 0 16px; }
.risk-badge .level { font-size: 40px; font-weight: 800; color: #0b0b0b; line-height: 1.1; }

.reason-row, .action-row {
    display: flex; gap: 10px; align-items: flex-start;
    padding: 8px 0; border-bottom: 1px solid rgba(11,11,11,0.06);
    font-size: 14.5px; color: #0b0b0b;
}
.reason-row:last-child, .action-row:last-child { border-bottom: none; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def stat_tile(label: str, value, icon: str = "") -> str:
    return f"""
    <div class="stat-tile">
        <div class="stat-label">{icon} {label}</div>
        <div class="stat-value">{value}</div>
    </div>
    """


st.markdown('<div class="hero-title">⏱️ AI Schedule Risk Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Identify tasks likely to be delayed, based on progress, '
    "dependencies, resources and deadlines — with simple mitigation suggestions.</div>",
    unsafe_allow_html=True,
)

tab_single, tab_batch, tab_about = st.tabs(["🔍 Assess a Task", "📊 Project Portfolio", "ℹ️ About"])


# ---------------------------------------------------------------- single task
with tab_single:
    with st.container(border=True):
        st.markdown("##### Enter task details")
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

        assess_clicked = st.button("Assess risk", type="primary")

    if assess_clicked:
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
        meta = RISK_META[level]

        st.write("")
        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(
                f"""
                <div class="risk-badge" style="background-color:{meta['color']}14;border:1.5px solid {meta['color']}55;">
                    <div class="eyebrow">Predicted risk for</div>
                    <div class="task-name">{task_name}</div>
                    <div class="level">{meta['icon']} {level}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if result["escalated"]:
                st.warning(f"Escalated from ML prediction ({result['ml_level']}) by rule guardrails.")
            st.markdown(
                stat_tile("ML model confidence", f"{max(result['ml_probabilities'].values()):.0%}", "🎯"),
                unsafe_allow_html=True,
            )

        with r2:
            proba_df = pd.DataFrame(
                {"Risk level": list(result["ml_probabilities"].keys()),
                 "Probability": list(result["ml_probabilities"].values())}
            )
            fig = px.bar(
                proba_df, x="Risk level", y="Probability", color="Risk level",
                color_discrete_map=RISK_COLORS, range_y=[0, 1], text_auto=".0%",
                title="ML model probability by risk level",
                category_orders={"Risk level": ["Low", "Medium", "High"]},
            )
            fig.update_traces(textfont_color=INK_PRIMARY, textposition="outside")
            fig.update_layout(
                showlegend=False, height=280, margin=dict(t=40, b=10),
                plot_bgcolor="#fcfcfb", paper_bgcolor="rgba(0,0,0,0)",
                font_color=INK_SECONDARY,
            )
            st.plotly_chart(fig, width="stretch")

        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("##### Why this risk level was triggered")
                if result["triggered_reasons"]:
                    rows = "".join(
                        f'<div class="reason-row">⚠️ <div>{reason}</div></div>'
                        for reason, _ in sorted(result["triggered_reasons"], key=lambda t: -t[1])
                    )
                else:
                    rows = '<div class="reason-row">✅ <div>No major risk indicators detected.</div></div>'
                st.markdown(rows, unsafe_allow_html=True)
        with c2:
            with st.container(border=True):
                st.markdown("##### Suggested mitigation actions")
                rows = "".join(f'<div class="action-row">👉 <div>{s}</div></div>' for s in result["mitigation_suggestions"])
                st.markdown(rows, unsafe_allow_html=True)


# ---------------------------------------------------------------- batch / portfolio
with tab_batch:
    with st.container(border=True):
        st.markdown("##### Score a whole project's task list")
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

        st.write("")
        counts = scored["final_level"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(stat_tile("Total tasks", len(scored), "📋"), unsafe_allow_html=True)
        c2.markdown(stat_tile("Low risk", int(counts.get("Low", 0)), RISK_META["Low"]["icon"]), unsafe_allow_html=True)
        c3.markdown(stat_tile("Medium risk", int(counts.get("Medium", 0)), RISK_META["Medium"]["icon"]), unsafe_allow_html=True)
        c4.markdown(stat_tile("High risk", int(counts.get("High", 0)), RISK_META["High"]["icon"]), unsafe_allow_html=True)

        st.write("")
        pie = px.pie(
            counts.reset_index(), names="final_level", values="count",
            color="final_level", color_discrete_map=RISK_COLORS,
            title="Portfolio risk breakdown", hole=0.5,
            category_orders={"final_level": ["Low", "Medium", "High"]},
        )
        pie.update_traces(textposition="outside", textfont_color=INK_SECONDARY, textinfo="label+percent")
        pie.update_layout(
            margin=dict(t=40, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color=INK_SECONDARY,
            showlegend=False,
        )
        with st.container(border=True):
            st.plotly_chart(pie, width="stretch")

        st.write("")
        with st.container(border=True):
            st.markdown("##### Task detail")
            level_filter = st.multiselect(
                "Filter by risk level", options=["Low", "Medium", "High"],
                default=["Medium", "High"],
            )
            display_cols = {
                "task_id": "Task ID",
                "task_name": "Task Name",
                "days_until_deadline": "Days Left",
                "percent_complete": "% Complete",
                "expected_percent_complete": "Expected %",
                "final_level": "Risk Level",
                "escalated": "Escalated",
                "mitigation_suggestions": "Suggested Actions",
            }
            filtered = scored[scored["final_level"].isin(level_filter)] if level_filter else scored
            renamed = filtered[[c for c in display_cols if c in filtered.columns]].rename(columns=display_cols)
            st.dataframe(
                renamed,
                width="stretch",
                hide_index=True,
                column_config={
                    "% Complete": st.column_config.ProgressColumn("% Complete", min_value=0, max_value=100, format="%d%%"),
                },
            )

            st.download_button(
                "Download scored results as CSV",
                data=scored.to_csv(index=False).encode("utf-8"),
                file_name="scored_tasks.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------- about
with tab_about:
    with st.container(border=True):
        st.markdown("##### About this prototype")
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
            with open(metrics_path) as f:
                metrics = json.load(f)
            st.write("")
            st.markdown(stat_tile("Model test accuracy", f"{metrics['accuracy']:.1%}", "📈"), unsafe_allow_html=True)
            st.caption("Target per project Assumptions: 70-80% acceptable for MVP.")
