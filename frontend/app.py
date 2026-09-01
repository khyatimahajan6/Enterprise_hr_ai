import html
import logging
import os
import sys

# Ensure root workspace is in sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.config import settings
from frontend.services.api_client import ChatBackendError, get_chat_answer
from frontend.services.data_loader import DataValidationError, load_data, with_performance_score

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Enterprise HR AI Intelligence Workspace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design system CSS
st.markdown("""
<style>
    :root {
        --bg-0: #080d1a;
        --bg-1: #0f172a;
        --surface: rgba(30, 41, 59, 0.5);
        --surface-border: rgba(255, 255, 255, 0.08);
        --text-primary: #f1f5f9;
        --text-muted: #94a3b8;
        --accent-blue: #38bdf8;
        --accent-indigo: #818cf8;
        --accent-purple: #c084fc;
        --risk-high: #ef4444;
        --risk-medium: #f59e0b;
        --risk-low: #10b981;
        --radius-md: 12px;
        --radius-lg: 16px;
    }

    .stApp {
        background: radial-gradient(circle at 10% 20%, var(--bg-1) 0%, var(--bg-0) 90%);
        color: var(--text-primary);
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px; }

    .app-header {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        border-bottom: 1px solid var(--surface-border);
        padding-bottom: 16px;
        margin-bottom: 24px;
    }
    .header-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-indigo) 50%, var(--accent-purple) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-subtitle { color: var(--text-muted); font-size: 0.95rem; font-weight: 500; }

    .section-eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--accent-blue);
        margin-bottom: 2px;
    }
    .section-heading {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 4px;
    }
    .section-desc { color: var(--text-muted); font-size: 0.9rem; margin-bottom: 18px; }

    div[data-testid="stMetricValue"] { font-size: 1.7rem !important; font-weight: 800 !important; color: var(--accent-blue) !important; }
    div[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-weight: 600 !important; font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 0.04em; }
    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--surface-border);
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        backdrop-filter: blur(10px);
        border-radius: var(--radius-lg);
        padding: 18px 22px;
        transition: border-color 0.15s ease;
    }
    div[data-testid="stMetric"]:hover { border-color: rgba(56, 189, 248, 0.35); }

    .badge { padding: 3px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.78rem; display: inline-block; }
    .badge-high { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.4); }
    .badge-medium { background: rgba(245,158,11,0.15); color: #fcd34d; border: 1px solid rgba(245,158,11,0.4); }
    .badge-low { background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.4); }

    .lookup-card {
        background: var(--surface);
        border: 1px solid var(--surface-border);
        border-radius: var(--radius-lg);
        padding: 20px 24px;
        margin-top: 8px;
    }
    .lookup-field-label { color: var(--text-muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 2px; }
    .lookup-field-value { font-size: 1.05rem; font-weight: 600; color: var(--text-primary); margin-bottom: 14px; }

    section[data-testid="stSidebar"] { background: rgba(8, 13, 26, 0.6); border-right: 1px solid var(--surface-border); }
</style>
""", unsafe_allow_html=True)

def risk_badge(level: str) -> str:
    cls = {"HIGH": "badge-high", "MEDIUM": "badge-medium", "LOW": "badge-low"}.get(level, "badge-medium")
    return f'<span class="badge {cls}">{html.escape(level)}</span>'

def section_header(eyebrow: str, heading: str, desc: str = "") -> None:
    st.markdown(f'<div class="section-eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-heading">{heading}</div>', unsafe_allow_html=True)
    if desc:
        st.markdown(f'<div class="section-desc">{desc}</div>', unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <div>
        <div class="header-title">⚡ Enterprise HR AI Intelligence Workspace</div>
        <div class="header-subtitle">Predictive Workforce Analytics · Skill Gap Engine · RAG HR Assistant</div>
    </div>
</div>
""", unsafe_allow_html=True)

try:
    intel_df, gaps_df = load_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except DataValidationError as e:
    st.error(f"Data schema problem: {e}")
    st.stop()

st.sidebar.markdown("### 🎛️ Filters")
st.sidebar.caption("Narrow the workforce view by department, risk, engagement, and income.")

dept_options = sorted(intel_df["Department"].unique())
selected_depts = st.sidebar.multiselect("Department", dept_options, default=dept_options)

risk_options = ["HIGH", "MEDIUM", "LOW"]
selected_risks = st.sidebar.multiselect("Attrition Risk Level", risk_options, default=risk_options)

eng_options = ["HIGH", "MEDIUM", "LOW"]
selected_engs = st.sidebar.multiselect("Engagement Category", eng_options, default=eng_options)

min_inc = float(intel_df["MonthlyIncome"].min())
max_inc = float(intel_df["MonthlyIncome"].max())
if min_inc == max_inc:
    st.sidebar.caption(f"All employees earn ${min_inc:,.0f}/mo — income filter disabled.")
    selected_income = (min_inc, max_inc)
else:
    selected_income = st.sidebar.slider(
        "Monthly Income Range ($)", min_value=min_inc, max_value=max_inc, value=(min_inc, max_inc)
    )

filtered_df = intel_df[
    intel_df["Department"].isin(selected_depts)
    & intel_df["Attrition_Risk_Level"].isin(selected_risks)
    & intel_df["Engagement_Category"].isin(selected_engs)
    & intel_df["MonthlyIncome"].between(*selected_income)
]

if filtered_df.empty:
    st.warning("No employees match the current filters. Adjust filters in the sidebar to see data.")
    st.stop()

# Top Metric Cards
c1, c2, c3, c4 = st.columns(4)
total_count = len(filtered_df)
at_risk_count = int((filtered_df["Attrition_Risk_Level"] == "HIGH").sum())
avg_engagement = filtered_df["EngagementScore"].mean()
skill_deficit_count = int((filtered_df["SkillGapCount"] >= 2).sum())

c1.metric("Active Workforce", f"{total_count:,}")
c2.metric("High Attrition Risk", f"{at_risk_count:,}",
          delta=f"{at_risk_count / total_count * 100:.1f}% Risk", delta_color="inverse")
c3.metric("Avg Engagement", f"{avg_engagement:.1f}%", delta="Workforce Index")
c4.metric("Skill Deficit Personnel", f"{skill_deficit_count:,}", delta="Upskilling Needed")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

tab_attr, tab_matrix, tab_skills, tab_chat, tab_lookup = st.tabs([
    "📉  Attrition Intelligence",
    "🎯  Risk vs Engagement Matrix",
    "🧩  Organization Skill Gaps",
    "💬  HR AI Assistant",
    "👤  Employee Lookup",
])

RISK_COLORS = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
    title_font_size=15,
    legend_title_text="",
    margin=dict(t=50, l=10, r=10, b=10),
)

# Tab 1: Attrition Intelligence
with tab_attr:
    section_header("ATTRITION", "Departmental Risk & Compensation Distribution",
                    "Where risk concentrates, and how pay relates to it.")

    col1, col2 = st.columns(2)
    with col1:
        dept_risk_df = filtered_df.groupby(["Department", "Attrition_Risk_Level"]).size().reset_index(name="Count")
        fig_bar = px.bar(dept_risk_df, x="Department", y="Count", color="Attrition_Risk_Level",
                          color_discrete_map=RISK_COLORS, template="plotly_dark", barmode="stack",
                          title="Attrition Risk by Department")
        fig_bar.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        fig_box = px.box(filtered_df, x="Attrition_Risk_Level", y="MonthlyIncome", color="Attrition_Risk_Level",
                          color_discrete_map=RISK_COLORS, points="all", template="plotly_dark",
                          title="Income Spread by Risk Tier")
        fig_box.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    section_header("ROSTER", "High Attrition Risk Employees")

    roster = filtered_df[filtered_df["Attrition_Risk_Level"] == "HIGH"].sort_values(
        "Attrition_Probability", ascending=False
    )
    if roster.empty:
        st.info("No high-risk employees in the current filter selection.")
    else:
        display_roster = roster[["EmployeeID", "Department", "JobRole", "MonthlyIncome",
                                  "OverTime", "Attrition_Probability", "EngagementScore"]].copy()
        
        # Download Button for CSV
        csv_data = display_roster.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download High Risk Attrition Roster (CSV)",
            data=csv_data,
            file_name="high_risk_attrition_roster.csv",
            mime="text/csv",
        )
        
        display_roster["Attrition_Probability"] = (display_roster["Attrition_Probability"] * 100).round(1).astype(str) + "%"
        display_roster["MonthlyIncome"] = display_roster["MonthlyIncome"].map(lambda v: f"${v:,.0f}")
        st.dataframe(display_roster, use_container_width=True, hide_index=True)

# Tab 2: Retention Matrix
with tab_matrix:
    section_header("RETENTION MATRIX", "Engagement vs Performance",
                    "Top-right, high-risk bubbles are your priority retention conversations.")

    if settings.show_synthetic_performance_score:
        st.caption("⚠️ Performance score shown is SYNTHETIC placeholder data, not a real rating.")
        df_matrix = with_performance_score(filtered_df)
        fig_quad = px.scatter(
            df_matrix, x="EngagementScore", y="Performance_Score", color="Attrition_Risk_Level",
            size="MonthlyIncome", hover_data=["EmployeeID", "JobRole", "Department", "Attrition_Probability"],
            color_discrete_map=RISK_COLORS, template="plotly_dark",
            title="Engagement vs Performance (bubble size = income) — DEMO DATA",
        )
        fig_quad.add_vline(x=75, line_dash="dash", line_color="#64748b")
        fig_quad.add_hline(y=80, line_dash="dash", line_color="#64748b")
        fig_quad.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_quad, use_container_width=True)
    else:
        st.info(
            "This matrix needs a real performance-rating column to be meaningful. "
            "Wire one in from your HRIS, or set SHOW_SYNTHETIC_PERF=true to preview "
            "the layout with clearly-labeled placeholder data."
        )

# Tab 3: Skill Gaps & Department Skill Matrix Heatmap
with tab_skills:
    section_header("SKILLS", "Organization Skill Deficit & Heatmap Matrix")

    if gaps_df.empty:
        st.info("No skill gap summary available.")
    else:
        # Download Button for Skill Gaps CSV
        gap_csv = gaps_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Skill Deficit Summary (CSV)",
            data=gap_csv,
            file_name="organization_skill_gaps.csv",
            mime="text/csv"
        )
        
        sg1, sg2 = st.columns([3, 2])
        with sg1:
            fig_gaps = px.bar(gaps_df.head(12), x="MissingEmployeeCount", y="Skill", orientation="h",
                               color="Severity", color_discrete_map=RISK_COLORS, template="plotly_dark",
                               title="Missing Skills Across Enterprise")
            fig_gaps.update_layout(**PLOTLY_LAYOUT, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_gaps, use_container_width=True)
        with sg2:
            st.dataframe(gaps_df[["Skill", "MissingEmployeeCount", "Severity"]],
                         use_container_width=True, hide_index=True)
                         
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("#### 🔥 Department vs Skill Gap Coverage Heatmap")
        
        # Build Department vs Skill Heatmap matrix
        heatmap_records = []
        for dept in dept_options:
            sub = intel_df[intel_df["Department"] == dept]
            all_missing = []
            for m in sub["MissingSkills"]:
                if m != "None":
                    all_missing.extend([s.strip() for s in m.split(',')])
            counts = pd.Series(all_missing).value_counts()
            for skill in gaps_df.head(10)["Skill"]:
                heatmap_records.append({"Department": dept, "Skill": skill, "Count": int(counts.get(skill, 0))})
                
        heat_df = pd.DataFrame(heatmap_records)
        heat_pivot = heat_df.pivot(index="Skill", columns="Department", values="Count").fillna(0)
        
        fig_heat = px.imshow(
            heat_pivot,
            labels=dict(x="Department", y="Missing Skill", color="Deficit Count"),
            x=heat_pivot.columns,
            y=heat_pivot.index,
            color_continuous_scale="Reds",
            template="plotly_dark",
            title="Departmental Vulnerability Heatmap by Skill"
        )
        fig_heat.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_heat, use_container_width=True)

# Tab 4: Chat
with tab_chat:
    section_header("ASSISTANT", "HR AI Assistant",
                    "Ask about attrition, disengaged cohorts, skill gaps, or training recommendations.")

    p1, p2, p3, p4 = st.columns(4)
    sample_q = None
    if p1.button("🔥 High Risk Engineers"):
        sample_q = "Which software engineers are at high risk of quitting?"
    if p2.button("💡 Sales Skill Gaps"):
        sample_q = "What are the top missing skills in the Sales department?"
    if p3.button("🎓 MLOps Upskilling"):
        sample_q = "Who needs MLOps training?"
    if p4.button("⚡ Lowest Engagement"):
        sample_q = "Show disengaged managers with low engagement scores"

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Ask me about attrition risk, skill gaps, engagement, or upskilling."}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask HR Assistant a question...") or sample_q

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing HR Intelligence knowledge base..."):
                try:
                    result = get_chat_answer(user_query)
                    st.markdown(result.answer)
                    if result.origin == "local_rag_fallback":
                        st.caption("⚠️ Answered via local fallback — primary backend was unreachable.")
                    if result.sources:
                        with st.expander("Evidence sources retrieved"):
                            st.json(result.sources)
                    st.session_state.messages.append({"role": "assistant", "content": result.answer})
                except ChatBackendError as e:
                    error_msg = "Sorry, I can't reach the HR knowledge base right now. Please try again shortly."
                    st.error(error_msg)
                    logging.getLogger("hr_workspace.chat").error("Chat failure: %s", e)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Tab 5: Employee Lookup & Peer Benchmarking Radar Chart
with tab_lookup:
    section_header("LOOKUP", "Single Employee Intelligence & Peer Benchmarking")

    emp_ids = filtered_df["EmployeeID"].tolist()
    selected_id = st.selectbox("Select Employee ID", emp_ids)
    emp_record = filtered_df.loc[filtered_df["EmployeeID"] == selected_id].iloc[0]

    st.markdown('<div class="lookup-card">', unsafe_allow_html=True)

    l1, l2, l3 = st.columns(3)
    for col, label, value in [
        (l1, "Department", html.escape(str(emp_record["Department"]))),
        (l2, "Job Role", html.escape(str(emp_record["JobRole"]))),
        (l3, "Monthly Income", f"${emp_record['MonthlyIncome']:,.2f}"),
    ]:
        col.markdown(f'<div class="lookup-field-label">{label}</div>'
                      f'<div class="lookup-field-value">{value}</div>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Attrition Probability", f"{emp_record['Attrition_Probability'] * 100:.1f}%")
    m1.markdown(risk_badge(emp_record["Attrition_Risk_Level"]), unsafe_allow_html=True)
    m2.metric("Engagement Score", f"{emp_record['EngagementScore']:.1f}%")
    m2.markdown(risk_badge(emp_record["Engagement_Category"]), unsafe_allow_html=True)
    m3.metric("Missing Skills Count", f"{emp_record['SkillGapCount']} Skills")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="lookup-field-label">Current Possessed Skills</div>'
                f'<div class="lookup-field-value">{html.escape(str(emp_record["CurrentSkills"]))}</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="lookup-field-label">Identified Missing Skills</div>'
                f'<div class="lookup-field-value">{html.escape(str(emp_record["MissingSkills"]))}</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="lookup-field-label">AI Recommended Training</div>'
                f'<div class="lookup-field-value">{html.escape(str(emp_record["UpskillingRecommendation"]))}</div>',
                unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    
    # Peer Benchmarking Radar Chart
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("#### 🕸️ Peer Benchmarking Radar (Employee vs Department Average)")
    
    dept_avg = intel_df[intel_df["Department"] == emp_record["Department"]]
    
    categories = ["Engagement Score", "Income Level", "Skill Coverage", "Retention Probability"]
    
    emp_vals = [
        float(emp_record["EngagementScore"]),
        float((emp_record["MonthlyIncome"] / max_inc) * 100),
        float(max(0, 100 - (emp_record["SkillGapCount"] * 25))),
        float((1 - emp_record["Attrition_Probability"]) * 100)
    ]
    
    dept_vals = [
        float(dept_avg["EngagementScore"].mean()),
        float((dept_avg["MonthlyIncome"].mean() / max_inc) * 100),
        float(max(0, 100 - (dept_avg["SkillGapCount"].mean() * 25))),
        float((1 - dept_avg["Attrition_Probability"].mean()) * 100)
    ]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=emp_vals, theta=categories, fill='toself', name=f"Employee #{selected_id}"))
    fig_radar.add_trace(go.Scatterpolar(r=dept_vals, theta=categories, fill='toself', name=f"{emp_record['Department']} Dept Avg"))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        template="plotly_dark",
        title=f"Peer Benchmarking: Employee #{selected_id} vs {emp_record['Department']} Department Average"
    )
    fig_radar.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig_radar, use_container_width=True)