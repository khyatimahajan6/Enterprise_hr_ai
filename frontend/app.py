import streamlit as st
import pandas as pd
import numpy as np
import os
import requests

# Page Config
st.set_page_config(
    page_title="Enterprise HR AI Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Glassmorphism CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.04);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .kpi-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.7) 0%, rgba(15,23,42,0.8) 100%);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 20px;
        border-radius: 14px;
        text-align: center;
    }
    .risk-high {
        color: #ff4d4d;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffaa00;
        font-weight: bold;
    }
    .risk-low {
        color: #00cc66;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Data Loaders
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    intel_path = os.path.join(base_dir, "data", "processed", "employee_intelligence.csv")
    gaps_path = os.path.join(base_dir, "data", "processed", "organization_skill_gap_summary.csv")
    
    intel_df = pd.read_csv(intel_path) if os.path.exists(intel_path) else pd.DataFrame()
    gaps_df = pd.read_csv(gaps_path) if os.path.exists(gaps_path) else pd.DataFrame()
    
    return intel_df, gaps_df

intel_df, gaps_df = load_data()

# Header Section
st.title("🤖 Enterprise HR AI Platform")
st.markdown("### Predictive Workforce Intelligence, Skill Gap Engine & Upskilling Pathways")
st.markdown("---")

if intel_df.empty:
    st.error("Data files not found! Please ensure data processing and ML pipeline have been executed.")
    st.stop()

# Sidebar Filters
st.sidebar.header("🔍 Global Filters")
departments = ["All"] + sorted(list(intel_df["Department"].unique()))
selected_dept = st.sidebar.selectbox("Filter by Department", departments)

if selected_dept != "All":
    filtered_df = intel_df[intel_df["Department"] == selected_dept]
else:
    filtered_df = intel_df

# Top KPI Summary Cards
col1, col2, col3, col4 = st.columns(4)

total_emp = len(filtered_df)
high_risk = len(filtered_df[filtered_df["Attrition_Risk_Level"] == "HIGH"])
avg_eng = filtered_df["EngagementScore"].mean()
high_gap_emp = len(filtered_df[filtered_df["SkillGapCount"] >= 3])

col1.metric("Total Workforce", f"{total_emp:,}")
col2.metric("At-Risk Employees", f"{high_risk:,}", delta=f"{(high_risk/total_emp*100):.1f}% Risk", delta_color="inverse")
col3.metric("Avg Engagement Score", f"{avg_eng:.1f}%", delta="Stable")
col4.metric("Skill Deficit Cohort", f"{high_gap_emp:,}", delta="Requires Training")

st.markdown("---")

# Main View Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📉 Attrition Intelligence", "🧩 Skill Gap Analysis", "📚 AI Upskilling Pathways", "👤 Single Employee Lookup"])

with tab1:
    st.subheader("Attrition Risk Distribution by Department & Role")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.write("#### Risk Breakdown by Risk Level")
        risk_counts = filtered_df["Attrition_Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        st.bar_chart(risk_counts.set_index("Risk Level"))
        
    with col_right:
        st.write("#### Top High Risk Roles")
        high_risk_roles = filtered_df[filtered_df["Attrition_Risk_Level"] == "HIGH"]["JobRole"].value_counts().head(5)
        st.dataframe(high_risk_roles, use_container_width=True)
        
    st.write("#### High Risk Attrition Roster")
    st.dataframe(
        filtered_df[filtered_df["Attrition_Risk_Level"] == "HIGH"][
            ["EmployeeID", "Department", "JobRole", "MonthlyIncome", "OverTime", "Attrition_Probability", "EngagementScore"]
        ].sort_values("Attrition_Probability", ascending=False),
        use_container_width=True
    )

with tab2:
    st.subheader("Organization-Wide Skill Shortage Matrix")
    
    if not gaps_df.empty:
        col_g1, col_g2 = st.columns([2, 1])
        
        with col_g1:
            st.write("#### Top Missing Skills Across Workforce")
            st.bar_chart(gaps_df.head(10).set_index("Skill")["MissingEmployeeCount"])
            
        with col_g2:
            st.write("#### Skill Deficit Severity")
            st.dataframe(gaps_df[["Skill", "MissingEmployeeCount", "Severity"]], use_container_width=True)

with tab3:
    st.subheader("Targeted AI Upskilling & Training Recommendations")
    st.markdown("Automated recommendation mapping missing skills to specialized professional courses.")
    
    rec_df = filtered_df[filtered_df["SkillGapCount"] > 0][
        ["EmployeeID", "Department", "JobRole", "MissingSkills", "UpskillingRecommendation"]
    ]
    st.dataframe(rec_df, use_container_width=True)

with tab4:
    st.subheader("Single Employee Intelligence Profile")
    
    emp_ids = filtered_df["EmployeeID"].tolist()
    selected_id = st.selectbox("Select Employee ID", emp_ids)
    
    emp_row = filtered_df[filtered_df["EmployeeID"] == selected_id].iloc[0]
    
    e1, e2, e3 = st.columns(3)
    e1.markdown(f"**Department:** {emp_row['Department']}")
    e2.markdown(f"**Job Role:** {emp_row['JobRole']}")
    e3.markdown(f"**Monthly Income:** ${emp_row['MonthlyIncome']:,.2f}")
    
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Attrition Probability", f"{emp_row['Attrition_Probability']*100:.1f}%", delta=emp_row['Attrition_Risk_Level'])
    with r2:
        st.metric("Engagement Score", f"{emp_row['EngagementScore']:.1f}%", delta=emp_row['Engagement_Category'])
        
    st.markdown("---")
    st.write("#### 🛠️ Skills Profile & Gap Analysis")
    st.markdown(f"**Current Possessed Skills:** `{emp_row['CurrentSkills']}`")
    st.markdown(f"**Identified Skill Gaps:** `{emp_row['MissingSkills']}`")
    st.markdown(f"**Recommended Course:** `{emp_row['UpskillingRecommendation']}`")
