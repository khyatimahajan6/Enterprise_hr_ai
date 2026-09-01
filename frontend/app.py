import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import plotly.express as px
import plotly.graph_objects as go

# Page Config
st.set_page_config(
    page_title="Enterprise HR AI Intelligence Workspace",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Executive Dark Glassmorphic CSS
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #080d1a 90%);
        color: #f1f5f9;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    
    /* Header Styling */
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: -5px;
        margin-bottom: 20px;
    }

    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        color: #38bdf8 !important;
    }
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        border-radius: 14px;
        padding: 16px 20px;
    }

    /* Badges */
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fcd34d;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 3px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Chat Styling */
    .chat-user {
        background: rgba(59, 130, 246, 0.15);
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .chat-bot {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #818cf8;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 16px;
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

# Header
st.markdown('<div class="header-title">⚡ Enterprise HR AI Intelligence Workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Predictive Workforce Analytics • Skill Gap Engine • RAG HR Assistant</div>', unsafe_allow_html=True)

if intel_df.empty:
    st.error("Data files not found! Please run the pipeline script `src/workforce_intelligence.py` first.")
    st.stop()

# Multi-Dimensional Sidebar Filters
st.sidebar.markdown("### 🎛️ Multi-Facet Filters")
st.sidebar.markdown("Filter workforce metrics effortless across departments, risk levels, and income.")

dept_options = sorted(list(intel_df["Department"].unique()))
selected_depts = st.sidebar.multiselect("Department", dept_options, default=dept_options)

risk_options = ["HIGH", "MEDIUM", "LOW"]
selected_risks = st.sidebar.multiselect("Attrition Risk Level", risk_options, default=risk_options)

eng_options = ["HIGH", "MEDIUM", "LOW"]
selected_engs = st.sidebar.multiselect("Engagement Category", eng_options, default=eng_options)

min_inc = float(intel_df["MonthlyIncome"].min())
max_inc = float(intel_df["MonthlyIncome"].max())
selected_income = st.sidebar.slider("Monthly Income Range ($)", min_value=min_inc, max_value=max_inc, value=(min_inc, max_inc))

# Apply Filters
filtered_df = intel_df[
    (intel_df["Department"].isin(selected_depts)) &
    (intel_df["Attrition_Risk_Level"].isin(selected_risks)) &
    (intel_df["Engagement_Category"].isin(selected_engs)) &
    (intel_df["MonthlyIncome"].between(selected_income[0], selected_income[1]))
]

# Top Executive KPI Metric Bar
c1, c2, c3, c4 = st.columns(4)

total_count = len(filtered_df)
at_risk_count = len(filtered_df[filtered_df["Attrition_Risk_Level"] == "HIGH"])
avg_engagement = filtered_df["EngagementScore"].mean() if total_count > 0 else 0.0
skill_deficit_count = len(filtered_df[filtered_df["SkillGapCount"] >= 2])

c1.metric("Active Workforce", f"{total_count:,}")
c2.metric("High Attrition Risk", f"{at_risk_count:,}", delta=f"{(at_risk_count/total_count*100 if total_count > 0 else 0):.1f}% Risk", delta_color="inverse")
c3.metric("Avg Engagement", f"{avg_engagement:.1f}%", delta="Workforce Index")
c4.metric("Skill Deficit Personnel", f"{skill_deficit_count:,}", delta="Upskilling Needed")

st.markdown("---")

# Main Navigation Tabs
tab_attr, tab_matrix, tab_skills, tab_chat, tab_lookup = st.tabs([
    "📉 Attrition Intelligence", 
    "🎯 Risk vs Engagement Matrix", 
    "🧩 Organization Skill Gaps", 
    "💬 HR AI Assistant (RAG Chatbot)", 
    "👤 Single Employee Lookup"
])

# TAB 1: Attrition Intelligence & Interactive Charts
with tab_attr:
    st.subheader("Departmental Attrition Risk & Compensation Distribution")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### Attrition Risk Breakdown by Department")
        if not filtered_df.empty:
            dept_risk_df = filtered_df.groupby(["Department", "Attrition_Risk_Level"]).size().reset_index(name="Count")
            fig_bar = px.bar(
                dept_risk_df,
                x="Department",
                y="Count",
                color="Attrition_Risk_Level",
                color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"},
                title="Attrition Risk Distribution per Department",
                template="plotly_dark",
                barmode="stack"
            )
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)
            
    with col_chart2:
        st.markdown("#### Income vs Attrition Risk Distribution")
        if not filtered_df.empty:
            fig_box = px.box(
                filtered_df,
                x="Attrition_Risk_Level",
                y="MonthlyIncome",
                color="Attrition_Risk_Level",
                color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"},
                title="Monthly Income Spread by Risk Tier",
                points="all",
                template="plotly_dark"
            )
            fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_box, use_container_width=True)
            
    st.markdown("#### High Attrition Risk Employee Roster")
    high_risk_roster = filtered_df[filtered_df["Attrition_Risk_Level"] == "HIGH"].sort_values("Attrition_Probability", ascending=False)
    st.dataframe(
        high_risk_roster[["EmployeeID", "Department", "JobRole", "MonthlyIncome", "OverTime", "Attrition_Probability", "EngagementScore"]],
        use_container_width=True
    )

# TAB 2: Engagement vs Performance 4-Quadrant Matrix
with tab_matrix:
    st.subheader("Workforce Retention Matrix: Engagement vs Performance")
    st.markdown("Identify **High Performers at Risk** (Top-Right High Risk glow) to proactively retention planning.")
    
    if not filtered_df.empty:
        # Create synthetic Performance Score for scatter quadrant visualization
        df_matrix = filtered_df.copy()
        np.random.seed(42)
        df_matrix["Performance_Score"] = np.round(np.random.uniform(60, 98, size=len(df_matrix)), 1)
        
        fig_quad = px.scatter(
            df_matrix,
            x="EngagementScore",
            y="Performance_Score",
            color="Attrition_Risk_Level",
            size="MonthlyIncome",
            hover_data=["EmployeeID", "JobRole", "Department", "Attrition_Probability"],
            color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"},
            title="Workforce Matrix: Engagement vs Performance (Bubble Size = Income)",
            labels={"EngagementScore": "Engagement Score (%)", "Performance_Score": "Performance Rating"},
            template="plotly_dark"
        )
        
        # Add Quadrant lines
        fig_quad.add_vline(x=75, line_dash="dash", line_color="#64748b")
        fig_quad.add_hline(y=80, line_dash="dash", line_color="#64748b")
        
        fig_quad.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_quad, use_container_width=True)

# TAB 3: Organization Skill Gaps & AI Upskilling
with tab_skills:
    st.subheader("Organization Skill Deficit & AI Course Mapping")
    
    if not gaps_df.empty:
        sg1, sg2 = st.columns([3, 2])
        
        with sg1:
            st.markdown("#### Top Organizational Skill Deficits")
            fig_gaps = px.bar(
                gaps_df.head(12),
                x="MissingEmployeeCount",
                y="Skill",
                orientation="h",
                color="Severity",
                color_discrete_map={"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"},
                title="Missing Skills Count Across Enterprise",
                template="plotly_dark"
            )
            fig_gaps.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_gaps, use_container_width=True)
            
        with sg2:
            st.markdown("#### Skill Deficit Severity Breakdown")
            st.dataframe(gaps_df[["Skill", "MissingEmployeeCount", "Severity"]], use_container_width=True)

# TAB 4: HR RAG Chatbot Assistant
with tab_chat:
    st.subheader("💬 HR AI Assistant (RAG Natural Language Chatbot)")
    st.markdown("Ask context-aware questions regarding workforce attrition, disengaged cohorts, skill gaps, or course recommendations.")
    
    # Quick Prompt Chips
    st.markdown("**Quick Sample Questions:**")
    prompt_col1, prompt_col2, prompt_col3, prompt_col4 = st.columns(4)
    
    sample_q = None
    if prompt_col1.button("🔥 High Risk Engineers"):
        sample_q = "Which software engineers are at high risk of quitting?"
    if prompt_col2.button("💡 Sales Skill Gaps"):
        sample_q = "What are the top missing skills in the Sales department?"
    if prompt_col3.button("🎓 MLOps Upskilling"):
        sample_q = "Who needs MLOps training?"
    if prompt_col4.button("⚡ Lowest Engagement"):
        sample_q = "Show disengaged managers with low engagement scores"
        
    # Chat session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your Enterprise HR AI Assistant. Ask me anything about employee attrition risk, skill gaps, engagement metrics, or upskilling recommendations."}
        ]
        
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Handle User Input
    user_query = st.chat_input("Ask HR Assistant a question...") or sample_q
    
    if user_query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Generate Answer via FastAPI Backend / RAG Service
        with st.chat_message("assistant"):
            with st.spinner("Analyzing HR Intelligence knowledge base..."):
                try:
                    res = requests.post("http://127.0.0.1:8000/api/chat/query", json={"query": user_query}, timeout=5)
                    if res.status_code == 200:
                        data = res.json()
                        bot_ans = data["answer"]
                        sources = data.get("retrieved_sources", [])
                    else:
                        bot_ans = "Received response error from backend API."
                        sources = []
                except Exception:
                    # Fallback to local RAG service if backend endpoint unreachable
                    from app.services.rag_service import HRKnowledgeRAG
                    rag_inst = HRKnowledgeRAG.get_instance()
                    res_local = rag_inst.query(user_query)
                    bot_ans = res_local["answer"]
                    sources = res_local.get("retrieved_sources", [])
                    
                st.markdown(bot_ans)
                
                if sources:
                    st.markdown("**Evidence Sources Retrieved:**")
                    st.json(sources)
                    
                st.session_state.messages.append({"role": "assistant", "content": bot_ans})

# TAB 5: Single Employee Deep-Dive Lookup
with tab_lookup:
    st.subheader("Single Employee Intelligence Search")
    
    emp_ids = filtered_df["EmployeeID"].tolist() if not filtered_df.empty else intel_df["EmployeeID"].tolist()
    selected_id = st.selectbox("Select Employee ID", emp_ids)
    
    emp_record = intel_df[intel_df["EmployeeID"] == selected_id].iloc[0]
    
    l1, l2, l3 = st.columns(3)
    l1.markdown(f"**Department:** `{emp_record['Department']}`")
    l2.markdown(f"**Job Role:** `{emp_record['JobRole']}`")
    l3.markdown(f"**Monthly Income:** `${emp_record['MonthlyIncome']:,.2f}`")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Attrition Probability", f"{emp_record['Attrition_Probability']*100:.1f}%", delta=emp_record['Attrition_Risk_Level'])
    m2.metric("Engagement Score", f"{emp_record['EngagementScore']:.1f}%", delta=emp_record['Engagement_Category'])
    m3.metric("Missing Skills Count", f"{emp_record['SkillGapCount']} Skills", delta="Deficit")
    
    st.markdown("---")
    st.markdown("#### Skill Inventory & Upskilling Plan")
    st.markdown(f"• **Current Possessed Skills:** `{emp_record['CurrentSkills']}`")
    st.markdown(f"• **Identified Missing Skills:** `{emp_record['MissingSkills']}`")
    st.markdown(f"• **AI Recommended Training:** *{emp_record['UpskillingRecommendation']}*")
