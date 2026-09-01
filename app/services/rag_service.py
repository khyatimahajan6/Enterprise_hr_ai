import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from app.utils.config import DATA_PROCESSED_DIR
from app.utils.logger import logger

class HRKnowledgeRAG:
    _instance = None

    def __init__(self):
        self.intel_df = pd.DataFrame()
        self.gaps_df = pd.DataFrame()
        self.load_data()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = HRKnowledgeRAG()
        return cls._instance

    def load_data(self):
        intel_path = os.path.join(DATA_PROCESSED_DIR, "employee_intelligence.csv")
        gaps_path = os.path.join(DATA_PROCESSED_DIR, "organization_skill_gap_summary.csv")

        if os.path.exists(intel_path):
            self.intel_df = pd.read_csv(intel_path)
        if os.path.exists(gaps_path):
            self.gaps_df = pd.read_csv(gaps_path)
        logger.info(f"RAG Engine loaded {len(self.intel_df)} employee intelligence records.")

    def query(self, user_prompt: str) -> Dict[str, Any]:
        prompt_lower = user_prompt.lower().strip()
        df = self.intel_df

        if df.empty:
            return {
                "answer": "The HR intelligence database is currently empty or unavailable.",
                "retrieved_sources": [],
                "confidence": "LOW"
            }

        retrieved = []
        answer_lines = []

        # Intent 1: Attrition / High Risk Employees
        if any(w in prompt_lower for w in ["attrition", "quitting", "leave", "leaving", "risk", "high risk", "at-risk"]):
            if "software" in prompt_lower or "engineer" in prompt_lower:
                subset = df[(df["Attrition_Risk_Level"] == "HIGH") & (df["JobRole"].str.contains("Software|Research", case=False))]
            elif "sales" in prompt_lower:
                subset = df[(df["Attrition_Risk_Level"] == "HIGH") & (df["Department"] == "Sales")]
            elif "manager" in prompt_lower:
                subset = df[(df["Attrition_Risk_Level"] == "HIGH") & (df["JobRole"].str.contains("Manager", case=False))]
            else:
                subset = df[df["Attrition_Risk_Level"] == "HIGH"]

            top_subset = subset.sort_values("Attrition_Probability", ascending=False).head(5)
            count = len(subset)

            answer_lines.append(f"Found **{count} employees** identified at **HIGH Attrition Risk** based on workload, overtime, and satisfaction metrics.")
            if not top_subset.empty:
                answer_lines.append("\n**Top At-Risk Personnel Highlights:**")
                for _, r in top_subset.iterrows():
                    answer_lines.append(
                        f"• **Employee #{r['EmployeeID']}** ({r['JobRole']}, {r['Department']}) — "
                        f"Risk Probability: **{r['Attrition_Probability']*100:.1f}%** | OverTime: {r['OverTime']} | "
                        f"Missing Skills: `{r['MissingSkills']}`"
                    )
                    retrieved.append({
                        "EmployeeID": int(r["EmployeeID"]),
                        "JobRole": r["JobRole"],
                        "Department": r["Department"],
                        "AttritionProbability": float(r["Attrition_Probability"]),
                        "RiskLevel": r["Attrition_Risk_Level"],
                        "MissingSkills": r["MissingSkills"]
                    })

        # Intent 2: Skill Gaps & Training / Upskilling
        elif any(w in prompt_lower for w in ["skill", "gap", "upskill", "course", "training", "learn", "mlops", "aws", "python", "docker"]):
            if "mlops" in prompt_lower:
                subset = df[df["MissingSkills"].str.contains("MLOps|Machine Learning|Python", case=False, na=False)]
                skill_name = "MLOps / Machine Learning"
            elif "salesforce" in prompt_lower:
                subset = df[df["MissingSkills"].str.contains("Salesforce|CRM", case=False, na=False)]
                skill_name = "Salesforce / CRM"
            elif "sales" in prompt_lower:
                subset = df[(df["Department"] == "Sales") & (df["SkillGapCount"] > 0)]
                skill_name = "Sales Department Skills"
            elif "r&d" in prompt_lower or "research" in prompt_lower:
                subset = df[(df["Department"].str.contains("Research", case=False)) & (df["SkillGapCount"] > 0)]
                skill_name = "R&D Skills"
            else:
                subset = df[df["SkillGapCount"] > 0]
                skill_name = "Key Organizational Skills"

            top_subset = subset.head(5)
            count = len(subset)
            answer_lines.append(f"Skill Gap Analysis for **{skill_name}**: Identified **{count} employees** requiring upskilling.")
            if not top_subset.empty:
                answer_lines.append("\n**Recommended Upskilling Pathways:**")
                for _, r in top_subset.iterrows():
                    answer_lines.append(
                        f"• **Employee #{r['EmployeeID']}** ({r['JobRole']}) — "
                        f"Missing: `{r['MissingSkills']}` → **Recommended:** *{r['UpskillingRecommendation']}*"
                    )
                    retrieved.append({
                        "EmployeeID": int(r["EmployeeID"]),
                        "JobRole": r["JobRole"],
                        "MissingSkills": r["MissingSkills"],
                        "Recommendation": r["UpskillingRecommendation"]
                    })

        # Intent 3: Engagement & Disengagement
        elif any(w in prompt_lower for w in ["engagement", "disengaged", "satisfaction", "low engagement", "score"]):
            subset = df.sort_values("EngagementScore", ascending=True).head(5)
            avg_eng = df["EngagementScore"].mean()
            answer_lines.append(f"Company-wide average engagement score is **{avg_eng:.1f}%**.")
            answer_lines.append("\n**Employees with Lowest Engagement Scores:**")
            for _, r in subset.iterrows():
                answer_lines.append(
                    f"• **Employee #{r['EmployeeID']}** ({r['JobRole']}, {r['Department']}) — "
                    f"Engagement Score: **{r['EngagementScore']:.1f}%** ({r['Engagement_Category']}) | Attrition Risk: {r['Attrition_Risk_Level']}"
                )
                retrieved.append({
                    "EmployeeID": int(r["EmployeeID"]),
                    "JobRole": r["JobRole"],
                    "EngagementScore": float(r["EngagementScore"]),
                    "EngagementCategory": r["Engagement_Category"]
                })

        # Fallback Intent: General HR Summary
        else:
            total_emp = len(df)
            high_risk_cnt = len(df[df["Attrition_Risk_Level"] == "HIGH"])
            avg_eng = df["EngagementScore"].mean()
            top_gaps = self.gaps_df.head(3)["Skill"].tolist() if not self.gaps_df.empty else ["MLOps", "Python", "Salesforce"]

            answer_lines.append(f"### HR Workforce Intelligence Overview")
            answer_lines.append(f"• **Total Workforce Analyzed:** {total_emp:,} employees")
            answer_lines.append(f"• **High Attrition Risk Personnel:** {high_risk_cnt} ({high_risk_cnt/total_emp*100:.1f}%)")
            answer_lines.append(f"• **Average Workforce Engagement:** {avg_eng:.1f}%")
            answer_lines.append(f"• **Critical Organization Skill Deficits:** {', '.join(top_gaps)}")
            answer_lines.append("\n*Tip: Try asking specific questions like 'Which software engineers are at high risk?' or 'Who needs MLOps training?'*")

        return {
            "answer": "\n".join(answer_lines),
            "retrieved_sources": retrieved,
            "confidence": "HIGH"
        }
