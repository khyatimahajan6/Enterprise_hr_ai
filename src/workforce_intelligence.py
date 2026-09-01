import os
import json
import joblib
import pandas as pd
import numpy as np

PROCESSED_DIR = r"c:\Users\khyat\OneDrive\Desktop\HrProject\enterprise_hr_ai\data\processed"
MODELS_DIR = r"c:\Users\khyat\OneDrive\Desktop\HrProject\enterprise_hr_ai\models"

def run_workforce_intelligence():
    print("=== Starting Workforce Intelligence Engine ===")
    
    # 1. Load Processed Datasets
    attr_df = pd.read_csv(os.path.join(PROCESSED_DIR, "employee_attrition_processed.csv"))
    eng_df = pd.read_csv(os.path.join(PROCESSED_DIR, "engagement_processed.csv"))
    skills_df = pd.read_csv(os.path.join(PROCESSED_DIR, "employee_skills_controlled.csv"))
    
    # 2. Load ML Attrition Pipeline
    model_path = os.path.join(MODELS_DIR, "attrition_pipeline.joblib")
    if os.path.exists(model_path):
        pipeline = joblib.load(model_path)
        
        # Prepare features for attrition prediction
        cat_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
        num_features = [
            'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
            'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
            'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
            'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
            'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
            'YearsSinceLastPromotion', 'YearsWithCurrManager'
        ]
        
        # Ensure engineered features exist
        attr_df['Income_Per_Year_At_Company'] = attr_df['MonthlyIncome'] / (attr_df['YearsAtCompany'] + 1)
        attr_df['Promotion_Gap_Ratio'] = attr_df['YearsSinceLastPromotion'] / (attr_df['YearsAtCompany'] + 1)
        attr_df['Satisfaction_Index'] = (attr_df['JobSatisfaction'] + attr_df['EnvironmentSatisfaction'] + 
                                         attr_df['RelationshipSatisfaction'] + attr_df['WorkLifeBalance']) / 4.0
        attr_df['Experience_Ratio'] = attr_df['YearsAtCompany'] / (attr_df['TotalWorkingYears'] + 1)
        
        eng_num_features = num_features + ['Income_Per_Year_At_Company', 'Promotion_Gap_Ratio', 'Satisfaction_Index', 'Experience_Ratio']
        
        X = attr_df[cat_features + eng_num_features]
        attr_probs = pipeline.predict_proba(X)[:, 1]
        attr_df['Attrition_Probability'] = np.round(attr_probs, 4)
    else:
        print("[!] Warning: Model pipeline not found. Generating default probabilities.")
        attr_df['Attrition_Probability'] = np.random.uniform(0.05, 0.40, size=len(attr_df))
        
    attr_df['Attrition_Risk_Level'] = pd.cut(
        attr_df['Attrition_Probability'],
        bins=[-0.01, 0.25, 0.50, 1.0],
        labels=['LOW', 'MEDIUM', 'HIGH']
    )
    
    # 3. Role Required Skills Map
    role_required_skills = {
        'Sales Executive': ['CRM Software', 'Salesforce', 'Negotiation', 'Communication', 'Lead Generation', 'Excel'],
        'Research Scientist': ['Python', 'Data Analysis', 'Statistics', 'R', 'Machine Learning', 'Lab Equipment'],
        'Laboratory Technician': ['Lab Safety', 'Sample Preparation', 'Quality Control', 'Data Entry', 'Excel'],
        'Manufacturing Director': ['Supply Chain', 'Operations Management', 'ERP', 'Budgeting', 'Lean Six Sigma'],
        'Healthcare Representative': ['Patient Care', 'Medical Records', 'Customer Service', 'EMR Software', 'Communication'],
        'Manager': ['Leadership', 'Project Management', 'Strategic Planning', 'Budgeting', 'Performance Management'],
        'Sales Representative': ['Cold Calling', 'Salesforce', 'Customer Service', 'Communication', 'CRM'],
        'Research Director': ['R&D Strategy', 'Grant Writing', 'Data Analysis', 'Python', 'Leadership'],
        'Human Resources': ['HRIS', 'Recruiting', 'Employee Relations', 'Payroll', 'Compliance', 'Workday'],
        'Software Engineer': ['Python', 'Java', 'SQL', 'Git', 'Docker', 'REST API', 'Data Structures'],
        'Data Analyst': ['SQL', 'Python', 'Tableau', 'PowerBI', 'Excel', 'Statistics'],
        'Cybersecurity Specialist': ['Network Security', 'Firewalls', 'Python', 'Linux', 'Incident Response'],
        'Marketing Executive': ['SEO', 'Google Analytics', 'Content Strategy', 'Social Media', 'Copywriting']
    }
    
    # 4. Upskilling Course Recommendation Registry
    course_registry = {
        'Python': 'Advanced Python Programming & Data Science Masterclass',
        'Machine Learning': 'Applied Machine Learning & MLOps Specialization',
        'Docker': 'Containerization & DevOps Engineering with Docker & Kubernetes',
        'SQL': 'Enterprise SQL Database Querying & Data Modeling',
        'Salesforce': 'Salesforce CRM Administrator & Cloud Management',
        'Lead Generation': 'B2B Digital Marketing & Modern Sales Pipelines',
        'R': 'Statistical Computing & Data Visualization in R',
        'Lab Safety': 'ISO Certified Laboratory Safety & Bio-hazard Compliance',
        'Quality Control': 'Six Sigma Quality Assurance & Process Improvement',
        'Supply Chain': 'Global Supply Chain Operations & Logistics Architecture',
        'Lean Six Sigma': 'Lean Six Sigma Green Belt Enterprise Certification',
        'EMR Software': 'Healthcare Electronic Medical Records Administration',
        'Strategic Planning': 'Executive Leadership & Strategic Workforce Planning',
        'Grant Writing': 'Research Grant Writing & Innovation Funding Workshop',
        'HRIS': 'Modern HRIS Technology & HR Analytics Management',
        'Workday': 'Workday Human Capital Management Fundamentals',
        'Tableau': 'Executive Business Intelligence Dashboards in Tableau',
        'PowerBI': 'Data Analytics & Reporting with Microsoft PowerBI',
        'Network Security': 'Certified Information Systems Security Professional (CISSP)',
        'SEO': 'Search Engine Optimization & Modern Digital Growth'
    }
    
    # Group employee skills
    emp_skills_grouped = skills_df.groupby('EmployeeID')['CurrentSkill'].apply(list).to_dict()
    
    records = []
    
    # Merge Attrition Cohort (1470 employees)
    for idx, row in attr_df.iterrows():
        emp_id = int(row['EmployeeNumber'])
        role = str(row['JobRole'])
        dept = str(row['Department'])
        att_prob = float(row['Attrition_Probability'])
        risk_lvl = str(row['Attrition_Risk_Level'])
        income = float(row['MonthlyIncome'])
        overtime = str(row['OverTime'])
        
        # Skill Gaps
        req_skills = role_required_skills.get(role, ['Communication', 'Excel', 'Problem Solving'])
        curr_skills = emp_skills_grouped.get(emp_id, [])
        
        missing_skills = [s for s in req_skills if s not in curr_skills]
        
        # Recommendations
        recs = [course_registry.get(s, f"Professional Skill Development: {s}") for s in missing_skills[:2]]
        rec_str = "; ".join(recs) if recs else "Role Proficiency Achieved"
        
        # Mock matching engagement score for joined view
        eng_score = round(float(np.random.uniform(55, 95)), 2)
        eng_cat = 'HIGH' if eng_score >= 75 else ('MEDIUM' if eng_score >= 50 else 'LOW')
        
        records.append({
            'EmployeeID': emp_id,
            'Name': f"Employee #{emp_id}",
            'Department': dept,
            'JobRole': role,
            'MonthlyIncome': income,
            'OverTime': overtime,
            'Attrition_Probability': att_prob,
            'Attrition_Risk_Level': risk_lvl,
            'EngagementScore': eng_score,
            'Engagement_Category': eng_cat,
            'CurrentSkills': ", ".join(curr_skills) if curr_skills else "None",
            'MissingSkills': ", ".join(missing_skills) if missing_skills else "None",
            'SkillGapCount': len(missing_skills),
            'UpskillingRecommendation': rec_str
        })
        
    intel_df = pd.DataFrame(records)
    out_path = os.path.join(PROCESSED_DIR, "employee_intelligence.csv")
    intel_df.to_csv(out_path, index=False)
    print(f"[+] Master Employee Intelligence Table created -> {out_path} ({intel_df.shape[0]} employees)")
    
    # Organization-Wide Skill Gap Summary
    all_missing = []
    for m in intel_df['MissingSkills']:
        if m != 'None':
            all_missing.extend([s.strip() for s in m.split(',')])
            
    gap_counts = pd.Series(all_missing).value_counts().reset_index()
    gap_counts.columns = ['Skill', 'MissingEmployeeCount']
    gap_counts['Severity'] = gap_counts['MissingEmployeeCount'].apply(
        lambda c: 'HIGH' if c >= 100 else ('MEDIUM' if c >= 50 else 'LOW')
    )
    
    gap_summary_path = os.path.join(PROCESSED_DIR, "organization_skill_gap_summary.csv")
    gap_counts.to_csv(gap_summary_path, index=False)
    print(f"[+] Organization Skill Gap Summary created -> {gap_summary_path} ({len(gap_counts)} unique skill gaps)")
    
    print("=== Workforce Intelligence Engine Complete ===")

if __name__ == "__main__":
    run_workforce_intelligence()
