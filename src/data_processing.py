import os
import pandas as pd
import numpy as np

RAW_DIR = r"c:\Users\khyat\OneDrive\Desktop\HrProject\enterprise_hr_ai\data\raw"
PROCESSED_DIR = r"c:\Users\khyat\OneDrive\Desktop\HrProject\enterprise_hr_ai\data\processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_attrition_data():
    path = os.path.join(RAW_DIR, "employee_attrition.csv")
    df = pd.read_csv(path)
    
    # Target encoding: Yes -> 1, No -> 0
    df['Attrition_Binary'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Data type cleaning & feature validation
    df['Age'] = df['Age'].astype(int)
    df['MonthlyIncome'] = df['MonthlyIncome'].astype(float)
    df['YearsAtCompany'] = df['YearsAtCompany'].astype(int)
    df['OverTime_Binary'] = df['OverTime'].map({'Yes': 1, 'No': 0})
    
    out_path = os.path.join(PROCESSED_DIR, "employee_attrition_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"[+] Processed employee_attrition.csv -> {out_path} ({df.shape[0]} rows)")
    return df

def process_engagement_data():
    path = os.path.join(RAW_DIR, "hr_performance_engagement.csv")
    df = pd.read_csv(path)
    
    # Strip whitespace in string columns
    df['Department'] = df['Department'].str.strip()
    df['Job Role'] = df['Job Role'].str.strip()
    
    # Calculate composite Engagement Score (0 - 100)
    # Weights: Attendance(20%), Task Completion(30%), Peer Rating normalized(20%), Manager Feedback normalized(20%), KPI Score(10%)
    peer_norm = (df['Peer Rating'] / 5.0) * 100
    mgr_norm = (df['Manager Feedback'] / 5.0) * 100
    
    df['EngagementScore'] = np.round(
        df['Attendance (%)'] * 0.20 +
        df['Task Completion (%)'] * 0.30 +
        peer_norm * 0.20 +
        mgr_norm * 0.20 +
        df['KPI Score'] * 0.10, 
        2
    )
    
    # Categorize Risk Level based on Engagement
    df['Engagement_Category'] = pd.cut(
        df['EngagementScore'], 
        bins=[0, 50, 75, 100], 
        labels=['LOW', 'MEDIUM', 'HIGH']
    )
    
    out_path = os.path.join(PROCESSED_DIR, "engagement_processed.csv")
    df.to_csv(out_path, index=False)
    print(f"[+] Processed hr_performance_engagement.csv -> {out_path} ({df.shape[0]} rows)")
    return df

def process_occupation_and_skills():
    occ_path = os.path.join(RAW_DIR, "occupation_data.csv")
    ess_path = os.path.join(RAW_DIR, "essential_skills.csv")
    soft_path = os.path.join(RAW_DIR, "software_skills.csv")
    
    occ_df = pd.read_csv(occ_path)
    ess_df = pd.read_csv(ess_path)
    soft_df = pd.read_csv(soft_path)
    
    # Save clean master occupation table
    occ_out = os.path.join(PROCESSED_DIR, "occupation_master.csv")
    occ_df.to_csv(occ_out, index=False)
    
    # Essential skills importance >= 3.0
    ess_clean = ess_df[ess_df['Scale ID'] == 'IM'].copy()
    ess_clean['Data Value'] = pd.to_numeric(ess_clean['Data Value'], errors='coerce')
    ess_clean = ess_clean[ess_clean['Data Value'] >= 3.0]
    ess_out = os.path.join(PROCESSED_DIR, "essential_skills_processed.csv")
    ess_clean.to_csv(ess_out, index=False)
    
    # Software skills clean
    soft_clean = soft_df.drop_duplicates(subset=['O*NET-SOC Code', 'Workplace Example']).copy()
    soft_out = os.path.join(PROCESSED_DIR, "software_skills_processed.csv")
    soft_clean.to_csv(soft_out, index=False)
    
    print(f"[+] Processed occupations ({occ_df.shape[0]} rows), essential skills ({ess_clean.shape[0]} rows), software skills ({soft_clean.shape[0]} rows)")

def generate_employee_skills_controlled(attr_df, perf_df):
    """
    Build controlled employee skills table mapping each employee to their current skills and target role requirements.
    """
    np.random.seed(42)
    
    role_skill_map = {
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
    
    records = []
    
    # Map employees from attrition dataset
    for idx, row in attr_df.iterrows():
        emp_id = int(row['EmployeeNumber'])
        role = str(row['JobRole'])
        avail_skills = role_skill_map.get(role, ['Communication', 'Excel', 'Problem Solving'])
        
        # Randomly select 50-80% of skills as currently possessed
        num_owned = max(1, int(len(avail_skills) * np.random.uniform(0.5, 0.85)))
        owned_skills = np.random.choice(avail_skills, size=num_owned, replace=False)
        
        for skill in owned_skills:
            records.append({'EmployeeID': emp_id, 'JobRole': role, 'CurrentSkill': skill})
            
    # Also add for perf_df employees if any missing
    for idx, row in perf_df.iterrows():
        emp_id = int(row['Employee ID'])
        role = str(row['Job Role'])
        avail_skills = role_skill_map.get(role, ['Communication', 'Excel', 'Problem Solving'])
        num_owned = max(1, int(len(avail_skills) * np.random.uniform(0.5, 0.85)))
        owned_skills = np.random.choice(avail_skills, size=num_owned, replace=False)
        for skill in owned_skills:
            records.append({'EmployeeID': emp_id, 'JobRole': role, 'CurrentSkill': skill})

    skills_df = pd.DataFrame(records).drop_duplicates()
    out_path = os.path.join(PROCESSED_DIR, "employee_skills_controlled.csv")
    skills_df.to_csv(out_path, index=False)
    print(f"[+] Generated employee skills controlled table -> {out_path} ({skills_df.shape[0]} skill records)")
    return skills_df

if __name__ == "__main__":
    print("=== Starting Data Processing ===")
    attr_df = process_attrition_data()
    perf_df = process_engagement_data()
    process_occupation_and_skills()
    generate_employee_skills_controlled(attr_df, perf_df)
    print("=== Data Processing Complete ===")
