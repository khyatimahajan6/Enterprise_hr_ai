import os
import json

NOTEBOOKS_DIR = r"c:\Users\khyat\OneDrive\Desktop\HrProject\enterprise_hr_ai\notebooks"
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def create_notebook(filename, title, description, code_cells):
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n\n", f"{description}"]
            }
        ],
        "metadata": {
            "language_info": {"name": "python"},
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    for code in code_cells:
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [code]
        })
        
    out_path = os.path.join(NOTEBOOKS_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[+] Generated notebook -> {out_path}")

def generate_all_notebooks():
    # 01 Data Understanding
    create_notebook(
        "01_data_understanding.ipynb",
        "01 Data Understanding",
        "Load raw datasets and inspect shape, columns, missing values, and join candidate keys.",
        [
            "import pandas as pd\nimport os\n\nDATA_PATH = '../data/raw'\nprint('Raw files:', os.listdir(DATA_PATH))",
            "df_attr = pd.read_csv(f'{DATA_PATH}/employee_attrition.csv')\nprint('Attrition Shape:', df_attr.shape)\ndf_attr.info()\ndf_attr.head(3)",
            "df_perf = pd.read_csv(f'{DATA_PATH}/hr_performance_engagement.csv')\nprint('Performance Shape:', df_perf.shape)\ndf_perf.head(3)",
            "df_occ = pd.read_csv(f'{DATA_PATH}/occupation_data.csv')\nprint('Occupation Shape:', df_occ.shape)\ndf_occ.head(3)"
        ]
    )
    
    # 02 Data Validation
    create_notebook(
        "02_data_validation.ipynb",
        "02 Data Validation",
        "Validate schemas, value ranges, category consistency, and primary key uniqueness.",
        [
            "import pandas as pd\n\ndf = pd.read_csv('../data/raw/employee_attrition.csv')",
            "assert df['Age'].between(18, 100).all(), 'Age out of valid range'\nprint('[✓] Age validation passed')",
            "assert df['EmployeeNumber'].is_unique, 'Duplicate EmployeeNumber found'\nprint('[✓] EmployeeNumber uniqueness passed')",
            "assert set(df['Attrition'].unique()) <= {'Yes', 'No'}, 'Unexpected Attrition values'\nprint('[✓] Attrition categories passed')"
        ]
    )

    # 03 Data Cleaning
    create_notebook(
        "03_data_cleaning.ipynb",
        "03 Data Cleaning",
        "Perform data type casting, null handling, category normalization, and save processed files.",
        [
            "import pandas as pd\nfrom src.data_processing import process_attrition_data, process_engagement_data, process_occupation_and_skills",
            "attr_df = process_attrition_data()\neng_df = process_engagement_data()\nprocess_occupation_and_skills()",
            "print('Processed datasets exported to data/processed/')"
        ]
    )

    # 04 Data Relationships
    create_notebook(
        "04_data_relationships.ipynb",
        "04 Data Relationships",
        "Verify entity relationships, key mappings, and table joins across employee, role, and skill datasets.",
        [
            "import pandas as pd\n\nattr = pd.read_csv('../data/processed/employee_attrition_processed.csv')\neng = pd.read_csv('../data/processed/engagement_processed.csv')\n\nprint('Attrition employees:', len(attr))\nprint('Engagement records:', len(eng))"
        ]
    )

    # 05 Feature Engineering
    create_notebook(
        "05_feature_engineering.ipynb",
        "05 Feature Engineering",
        "Create domain-specific engineered features: income per year at company, promotion gap ratio, and satisfaction index.",
        [
            "import pandas as pd\n\ndf = pd.read_csv('../data/processed/employee_attrition_processed.csv')\ndf['Income_Per_Year'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)\ndf['Promotion_Gap_Ratio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)\ndf['Satisfaction_Index'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + df['WorkLifeBalance']) / 3.0\ndf.head(3)"
        ]
    )

    # 06 Baseline Model
    create_notebook(
        "06_baseline_model.ipynb",
        "06 Baseline Model",
        "Build baseline Logistic Regression model for employee attrition risk probability estimation.",
        [
            "import pandas as pd\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import roc_auc_score, classification_report",
            "df = pd.read_csv('../data/processed/employee_attrition_processed.csv')\nX = df[['Age', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 'WorkLifeBalance']]\ny = df['Attrition_Binary']\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)\nclf = LogisticRegression().fit(X_tr, y_tr)\nprint('ROC-AUC:', roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))"
        ]
    )

    # 07 Model Comparison
    create_notebook(
        "07_model_comparison.ipynb",
        "07 Model Comparison",
        "Compare Logistic Regression, Random Forest, and XGBoost models on ROC-AUC, Recall, Precision, and F1.",
        [
            "from src.train_ml_model import train_attrition_models\ntrain_attrition_models()"
        ]
    )

    # 08 Model Explainability
    create_notebook(
        "08_model_explainability.ipynb",
        "08 Model Explainability (SHAP)",
        "Extract global feature importances and local employee attrition risk drivers using SHAP values.",
        [
            "import pandas as pd\n\nshap_df = pd.read_csv('../models/v1/shap_feature_importance.csv')\nshap_df.head(10)"
        ]
    )

    # 09 Model Versioning
    create_notebook(
        "09_model_versioning.ipynb",
        "09 Model Versioning",
        "Track model artifacts, performance metrics, and metadata in models/v1/metadata.json.",
        [
            "import json\nwith open('../models/v1/metadata.json') as f:\n    meta = json.load(f)\nprint(json.dumps(meta, indent=2))"
        ]
    )

    # 09 Engagement Intelligence
    create_notebook(
        "09_engagement_intelligence.ipynb",
        "09 Engagement Intelligence",
        "Analyze employee engagement scores, department averages, and identify disengaged cohorts.",
        [
            "import pandas as pd\n\ndf = pd.read_csv('../data/processed/engagement_processed.csv')\nprint(df.groupby('Department')['EngagementScore'].mean().sort_values(ascending=False))"
        ]
    )

    # 10 Role Intelligence
    create_notebook(
        "10_role_intelligence.ipynb",
        "10 Role Intelligence",
        "Reference master occupation taxonomy and extract required skill profiles.",
        [
            "import pandas as pd\n\nocc = pd.read_csv('../data/processed/occupation_master.csv')\nocc.head(5)"
        ]
    )

    # 11 Employee Skills
    create_notebook(
        "11_employee_skills.ipynb",
        "11 Employee Skills Table",
        "Inspect inventory of employee skills and competency mappings.",
        [
            "import pandas as pd\n\nskills = pd.read_csv('../data/processed/employee_skills_controlled.csv')\nskills.head(10)"
        ]
    )

    # 12 Skill Gap Engine
    create_notebook(
        "12_skill_gap_engine.ipynb",
        "12 Skill Gap Engine",
        "Compute employee skill gaps via set difference (Required Skills - Current Skills).",
        [
            "req = {'Python', 'SQL', 'Docker', 'AWS', 'MLOps'}\nhas = {'Python', 'SQL'}\ngap = req - has\nprint('Missing Skills Gap:', gap)"
        ]
    )

    # 13 Organization Skill Gap
    create_notebook(
        "13_organization_skill_gap.ipynb",
        "13 Organization-Wide Skill Gap",
        "Aggregate missing skills across all employees and determine critical severity levels.",
        [
            "import pandas as pd\n\ngaps = pd.read_csv('../data/processed/organization_skill_gap_summary.csv')\ngaps.head(10)"
        ]
    )

    # 14 Recommendation Engine
    create_notebook(
        "14_recommendation_engine.ipynb",
        "14 Upskilling Recommendation Engine",
        "Map missing skill gaps to targeted professional training courses.",
        [
            "import pandas as pd\n\nintel = pd.read_csv('../data/processed/employee_intelligence.csv')\nintel[['EmployeeID', 'JobRole', 'MissingSkills', 'UpskillingRecommendation']].head(5)"
        ]
    )

    # 15 Employee Intelligence
    create_notebook(
        "15_employee_intelligence.ipynb",
        "15 Employee Intelligence Master Dataset",
        "Consolidate attrition risk, engagement scores, skill gaps, and recommendations into master dataset.",
        [
            "import pandas as pd\n\nintel = pd.read_csv('../data/processed/employee_intelligence.csv')\nprint('Master Dataset Shape:', intel.shape)\nintel.head(5)"
        ]
    )

if __name__ == "__main__":
    generate_all_notebooks()
