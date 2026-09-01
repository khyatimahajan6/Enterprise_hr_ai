# 🏢 Enterprise HR AI

An AI-powered HR analytics platform that helps organisations understand **employee attrition, workforce skill gaps, employee performance, and learning needs**.

## 🚀 Key Features

### 🔴 Employee Attrition Prediction

* Predicts the probability of an employee leaving the organisation.
* Uses employee demographics, job information, compensation, satisfaction, overtime, experience, and other HR factors.
* Provides a **high-risk employee roster** for HR teams.
* Handles the imbalanced attrition target using appropriate ML evaluation metrics.

### 🔥 Department Skill Gap Analysis

* Identifies missing skills across the organisation.
* Maps required occupational skills to employee roles.
* Provides a **department × missing-skill heatmap** using Plotly.
* Helps HR identify departments with the highest skill vulnerabilities.
* Supports prioritisation of future training programs.

### 🕸️ Employee Peer Benchmarking

* Provides an employee-level comparison against their department benchmark.
* Compares:

  * Engagement Score
  * Monthly Income
  * Skill Coverage
  * Retention Probability
* Displays the comparison using an interactive **radar chart**.

### 📚 Learning & Skill Recommendations

* Identifies the gap between an employee's current skills and the skills required for their role.
* Uses occupational skill data to determine development areas.
* Helps generate targeted learning priorities for employees.

### 📊 HR Analytics Dashboard

* Interactive **Streamlit dashboard** for HR users.
* Employee lookup and individual analytics.
* Department-level workforce insights.
* Interactive charts and visualisations.
* Designed to provide insights without requiring users to work directly with notebooks.

### 📥 Executive Data Exports

One-click CSV exports from the dashboard:

* `high_risk_attrition_roster.csv` — employees with elevated attrition risk.
* `organization_skill_gaps.csv` — organisation-wide skill deficit summary.

### ⚡ FastAPI Backend

* REST API layer for serving the ML and workforce analytics functionality.
* Separates the application logic from the dashboard.
* Interactive API documentation available through Swagger.

## 🏗️ Architecture

```text
Raw HR Data
     ↓
Data Cleaning & Validation
     ↓
Feature Engineering
     ↓
 ┌───────────────┬──────────────────┐
 ↓               ↓                  ↓
Attrition ML   Skill Analysis   Engagement
 ↓               ↓                  ↓
 └───────────────┴──────────────────┘
                 ↓
             FastAPI
                 ↓
          Streamlit Dashboard
                 ↓
       HR Insights & Exports
```

## 🛠️ Tech Stack

**Python · Pandas · NumPy · Scikit-learn · Plotly · FastAPI · Streamlit · Pytest · Joblib**

## 📂 Project Structure

```text
enterprise_hr_ai/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── models/
├── app/
├── frontend/
├── tests/
└── requirements.txt
```

## 📊 Datasets

The system uses:

* `employee_attrition.csv`
* `hr_performance_engagement.csv`
* `occupation_data.csv`
* `essential_skills.csv`
* `software_skills.csv`

## ▶️ Run Locally

```bash
pip install -r requirements.txt
```

Start the backend:

```bash
uvicorn app.main:app --reload --port 8000
```

Start the dashboard:

```bash
streamlit run frontend/app.py
```

* API: `http://localhost:8000`
* Swagger: `http://localhost:8000/docs`
* Dashboard: `http://localhost:8501`

## 🧪 Testing

**9/9 tests passed ✅**

```bash
pytest
```

## 🚧 Future Improvements

* Dockerisation
* MLflow experiment tracking
* Data-drift monitoring
* Automated model retraining
* CI/CD and cloud deployment
