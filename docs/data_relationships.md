# Data Relationships & Schema Documentation

This document describes how the five raw datasets interact within the Enterprise HR AI pipeline.

## Entity Relationship Overview

```
                   +-----------------------+
                   |  EMPLOYEE ATTRITION   |
                   |  (ML Training Model)  |
                   +-----------+-----------+
                               |
                        EmployeeNumber
                               |
                   +-----------v-----------+
                   |  EMPLOYEE INTELLIGENCE | <--- Employee ID ---> [ Engagement Data ]
                   |    (Master Dataset)   |
                   +-----------+-----------+
                               |
                            JobRole
                               |
                   +-----------v-----------+
                   |    OCCUPATION MASTER  |
                   +-----------+-----------+
                               |
                        O*NET-SOC Code
                               |
         +---------------------+---------------------+
         |                                           |
+--------v----------+                       +--------v----------+
|  ESSENTIAL SKILLS |                       |  SOFTWARE SKILLS  |
+-------------------+                       +-------------------+
```

## Table Definitions & Join Keys

| Source Table | Target Table | Join Key | Cardinality | Business Purpose |
|---|---|---|---|---|
| `employee_attrition.csv` | `engagement_processed.csv` | `EmployeeNumber` / `Employee ID` | 1:1 | Combines attrition demography with workplace engagement and performance scores. |
| `employee_attrition.csv` | `occupation_master.csv` | `JobRole` -> `Title` | N:1 | Maps company specific job roles to O*NET occupation codes. |
| `occupation_master.csv` | `essential_skills.csv` | `O*NET-SOC Code` | 1:N | Retrieves required cognitive, technical, and analytical skills for each role. |
| `occupation_master.csv` | `software_skills.csv` | `O*NET-SOC Code` | 1:N | Retrieves required software tools (Python, SQL, AWS, Salesforce, etc.) for each role. |
| `employee_attrition.csv` | `employee_skills_controlled.csv` | `EmployeeID` | 1:N | Provides the inventory of current skills possessed by each employee for gap analysis. |

---

## Data Pipelines & Artifacts Generated

- `data/processed/employee_attrition_processed.csv`
- `data/processed/engagement_processed.csv`
- `data/processed/occupation_master.csv`
- `data/processed/essential_skills_processed.csv`
- `data/processed/software_skills_processed.csv`
- `data/processed/employee_skills_controlled.csv`
- `data/processed/employee_intelligence.csv`
