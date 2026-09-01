import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
import shap

PROCESSED_DIR = r"c:\Users\khyat\OneDrive\Desktop\HrProject\enterprise_hr_ai\data\processed"
MODELS_DIR = r"c:\Users\khyat\OneDrive\Desktop\HrProject\enterprise_hr_ai\models"
V1_DIR = os.path.join(MODELS_DIR, "v1")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(V1_DIR, exist_ok=True)

def train_attrition_models():
    data_path = os.path.join(PROCESSED_DIR, "employee_attrition_processed.csv")
    df = pd.read_csv(data_path)
    
    # Feature Engineering
    df['Income_Per_Year_At_Company'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
    df['Promotion_Gap_Ratio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    df['Satisfaction_Index'] = (df['JobSatisfaction'] + df['EnvironmentSatisfaction'] + 
                                df['RelationshipSatisfaction'] + df['WorkLifeBalance']) / 4.0
    df['Experience_Ratio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    
    # Target and Feature Sets
    y = df['Attrition_Binary']
    
    cat_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
    num_features = [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
        'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
        'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager',
        'Income_Per_Year_At_Company', 'Promotion_Gap_Ratio', 'Satisfaction_Index', 'Experience_Ratio'
    ]
    
    X = df[cat_features + num_features]
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss')
    }
    
    results = {}
    best_name = None
    best_score = -1
    best_pipeline = None
    
    print("=== Training & Evaluating Models ===")
    for name, clf in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.35).astype(int)  # Lean toward recall for HR risk identification
        
        auc = roc_auc_score(y_test, probs)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        results[name] = {
            "ROC_AUC": round(float(auc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_Score": round(float(f1), 4)
        }
        
        print(f"[{name}] ROC-AUC: {auc:.4f} | Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f}")
        
        if auc > best_score:
            best_score = auc
            best_name = name
            best_pipeline = pipeline
            
    print(f"\n[+] Winning Model Selected: {best_name} (ROC-AUC: {best_score:.4f})")
    
    # Save root pipeline and v1 pipeline
    root_model_path = os.path.join(MODELS_DIR, "attrition_pipeline.joblib")
    v1_model_path = os.path.join(V1_DIR, "attrition_pipeline.joblib")
    
    joblib.dump(best_pipeline, root_model_path)
    joblib.dump(best_pipeline, v1_model_path)
    print(f"[+] Saved model pipeline -> {root_model_path} and {v1_model_path}")
    
    # Save Model Versioning Metadata
    metadata = {
        "model_name": "Enterprise Employee Attrition Model",
        "version": "v1.0",
        "winning_algorithm": best_name,
        "training_date": "2026-09-01",
        "evaluation_metrics": results[best_name],
        "all_model_comparisons": results,
        "categorical_features": cat_features,
        "numeric_features": num_features,
        "total_samples": len(df),
        "target_balance": {
            "Stay": int((y == 0).sum()),
            "Leave": int((y == 1).sum())
        }
    }
    
    metadata_path = os.path.join(V1_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved model metadata -> {metadata_path}")
    
    # SHAP Feature Importance extraction
    try:
        X_test_transformed = best_pipeline.named_steps['preprocessor'].transform(X_test)
        ohe = best_pipeline.named_steps['preprocessor'].named_transformers_['cat']
        encoded_cat_names = list(ohe.get_feature_names_out(cat_features))
        all_feature_names = num_features + encoded_cat_names
        
        classifier = best_pipeline.named_steps['classifier']
        
        if best_name in ["XGBoost", "RandomForest"]:
            explainer = shap.TreeExplainer(classifier)
            shap_vals = explainer.shap_values(X_test_transformed)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
        else:
            explainer = shap.LinearExplainer(classifier, X_test_transformed)
            shap_vals = explainer.shap_values(X_test_transformed)
            
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        shap_summary = pd.DataFrame({
            'Feature': all_feature_names,
            'SHAP_Importance': mean_abs_shap
        }).sort_values(by='SHAP_Importance', ascending=False)
        
        shap_out_path = os.path.join(V1_DIR, "shap_feature_importance.csv")
        shap_summary.to_csv(shap_out_path, index=False)
        print(f"[+] Exported SHAP feature importances -> {shap_out_path}")
    except Exception as e:
        print(f"[!] SHAP calculation notice: {e}")

if __name__ == "__main__":
    train_attrition_models()
