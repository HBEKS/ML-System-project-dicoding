# modelling_tuning.py (yang sudah diperbaiki)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================
# SETUP MLflow DAGSHUB (Online Tracking)
# ============================================
DAGSHUB_REPO = "HBEKS/submission-ml-system-daniel-wuliutomo"
DAGSHUB_TOKEN = "5da0043f53592cf1c84627150bfa84ac80d11f99"

mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_REPO}.mlflow")

# Login ke DagsHub
import os
os.environ["MLFLOW_TRACKING_USERNAME"] = "HBEKS"
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

# Set experiment (dengan try-except agar aman)
experiment_name = "Eksperimen_Skilled_Advanced"
try:
    mlflow.set_experiment(experiment_name)
    print(f"Experiment '{experiment_name}' sudah ada, menggunakan yang existing")
except:
    # Jika experiment tidak ada, buat baru
    experiment_id = mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)
    print(f"Experiment '{experiment_name}' baru dibuat dengan ID: {experiment_id}")

# ============================================
# LOAD DATA
# ============================================
df = pd.read_csv('loan_preprocessing/loan_processed.csv')

X = df.drop('loan_status', axis=1)
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# HYPERPARAMETER GRID
# ============================================
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("="*60)
print("MODELLING TUNING SKILLED + ADVANCED (DAGSHUB)")
print("="*60)

# GridSearchCV
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# ============================================
# TRAINING DENGAN MANUAL LOGGING
# ============================================
with mlflow.start_run(run_name="RandomForest_Tuning_Advanced"):
    
    # --- 1. Log parameter grid ---
    mlflow.log_param("param_grid", str(param_grid))
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scoring_metric", "f1")
    
    # --- 2. Fit grid search ---
    grid_search.fit(X_train, y_train)
    
    # --- 3. Best parameters ---
    best_params = grid_search.best_params_
    for param, value in best_params.items():
        mlflow.log_param(f"best_{param}", value)
    
    # --- 4. Best model ---
    best_model = grid_search.best_estimator_
    
    # --- 5. Predict ---
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    # --- 6. Metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # --- 8. ARTEFAK 1: Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Approved', 'Approved'],
                yticklabels=['Not Approved', 'Approved'])
    plt.title('Confusion Matrix - Best Model (Tuning)')
    plt.tight_layout()
    plt.savefig('confusion_matrix_tuning.png')
    mlflow.log_artifact('confusion_matrix_tuning.png')
    plt.close()
    
    # --- 9. ARTEFAK 2: ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Best Model (Tuning)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve_tuning.png')
    mlflow.log_artifact('roc_curve_tuning.png')
    plt.close()
    
    # --- 10. ARTEFAK 3: Feature Importance ---
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature', palette='viridis')
    plt.title('Top 10 Feature Importance - Best Model (Tuning)')
    plt.tight_layout()
    plt.savefig('feature_importance_tuning.png')
    mlflow.log_artifact('feature_importance_tuning.png')
    plt.close()
    
    # --- 11. Log model ---
    mlflow.sklearn.log_model(best_model, "best_tuned_model")
    
    # --- 12. Print results ---
    print("\n" + "="*50)
    print("BEST PARAMETERS:")
    print("="*50)
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    print("\n" + "="*50)
    print("TEST METRICS:")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    
    print("\n" + "="*50)
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"MLflow UI: https://dagshub.com/{DAGSHUB_REPO}/experiments")