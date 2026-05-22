# modelling.py
# Basic + Advanced: MLflow dengan autolog + Localhost

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================
# SETUP MLflow LOKAL (Localhost)
# ============================================
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Eksperimen_Basic_Lokal")

# ============================================
# LOAD DATA
# ============================================
# Load data hasil preprocessing
df = pd.read_csv('loan_preprocessing/loan_processed.csv')

# Pisahkan fitur dan target
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("="*60)
print("MODELLING BASIC + ADVANCED (LOCALHOST)")
print("="*60)

# Train model dengan autolog + manual logging
with mlflow.start_run(run_name="RandomForest_Basic_Advanced_Lokal"):
    
    # --- 1. Log parameters ---
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)
    
    # --- 2. Training dengan autolog ---
    mlflow.sklearn.autolog()
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # --- 3. Predict ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # --- 4. Metrics ---
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # --- ARTEFAK TAMBAHAN 1: Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Approved', 'Approved'],
                yticklabels=['Not Approved', 'Approved'])
    plt.title('Confusion Matrix - Random Forest')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    plt.close()
    
    # --- ARTEFAK TAMBAHAN 2: ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    mlflow.log_artifact('roc_curve.png')
    plt.close()
    
    # --- ARTEFAK TAMBAHAN 3: Feature Importance ---
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature', palette='viridis')
    plt.title('Top 10 Feature Importance - Random Forest')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    mlflow.log_artifact('feature_importance.png')
    plt.close()
    
    # --- Log model ---
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    # --- Print hasil ---
    print("\n" + "="*50)
    print("HASIL MODEL RANDOM FOREST")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\n" + "="*50)
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print("\nJalankan 'mlflow ui' di terminal untuk melihat dashboard")
    print("Kemudian buka http://127.0.0.1:5000/ di browser")