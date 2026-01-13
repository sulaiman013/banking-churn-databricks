# Databricks notebook source
# MAGIC %md
# MAGIC # Churn Prediction Model Training
# MAGIC
# MAGIC This notebook trains a machine learning model to predict customer churn using features from the Gold layer.
# MAGIC
# MAGIC ## Overview
# MAGIC - **Input**: `bank_proj.gold.customer_features` table
# MAGIC - **Target**: `churn_risk_score` (rule-based) → we'll create synthetic labels for demo
# MAGIC - **Algorithm**: Random Forest Classifier
# MAGIC - **Tracking**: MLflow experiment tracking
# MAGIC - **Output**: Registered model in MLflow Model Registry

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup & Configuration

# COMMAND ----------

# Import libraries
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt

# COMMAND ----------

# Configuration
CATALOG = "bank_proj"
GOLD_SCHEMA = "gold"
ML_SCHEMA = "ml"
FEATURES_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.customer_features"
PREDICTIONS_TABLE = f"{CATALOG}.{ML_SCHEMA}.churn_predictions"

# MLflow settings
EXPERIMENT_NAME = "/Shared/banking-churn-prediction"
MODEL_NAME = "banking_churn_model"

print(f"Features table: {FEATURES_TABLE}")
print(f"Predictions table: {PREDICTIONS_TABLE}")
print(f"MLflow experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# Set up MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Feature Data

# COMMAND ----------

# Load features from Gold layer
df_features = spark.table(FEATURES_TABLE)
print(f"Total customers: {df_features.count()}")
df_features.printSchema()

# COMMAND ----------

# Convert to Pandas for sklearn
pdf = df_features.toPandas()
print(f"DataFrame shape: {pdf.shape}")
pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Synthetic Churn Labels (Demo)
# MAGIC
# MAGIC Since we don't have historical churn data, we'll create synthetic labels based on the risk score.
# MAGIC In production, you would use actual historical churn events.

# COMMAND ----------

# Create synthetic churn label based on risk score + some noise
# High risk (60+) -> 70% chance of churn
# Medium risk (40-60) -> 30% chance of churn
# Low risk (<40) -> 10% chance of churn

np.random.seed(42)

def generate_churn_label(risk_score):
    if risk_score >= 60:
        return 1 if np.random.random() < 0.70 else 0
    elif risk_score >= 40:
        return 1 if np.random.random() < 0.30 else 0
    else:
        return 1 if np.random.random() < 0.10 else 0

pdf['churn_label'] = pdf['churn_risk_score'].apply(generate_churn_label)

# Check class distribution
print("Churn label distribution:")
print(pdf['churn_label'].value_counts(normalize=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Feature Selection & Preparation

# COMMAND ----------

# Select features for the model
# Exclude IDs, email, and the risk score (since we derived labels from it)
feature_columns = [
    # Demographics
    'gender_encoded',
    'customer_type_encoded',
    'region_encoded',

    # Tenure
    'tenure_days',
    'tenure_risk_score',

    # Transactions
    'total_transactions',
    'transaction_frequency',
    'transactions_last_30d',
    'days_since_last_transaction',
    'inactivity_risk_score',

    # Support/Complaints
    'total_support_cases',
    'complaint_rate',
    'has_open_complaint',
    'has_high_priority_complaint',
    'avg_resolution_days',

    # Digital engagement
    'total_digital_sessions',
    'app_usage_frequency',
    'is_digitally_active',
    'sessions_last_30d',
    'days_since_last_engagement',

    # Relationship
    'has_rm_attention',

    # Composite
    'engagement_health_score'
]

# Verify all columns exist
missing_cols = [col for col in feature_columns if col not in pdf.columns]
if missing_cols:
    print(f"Warning: Missing columns: {missing_cols}")
    feature_columns = [col for col in feature_columns if col in pdf.columns]

print(f"Using {len(feature_columns)} features:")
for col in feature_columns:
    print(f"  - {col}")

# COMMAND ----------

# Prepare X and y
X = pdf[feature_columns].copy()
y = pdf['churn_label'].copy()

# Handle any missing values
X = X.fillna(0)

# Check for any remaining issues
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Missing values per column:\n{X.isnull().sum()}")

# COMMAND ----------

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Maintain class balance
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training churn rate: {y_train.mean():.2%}")
print(f"Test churn rate: {y_test.mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Train Model with MLflow Tracking

# COMMAND ----------

# Start MLflow run
with mlflow.start_run(run_name=f"rf_churn_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:

    # Log parameters
    params = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_features": len(feature_columns),
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0]
    }
    mlflow.log_params(params)

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=params["random_state"],
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba)
    }

    # Log metrics
    mlflow.log_metrics(metrics)

    print("=" * 50)
    print("MODEL PERFORMANCE")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n" + "=" * 50)
    print("TOP 10 FEATURE IMPORTANCES")
    print("=" * 50)
    print(feature_importance.head(10).to_string(index=False))

    # Log feature importance as artifact
    feature_importance.to_csv("/tmp/feature_importance.csv", index=False)
    mlflow.log_artifact("/tmp/feature_importance.csv")

    # Classification report
    print("\n" + "=" * 50)
    print("CLASSIFICATION REPORT")
    print("=" * 50)
    print(classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n" + "=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    print(f"                 Predicted")
    print(f"                 No    Yes")
    print(f"Actual No       {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Yes      {cm[1,0]:4d}  {cm[1,1]:4d}")

    # Log model with signature
    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        registered_model_name=MODEL_NAME
    )

    print(f"\nModel logged to MLflow run: {run.info.run_id}")
    print(f"Model registered as: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Feature Importance Visualization

# COMMAND ----------

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 8))
feature_importance_top = feature_importance.head(15)
ax.barh(feature_importance_top['feature'], feature_importance_top['importance'])
ax.set_xlabel('Importance')
ax.set_ylabel('Feature')
ax.set_title('Top 15 Feature Importances for Churn Prediction')
ax.invert_yaxis()
plt.tight_layout()
plt.show()

# Save plot
fig.savefig("/tmp/feature_importance_plot.png", dpi=300, bbox_inches='tight')
mlflow.log_artifact("/tmp/feature_importance_plot.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Model Comparison (Optional)
# MAGIC
# MAGIC Train additional models to compare performance

# COMMAND ----------

# Train Gradient Boosting for comparison
with mlflow.start_run(run_name=f"gb_churn_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:

    params_gb = {
        "model_type": "GradientBoostingClassifier",
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42
    }
    mlflow.log_params(params_gb)

    model_gb = GradientBoostingClassifier(
        n_estimators=params_gb["n_estimators"],
        max_depth=params_gb["max_depth"],
        learning_rate=params_gb["learning_rate"],
        random_state=params_gb["random_state"]
    )

    model_gb.fit(X_train, y_train)

    y_pred_gb = model_gb.predict(X_test)
    y_pred_proba_gb = model_gb.predict_proba(X_test)[:, 1]

    metrics_gb = {
        "accuracy": accuracy_score(y_test, y_pred_gb),
        "precision": precision_score(y_test, y_pred_gb),
        "recall": recall_score(y_test, y_pred_gb),
        "f1_score": f1_score(y_test, y_pred_gb),
        "roc_auc": roc_auc_score(y_test, y_pred_proba_gb)
    }

    mlflow.log_metrics(metrics_gb)

    print("Gradient Boosting Performance:")
    for metric, value in metrics_gb.items():
        print(f"  {metric}: {value:.4f}")

    signature_gb = infer_signature(X_train, model_gb.predict(X_train))
    mlflow.sklearn.log_model(model_gb, "model", signature=signature_gb)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary
# MAGIC
# MAGIC ### What we accomplished:
# MAGIC 1. Loaded customer features from the Gold layer
# MAGIC 2. Created synthetic churn labels (for demo purposes)
# MAGIC 3. Trained Random Forest and Gradient Boosting models
# MAGIC 4. Tracked experiments with MLflow
# MAGIC 5. Registered the best model in MLflow Model Registry
# MAGIC
# MAGIC ### Next Steps:
# MAGIC 1. Run `02_score_customers.py` to score all customers
# MAGIC 2. Review model in MLflow UI
# MAGIC 3. Promote model to Production stage
# MAGIC 4. Set up automated retraining pipeline

# COMMAND ----------

# Print MLflow experiment URL
print(f"\nView experiments at: {mlflow.get_tracking_uri()}")
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Registered Model: {MODEL_NAME}")
