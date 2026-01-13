# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Churn Scoring
# MAGIC
# MAGIC This notebook scores all customers using the trained churn prediction model.
# MAGIC
# MAGIC ## Overview
# MAGIC - **Input**: `bank_proj.gold.customer_features` table
# MAGIC - **Model**: Loaded from MLflow Model Registry
# MAGIC - **Output**: `bank_proj.ml.churn_predictions` table

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup & Configuration

# COMMAND ----------

# Import libraries
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, TimestampType

# COMMAND ----------

# Configuration
CATALOG = "bank_proj"
GOLD_SCHEMA = "gold"
ML_SCHEMA = "ml"

FEATURES_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.customer_features"
CUSTOMER_360_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.customer_360"
PREDICTIONS_TABLE = f"{CATALOG}.{ML_SCHEMA}.churn_predictions"

MODEL_NAME = "banking_churn_model"
MODEL_STAGE = "None"  # Use "Production" once model is promoted

print(f"Features table: {FEATURES_TABLE}")
print(f"Predictions table: {PREDICTIONS_TABLE}")
print(f"Model: {MODEL_NAME} (stage: {MODEL_STAGE})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Model from Registry

# COMMAND ----------

# Load the latest version of the model
try:
    # Try to load from Production stage first
    model_uri = f"models:/{MODEL_NAME}/Production"
    model = mlflow.sklearn.load_model(model_uri)
    model_stage_used = "Production"
    print(f"Loaded model from Production stage")
except Exception as e:
    print(f"No Production model found, loading latest version...")
    # Load latest version
    model_uri = f"models:/{MODEL_NAME}/latest"
    model = mlflow.sklearn.load_model(model_uri)
    model_stage_used = "latest"
    print(f"Loaded latest model version")

print(f"Model type: {type(model).__name__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Feature Data

# COMMAND ----------

# Load all customer features
df_features = spark.table(FEATURES_TABLE)
print(f"Total customers to score: {df_features.count()}")

# Convert to Pandas
pdf = df_features.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Prepare Features for Scoring

# COMMAND ----------

# Feature columns (must match training)
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

# Filter to available columns
available_columns = [col for col in feature_columns if col in pdf.columns]
print(f"Using {len(available_columns)} features for scoring")

# Prepare feature matrix
X = pdf[available_columns].fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Score All Customers

# COMMAND ----------

# Generate predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)

# Get churn probability (probability of class 1)
churn_probability = probabilities[:, 1]

print(f"Scored {len(predictions)} customers")
print(f"Predicted churners: {sum(predictions)} ({sum(predictions)/len(predictions):.1%})")

# COMMAND ----------

# Create predictions DataFrame
predictions_df = pd.DataFrame({
    'unified_customer_id': pdf['unified_customer_id'],
    'erp_customer_id': pdf['erp_customer_id'],
    'email': pdf['email'],
    'churn_prediction': predictions,
    'churn_probability': churn_probability,
    'rule_based_risk_score': pdf['churn_risk_score'],
    'engagement_health_score': pdf['engagement_health_score']
})

# Add risk tier based on probability
def get_risk_tier(prob):
    if prob >= 0.7:
        return 'Critical'
    elif prob >= 0.5:
        return 'High'
    elif prob >= 0.3:
        return 'Medium'
    else:
        return 'Low'

predictions_df['ml_risk_tier'] = predictions_df['churn_probability'].apply(get_risk_tier)

# Add scoring metadata
predictions_df['model_name'] = MODEL_NAME
predictions_df['model_stage'] = model_stage_used
predictions_df['scored_at'] = datetime.now()

print("\nPredictions summary:")
print(predictions_df['ml_risk_tier'].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save Predictions to Unity Catalog

# COMMAND ----------

# Convert to Spark DataFrame
spark_predictions = spark.createDataFrame(predictions_df)

# Show schema
spark_predictions.printSchema()

# COMMAND ----------

# Ensure ML schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{ML_SCHEMA}")

# Write predictions to table
spark_predictions.write \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(PREDICTIONS_TABLE)

print(f"Predictions saved to {PREDICTIONS_TABLE}")
print(f"Total records: {spark.table(PREDICTIONS_TABLE).count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Create High-Risk Customer View

# COMMAND ----------

# Create view of high-risk customers with contact info
high_risk_query = f"""
CREATE OR REPLACE VIEW {CATALOG}.{ML_SCHEMA}.high_risk_ml_customers AS
SELECT
    p.unified_customer_id,
    p.erp_customer_id,
    p.email,
    c.customer_name,
    c.phone,
    c.territory,
    c.customer_group,
    p.churn_probability,
    p.ml_risk_tier,
    p.rule_based_risk_score,
    p.engagement_health_score,

    -- Recommended action based on ML risk
    CASE
        WHEN p.ml_risk_tier = 'Critical' THEN 'Immediate outreach required - assign senior RM'
        WHEN p.ml_risk_tier = 'High' THEN 'Schedule retention call within 48 hours'
        WHEN p.ml_risk_tier = 'Medium' THEN 'Include in proactive engagement campaign'
        ELSE 'Monitor - no immediate action needed'
    END as recommended_action,

    -- Priority score (combines ML probability and customer value)
    ROUND(p.churn_probability * 60 + LEAST(40, COALESCE(c.total_transaction_amount, 0) / 10000), 1) as priority_score,

    p.model_name,
    p.scored_at

FROM {PREDICTIONS_TABLE} p
LEFT JOIN {CUSTOMER_360_TABLE} c
    ON p.unified_customer_id = c.unified_customer_id
WHERE p.ml_risk_tier IN ('Critical', 'High', 'Medium')
ORDER BY p.churn_probability DESC
"""

spark.sql(high_risk_query)
print(f"Created view: {CATALOG}.{ML_SCHEMA}.high_risk_ml_customers")

# COMMAND ----------

# Preview high-risk customers
display(spark.sql(f"SELECT * FROM {CATALOG}.{ML_SCHEMA}.high_risk_ml_customers LIMIT 20"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Scoring Summary

# COMMAND ----------

# Summary statistics
summary_query = f"""
SELECT
    ml_risk_tier,
    COUNT(*) as customer_count,
    ROUND(AVG(churn_probability), 3) as avg_churn_prob,
    ROUND(AVG(rule_based_risk_score), 1) as avg_rule_risk,
    ROUND(AVG(engagement_health_score), 1) as avg_engagement
FROM {PREDICTIONS_TABLE}
GROUP BY ml_risk_tier
ORDER BY
    CASE ml_risk_tier
        WHEN 'Critical' THEN 1
        WHEN 'High' THEN 2
        WHEN 'Medium' THEN 3
        WHEN 'Low' THEN 4
    END
"""

print("=" * 60)
print("SCORING SUMMARY BY RISK TIER")
print("=" * 60)
display(spark.sql(summary_query))

# COMMAND ----------

# Comparison: ML vs Rule-based predictions
comparison_query = f"""
SELECT
    ml_risk_tier,

    -- Rule-based risk distribution within each ML tier
    ROUND(AVG(CASE WHEN rule_based_risk_score >= 60 THEN 1 ELSE 0 END) * 100, 1) as pct_rule_high_risk,
    ROUND(AVG(CASE WHEN rule_based_risk_score >= 40 AND rule_based_risk_score < 60 THEN 1 ELSE 0 END) * 100, 1) as pct_rule_medium_risk,
    ROUND(AVG(CASE WHEN rule_based_risk_score < 40 THEN 1 ELSE 0 END) * 100, 1) as pct_rule_low_risk,

    COUNT(*) as total_customers

FROM {PREDICTIONS_TABLE}
GROUP BY ml_risk_tier
ORDER BY
    CASE ml_risk_tier
        WHEN 'Critical' THEN 1
        WHEN 'High' THEN 2
        WHEN 'Medium' THEN 3
        WHEN 'Low' THEN 4
    END
"""

print("\n" + "=" * 60)
print("ML vs RULE-BASED RISK COMPARISON")
print("=" * 60)
display(spark.sql(comparison_query))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary
# MAGIC
# MAGIC ### What we accomplished:
# MAGIC 1. Loaded the trained model from MLflow Model Registry
# MAGIC 2. Scored all customers in the feature table
# MAGIC 3. Assigned ML-based risk tiers (Critical, High, Medium, Low)
# MAGIC 4. Saved predictions to `bank_proj.ml.churn_predictions`
# MAGIC 5. Created a high-risk customer view with recommended actions
# MAGIC
# MAGIC ### Output Tables:
# MAGIC - `bank_proj.ml.churn_predictions` - All customer predictions
# MAGIC - `bank_proj.ml.high_risk_ml_customers` - Filtered view for retention team
# MAGIC
# MAGIC ### Next Steps:
# MAGIC 1. Connect Power BI to the predictions table
# MAGIC 2. Set up Databricks Workflow for daily scoring
# MAGIC 3. Monitor model performance over time
# MAGIC 4. Implement A/B testing for retention campaigns

# COMMAND ----------

# Final counts
print("\n" + "=" * 60)
print("FINAL OUTPUT")
print("=" * 60)
print(f"Total customers scored: {spark.table(PREDICTIONS_TABLE).count()}")
print(f"High-risk customers (Critical + High + Medium): {spark.table(f'{CATALOG}.{ML_SCHEMA}.high_risk_ml_customers').count()}")
print(f"\nPredictions table: {PREDICTIONS_TABLE}")
print(f"High-risk view: {CATALOG}.{ML_SCHEMA}.high_risk_ml_customers")
print(f"\nScoring completed at: {datetime.now()}")
