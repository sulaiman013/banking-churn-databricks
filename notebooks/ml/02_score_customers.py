# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Churn Scoring
# MAGIC
# MAGIC This notebook scores all customers using the trained churn prediction model.
# MAGIC
# MAGIC ## Overview
# MAGIC - **Input**: `bank_proj.gold.customer_features` table
# MAGIC - **Model**: Loaded from saved pickle file (Free Edition compatible)
# MAGIC - **Output**: `bank_proj.ml.churn_predictions` table
# MAGIC
# MAGIC *Compatible with Databricks Free Edition (serverless compute)*

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup & Configuration

# COMMAND ----------

# Import libraries (all pre-installed in Databricks serverless)
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# PySpark
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, TimestampType

print("Libraries imported successfully!")

# COMMAND ----------

# Configuration
CATALOG = "bank_proj"
GOLD_SCHEMA = "gold"
ML_SCHEMA = "ml"

FEATURES_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.customer_features"
CUSTOMER_360_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.customer_360"
PREDICTIONS_TABLE = f"{CATALOG}.{ML_SCHEMA}.churn_predictions"

print(f"Features table: {FEATURES_TABLE}")
print(f"Predictions table: {PREDICTIONS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Feature Data

# COMMAND ----------

# Load all customer features
df_features = spark.table(FEATURES_TABLE)
print(f"Total customers to score: {df_features.count()}")

# Convert to Pandas
pdf = df_features.toPandas()
print(f"DataFrame shape: {pdf.shape}")

# COMMAND ----------

# Preview data
pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Engineering (must match training)

# COMMAND ----------

# Create the same engineered features as in training
df_ml = pdf.copy()

# Interaction features
df_ml['inactivity_complaint_risk'] = df_ml['inactivity_risk_score'] * df_ml['has_open_complaint']
df_ml['tenure_engagement_ratio'] = df_ml['tenure_days'] / (df_ml['engagement_health_score'] + 1)
df_ml['digital_transaction_ratio'] = df_ml['total_digital_sessions'] / (df_ml['total_transactions'] + 1)
df_ml['recent_activity_score'] = df_ml['transactions_last_30d'] * 2 + df_ml['sessions_last_30d']

# Engagement intensity
df_ml['avg_sessions_per_month'] = df_ml['total_digital_sessions'] / (df_ml['tenure_days'] / 30 + 1)
df_ml['avg_transactions_per_month'] = df_ml['total_transactions'] / (df_ml['tenure_days'] / 30 + 1)

# Risk combinations
df_ml['combined_risk'] = (
    df_ml['tenure_risk_score'] +
    df_ml['inactivity_risk_score'] +
    (df_ml['has_open_complaint'] * 30) +
    (df_ml['has_high_priority_complaint'] * 20)
)

# Engagement decline indicator
df_ml['engagement_decline'] = (
    (df_ml['total_digital_sessions'] > 5) &
    (df_ml['sessions_last_30d'] == 0)
).astype(int)

# Transaction decline indicator
df_ml['transaction_decline'] = (
    (df_ml['total_transactions'] > 3) &
    (df_ml['transactions_last_30d'] == 0)
).astype(int)

# High inactivity flag
df_ml['high_inactivity'] = (df_ml['days_since_last_transaction'] > 60).astype(int)

# Complaint severity score
df_ml['complaint_severity'] = (
    df_ml['total_support_cases'] +
    df_ml['has_open_complaint'] * 5 +
    df_ml['has_high_priority_complaint'] * 10
)

# Overall risk indicator
df_ml['overall_risk_flag'] = (
    (df_ml['churn_risk_score'] > 60) |
    (df_ml['engagement_health_score'] < 40) |
    (df_ml['days_since_last_transaction'] > 60)
).astype(int)

print("Engineered features created successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Prepare Features for Scoring

# COMMAND ----------

# Feature columns (must match training exactly)
feature_columns = [
    # Original demographics
    'gender_encoded', 'customer_type_encoded', 'region_encoded',

    # Tenure
    'tenure_days', 'tenure_risk_score',

    # Transactions
    'total_transactions', 'transaction_frequency', 'transactions_last_30d',
    'days_since_last_transaction', 'inactivity_risk_score',

    # Support
    'total_support_cases', 'complaint_rate', 'has_open_complaint',
    'has_high_priority_complaint', 'avg_resolution_days',

    # Digital
    'total_digital_sessions', 'app_usage_frequency', 'is_digitally_active',
    'sessions_last_30d', 'days_since_last_engagement',

    # Relationship
    'has_rm_attention',

    # Composite scores
    'engagement_health_score', 'churn_risk_score',

    # Engineered features
    'inactivity_complaint_risk', 'tenure_engagement_ratio',
    'digital_transaction_ratio', 'recent_activity_score',
    'avg_sessions_per_month', 'avg_transactions_per_month',
    'combined_risk', 'engagement_decline', 'transaction_decline',
    'high_inactivity', 'complaint_severity', 'overall_risk_flag'
]

# Filter to available columns
available_features = [col for col in feature_columns if col in df_ml.columns]
print(f"Using {len(available_features)} features for scoring")

# Prepare feature matrix
X = df_ml[available_features].copy()
X = X.fillna(0)
X = X.replace([np.inf, -np.inf], 0)

print(f"Feature matrix shape: {X.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Load or Train Model

# COMMAND ----------

# Try to load saved model, otherwise train a new one
MODEL_PATH = "/tmp/churn_model_final.pkl"
SCALER_PATH = "/tmp/churn_scaler.pkl"

try:
    # Try to load existing model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Loaded existing model and scaler from /tmp/")
    model_source = "Loaded from file"
except:
    print("No saved model found. Training new model for scoring...")

    # Create target variable (same logic as training)
    np.random.seed(42)

    def create_churn_label(row):
        churn_probability = 0.0
        if row['days_since_last_transaction'] > 90:
            churn_probability += 0.40
        elif row['days_since_last_transaction'] > 60:
            churn_probability += 0.25
        elif row['days_since_last_transaction'] > 30:
            churn_probability += 0.12
        if row['has_open_complaint'] == 1:
            churn_probability += 0.30
        if row['has_high_priority_complaint'] == 1:
            churn_probability += 0.20
        if row['is_digitally_active'] == 0 and row['total_digital_sessions'] > 0:
            churn_probability += 0.25
        if row['engagement_health_score'] < 30:
            churn_probability += 0.20
        if row['transactions_last_30d'] == 0 and row['total_transactions'] > 0:
            churn_probability += 0.18
        churn_probability = min(churn_probability, 0.95)
        noise = np.random.uniform(-0.08, 0.08)
        final_prob = max(0, min(1, churn_probability + noise))
        return 1 if np.random.random() < final_prob else 0

    df_ml['churn_label'] = df_ml.apply(create_churn_label, axis=1)
    y = df_ml['churn_label']

    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled, y)

    # Save for future use
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print("New model trained and saved!")
    model_source = "Newly trained"

print(f"Model type: {type(model).__name__}")
print(f"Model source: {model_source}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Score All Customers

# COMMAND ----------

# Scale features
X_scaled = scaler.transform(X)

# Generate predictions
predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)

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
predictions_df['model_name'] = "sklearn_ensemble"
predictions_df['model_source'] = model_source
predictions_df['scored_at'] = datetime.now()

print("\nPredictions summary:")
print(predictions_df['ml_risk_tier'].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Visualize Scoring Results

# COMMAND ----------

# Visualization of predictions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Churn probability distribution
axes[0, 0].hist(churn_probability, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(x=0.3, color='orange', linestyle='--', label='Medium threshold')
axes[0, 0].axvline(x=0.5, color='red', linestyle='--', label='High threshold')
axes[0, 0].axvline(x=0.7, color='darkred', linestyle='--', label='Critical threshold')
axes[0, 0].set_xlabel('Churn Probability')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Distribution of Churn Probabilities', fontweight='bold')
axes[0, 0].legend()

# 2. Risk tier distribution
tier_counts = predictions_df['ml_risk_tier'].value_counts()
tier_order = ['Critical', 'High', 'Medium', 'Low']
tier_counts = tier_counts.reindex(tier_order)
colors = ['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c']
axes[0, 1].bar(tier_counts.index, tier_counts.values, color=colors)
axes[0, 1].set_xlabel('Risk Tier')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Customers by Risk Tier', fontweight='bold')
for i, v in enumerate(tier_counts.values):
    axes[0, 1].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

# 3. ML vs Rule-based comparison
axes[1, 0].scatter(predictions_df['rule_based_risk_score'],
                   predictions_df['churn_probability'],
                   alpha=0.5, c='steelblue', s=20)
axes[1, 0].set_xlabel('Rule-Based Risk Score')
axes[1, 0].set_ylabel('ML Churn Probability')
axes[1, 0].set_title('ML vs Rule-Based Risk Comparison', fontweight='bold')
# Add trend line
z = np.polyfit(predictions_df['rule_based_risk_score'], predictions_df['churn_probability'], 1)
p = np.poly1d(z)
x_line = np.linspace(0, 100, 100)
axes[1, 0].plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
axes[1, 0].legend()

# 4. Risk tier pie chart
axes[1, 1].pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%',
               colors=colors, explode=(0.05, 0.02, 0, 0))
axes[1, 1].set_title('Risk Tier Distribution', fontweight='bold')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save Predictions to Unity Catalog

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
# MAGIC ## 9. Create High-Risk Customer View

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
# MAGIC ## 10. Scoring Summary

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
# MAGIC ## 11. Summary
# MAGIC
# MAGIC ### What we accomplished:
# MAGIC 1. Loaded customer features from Unity Catalog
# MAGIC 2. Applied same feature engineering as training
# MAGIC 3. Loaded/trained model (sklearn - Free Edition compatible)
# MAGIC 4. Scored all customers with churn probabilities
# MAGIC 5. Assigned ML-based risk tiers (Critical, High, Medium, Low)
# MAGIC 6. Created visualizations of scoring results
# MAGIC 7. Saved predictions to `bank_proj.ml.churn_predictions`
# MAGIC 8. Created high-risk customer view with recommended actions
# MAGIC
# MAGIC ### Output Tables:
# MAGIC - `bank_proj.ml.churn_predictions` - All customer predictions
# MAGIC - `bank_proj.ml.high_risk_ml_customers` - Filtered view for retention team
# MAGIC
# MAGIC ### Next Steps:
# MAGIC 1. Connect Power BI to the predictions table
# MAGIC 2. Set up Databricks Workflow for daily scoring
# MAGIC 3. Run model monitoring notebook
# MAGIC 4. Implement retention campaigns for high-risk customers

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
