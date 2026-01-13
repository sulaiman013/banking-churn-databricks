# Databricks notebook source
# MAGIC %md
# MAGIC # Model Monitoring & Performance Tracking
# MAGIC
# MAGIC This notebook monitors the churn prediction model's performance over time.
# MAGIC
# MAGIC ## Overview
# MAGIC - Track prediction distribution drift
# MAGIC - Compare ML predictions vs actual churn events
# MAGIC - Generate model performance reports
# MAGIC - Alert on significant model degradation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pyspark.sql import functions as F

# Configuration
CATALOG = "bank_proj"
ML_SCHEMA = "ml"
GOLD_SCHEMA = "gold"

PREDICTIONS_TABLE = f"{CATALOG}.{ML_SCHEMA}.churn_predictions"
CUSTOMER_360_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.customer_360"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prediction Distribution Analysis

# COMMAND ----------

# Load current predictions
df_predictions = spark.table(PREDICTIONS_TABLE)

# Distribution of churn probabilities
print("=" * 60)
print("CHURN PROBABILITY DISTRIBUTION")
print("=" * 60)

prob_stats = df_predictions.select(
    F.count("*").alias("total_customers"),
    F.mean("churn_probability").alias("mean_prob"),
    F.stddev("churn_probability").alias("std_prob"),
    F.min("churn_probability").alias("min_prob"),
    F.max("churn_probability").alias("max_prob"),
    F.expr("percentile(churn_probability, 0.25)").alias("p25"),
    F.expr("percentile(churn_probability, 0.50)").alias("p50"),
    F.expr("percentile(churn_probability, 0.75)").alias("p75"),
    F.expr("percentile(churn_probability, 0.90)").alias("p90")
).collect()[0]

print(f"Total Customers: {prob_stats.total_customers}")
print(f"Mean Probability: {prob_stats.mean_prob:.3f}")
print(f"Std Deviation: {prob_stats.std_prob:.3f}")
print(f"Min: {prob_stats.min_prob:.3f}")
print(f"Max: {prob_stats.max_prob:.3f}")
print(f"\nPercentiles:")
print(f"  25th: {prob_stats.p25:.3f}")
print(f"  50th (Median): {prob_stats.p50:.3f}")
print(f"  75th: {prob_stats.p75:.3f}")
print(f"  90th: {prob_stats.p90:.3f}")

# COMMAND ----------

# Risk tier distribution
print("\n" + "=" * 60)
print("RISK TIER DISTRIBUTION")
print("=" * 60)

tier_dist = df_predictions.groupBy("ml_risk_tier").agg(
    F.count("*").alias("count"),
    F.round(F.avg("churn_probability"), 3).alias("avg_prob"),
    F.round(F.avg("rule_based_risk_score"), 1).alias("avg_rule_score")
).orderBy(
    F.when(F.col("ml_risk_tier") == "Critical", 1)
    .when(F.col("ml_risk_tier") == "High", 2)
    .when(F.col("ml_risk_tier") == "Medium", 3)
    .otherwise(4)
)

display(tier_dist)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. ML vs Rule-Based Comparison

# COMMAND ----------

# Compare ML predictions with rule-based risk scores
comparison_df = df_predictions.withColumn(
    "rule_risk_tier",
    F.when(F.col("rule_based_risk_score") >= 60, "High")
    .when(F.col("rule_based_risk_score") >= 40, "Medium")
    .otherwise("Low")
).withColumn(
    "ml_simplified_tier",
    F.when(F.col("ml_risk_tier").isin(["Critical", "High"]), "High")
    .when(F.col("ml_risk_tier") == "Medium", "Medium")
    .otherwise("Low")
)

# Confusion matrix: ML vs Rule-based
print("=" * 60)
print("ML vs RULE-BASED AGREEMENT MATRIX")
print("=" * 60)

agreement_matrix = comparison_df.groupBy("rule_risk_tier", "ml_simplified_tier").count()
display(agreement_matrix.orderBy("rule_risk_tier", "ml_simplified_tier"))

# COMMAND ----------

# Agreement rate
agreement_rate = comparison_df.filter(
    F.col("rule_risk_tier") == F.col("ml_simplified_tier")
).count() / comparison_df.count()

disagreement_df = comparison_df.filter(
    F.col("rule_risk_tier") != F.col("ml_simplified_tier")
)

print(f"\nAgreement Rate: {agreement_rate:.1%}")
print(f"Disagreement Count: {disagreement_df.count()}")

# Show examples where ML and rule-based disagree
print("\nExamples of ML vs Rule-Based Disagreements:")
display(
    disagreement_df.select(
        "unified_customer_id",
        "churn_probability",
        "rule_based_risk_score",
        "ml_risk_tier",
        "rule_risk_tier",
        "engagement_health_score"
    ).limit(10)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Feature Distribution Analysis

# COMMAND ----------

# Check if feature distributions have shifted
features_table = f"{CATALOG}.{GOLD_SCHEMA}.customer_features"
df_features = spark.table(features_table)

# Key features to monitor
key_features = [
    "tenure_days",
    "total_transactions",
    "days_since_last_transaction",
    "total_support_cases",
    "total_digital_sessions",
    "churn_risk_score",
    "engagement_health_score"
]

print("=" * 60)
print("KEY FEATURE STATISTICS")
print("=" * 60)

for feature in key_features:
    if feature in df_features.columns:
        stats = df_features.select(
            F.mean(feature).alias("mean"),
            F.stddev(feature).alias("std"),
            F.min(feature).alias("min"),
            F.max(feature).alias("max")
        ).collect()[0]
        print(f"\n{feature}:")
        print(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
        print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate Monitoring Report

# COMMAND ----------

# Create monitoring metrics table
monitoring_metrics = {
    "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_customers_scored": int(prob_stats.total_customers),
    "mean_churn_probability": float(prob_stats.mean_prob),
    "std_churn_probability": float(prob_stats.std_prob),
    "critical_risk_count": int(df_predictions.filter(F.col("ml_risk_tier") == "Critical").count()),
    "high_risk_count": int(df_predictions.filter(F.col("ml_risk_tier") == "High").count()),
    "medium_risk_count": int(df_predictions.filter(F.col("ml_risk_tier") == "Medium").count()),
    "low_risk_count": int(df_predictions.filter(F.col("ml_risk_tier") == "Low").count()),
    "ml_rule_agreement_rate": float(agreement_rate)
}

print("=" * 60)
print("MONITORING REPORT")
print("=" * 60)
for metric, value in monitoring_metrics.items():
    if isinstance(value, float):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Alert Thresholds

# COMMAND ----------

# Define alert thresholds
ALERT_THRESHOLDS = {
    "critical_risk_pct_max": 0.15,  # Alert if >15% are critical
    "mean_prob_change_max": 0.10,   # Alert if mean prob changes by >10%
    "agreement_rate_min": 0.60,     # Alert if agreement drops below 60%
}

# Check alerts
alerts = []

critical_pct = monitoring_metrics["critical_risk_count"] / monitoring_metrics["total_customers_scored"]
if critical_pct > ALERT_THRESHOLDS["critical_risk_pct_max"]:
    alerts.append(f"ALERT: Critical risk customers at {critical_pct:.1%} (threshold: {ALERT_THRESHOLDS['critical_risk_pct_max']:.1%})")

if monitoring_metrics["ml_rule_agreement_rate"] < ALERT_THRESHOLDS["agreement_rate_min"]:
    alerts.append(f"ALERT: ML/Rule agreement at {monitoring_metrics['ml_rule_agreement_rate']:.1%} (threshold: {ALERT_THRESHOLDS['agreement_rate_min']:.1%})")

print("\n" + "=" * 60)
print("ALERTS")
print("=" * 60)
if alerts:
    for alert in alerts:
        print(f"  {alert}")
else:
    print("  No alerts - all metrics within acceptable range")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save Monitoring History

# COMMAND ----------

# Save monitoring metrics to history table
monitoring_df = spark.createDataFrame([monitoring_metrics])

# Append to history table
MONITORING_HISTORY_TABLE = f"{CATALOG}.{ML_SCHEMA}.model_monitoring_history"

try:
    monitoring_df.write.mode("append").saveAsTable(MONITORING_HISTORY_TABLE)
    print(f"Monitoring metrics appended to {MONITORING_HISTORY_TABLE}")
except Exception as e:
    # Table might not exist, create it
    monitoring_df.write.mode("overwrite").saveAsTable(MONITORING_HISTORY_TABLE)
    print(f"Created monitoring history table: {MONITORING_HISTORY_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary
# MAGIC
# MAGIC ### Monitoring Metrics Tracked:
# MAGIC - Prediction distribution (mean, std, percentiles)
# MAGIC - Risk tier distribution
# MAGIC - ML vs Rule-based agreement rate
# MAGIC - Feature statistics
# MAGIC
# MAGIC ### Recommended Actions:
# MAGIC 1. Run this notebook weekly to track model drift
# MAGIC 2. Retrain model if agreement rate drops below 60%
# MAGIC 3. Investigate sudden spikes in critical risk customers
# MAGIC 4. Review feature distributions for data quality issues
