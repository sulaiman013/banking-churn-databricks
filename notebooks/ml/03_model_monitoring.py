# Databricks notebook source
# MAGIC %md
# MAGIC # Model Monitoring & Performance Tracking
# MAGIC
# MAGIC This notebook monitors the churn prediction model's performance over time.
# MAGIC
# MAGIC ## Overview
# MAGIC - Track prediction distribution drift
# MAGIC - Compare ML predictions vs rule-based scoring
# MAGIC - Monitor feature distributions for data drift
# MAGIC - Generate alerts on significant changes
# MAGIC - Save monitoring history for trend analysis
# MAGIC
# MAGIC *Compatible with Databricks Free Edition (serverless compute)*

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# PySpark
from pyspark.sql import functions as F

print("Libraries imported successfully!")

# COMMAND ----------

# Configuration
CATALOG = "bank_proj"
ML_SCHEMA = "ml"
GOLD_SCHEMA = "gold"

PREDICTIONS_TABLE = f"{CATALOG}.{ML_SCHEMA}.churn_predictions"
CUSTOMER_360_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.customer_360"
FEATURES_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.customer_features"
MONITORING_HISTORY_TABLE = f"{CATALOG}.{ML_SCHEMA}.model_monitoring_history"

print(f"Predictions table: {PREDICTIONS_TABLE}")
print(f"Monitoring history: {MONITORING_HISTORY_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Current Predictions

# COMMAND ----------

# Load current predictions
df_predictions = spark.table(PREDICTIONS_TABLE)
total_customers = df_predictions.count()
print(f"Total customers with predictions: {total_customers}")

# Convert to pandas for analysis
pdf_predictions = df_predictions.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Prediction Distribution Analysis

# COMMAND ----------

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
    F.expr("percentile(churn_probability, 0.90)").alias("p90"),
    F.expr("percentile(churn_probability, 0.95)").alias("p95")
).collect()[0]

print(f"Total Customers: {prob_stats.total_customers}")
print(f"Mean Probability: {prob_stats.mean_prob:.4f}")
print(f"Std Deviation: {prob_stats.std_prob:.4f}")
print(f"Min: {prob_stats.min_prob:.4f}")
print(f"Max: {prob_stats.max_prob:.4f}")
print(f"\nPercentiles:")
print(f"  25th: {prob_stats.p25:.4f}")
print(f"  50th (Median): {prob_stats.p50:.4f}")
print(f"  75th: {prob_stats.p75:.4f}")
print(f"  90th: {prob_stats.p90:.4f}")
print(f"  95th: {prob_stats.p95:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Visualize Prediction Distributions

# COMMAND ----------

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Probability histogram
axes[0, 0].hist(pdf_predictions['churn_probability'], bins=50, color='steelblue',
                edgecolor='black', alpha=0.7)
axes[0, 0].axvline(x=prob_stats.mean_prob, color='red', linestyle='--',
                   label=f'Mean: {prob_stats.mean_prob:.3f}')
axes[0, 0].axvline(x=prob_stats.p50, color='orange', linestyle='--',
                   label=f'Median: {prob_stats.p50:.3f}')
axes[0, 0].set_xlabel('Churn Probability')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Churn Probability Distribution', fontweight='bold')
axes[0, 0].legend()

# 2. Risk tier distribution
tier_counts = pdf_predictions['ml_risk_tier'].value_counts()
tier_order = ['Critical', 'High', 'Medium', 'Low']
tier_counts = tier_counts.reindex(tier_order).fillna(0)
colors = ['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c']
bars = axes[0, 1].bar(tier_counts.index, tier_counts.values, color=colors)
axes[0, 1].set_xlabel('Risk Tier')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Customers by Risk Tier', fontweight='bold')
for bar, count in zip(bars, tier_counts.values):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{int(count)}', ha='center', fontweight='bold')

# 3. Risk tier percentages (pie chart)
axes[0, 2].pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%',
               colors=colors, explode=(0.05, 0.02, 0, 0))
axes[0, 2].set_title('Risk Tier Distribution (%)', fontweight='bold')

# 4. Box plot of probabilities by tier
tier_mapping = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
pdf_predictions['tier_order'] = pdf_predictions['ml_risk_tier'].map(tier_mapping)
pdf_sorted = pdf_predictions.sort_values('tier_order', ascending=False)
sns.boxplot(data=pdf_sorted, x='ml_risk_tier', y='churn_probability',
            order=tier_order, palette=colors, ax=axes[1, 0])
axes[1, 0].set_xlabel('Risk Tier')
axes[1, 0].set_ylabel('Churn Probability')
axes[1, 0].set_title('Probability Distribution by Tier', fontweight='bold')

# 5. ML vs Rule-based scatter
axes[1, 1].scatter(pdf_predictions['rule_based_risk_score'],
                   pdf_predictions['churn_probability'],
                   alpha=0.4, c='steelblue', s=15)
# Add trend line
z = np.polyfit(pdf_predictions['rule_based_risk_score'],
               pdf_predictions['churn_probability'], 1)
p = np.poly1d(z)
x_line = np.linspace(0, 100, 100)
axes[1, 1].plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
axes[1, 1].set_xlabel('Rule-Based Risk Score')
axes[1, 1].set_ylabel('ML Churn Probability')
axes[1, 1].set_title('ML vs Rule-Based Comparison', fontweight='bold')
axes[1, 1].legend()

# 6. Cumulative distribution
sorted_probs = np.sort(pdf_predictions['churn_probability'])
cdf = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
axes[1, 2].plot(sorted_probs, cdf, color='steelblue', linewidth=2)
axes[1, 2].axhline(y=0.9, color='orange', linestyle='--', label='90th percentile')
axes[1, 2].axhline(y=0.75, color='green', linestyle='--', label='75th percentile')
axes[1, 2].set_xlabel('Churn Probability')
axes[1, 2].set_ylabel('Cumulative Proportion')
axes[1, 2].set_title('Cumulative Distribution', fontweight='bold')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Risk Tier Analysis

# COMMAND ----------

# Risk tier distribution with detailed stats
print("\n" + "=" * 60)
print("RISK TIER DISTRIBUTION")
print("=" * 60)

tier_dist = df_predictions.groupBy("ml_risk_tier").agg(
    F.count("*").alias("count"),
    F.round(F.avg("churn_probability"), 4).alias("avg_prob"),
    F.round(F.min("churn_probability"), 4).alias("min_prob"),
    F.round(F.max("churn_probability"), 4).alias("max_prob"),
    F.round(F.avg("rule_based_risk_score"), 1).alias("avg_rule_score"),
    F.round(F.avg("engagement_health_score"), 1).alias("avg_engagement")
).orderBy(
    F.when(F.col("ml_risk_tier") == "Critical", 1)
    .when(F.col("ml_risk_tier") == "High", 2)
    .when(F.col("ml_risk_tier") == "Medium", 3)
    .otherwise(4)
)

display(tier_dist)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. ML vs Rule-Based Comparison

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

# Agreement matrix
print("=" * 60)
print("ML vs RULE-BASED AGREEMENT MATRIX")
print("=" * 60)

agreement_matrix = comparison_df.groupBy("rule_risk_tier", "ml_simplified_tier").count()
display(agreement_matrix.orderBy("rule_risk_tier", "ml_simplified_tier"))

# COMMAND ----------

# Calculate agreement rate
total_count = comparison_df.count()
agreement_count = comparison_df.filter(
    F.col("rule_risk_tier") == F.col("ml_simplified_tier")
).count()
agreement_rate = agreement_count / total_count

disagreement_df = comparison_df.filter(
    F.col("rule_risk_tier") != F.col("ml_simplified_tier")
)
disagreement_count = disagreement_df.count()

print(f"\nAgreement Rate: {agreement_rate:.1%}")
print(f"Disagreement Count: {disagreement_count} ({disagreement_count/total_count:.1%})")

# COMMAND ----------

# Visualize agreement matrix as heatmap
agreement_pdf = comparison_df.groupBy("rule_risk_tier", "ml_simplified_tier").count().toPandas()
pivot_table = agreement_pdf.pivot(index='rule_risk_tier', columns='ml_simplified_tier', values='count')
pivot_table = pivot_table.reindex(index=['High', 'Medium', 'Low'], columns=['High', 'Medium', 'Low'])
pivot_table = pivot_table.fillna(0)

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='Blues',
            linewidths=0.5, cbar_kws={'label': 'Count'})
plt.xlabel('ML Risk Tier (Simplified)')
plt.ylabel('Rule-Based Risk Tier')
plt.title(f'ML vs Rule-Based Agreement Matrix\n(Agreement Rate: {agreement_rate:.1%})',
          fontweight='bold')
plt.tight_layout()
plt.show()

# COMMAND ----------

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
# MAGIC ## 7. Feature Distribution Analysis

# COMMAND ----------

# Check if feature distributions have shifted
df_features = spark.table(FEATURES_TABLE)

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

feature_stats = []
for feature in key_features:
    if feature in df_features.columns:
        stats = df_features.select(
            F.mean(feature).alias("mean"),
            F.stddev(feature).alias("std"),
            F.min(feature).alias("min"),
            F.max(feature).alias("max"),
            F.expr(f"percentile({feature}, 0.5)").alias("median")
        ).collect()[0]
        feature_stats.append({
            'feature': feature,
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            'max': stats['max'],
            'median': stats['median']
        })
        print(f"\n{feature}:")
        print(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")

# COMMAND ----------

# Visualize feature distributions
pdf_features = df_features.select(key_features).toPandas()

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, feature in enumerate(key_features):
    if i < len(axes):
        try:
            axes[i].hist(pdf_features[feature].dropna(), bins=30, color='steelblue',
                        edgecolor='black', alpha=0.7)
        except:
            axes[i].hist(pdf_features[feature].dropna(), bins=30, color='steelblue', alpha=0.7)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Count')
        axes[i].set_title(f'{feature}', fontsize=10, fontweight='bold')

# Hide empty subplot
if len(key_features) < len(axes):
    axes[-1].set_visible(False)

plt.suptitle('Feature Distributions', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Generate Monitoring Report

# COMMAND ----------

# Create comprehensive monitoring metrics
monitoring_metrics = {
    "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "total_customers_scored": int(prob_stats.total_customers),
    "mean_churn_probability": float(prob_stats.mean_prob) if prob_stats.mean_prob else 0.0,
    "std_churn_probability": float(prob_stats.std_prob) if prob_stats.std_prob else 0.0,
    "median_churn_probability": float(prob_stats.p50) if prob_stats.p50 else 0.0,
    "p90_churn_probability": float(prob_stats.p90) if prob_stats.p90 else 0.0,
    "critical_risk_count": int(df_predictions.filter(F.col("ml_risk_tier") == "Critical").count()),
    "high_risk_count": int(df_predictions.filter(F.col("ml_risk_tier") == "High").count()),
    "medium_risk_count": int(df_predictions.filter(F.col("ml_risk_tier") == "Medium").count()),
    "low_risk_count": int(df_predictions.filter(F.col("ml_risk_tier") == "Low").count()),
    "ml_rule_agreement_rate": float(agreement_rate)
}

# Calculate percentages
monitoring_metrics["critical_risk_pct"] = monitoring_metrics["critical_risk_count"] / monitoring_metrics["total_customers_scored"]
monitoring_metrics["high_risk_pct"] = monitoring_metrics["high_risk_count"] / monitoring_metrics["total_customers_scored"]
monitoring_metrics["at_risk_pct"] = (monitoring_metrics["critical_risk_count"] + monitoring_metrics["high_risk_count"] + monitoring_metrics["medium_risk_count"]) / monitoring_metrics["total_customers_scored"]

print("=" * 60)
print("MONITORING REPORT")
print("=" * 60)
print(f"\nReport Date: {monitoring_metrics['report_date']}")
print(f"\n--- Customer Counts ---")
print(f"Total Customers Scored: {monitoring_metrics['total_customers_scored']:,}")
print(f"Critical Risk: {monitoring_metrics['critical_risk_count']:,} ({monitoring_metrics['critical_risk_pct']:.1%})")
print(f"High Risk: {monitoring_metrics['high_risk_count']:,} ({monitoring_metrics['high_risk_pct']:.1%})")
print(f"Medium Risk: {monitoring_metrics['medium_risk_count']:,}")
print(f"Low Risk: {monitoring_metrics['low_risk_count']:,}")
print(f"\n--- Probability Statistics ---")
print(f"Mean Probability: {monitoring_metrics['mean_churn_probability']:.4f}")
print(f"Std Deviation: {monitoring_metrics['std_churn_probability']:.4f}")
print(f"Median Probability: {monitoring_metrics['median_churn_probability']:.4f}")
print(f"90th Percentile: {monitoring_metrics['p90_churn_probability']:.4f}")
print(f"\n--- Model Agreement ---")
print(f"ML vs Rule Agreement Rate: {monitoring_metrics['ml_rule_agreement_rate']:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Alert Thresholds & Checks

# COMMAND ----------

# Define alert thresholds
ALERT_THRESHOLDS = {
    "critical_risk_pct_max": 0.15,    # Alert if >15% are critical
    "high_risk_pct_max": 0.25,        # Alert if >25% are high risk
    "mean_prob_max": 0.50,            # Alert if mean prob > 50%
    "agreement_rate_min": 0.60,       # Alert if agreement drops below 60%
    "at_risk_total_max": 0.50,        # Alert if >50% at risk (critical + high + medium)
}

# Check alerts
alerts = []
warnings_list = []

# Critical risk check
if monitoring_metrics["critical_risk_pct"] > ALERT_THRESHOLDS["critical_risk_pct_max"]:
    alerts.append(f"🚨 CRITICAL: Critical risk customers at {monitoring_metrics['critical_risk_pct']:.1%} (threshold: {ALERT_THRESHOLDS['critical_risk_pct_max']:.1%})")

# High risk check
if monitoring_metrics["high_risk_pct"] > ALERT_THRESHOLDS["high_risk_pct_max"]:
    alerts.append(f"🚨 HIGH: High risk customers at {monitoring_metrics['high_risk_pct']:.1%} (threshold: {ALERT_THRESHOLDS['high_risk_pct_max']:.1%})")

# Agreement rate check
if monitoring_metrics["ml_rule_agreement_rate"] < ALERT_THRESHOLDS["agreement_rate_min"]:
    alerts.append(f"🚨 MODEL DRIFT: ML/Rule agreement at {monitoring_metrics['ml_rule_agreement_rate']:.1%} (threshold: {ALERT_THRESHOLDS['agreement_rate_min']:.1%})")

# Mean probability check
if monitoring_metrics["mean_churn_probability"] > ALERT_THRESHOLDS["mean_prob_max"]:
    warnings_list.append(f"⚠️ WARNING: Mean churn probability at {monitoring_metrics['mean_churn_probability']:.1%} (threshold: {ALERT_THRESHOLDS['mean_prob_max']:.1%})")

# Total at risk check
if monitoring_metrics["at_risk_pct"] > ALERT_THRESHOLDS["at_risk_total_max"]:
    warnings_list.append(f"⚠️ WARNING: Total at-risk customers at {monitoring_metrics['at_risk_pct']:.1%} (threshold: {ALERT_THRESHOLDS['at_risk_total_max']:.1%})")

print("\n" + "=" * 60)
print("ALERTS & WARNINGS")
print("=" * 60)

if alerts:
    print("\n🚨 ALERTS:")
    for alert in alerts:
        print(f"  {alert}")
else:
    print("\n✅ No critical alerts")

if warnings_list:
    print("\n⚠️ WARNINGS:")
    for warning in warnings_list:
        print(f"  {warning}")
else:
    print("✅ No warnings")

if not alerts and not warnings_list:
    print("\n✅ All metrics within acceptable range")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Save Monitoring History

# COMMAND ----------

# Convert monitoring metrics to DataFrame
# Remove calculated percentages for storage (they can be derived)
storage_metrics = {k: v for k, v in monitoring_metrics.items()
                   if not k.endswith('_pct')}

monitoring_df = spark.createDataFrame([storage_metrics])

# Ensure ML schema exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{ML_SCHEMA}")

# Append to history table
try:
    # Try to append to existing table
    monitoring_df.write.mode("append").saveAsTable(MONITORING_HISTORY_TABLE)
    print(f"✅ Monitoring metrics appended to {MONITORING_HISTORY_TABLE}")
except Exception as e:
    # Table might not exist, create it
    monitoring_df.write.mode("overwrite").saveAsTable(MONITORING_HISTORY_TABLE)
    print(f"✅ Created monitoring history table: {MONITORING_HISTORY_TABLE}")

# COMMAND ----------

# View monitoring history
print("\n" + "=" * 60)
print("MONITORING HISTORY (Last 10 Records)")
print("=" * 60)

try:
    history_df = spark.table(MONITORING_HISTORY_TABLE).orderBy(F.col("report_date").desc()).limit(10)
    display(history_df)
except Exception as e:
    print(f"Note: History table just created, only one record available")
    display(spark.table(MONITORING_HISTORY_TABLE))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Trend Analysis (if history exists)

# COMMAND ----------

# Check if we have historical data for trend analysis
try:
    history_count = spark.table(MONITORING_HISTORY_TABLE).count()
    if history_count >= 2:
        print("=" * 60)
        print("TREND ANALYSIS")
        print("=" * 60)

        history_pdf = spark.table(MONITORING_HISTORY_TABLE).orderBy("report_date").toPandas()

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))

        # 1. Mean probability trend
        axes[0, 0].plot(history_pdf['report_date'], history_pdf['mean_churn_probability'],
                        marker='o', color='steelblue', linewidth=2)
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Mean Probability')
        axes[0, 0].set_title('Mean Churn Probability Over Time', fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Critical + High risk trend
        axes[0, 1].plot(history_pdf['report_date'], history_pdf['critical_risk_count'],
                        marker='o', color='red', linewidth=2, label='Critical')
        axes[0, 1].plot(history_pdf['report_date'], history_pdf['high_risk_count'],
                        marker='s', color='orange', linewidth=2, label='High')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('High Risk Customers Over Time', fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Agreement rate trend
        axes[1, 0].plot(history_pdf['report_date'], history_pdf['ml_rule_agreement_rate'],
                        marker='o', color='green', linewidth=2)
        axes[1, 0].axhline(y=0.6, color='red', linestyle='--', label='Threshold (60%)')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Agreement Rate')
        axes[1, 0].set_title('ML vs Rule Agreement Rate', fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Total customers trend
        axes[1, 1].plot(history_pdf['report_date'], history_pdf['total_customers_scored'],
                        marker='o', color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Total Customers')
        axes[1, 1].set_title('Total Customers Scored Over Time', fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    else:
        print("Not enough historical data for trend analysis (need at least 2 records)")
except Exception as e:
    print(f"Trend analysis not available: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Summary & Recommendations

# COMMAND ----------

print("=" * 80)
print("MONITORING SUMMARY")
print("=" * 80)
print(f"""
Report Generated: {monitoring_metrics['report_date']}

📊 KEY METRICS:
   Total Customers Scored: {monitoring_metrics['total_customers_scored']:,}
   Mean Churn Probability: {monitoring_metrics['mean_churn_probability']:.2%}
   ML/Rule Agreement Rate: {monitoring_metrics['ml_rule_agreement_rate']:.1%}

🎯 RISK DISTRIBUTION:
   Critical: {monitoring_metrics['critical_risk_count']:,} customers
   High: {monitoring_metrics['high_risk_count']:,} customers
   Medium: {monitoring_metrics['medium_risk_count']:,} customers
   Low: {monitoring_metrics['low_risk_count']:,} customers

📋 RECOMMENDED ACTIONS:
""")

# Generate recommendations based on current metrics
recommendations = []

if monitoring_metrics['critical_risk_pct'] > 0.10:
    recommendations.append("1. Review critical risk customers immediately - schedule retention calls")

if monitoring_metrics['ml_rule_agreement_rate'] < 0.70:
    recommendations.append("2. Investigate model drift - consider retraining the model")

if monitoring_metrics['mean_churn_probability'] > 0.40:
    recommendations.append("3. Launch proactive engagement campaign for at-risk customers")

if monitoring_metrics['at_risk_pct'] > 0.40:
    recommendations.append("4. Increase retention team capacity to handle high-risk volume")

recommendations.append("5. Run this monitoring notebook weekly to track trends")
recommendations.append("6. Set up Databricks Workflow for automated monitoring")

for rec in recommendations:
    print(f"   {rec}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Next Steps
# MAGIC
# MAGIC ### Monitoring Metrics Tracked:
# MAGIC - Prediction distribution (mean, std, percentiles)
# MAGIC - Risk tier distribution
# MAGIC - ML vs Rule-based agreement rate
# MAGIC - Feature statistics
# MAGIC - Historical trends
# MAGIC
# MAGIC ### Recommended Schedule:
# MAGIC 1. **Daily**: Run scoring notebook (02_score_customers.py)
# MAGIC 2. **Weekly**: Run this monitoring notebook
# MAGIC 3. **Monthly**: Review model performance and consider retraining
# MAGIC
# MAGIC ### Alert Actions:
# MAGIC - **Critical Risk > 15%**: Immediate review of top customers
# MAGIC - **Agreement Rate < 60%**: Investigate model drift, consider retraining
# MAGIC - **Mean Probability > 50%**: Review data quality and customer base changes
