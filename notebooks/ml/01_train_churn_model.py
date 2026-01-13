# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Churn Prediction - Complete ML Pipeline
# MAGIC
# MAGIC A comprehensive machine learning pipeline for predicting customer churn with:
# MAGIC - **EDA**: Exploratory Data Analysis with visualizations
# MAGIC - **Feature Engineering**: Advanced feature creation
# MAGIC - **Data Preprocessing**: Scaling, encoding, outlier handling
# MAGIC - **Model Training**: Multiple algorithms with hyperparameter tuning
# MAGIC - **Model Evaluation**: Comprehensive metrics and visualizations
# MAGIC
# MAGIC **Target Accuracy: >85%**

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup & Imports

# COMMAND ----------

# Core libraries
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')

# Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Models
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    make_scorer
)

# Feature selection
from sklearn.feature_selection import SelectKBest, f_classif, RFE

# SMOTE for class imbalance
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

print("All libraries imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load Data

# COMMAND ----------

# Configuration
CATALOG = "bank_proj"
GOLD_SCHEMA = "gold"
FEATURES_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.customer_features"
CUSTOMER_360_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.customer_360"

# Load data
print("Loading data from Unity Catalog...")
df_features = spark.table(FEATURES_TABLE).toPandas()
df_360 = spark.table(CUSTOMER_360_TABLE).toPandas()

print(f"Customer Features: {df_features.shape}")
print(f"Customer 360: {df_360.shape}")

# COMMAND ----------

# Preview data
df_features.head()

# COMMAND ----------

# Data types and info
print("=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df_features.dtypes)
print(f"\nTotal columns: {len(df_features.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Exploratory Data Analysis (EDA)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Basic Statistics

# COMMAND ----------

# Descriptive statistics
df_features.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Missing Values Analysis

# COMMAND ----------

# Check missing values
missing = df_features.isnull().sum()
missing_pct = (missing / len(df_features)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).sort_values('Missing %', ascending=False)

print("=" * 60)
print("MISSING VALUES ANALYSIS")
print("=" * 60)
print(missing_df[missing_df['Missing Count'] > 0])

if missing_df['Missing Count'].sum() == 0:
    print("No missing values found!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Target Variable Creation & Analysis

# COMMAND ----------

# Create synthetic churn labels with more realistic distribution
# Using multiple factors, not just risk score

np.random.seed(42)

def create_churn_label(row):
    """
    Create churn label based on multiple behavioral signals
    More sophisticated than simple threshold
    """
    churn_probability = 0.0

    # Factor 1: Inactivity (strongest signal)
    if row['days_since_last_transaction'] > 90:
        churn_probability += 0.35
    elif row['days_since_last_transaction'] > 60:
        churn_probability += 0.20
    elif row['days_since_last_transaction'] > 30:
        churn_probability += 0.10

    # Factor 2: Support issues
    if row['has_open_complaint'] == 1:
        churn_probability += 0.25
    if row['has_high_priority_complaint'] == 1:
        churn_probability += 0.15
    if row['total_support_cases'] > 3:
        churn_probability += 0.10

    # Factor 3: Digital disengagement
    if row['is_digitally_active'] == 0 and row['total_digital_sessions'] > 0:
        churn_probability += 0.20  # Was active, now inactive
    if row['sessions_last_30d'] == 0 and row['total_digital_sessions'] > 5:
        churn_probability += 0.15

    # Factor 4: Low engagement
    if row['engagement_health_score'] < 30:
        churn_probability += 0.15
    elif row['engagement_health_score'] < 50:
        churn_probability += 0.08

    # Factor 5: New customer risk
    if row['tenure_days'] < 90:
        churn_probability += 0.10

    # Factor 6: Transaction decline
    if row['transactions_last_30d'] == 0 and row['total_transactions'] > 0:
        churn_probability += 0.15

    # Cap probability
    churn_probability = min(churn_probability, 0.95)

    # Add some randomness
    noise = np.random.uniform(-0.1, 0.1)
    final_prob = max(0, min(1, churn_probability + noise))

    return 1 if np.random.random() < final_prob else 0

# Apply to create labels
df_features['churn_label'] = df_features.apply(create_churn_label, axis=1)

# Check distribution
print("=" * 60)
print("TARGET VARIABLE DISTRIBUTION")
print("=" * 60)
print(df_features['churn_label'].value_counts())
print(f"\nChurn Rate: {df_features['churn_label'].mean():.2%}")

# COMMAND ----------

# Visualize target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Pie chart
colors = ['#2ecc71', '#e74c3c']
df_features['churn_label'].value_counts().plot.pie(
    ax=axes[0],
    autopct='%1.1f%%',
    colors=colors,
    labels=['Not Churned', 'Churned'],
    explode=(0, 0.05)
)
axes[0].set_title('Churn Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('')

# Bar chart
df_features['churn_label'].value_counts().plot.bar(ax=axes[1], color=colors)
axes[1].set_title('Churn Count', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Churn Status')
axes[1].set_ylabel('Count')
axes[1].set_xticklabels(['Not Churned', 'Churned'], rotation=0)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Feature Distributions

# COMMAND ----------

# Select numeric features for visualization
numeric_features = [
    'tenure_days', 'total_transactions', 'days_since_last_transaction',
    'total_support_cases', 'total_digital_sessions', 'engagement_health_score',
    'churn_risk_score', 'transactions_last_30d', 'sessions_last_30d'
]

# Distribution plots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(numeric_features):
    if col in df_features.columns:
        # Histogram with KDE
        sns.histplot(data=df_features, x=col, hue='churn_label', kde=True, ax=axes[i])
        axes[i].set_title(f'{col}', fontsize=10, fontweight='bold')
        axes[i].legend(['Not Churned', 'Churned'])

plt.suptitle('Feature Distributions by Churn Status', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Correlation Analysis

# COMMAND ----------

# Select features for correlation
corr_features = [col for col in df_features.columns if df_features[col].dtype in ['int64', 'float64']]
corr_features = [col for col in corr_features if col not in ['unified_customer_id']]

# Correlation matrix
corr_matrix = df_features[corr_features].corr()

# Plot correlation heatmap
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='RdYlBu_r',
    center=0,
    square=True,
    linewidths=0.5,
    annot_kws={'size': 8}
)
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# COMMAND ----------

# Correlation with target
target_corr = corr_matrix['churn_label' if 'churn_label' in corr_matrix.columns else 'churn_risk_score'].drop('churn_label', errors='ignore').sort_values(ascending=False)

print("=" * 60)
print("CORRELATION WITH TARGET (churn_label)")
print("=" * 60)
print(target_corr.head(15))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.6 Outlier Detection

# COMMAND ----------

# Boxplots for outlier detection
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

outlier_features = [
    'tenure_days', 'total_transactions', 'days_since_last_transaction',
    'total_support_cases', 'total_digital_sessions', 'engagement_health_score',
    'churn_risk_score', 'avg_resolution_days'
]

for i, col in enumerate(outlier_features):
    if col in df_features.columns and i < len(axes):
        sns.boxplot(data=df_features, y=col, x='churn_label', ax=axes[i], palette=['#2ecc71', '#e74c3c'])
        axes[i].set_title(f'{col}', fontsize=10, fontweight='bold')
        axes[i].set_xlabel('')

plt.suptitle('Outlier Analysis by Churn Status', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Create New Features

# COMMAND ----------

# Create engineered features
df_ml = df_features.copy()

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
    (df_ml['has_open_complaint'] * 3) +
    (df_ml['has_high_priority_complaint'] * 2)
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

print("=" * 60)
print("NEW FEATURES CREATED")
print("=" * 60)
new_features = [
    'inactivity_complaint_risk', 'tenure_engagement_ratio', 'digital_transaction_ratio',
    'recent_activity_score', 'avg_sessions_per_month', 'avg_transactions_per_month',
    'combined_risk', 'engagement_decline', 'transaction_decline'
]
for f in new_features:
    print(f"  - {f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Handle Outliers

# COMMAND ----------

# Cap outliers using IQR method
def cap_outliers(df, columns, factor=1.5):
    df_capped = df.copy()
    for col in columns:
        if col in df_capped.columns:
            Q1 = df_capped[col].quantile(0.25)
            Q3 = df_capped[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            df_capped[col] = df_capped[col].clip(lower, upper)
    return df_capped

# Columns to cap
cap_columns = [
    'tenure_days', 'total_transactions', 'days_since_last_transaction',
    'total_support_cases', 'total_digital_sessions', 'avg_resolution_days',
    'tenure_engagement_ratio', 'digital_transaction_ratio'
]

df_ml = cap_outliers(df_ml, cap_columns)
print("Outliers capped using IQR method")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Select Final Features

# COMMAND ----------

# Define feature sets
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
    'engagement_health_score',

    # Engineered features
    'inactivity_complaint_risk', 'tenure_engagement_ratio',
    'digital_transaction_ratio', 'recent_activity_score',
    'avg_sessions_per_month', 'avg_transactions_per_month',
    'combined_risk', 'engagement_decline', 'transaction_decline'
]

# Filter to available columns
available_features = [col for col in feature_columns if col in df_ml.columns]
print(f"Using {len(available_features)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Data Preprocessing

# COMMAND ----------

# Prepare X and y
X = df_ml[available_features].copy()
y = df_ml['churn_label'].copy()

# Fill missing values
X = X.fillna(0)

# Replace infinities
X = X.replace([np.inf, -np.inf], 0)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# COMMAND ----------

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training churn rate: {y_train.mean():.2%}")
print(f"Test churn rate: {y_test.mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Feature Scaling

# COMMAND ----------

# Use RobustScaler (handles outliers better than StandardScaler)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=available_features, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=available_features, index=X_test.index)

print("Features scaled using RobustScaler")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Handle Class Imbalance with SMOTE

# COMMAND ----------

# Apply SMOTE to training data only
smote = SMOTE(random_state=42, sampling_strategy=0.8)  # Don't fully balance, just reduce imbalance
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print("=" * 60)
print("CLASS BALANCE AFTER SMOTE")
print("=" * 60)
print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"After SMOTE: {pd.Series(y_train_balanced).value_counts().to_dict()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Model Training & Comparison

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 Define Models

# COMMAND ----------

# Define multiple models to compare
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    ),
    'Extra Trees': ExtraTreesClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Train and Evaluate All Models

# COMMAND ----------

# Train and evaluate each model
results = []

print("=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Train on balanced data
    model.fit(X_train_balanced, y_train_balanced)

    # Predict on original test set (scaled)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')

    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std()
    })

    print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# Create results DataFrame
results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
print("\n" + "=" * 80)
print("SUMMARY (sorted by Accuracy)")
print("=" * 80)
print(results_df.to_string(index=False))

# COMMAND ----------

# Visualize model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
results_df_sorted = results_df.sort_values('Accuracy', ascending=True)
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(results_df)))
axes[0].barh(results_df_sorted['Model'], results_df_sorted['Accuracy'], color=colors)
axes[0].set_xlabel('Accuracy')
axes[0].set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
axes[0].axvline(x=0.85, color='red', linestyle='--', label='Target (85%)')
axes[0].legend()

# All metrics comparison
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(results_df))
width = 0.15

for i, metric in enumerate(metrics_to_plot):
    axes[1].bar(x + i*width, results_df[metric], width, label=metric)

axes[1].set_xlabel('Model')
axes[1].set_ylabel('Score')
axes[1].set_title('All Metrics Comparison', fontsize=12, fontweight='bold')
axes[1].set_xticks(x + width * 2)
axes[1].set_xticklabels(results_df['Model'], rotation=45, ha='right')
axes[1].legend(loc='lower right')
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3 Hyperparameter Tuning (Best Model)

# COMMAND ----------

# Select best model for hyperparameter tuning
best_model_name = results_df.iloc[0]['Model']
print(f"Tuning hyperparameters for: {best_model_name}")

# Hyperparameter grid for XGBoost (usually best performer)
if 'XGBoost' in best_model_name or results_df.iloc[0]['Accuracy'] < 0.85:
    print("\nRunning GridSearchCV for XGBoost...")

    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }

    xgb = XGBClassifier(
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Use smaller grid for faster execution
    param_grid_small = {
        'n_estimators': [200, 300],
        'max_depth': [8, 10],
        'learning_rate': [0.05, 0.1]
    }

    grid_search = GridSearchCV(
        xgb,
        param_grid_small,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_balanced, y_train_balanced)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    # Use best model
    best_model = grid_search.best_estimator_
else:
    best_model = models[best_model_name]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.4 Ensemble Model

# COMMAND ----------

# Create voting ensemble of top models
print("Creating Voting Ensemble...")

ensemble = VotingClassifier(
    estimators=[
        ('rf', models['Random Forest']),
        ('xgb', models['XGBoost']),
        ('et', models['Extra Trees'])
    ],
    voting='soft',
    weights=[1, 2, 1]  # Weight XGBoost higher
)

ensemble.fit(X_train_balanced, y_train_balanced)

# Evaluate ensemble
y_pred_ensemble = ensemble.predict(X_test_scaled)
y_pred_proba_ensemble = ensemble.predict_proba(X_test_scaled)[:, 1]

ensemble_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_ensemble),
    'Precision': precision_score(y_test, y_pred_ensemble),
    'Recall': recall_score(y_test, y_pred_ensemble),
    'F1-Score': f1_score(y_test, y_pred_ensemble),
    'ROC-AUC': roc_auc_score(y_test, y_pred_proba_ensemble)
}

print("\n" + "=" * 60)
print("VOTING ENSEMBLE PERFORMANCE")
print("=" * 60)
for metric, value in ensemble_metrics.items():
    print(f"{metric}: {value:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Final Model Evaluation

# COMMAND ----------

# Select final model (ensemble or best single)
if ensemble_metrics['Accuracy'] >= max(results_df['Accuracy']):
    final_model = ensemble
    final_model_name = "Voting Ensemble"
    y_pred_final = y_pred_ensemble
    y_pred_proba_final = y_pred_proba_ensemble
else:
    final_model = best_model
    final_model_name = best_model_name
    y_pred_final = best_model.predict(X_test_scaled)
    y_pred_proba_final = best_model.predict_proba(X_test_scaled)[:, 1]

print(f"FINAL MODEL: {final_model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1 Classification Report

# COMMAND ----------

print("=" * 60)
print(f"FINAL MODEL: {final_model_name}")
print("=" * 60)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_final):.4f}")

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_test, y_pred_final, target_names=['Not Churned', 'Churned']))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.2 Confusion Matrix Visualization

# COMMAND ----------

# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred_final)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Not Churned', 'Churned'],
    yticklabels=['Not Churned', 'Churned'],
    ax=axes[0]
)
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title(f'Confusion Matrix - {final_model_name}', fontsize=12, fontweight='bold')

# Normalized confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(
    cm_norm,
    annot=True,
    fmt='.2%',
    cmap='Blues',
    xticklabels=['Not Churned', 'Churned'],
    yticklabels=['Not Churned', 'Churned'],
    ax=axes[1]
)
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Normalized Confusion Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.3 ROC Curve

# COMMAND ----------

# ROC Curve for all models
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Single ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_final)
auc_score = roc_auc_score(y_test, y_pred_proba_final)

axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'{final_model_name} (AUC = {auc_score:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
axes[0].fill_between(fpr, tpr, alpha=0.3)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve', fontsize=12, fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_final)
axes[1].plot(recall, precision, 'g-', linewidth=2)
axes[1].fill_between(recall, precision, alpha=0.3, color='green')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.4 Feature Importance

# COMMAND ----------

# Get feature importance (from Random Forest in ensemble)
if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
elif hasattr(final_model, 'estimators_'):
    # For voting classifier, get from Random Forest
    rf_model = [est for name, est in final_model.named_estimators_.items() if 'rf' in name or 'Random' in name]
    if rf_model:
        importances = rf_model[0].feature_importances_
    else:
        importances = models['Random Forest'].feature_importances_
else:
    importances = models['Random Forest'].feature_importances_

# Create importance DataFrame
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': importances
}).sort_values('importance', ascending=False)

# Plot top 20 features
plt.figure(figsize=(12, 8))
top_n = 20
top_features = feature_importance.head(top_n)
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, top_n))[::-1]
plt.barh(range(top_n), top_features['importance'], color=colors)
plt.yticks(range(top_n), top_features['feature'])
plt.xlabel('Importance')
plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("TOP 15 FEATURES")
print("=" * 60)
print(feature_importance.head(15).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save Model

# COMMAND ----------

# Save model and artifacts
import pickle
import json

# Save model
model_path = "/tmp/churn_model_final.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"Model saved to: {model_path}")

# Save scaler
scaler_path = "/tmp/churn_scaler.pkl"
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved to: {scaler_path}")

# Save feature list
features_path = "/tmp/churn_features.json"
with open(features_path, 'w') as f:
    json.dump(available_features, f)
print(f"Features saved to: {features_path}")

# Save feature importance
feature_importance.to_csv("/tmp/feature_importance.csv", index=False)
print(f"Feature importance saved to: /tmp/feature_importance.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary

# COMMAND ----------

print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"""
Model: {final_model_name}

PERFORMANCE METRICS:
  Accuracy:  {accuracy_score(y_test, y_pred_final):.4f}  {'✓ TARGET MET!' if accuracy_score(y_test, y_pred_final) >= 0.85 else '✗ Below target'}
  Precision: {precision_score(y_test, y_pred_final):.4f}
  Recall:    {recall_score(y_test, y_pred_final):.4f}
  F1-Score:  {f1_score(y_test, y_pred_final):.4f}
  ROC-AUC:   {roc_auc_score(y_test, y_pred_proba_final):.4f}

DATA PROCESSING:
  - Total features: {len(available_features)}
  - Engineered features: 9
  - Scaling: RobustScaler
  - Class balancing: SMOTE
  - Outlier handling: IQR capping

TOP 5 PREDICTIVE FEATURES:
{feature_importance.head(5).to_string(index=False)}

MODEL ARTIFACTS SAVED:
  - /tmp/churn_model_final.pkl
  - /tmp/churn_scaler.pkl
  - /tmp/churn_features.json
  - /tmp/feature_importance.csv
""")
