# Interview Preparation Guide
## Banking Customer Churn Prediction Project

This guide prepares you to confidently explain every aspect of your project in technical interviews.

---

# TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [dbt Fundamentals](#2-dbt-fundamentals)
3. [Medallion Architecture](#3-medallion-architecture)
4. [Customer Entity Resolution](#4-customer-entity-resolution)
5. [SQL & Data Modeling](#5-sql--data-modeling)
6. [Feature Engineering](#6-feature-engineering)
7. [Machine Learning](#7-machine-learning)
8. [CI/CD & GitHub Actions](#8-cicd--github-actions)
9. [Databricks & Unity Catalog](#9-databricks--unity-catalog)
10. [Data Quality & Testing](#10-data-quality--testing)
11. [Common Behavioral Questions](#11-common-behavioral-questions)

---

# 1. PROJECT OVERVIEW

## What You Built
A customer churn prediction platform for "Apex National Bank" that:
- Unifies customer data from 4 different source systems
- Transforms raw data through Bronze → Silver → Gold layers
- Engineers ML features for churn prediction
- Trains and deploys a churn model (0.86 AUC)
- Automates everything with CI/CD

## The Business Problem
Banks lose $5-6M annually to customer churn. Customer data is fragmented across:
- **ERPNext** (Core Banking): Transactions, accounts
- **Salesforce** (CRM): Complaints, interactions
- **Supabase** (Digital): App logins, sessions
- **Google Sheets** (Legacy): Branch data, manual notes

**Pain Point**: No single view of customer. Can't predict who will leave.

## Your Solution
```
4 Source Systems → Bronze (raw) → Silver (cleaned) → Gold (features) → ML Model → Predictions
```

### Interview Question: "Tell me about a data project you built."

**Your Answer:**
"I built an end-to-end customer churn prediction platform for a retail bank POC. The challenge was unifying fragmented customer data from four different systems - each identified customers differently.

I implemented a Medallion architecture using dbt-databricks: Bronze layer for raw ingestion, Silver for cleaning and customer unification using email-based entity resolution, and Gold for ML feature engineering.

The ML pipeline achieved 0.86 ROC-AUC using sklearn's HistGradientBoostingClassifier. Everything runs on Databricks Free Edition with GitHub Actions CI/CD - every PR triggers automated testing, and merges auto-deploy to production.

The business impact: what was 40 hours of manual monthly reporting is now fully automated, and the bank can proactively identify at-risk customers for retention campaigns."

---

# 2. DBT FUNDAMENTALS

## What is dbt?

**Definition**: dbt (data build tool) is a transformation framework that lets you write SQL SELECT statements and handles the CREATE TABLE/VIEW, dependency ordering, and testing automatically.

**Key Insight**: dbt is the "T" in ELT. It doesn't extract or load data - it transforms data that's already in your warehouse.

## Core Concepts

### 2.1 Models
A model is a SQL SELECT statement in a `.sql` file. dbt compiles it and runs it against your database.

```sql
-- models/staging/stg_erp_customers.sql
select
    customer_id,
    customer_name,
    email
from {{ source('bronze', 'erp_customers') }}
```

dbt turns this into:
```sql
CREATE VIEW staging.stg_erp_customers AS
SELECT customer_id, customer_name, email
FROM bronze.erp_customers
```

### 2.2 ref() Function
**What it does**: Creates dependencies between models.

```sql
-- This model depends on stg_erp_customers
select * from {{ ref('stg_erp_customers') }}
```

**Why it matters**:
- dbt builds models in correct order (parents before children)
- If `stg_erp_customers` fails, downstream models are skipped
- Enables `dbt build --select +model_name` to build all parents

### Interview Question: "What does ref() do in dbt?"

**Your Answer:**
"ref() creates a dependency between dbt models. When I write `{{ ref('stg_erp_customers') }}`, two things happen: First, dbt knows to build stg_erp_customers before any model that references it. Second, it resolves to the correct schema and table name based on my target environment - so in dev it might be `dev_staging.stg_erp_customers` and in prod it's `staging.stg_erp_customers`. This is how dbt manages the DAG (directed acyclic graph) of transformations."

### 2.3 source() Function
**What it does**: References raw tables that exist outside dbt.

```sql
select * from {{ source('bronze', 'erp_customers') }}
```

**Why not just write the table name directly?**
- Sources are defined in `_sources.yml` with documentation
- You can add freshness checks
- If source table moves, update one place, not every model

### 2.4 Materializations

| Type | What it creates | When to use |
|------|-----------------|-------------|
| `view` | CREATE VIEW | Staging models, lightweight transforms |
| `table` | CREATE TABLE | Heavy transforms, frequently queried |
| `incremental` | INSERT INTO (append) | Large fact tables, event data |
| `ephemeral` | No object (CTE) | Helper logic, not queried directly |

**In your project:**
- Staging models = `view` (lightweight wrappers)
- Silver/Gold models = `table` (heavy transforms, queried by ML)

### Interview Question: "When would you use a view vs a table in dbt?"

**Your Answer:**
"Views for staging models because they're just column renaming and type casting - I want to always see fresh source data. Tables for Silver and Gold layers because they involve complex joins and aggregations that would be expensive to recompute on every query. For example, my `dim_customer_unified` does a 4-way join with deduplication - that should be materialized once, not computed every time someone queries it.

I'd use incremental for event data like transactions if the table grew to millions of rows - no need to rebuild the whole table when only new transactions need processing."

### 2.5 Tests

dbt has built-in tests:
- `unique` - column has no duplicates
- `not_null` - column has no NULLs
- `accepted_values` - column only contains expected values
- `relationships` - foreign key exists in parent table

```yaml
# _staging.yml
models:
  - name: stg_erp_customers
    columns:
      - name: customer_id
        tests:
          - not_null
          - unique
```

### Interview Question: "How do you ensure data quality in dbt?"

**Your Answer:**
"I use dbt's built-in tests on key columns. Every primary key gets `not_null` and `unique` tests. Foreign keys get `relationships` tests to ensure referential integrity. For business rules, I write custom tests in the `tests/` folder.

In my project, I caught a data quality issue where 18 transactions had no matching customer - the `not_null` test on `unified_customer_id` failed. I changed it to `severity: warn` because orphaned transactions are acceptable, but I want visibility into them. The test documents the issue without blocking deployment."

---

# 3. MEDALLION ARCHITECTURE

## What is it?
A data organization pattern with three layers:

```
Bronze (Raw) → Silver (Cleaned) → Gold (Business-Ready)
```

## Why use it?

### Interview Question: "Why did you choose Medallion architecture?"

**Your Answer:**
"Three main reasons:

1. **Data Lineage & Debugging**: If something breaks in Gold, I can trace back through Silver to Bronze and see exactly where the issue is. Bronze is always the raw truth.

2. **Reprocessing**: If business logic changes, I don't re-ingest data. I just rebuild Silver and Gold from Bronze. We had this exact scenario - we changed the customer unification logic and just reran `dbt build --select silver+`.

3. **Separation of Concerns**: Bronze is owned by data engineers (ingestion). Silver is data modeling (cleaning, conforming). Gold is analytics engineering (business metrics, ML features). Different teams can work independently."

## Your Implementation

### Bronze Layer
**Location**: `bank_proj.bronze.*`
**What's there**: Raw tables from source systems, exactly as received
**Example tables**: `erp_customers`, `sf_contacts`, `sb_app_sessions`, `gs_branches`

### Silver Layer (Your dbt Staging + Silver)
**Location**: `bank_proj.staging.*` and `bank_proj.silver.*`
**What happens**:
- Column renaming (`Id` → `customer_id`)
- Type casting (`string` → `timestamp`)
- Basic cleaning
- Customer unification (the hard part)

### Gold Layer
**Location**: `bank_proj.gold.*`
**What's there**: Business-ready tables
- `customer_360` - Complete customer view
- `customer_features` - ML feature table
- `agg_churn_by_segment` - Pre-aggregated metrics

### Interview Question: "Walk me through your data pipeline."

**Your Answer:**
"Data flows through three layers:

**Bronze**: Raw data lands here from our ingestion notebook. ERPNext customers, Salesforce contacts, Supabase sessions, Google Sheets branches. No transformation - just `SELECT *` with metadata columns.

**Silver**: This is where the magic happens. Staging models rename columns and cast types. Then I unify customers across all four systems using email as the matching key. The `dim_customer_unified` model does a 4-way LEFT JOIN from ERPNext (master system) to Salesforce, Supabase, and Google Sheets. I also deduplicate and create surrogate keys with MD5 hashes.

**Gold**: Business-ready tables. `customer_360` joins the unified customer with aggregated transactions, support cases, and digital engagement. `customer_features` engineers 20+ features for ML - things like `days_since_last_transaction`, `has_open_complaint`, `sessions_last_30d`. These features directly feed the churn model."

---

# 4. CUSTOMER ENTITY RESOLUTION

## The Problem
Each system identifies customers differently:
- ERPNext: `CUST-001234`
- Salesforce: `003Hs00001ABC123`
- Supabase: `uuid-v4`
- Google Sheets: `customer_email`

How do you know these are the same person?

## Your Solution: Email-Based Matching

```sql
-- dim_customer_unified.sql (simplified)
select
    md5(lower(trim(erp.email))) as unified_customer_id,
    erp.erp_customer_id,
    sf.sf_contact_id,
    app.app_customer_email
from erp_customers erp
left join sf_contacts sf
    on lower(trim(erp.email)) = lower(trim(sf.sf_email))
left join app_customers app
    on lower(trim(erp.email)) = lower(trim(app.app_customer_email))
```

### Interview Question: "How did you unify customers across different systems?"

**Your Answer:**
"I used email-based entity resolution. Email is the one attribute that exists in all four systems and uniquely identifies a person.

The process:
1. Normalize emails: `lower(trim(email))` to handle ' John@Bank.com ' vs 'john@bank.com'
2. ERPNext is the master system (most complete customer data)
3. LEFT JOIN Salesforce contacts on normalized email
4. LEFT JOIN Supabase app users on normalized email
5. LEFT JOIN Google Sheets notes on normalized email
6. Create a unified surrogate key: `md5(normalized_email)`

Why MD5? It creates a consistent hash regardless of which system the customer appears in. If someone has email 'john@bank.com', their `unified_customer_id` is always `abc123def...`

I also added deduplication with ROW_NUMBER() because some systems had duplicate emails - I keep the first record by ERPNext customer ID."

### Interview Question: "What are the limitations of email-based matching?"

**Your Answer:**
"Good question - there are several:

1. **Customers without email**: Some older customers might not have email on file. My model falls back to ERPNext customer ID for these.

2. **Shared emails**: Family members sometimes share an email. This would incorrectly merge two different customers.

3. **Changed emails**: If a customer updates their email, they appear as a new customer.

4. **Typos**: 'john@bank.com' and 'jonh@bank.com' won't match.

In production, I'd enhance this with:
- Phone number as secondary match key
- Fuzzy name matching
- Address matching
- A proper MDM (Master Data Management) solution for complex cases"

---

# 5. SQL & DATA MODELING

## Key SQL Concepts in Your Project

### 5.1 Window Functions

**ROW_NUMBER()** - Used for deduplication
```sql
row_number() over (
    partition by md5(coalesce(lower(trim(email)), customer_id))
    order by customer_id
) as rn
...
where rn = 1  -- Keep only first record per customer
```

**What it does**: Numbers rows within each partition. `rn = 1` keeps the first row per group.

### Interview Question: "How did you handle duplicate customers?"

**Your Answer:**
"I used ROW_NUMBER() window function. The logic:
1. PARTITION BY the unified customer key (MD5 hash of email)
2. ORDER BY ERPNext customer ID (deterministic ordering)
3. Filter WHERE rn = 1 to keep only the first record

This guarantees exactly one row per customer. The key insight is that the MD5 hash in the PARTITION BY must match the key I'm deduplicating on."

### 5.2 COALESCE

```sql
coalesce(erp.customer_name, sf.sf_name) as customer_name
```

**What it does**: Returns the first non-NULL value.

**In your project**: ERPNext is the master system, but if a field is missing, fall back to Salesforce.

### 5.3 JOINs

**LEFT JOIN** - Keep all ERPNext customers, match Salesforce if exists
```sql
from erp_customers erp
left join sf_contacts sf on erp.email = sf.email
```

**FULL OUTER JOIN** - Keep records from both sides (used in dim_geography)
```sql
from territories t
full outer join branches b on t.territory = b.region
```

### Interview Question: "Why LEFT JOIN instead of INNER JOIN for customer unification?"

**Your Answer:**
"I want to keep ALL ERPNext customers, even if they don't exist in Salesforce. INNER JOIN would drop customers who haven't filed a support case or don't have a Salesforce contact.

ERPNext is our system of record - if someone is a customer there, they're a customer, period. The LEFT JOIN enriches them with Salesforce data when available, but doesn't exclude them when it's not."

### 5.4 CTEs (Common Table Expressions)

```sql
with customers as (
    select * from {{ ref('dim_customer_unified') }}
),
transactions as (
    select * from {{ ref('fct_transactions') }}
),
final as (
    select c.*, sum(t.amount) as total_spend
    from customers c
    left join transactions t on c.id = t.customer_id
    group by c.id
)
select * from final
```

**Why CTEs?**
- Readable step-by-step logic
- Each CTE has a descriptive name
- Easier to debug (can SELECT from intermediate CTEs)

---

# 6. FEATURE ENGINEERING

## What are Features?
Features are measurable properties that the ML model uses to make predictions. Good features capture signals that indicate whether a customer will churn.

## Features in Your Project

| Feature | Calculation | Churn Signal |
|---------|-------------|--------------|
| `days_since_last_transaction` | `datediff(current_date, last_txn)` | Long gaps = disengagement |
| `has_open_complaint` | `open_cases > 0` | Unresolved issues = frustrated |
| `sessions_last_30d` | Count of app logins | Declining usage = leaving |
| `total_transaction_amount` | Sum of all txn amounts | High value = worth retaining |
| `tenure_days` | Days since first transaction | New customers churn more |
| `complaint_resolution_days` | Avg time to close cases | Slow service = frustrated |

### Interview Question: "How did you engineer features for churn prediction?"

**Your Answer:**
"I created about 20 features in `customer_features.sql` based on three categories:

**Behavioral features**: Transaction recency, frequency, monetary value. `days_since_last_transaction` is the strongest single predictor - customers who haven't transacted in 90+ days are 3x more likely to churn.

**Engagement features**: App sessions in last 30 days, total events, session duration. Declining digital engagement often precedes churn.

**Sentiment features**: Open complaints, high-priority cases, average resolution time. A customer with an unresolved complaint for 30+ days is a churn risk.

I also created compound features like `engagement_health_score` that combines multiple signals into a 0-100 score, and `churn_risk_score` as a rule-based heuristic before ML."

### Interview Question: "How do you handle missing values in features?"

**Your Answer:**
"I use COALESCE to provide sensible defaults:

```sql
coalesce(days_since_last_transaction, 9999) -- Never transacted = very old
coalesce(sessions_last_30d, 0) -- No sessions = 0
coalesce(open_cases, 0) -- No cases = 0
```

The key is understanding what NULL means semantically. A customer with no transactions isn't 'unknown' - they've legitimately never transacted, which is itself a strong signal."

---

# 7. MACHINE LEARNING

## Your Model Architecture

```
customer_features (Gold table)
        ↓
  Train/Test Split (80/20)
        ↓
  RobustScaler (normalize features)
        ↓
  HistGradientBoostingClassifier (handles missing values)
        ↓
  class_weight='balanced' (handle imbalanced classes)
        ↓
  Ensemble Voting (6 algorithms)
        ↓
  0.86 ROC-AUC
```

### Interview Question: "Walk me through your ML pipeline."

**Your Answer:**
"The pipeline has four stages:

**1. Data Prep**: Read `customer_features` from Unity Catalog. Split 80/20 train/test with stratification (churn is only 20% of data).

**2. Preprocessing**: RobustScaler for numeric features because it handles outliers better than StandardScaler - some customers have extreme transaction values.

**3. Training**: I tested 6 algorithms - Logistic Regression, Random Forest, Gradient Boosting, HistGradientBoosting, AdaBoost, and Gaussian NB. Used cross-validation to evaluate each.

**4. Ensemble**: Created a VotingClassifier combining the top 3 performers with soft voting. The ensemble achieved 0.86 ROC-AUC, better than any individual model."

### Interview Question: "Why HistGradientBoostingClassifier instead of XGBoost?"

**Your Answer:**
"Databricks Free Edition blocks pip install for security reasons - you can only use pre-installed packages. XGBoost isn't pre-installed, but sklearn's HistGradientBoostingClassifier is.

Fortunately, HistGradientBoostingClassifier is very similar to XGBoost:
- Both use histogram-based gradient boosting
- Both handle missing values natively
- Both are fast on medium-sized datasets

In my tests, HistGradientBoosting achieved 0.84 AUC alone, which is within 1-2% of what XGBoost typically achieves. The constraint forced a good solution."

### Interview Question: "How did you handle class imbalance?"

**Your Answer:**
"Churn is imbalanced - only about 20% of customers actually churn. I handled this three ways:

1. **Stratified split**: Train/test split preserves the 80/20 ratio in both sets.

2. **class_weight='balanced'**: The model automatically upweights the minority class during training.

3. **Evaluation metrics**: I focused on ROC-AUC and F1-score rather than accuracy. A model that predicts 'no churn' for everyone would be 80% accurate but useless.

I couldn't use SMOTE (synthetic oversampling) because imbalanced-learn isn't available in Databricks Free Edition, but class_weight achieves similar results for tree-based models."

### Interview Question: "What does ROC-AUC mean and why did you choose it?"

**Your Answer:**
"ROC-AUC measures how well the model ranks customers by churn probability. An AUC of 0.86 means: if I randomly pick one churner and one non-churner, the model correctly ranks the churner as higher risk 86% of the time.

I chose it over accuracy because:
1. **Works with imbalanced data**: Accuracy is misleading when classes are imbalanced.
2. **Threshold-independent**: AUC evaluates the model across all possible thresholds, not just 0.5.
3. **Business-relevant**: The bank wants to rank customers by risk and target the top 10% - AUC directly measures ranking ability."

---

# 8. CI/CD & GITHUB ACTIONS

## What is CI/CD?
- **CI (Continuous Integration)**: Automatically test code when PRs are opened
- **CD (Continuous Deployment)**: Automatically deploy when code merges to main

## Your Implementation

### On Pull Request:
```yaml
# .github/workflows/dbt-ci.yml
on:
  pull_request:
    branches: [main, master]
```
- Runs `dbt build` in isolated test schema
- All tests must pass before merge

### On Merge to Main:
```yaml
# .github/workflows/dbt-deploy.yml
on:
  push:
    branches: [main, master]
```
- Runs `dbt build` in production schema
- Updates all Silver and Gold tables

### Interview Question: "How did you implement CI/CD for your dbt project?"

**Your Answer:**
"I set up two GitHub Actions workflows:

**CI (Pull Requests)**: When someone opens a PR that touches dbt files, GitHub Actions spins up a runner, installs dbt-databricks, and runs `dbt build`. This catches issues before they hit production - column mismatches, test failures, syntax errors.

**CD (Deployment)**: When a PR merges to main, another workflow runs `dbt build` against the production schemas. All tables get refreshed with the latest logic.

The key is environment isolation - CI runs in a test schema so it doesn't affect production data. I use GitHub Secrets to store the Databricks token securely."

### Interview Question: "What happens if CI fails?"

**Your Answer:**
"The PR is blocked from merging. We had this happen when I pushed code with a test referencing a non-existent column - `branch_code` instead of `branch_id`. The CI caught it, I fixed the test, pushed again, and it passed.

This is exactly what CI is for - catch bugs before they hit production. Without CI, that column error would have broken our production pipeline."

---

# 9. DATABRICKS & UNITY CATALOG

## What is Databricks?
A unified analytics platform for data engineering, data science, and ML. Built on Apache Spark.

## What is Unity Catalog?
Databricks' data governance layer. Think of it as a three-level namespace:

```
Catalog → Schema → Table
bank_proj → bronze → erp_customers
bank_proj → silver → dim_customer_unified
bank_proj → gold → customer_features
```

### Interview Question: "What is Unity Catalog and why did you use it?"

**Your Answer:**
"Unity Catalog is Databricks' data governance solution. It provides:

1. **Three-level namespace**: Catalog.Schema.Table instead of just Schema.Table. This lets me organize data by domain (bank_proj) and layer (bronze, silver, gold).

2. **Access control**: I can grant permissions at catalog, schema, or table level. Data engineers can write to bronze, analysts can read from gold.

3. **Data lineage**: Unity Catalog tracks where data came from and where it flows to. I can see that `customer_360` depends on `dim_customer_unified` which depends on `stg_erp_customers`.

4. **Volumes**: Persistent file storage that survives session restarts. I used this to store my trained ML model since /tmp is ephemeral on serverless."

## Databricks Free Edition Constraints

### Interview Question: "What challenges did you face with Databricks Free Edition?"

**Your Answer:**
"Three main constraints:

1. **No pip install**: Serverless compute blocks installing packages for security. I couldn't use XGBoost or imbalanced-learn. Solution: use pre-installed sklearn alternatives.

2. **Ephemeral /tmp**: Files in /tmp disappear when the cluster restarts. My first model training worked, but the model vanished. Solution: save to Unity Catalog Volumes at `/Volumes/bank_proj/bronze/credentials/`.

3. **No long-running clusters**: Free Edition only offers serverless, which is session-based. Solution: design pipelines to be idempotent - can safely rerun from scratch.

These constraints actually forced better engineering practices. Volume storage is more production-appropriate than /tmp anyway."

---

# 10. DATA QUALITY & TESTING

## Your Testing Strategy

| Test Type | What It Checks | Example |
|-----------|---------------|---------|
| `not_null` | No NULL values | Primary keys |
| `unique` | No duplicates | Primary keys |
| `accepted_values` | Value in allowed list | Status fields |
| `relationships` | FK exists in parent | Foreign keys |
| Custom SQL | Business rules | "No future dates" |

## Test Severities

```yaml
tests:
  - not_null:
      severity: warn  # Log but don't fail
```

- `error` (default): Fails the build
- `warn`: Logs warning, build continues

### Interview Question: "How do you ensure data quality?"

**Your Answer:**
"I test at multiple levels:

**Schema tests**: Every primary key has `not_null` and `unique` tests. Every foreign key has a `relationships` test.

**Business logic tests**: Custom SQL in the `tests/` folder. For example, 'transaction amount must be positive' or 'case closed date must be after created date'.

**Severity levels**: Some issues are warnings, not errors. I have 18 transactions without matching customers - that's a data quality issue to investigate, but it shouldn't block deployment. I set that test to `severity: warn`.

**Freshness checks**: dbt can check if source data is stale. If bronze tables haven't been updated in 24 hours, the pipeline warns me."

---

# 11. COMMON BEHAVIORAL QUESTIONS

### "Tell me about a time you debugged a difficult data issue."

**Your Answer:**
"When I first ran the full pipeline, the `dim_geography` model created duplicate keys. The `unique` test failed with 5 duplicates.

I traced through the logic: I was doing a FULL OUTER JOIN between territories and branches, but multiple branches exist in the same region. So region 'North' joined to 5 different branches, creating 5 rows with the same geography_key.

The fix: aggregate branches by region first (count, sum of staff, etc.), THEN join. This guarantees one row per region. The test passed after the fix.

The lesson: always think about cardinality before joining. A many-to-one join inflates row count."

### "How would you scale this to production?"

**Your Answer:**
"The architecture is production-ready - I'd change the data scale, not the design:

1. **Incremental models**: Convert `fct_transactions` to incremental. Only process new transactions, not the full history.

2. **Partitioning**: Partition large tables by date. Queries for 'last 30 days' only scan relevant partitions.

3. **Cluster compute**: Move from serverless to a proper cluster for ML training on millions of rows.

4. **Real-time scoring**: Add a streaming layer for real-time churn predictions, not just daily batch.

5. **Monitoring**: Add data observability tools - row counts, schema changes, drift detection."

### "What would you do differently?"

**Your Answer:**
"Three things:

1. **Better entity resolution**: Email matching has limitations. I'd add phone number as secondary key, maybe fuzzy name matching.

2. **More sophisticated ML**: Explore neural networks, time-series features (is usage declining week over week?), and model explainability with SHAP values.

3. **Data contracts**: Define explicit schemas between layers. If upstream changes a column, break loudly rather than silently propagating nulls."

---

# STUDY CHECKLIST

Before your interview, make sure you can:

- [ ] Explain Medallion architecture and why each layer exists
- [ ] Write a dbt model from memory (with ref, source, config)
- [ ] Explain ref() vs source() without hesitation
- [ ] Describe your customer unification logic step by step
- [ ] Write a ROW_NUMBER() deduplication query
- [ ] Explain why you chose specific ML algorithms
- [ ] Describe what happens when a PR is opened (CI flow)
- [ ] Explain Unity Catalog's three-level namespace
- [ ] Describe how you'd scale to 10x data volume

---

# QUICK REFERENCE CARD

Print this for last-minute review:

```
dbt ref() = dependency between models, resolves schema names
dbt source() = reference to raw tables outside dbt
Medallion = Bronze (raw) → Silver (clean) → Gold (business)
Entity Resolution = Matching same entity across different systems

My project:
- 4 sources: ERPNext, Salesforce, Supabase, Google Sheets
- Unification: Email-based matching with MD5 surrogate key
- Features: 20+ engineered features for ML
- Model: HistGradientBoosting ensemble, 0.86 AUC
- CI/CD: GitHub Actions, PR triggers test, merge triggers deploy
- Constraint: Databricks Free Edition (no pip, ephemeral /tmp)
```
