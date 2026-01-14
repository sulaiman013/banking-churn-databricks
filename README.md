# Banking Customer Churn Prediction Platform

A production-grade end-to-end data platform for **Apex National Bank** demonstrating multi-source data integration, customer unification, and ML-based churn prediction using modern data stack technologies.

![Architecture Overview](docs/diagrams/00_architecture_overview.png)

---

## Table of Contents

- [Business Problem](#business-problem)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Data Sources](#data-sources)
- [Data Model](#data-model)
  - [Bronze Layer (Raw)](#bronze-layer-raw)
  - [Silver Layer (Cleaned)](#silver-layer-cleaned)
  - [Gold Layer (Analytics)](#gold-layer-analytics)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [dbt Models Documentation](#dbt-models-documentation)
- [CI/CD Pipeline](#cicd-pipeline)
- [ML Churn Prediction Model](#ml-churn-prediction-model)
- [Dashboards](#dashboards)
- [Key Features](#key-features)
- [Learnings & Best Practices](#learnings--best-practices)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

---

## Business Problem

**Apex National Bank** faces a critical challenge: customer data is fragmented across 4 different systems with no unified view. This leads to:

| Pain Point | Impact |
|------------|--------|
| **Siloed Data** | Customer info scattered across ERPNext, Salesforce, Supabase, and Google Sheets |
| **Manual Reporting** | 40+ hours/month spent on manual data reconciliation |
| **No Churn Visibility** | Unable to identify at-risk customers until they leave |
| **Revenue Loss** | Estimated $5-6M annual cost due to preventable churn |
| **Reactive Support** | No proactive intervention for dissatisfied customers |

### The Challenge

Each system uses different customer identifiers:
- **ERPNext (Core Banking)**: `customer_id` (e.g., `CUST-00001`)
- **Salesforce (CRM)**: `contact_id` (e.g., `003xx000004TmEQAA0`)
- **Supabase (Digital)**: `customer_email` (e.g., `john@email.com`)
- **Google Sheets (Legacy)**: `customer_email` + manual notes

**How do you know if `CUST-00001` in ERPNext is the same person as `003xx000004TmEQ` in Salesforce?**

---

## Solution Overview

This project implements a **unified data platform** that:

1. **Ingests** data from 4 disparate source systems into a central data lake
2. **Unifies** customer identities using email as the primary matching key
3. **Transforms** raw data through Bronze → Silver → Gold medallion architecture
4. **Engineers** ML features from behavioral, transactional, and support data
5. **Predicts** customer churn risk using machine learning
6. **Enables** proactive retention through actionable insights

### Business Value Delivered

| Metric | Before | After |
|--------|--------|-------|
| Data Reconciliation Time | 40 hrs/month | Automated |
| Customer View | 4 separate systems | Single 360° view |
| Churn Prediction | None | ML-based risk scoring |
| Retention Actions | Reactive | Proactive with recommendations |
| Data Freshness | Weekly manual updates | Daily automated refresh |

---

## Architecture

### High-Level Overview

![Architecture Overview](docs/diagrams/00_architecture_overview.png)

### Part 1: Data Sources & Ingestion

Four disparate source systems feed into the Bronze layer through Python-based ingestion.

![Data Sources & Ingestion](docs/diagrams/01_data_sources_ingestion.png)

### Part 2: Transformation & Modeling

dbt transforms raw data through Bronze → Silver → Gold using the Medallion architecture.

![Transformation & Modeling](docs/diagrams/02_transformation_modeling.png)

### Part 3: ML Pipeline & Consumption

Gold layer features feed the ML model, producing churn predictions for dashboards and retention teams.

![ML Pipeline & Consumption](docs/diagrams/03_ml_consumption.png)

### Customer Entity Resolution

The core challenge: unifying customers across 4 different ID systems using email-based matching.

![Entity Resolution](docs/diagrams/04_entity_resolution.png)

### CI/CD Pipeline

GitHub Actions automates testing on PRs and deployment on merge to main.

![CI/CD Pipeline](docs/diagrams/05_cicd_pipeline.png)

---

### Data Flow Summary

| Step | Action | Technology |
|------|--------|------------|
| 1. INGEST | Raw data from 4 sources → Bronze | Python + Databricks |
| 2. STAGE | Bronze → Staging views | dbt (column rename, type cast) |
| 3. TRANSFORM | Staging → Silver tables | dbt (clean, unify, business logic) |
| 4. AGGREGATE | Silver → Gold tables | dbt (features, metrics) |
| 5. PREDICT | Gold features → ML model | sklearn + Ensemble |
| 6. ACT | Predictions → Dashboards | Power BI + Databricks SQL |

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Cloud Platform** | Databricks (Free Edition) | Unified analytics platform |
| **Data Catalog** | Unity Catalog | Data governance & discovery |
| **Transformation** | dbt-databricks | SQL-based transformations |
| **Orchestration** | Databricks Workflows | Job scheduling |
| **ML Platform** | MLflow | Model training & registry |
| **Version Control** | GitHub | Code repository |
| **CI/CD** | GitHub Actions | Automated testing & deployment |
| **Dashboards** | Power BI / Databricks SQL | Visualization |
| **Language** | SQL, Python | Transformations & ML |

### Source Systems

| System | Type | Data | Authentication |
|--------|------|------|----------------|
| **ERPNext** | Open-source ERP | Customers, Transactions, Products | API Token |
| **Salesforce** | CRM | Contacts, Cases, Tasks | OAuth 2.0 |
| **Supabase** | PostgreSQL | App Sessions, Events | Service Key |
| **Google Sheets** | Spreadsheet | Branch Data, Manual Notes | Service Account |

---

## Data Sources

### 1. ERPNext (Core Banking System)

The primary system of record for customer master data and financial transactions.

| Table | Description | Key Columns | Record Count |
|-------|-------------|-------------|--------------|
| `erp_customers` | Customer master data | customer_id, customer_name, email, territory, customer_group | ~100 |
| `erp_sales_invoices` | Financial transactions | transaction_id, customer_id, amount, status, date | ~500 |
| `erp_territories` | Geographic regions | territory_name, parent_territory | ~10 |
| `erp_items` | Product catalog | item_code, item_name, item_group | ~20 |

**Sample erp_customers schema:**
```
customer_id       | STRING   | Primary key (e.g., CUST-00001)
customer_name     | STRING   | Full name
email            | STRING   | Email address (used for unification)
mobile_no        | STRING   | Phone number
customer_group   | STRING   | Segment (Retail, Corporate, etc.)
customer_type    | STRING   | Individual or Company
territory        | STRING   | Geographic region
gender           | STRING   | Male/Female
creation         | TIMESTAMP| Account creation date
```

### 2. Salesforce (CRM System)

Customer relationship management including support cases and interactions.

| Table | Description | Key Columns | Record Count |
|-------|-------------|-------------|--------------|
| `sf_contacts` | CRM contact records | contact_id, email, name, lead_source | ~100 |
| `sf_cases` | Support tickets/complaints | case_id, contact_id, status, priority | ~200 |
| `sf_tasks` | Activities and follow-ups | task_id, contact_id, subject, status | ~150 |

**Sample sf_cases schema:**
```
sf_case_id       | STRING   | Primary key
sf_contact_id    | STRING   | Foreign key to contacts
subject          | STRING   | Case title
description      | STRING   | Full description
type             | STRING   | Complaint, Inquiry, etc.
status           | STRING   | Open, Closed, In Progress
priority         | STRING   | High, Medium, Low
origin           | STRING   | Phone, Email, Web
closed_date      | TIMESTAMP| Resolution date
created_date     | TIMESTAMP| Case creation date
```

### 3. Supabase (Digital Channels)

Mobile and web application usage data.

| Table | Description | Key Columns | Record Count |
|-------|-------------|-------------|--------------|
| `sb_app_sessions` | App login sessions | session_id, customer_email, start_at, end_at | ~1000 |
| `sb_app_events` | In-app user actions | event_id, session_id, event_type, timestamp | ~5000 |

**Sample sb_app_sessions schema:**
```
session_id       | STRING   | Primary key (UUID)
customer_email   | STRING   | User email (used for unification)
session_start_at | TIMESTAMP| Session start time
session_end_at   | TIMESTAMP| Session end time
device_type      | STRING   | Mobile, Desktop, Tablet
app_version      | STRING   | Application version
```

### 4. Google Sheets (Legacy Data)

Branch information and relationship manager notes.

| Table | Description | Key Columns | Record Count |
|-------|-------------|-------------|--------------|
| `gs_branches` | Bank branch data | branch_id, branch_name, region, manager | ~15 |
| `gs_customer_notes` | RM manual notes | customer_email, note_text, note_date | ~50 |

---

## Data Model

### Medallion Architecture

This project implements the **Medallion Architecture** (Bronze → Silver → Gold), a data design pattern that incrementally improves data quality as it flows through each layer.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MEDALLION ARCHITECTURE                               │
│                                                                              │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐           │
│   │   BRONZE    │         │   SILVER    │         │    GOLD     │           │
│   │             │         │             │         │             │           │
│   │  Raw Data   │ ──────▶ │  Cleaned &  │ ──────▶ │  Business   │           │
│   │  As-Is      │         │  Conformed  │         │  Ready      │           │
│   │             │         │             │         │             │           │
│   │ • Exact copy│         │ • Type cast │         │ • Aggregated│           │
│   │ • All cols  │         │ • Deduplicated│       │ • ML features│          │
│   │ • Append-   │         │ • Unified   │         │ • KPIs      │           │
│   │   only      │         │ • Validated │         │ • Dashboards│           │
│   └─────────────┘         └─────────────┘         └─────────────┘           │
│                                                                              │
│   Schema: bronze          Schema: silver          Schema: gold              │
│   Materialization: table  Materialization: table  Materialization: table    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Bronze Layer (Raw)

Raw data ingested from source systems without transformation. Serves as the immutable source of truth.

| Table | Source | Description |
|-------|--------|-------------|
| `bronze.erp_customers` | ERPNext API | Raw customer master data |
| `bronze.erp_sales_invoices` | ERPNext API | Raw transaction records |
| `bronze.erp_territories` | ERPNext API | Geographic hierarchy |
| `bronze.erp_items` | ERPNext API | Product catalog |
| `bronze.sf_contacts` | Salesforce API | CRM contact records |
| `bronze.sf_cases` | Salesforce API | Support tickets |
| `bronze.sf_tasks` | Salesforce API | Activities |
| `bronze.sb_app_sessions` | Supabase | App session logs |
| `bronze.sb_app_events` | Supabase | User event stream |
| `bronze.gs_branches` | Google Sheets | Branch master data |
| `bronze.gs_customer_notes` | Google Sheets | RM notes |

**Metadata columns added during ingestion:**
- `_source_system`: Origin system identifier
- `_ingested_at`: Timestamp of ingestion

### Silver Layer (Cleaned)

Cleaned, validated, and unified data. This layer implements:
- Column standardization (consistent naming conventions)
- Data type casting
- Customer unification across systems
- Business logic and derived fields

#### Staging Models (Views)

Thin wrappers on Bronze tables for column renaming and type casting.

| Model | Source | Key Transformations |
|-------|--------|---------------------|
| `stg_erp_customers` | bronze.erp_customers | Rename columns, cast types |
| `stg_erp_transactions` | bronze.erp_sales_invoices | Parse dates, cast amounts |
| `stg_erp_territories` | bronze.erp_territories | Standardize names |
| `stg_erp_items` | bronze.erp_items | Clean product data |
| `stg_sf_contacts` | bronze.sf_contacts | Combine name fields |
| `stg_sf_cases` | bronze.sf_cases | Derive is_closed flag |
| `stg_sf_tasks` | bronze.sf_tasks | Standardize status |
| `stg_app_sessions` | bronze.sb_app_sessions | Parse timestamps |
| `stg_app_events` | bronze.sb_app_events | Categorize events |
| `stg_branches` | bronze.gs_branches | Clean branch data |
| `stg_customer_notes` | bronze.gs_customer_notes | Parse note dates |

#### Dimension Tables

| Model | Description | Grain | Key Columns |
|-------|-------------|-------|-------------|
| `dim_customer_unified` | **Master customer dimension** - unifies customers from ERPNext, Salesforce, and Supabase using email matching | One row per unique customer | unified_customer_id, erp_customer_id, sf_contact_id, email, customer_name |
| `dim_geography` | Geographic hierarchy | One row per territory | territory_id, territory_name, parent_territory |
| `dim_branch` | Bank branch locations | One row per branch | branch_id, branch_name, region, manager |
| `dim_product` | Product/service catalog | One row per product | product_id, product_name, product_group |

#### Fact Tables

| Model | Description | Grain | Key Metrics |
|-------|-------------|-------|-------------|
| `fct_transactions` | Financial transactions linked to unified customers | One row per transaction | transaction_amount, transaction_status |
| `fct_support_cases` | Support tickets with resolution metrics | One row per case | priority_score, resolution_days, is_closed |
| `fct_digital_engagement` | Daily app usage aggregated per customer | One row per customer per day | session_count, event_count, engagement_score |

### Gold Layer (Analytics)

Business-ready aggregations and ML features. Optimized for dashboard performance and model training.

| Model | Description | Use Case |
|-------|-------------|----------|
| `customer_360` | Complete customer view with all behavioral metrics | Customer service, analytics |
| `customer_features` | ML feature table with engineered features | Churn model training/scoring |
| `agg_churn_by_segment` | Pre-aggregated metrics by dimension | Dashboard performance |
| `high_risk_customers` | Prioritized retention action list | Retention team workflow |

---

## Project Structure

```
banking-churn-databricks/
│
├── .github/
│   └── workflows/
│       ├── dbt-ci.yml              # PR testing pipeline
│       └── dbt-deploy.yml          # Main branch deployment
│
├── data/
│   └── raw/                        # Source data files
│       ├── erp_customers.csv
│       ├── erp_customers.json
│       ├── erp_sales_invoices.csv
│       ├── erp_sales_invoices.json
│       ├── erp_territories.csv
│       ├── erp_territories.json
│       ├── erp_items.csv
│       ├── erp_items.json
│       ├── sf_contacts.csv
│       ├── sf_contacts.json
│       ├── sf_cases.csv
│       ├── sf_cases.json
│       ├── sf_tasks.csv
│       ├── sf_tasks.json
│       ├── sb_sessions.csv
│       ├── sb_sessions.json
│       ├── sb_events.csv
│       ├── sb_events.json
│       ├── gs_branches.csv
│       ├── gs_branches.json
│       ├── gs_customer_notes.csv
│       └── gs_customer_notes.json
│
├── dbt/
│   ├── dbt_project.yml             # dbt project configuration
│   ├── profiles.yml                # Connection profiles
│   ├── packages.yml                # dbt package dependencies
│   │
│   ├── models/
│   │   ├── staging/                # Bronze → Staging transformations
│   │   │   ├── _sources.yml        # Source definitions
│   │   │   ├── _staging.yml        # Model documentation
│   │   │   ├── stg_erp_customers.sql
│   │   │   ├── stg_erp_transactions.sql
│   │   │   ├── stg_erp_territories.sql
│   │   │   ├── stg_erp_items.sql
│   │   │   ├── stg_sf_contacts.sql
│   │   │   ├── stg_sf_cases.sql
│   │   │   ├── stg_sf_tasks.sql
│   │   │   ├── stg_app_sessions.sql
│   │   │   ├── stg_app_events.sql
│   │   │   ├── stg_branches.sql
│   │   │   └── stg_customer_notes.sql
│   │   │
│   │   ├── silver/                 # Cleaned & unified data
│   │   │   ├── _silver.yml         # Model documentation
│   │   │   ├── dim_customer_unified.sql
│   │   │   ├── dim_geography.sql
│   │   │   ├── dim_branch.sql
│   │   │   ├── dim_product.sql
│   │   │   ├── fct_transactions.sql
│   │   │   ├── fct_support_cases.sql
│   │   │   └── fct_digital_engagement.sql
│   │   │
│   │   └── gold/                   # Analytics & ML features
│   │       ├── _gold.yml           # Model documentation
│   │       ├── customer_360.sql
│   │       ├── customer_features.sql
│   │       ├── agg_churn_by_segment.sql
│   │       └── high_risk_customers.sql
│   │
│   ├── macros/
│   │   └── generate_schema_name.sql  # Custom schema naming
│   │
│   ├── seeds/                      # Static reference data
│   │
│   ├── tests/                      # Custom data tests
│   │
│   └── target/                     # Compiled SQL (git-ignored)
│
├── notebooks/
│   ├── 01_bronze_ingestion.py      # Data ingestion notebook
│   ├── 02_exploration.py           # Data exploration
│   └── ml/
│       ├── 01_train_churn_model.py # Model training
│       └── 02_score_customers.py   # Batch scoring
│
├── docs/
│   ├── BRONZE_INGESTION_SETUP.md   # Ingestion documentation
│   └── lesson.md                   # Learning notes
│
├── .gitignore
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

---

## Setup Instructions

### Prerequisites

- Python 3.9+
- Git
- Databricks account (Free Edition works)
- GitHub account

### 1. Clone the Repository

```bash
git clone https://github.com/sulaiman013/banking-churn-databricks.git
cd banking-churn-databricks
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Databricks Connection

Create a `.env` file (never commit this!):

```env
DATABRICKS_HOST=your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-personal-access-token
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your-warehouse-id
```

Update `dbt/profiles.yml`:

```yaml
bank_churn:
  outputs:
    dev:
      type: databricks
      host: "{{ env_var('DATABRICKS_HOST') }}"
      http_path: "{{ env_var('DATABRICKS_HTTP_PATH') }}"
      token: "{{ env_var('DATABRICKS_TOKEN') }}"
      catalog: bank_proj
      schema: default
      threads: 4
  target: dev
```

### 4. Set Up Databricks

1. **Create Unity Catalog:**
   ```sql
   CREATE CATALOG IF NOT EXISTS bank_proj;
   CREATE SCHEMA IF NOT EXISTS bank_proj.bronze;
   CREATE SCHEMA IF NOT EXISTS bank_proj.silver;
   CREATE SCHEMA IF NOT EXISTS bank_proj.gold;
   CREATE SCHEMA IF NOT EXISTS bank_proj.ml;
   ```

2. **Run Bronze Ingestion:**
   - Open `notebooks/01_bronze_ingestion.py` in Databricks
   - Execute to load raw data into Bronze tables

### 5. Run dbt Models

```bash
cd dbt

# Test connection
dbt debug

# Install packages
dbt deps

# Run all models
dbt run

# Run tests
dbt test

# Generate documentation
dbt docs generate
dbt docs serve
```

### 6. Configure GitHub Actions (CI/CD)

Add these secrets to your GitHub repository:
- Settings → Secrets and variables → Actions → New repository secret

| Secret Name | Value |
|-------------|-------|
| `DATABRICKS_HOST` | Your Databricks workspace URL |
| `DATABRICKS_TOKEN` | Your Personal Access Token |

---

## dbt Models Documentation

### Staging Models

#### `stg_erp_customers`
```sql
-- Standardizes ERPNext customer data
-- Source: bronze.erp_customers

SELECT
    customer_id,
    customer_name,
    email,
    mobile_no AS mobile_phone,
    customer_group,
    customer_type,
    territory,
    gender,
    creation AS created_at
FROM {{ source('bronze', 'erp_customers') }}
```

#### `stg_sf_cases`
```sql
-- Standardizes Salesforce support cases
-- Derives is_closed flag from closed_date

SELECT
    sf_case_id AS case_id,
    sf_contact_id AS contact_id,
    subject AS case_subject,
    type AS case_type,
    status AS case_status,
    priority AS case_priority,
    CASE WHEN closed_date IS NOT NULL THEN true ELSE false END AS is_closed,
    closed_date AS closed_at,
    created_date AS created_at
FROM {{ source('bronze', 'sf_cases') }}
```

### Silver Models

#### `dim_customer_unified`
The most critical model - unifies customer identity across all systems.

**Unification Logic:**
```sql
-- Email is the primary matching key
-- ERPNext is the master system

SELECT
    md5(coalesce(lower(trim(erp.email)), erp.erp_customer_id)) AS unified_customer_id,

    -- Source system IDs
    erp.erp_customer_id,
    sf.sf_contact_id,
    app.app_customer_email,

    -- Unified attributes (prefer ERPNext, fallback to Salesforce)
    coalesce(erp.customer_name, sf.sf_name) AS customer_name,
    coalesce(erp.email, sf.sf_email) AS email,
    ...

FROM erp_customers erp
LEFT JOIN sf_contacts sf
    ON lower(trim(erp.email)) = lower(trim(sf.sf_email))
LEFT JOIN app_customers app
    ON lower(trim(erp.email)) = lower(trim(app.app_customer_email))
```

#### `fct_digital_engagement`
Daily aggregation of app usage per customer.

**Key Calculations:**
- `session_count`: Number of app sessions per day
- `session_duration_seconds`: Total time spent in app
- `event_count`: Number of in-app actions
- `engagement_score`: Weighted composite metric

```sql
engagement_score =
    (session_count * 10) +
    (total_session_duration_seconds / 60) +
    (event_count * 2)
```

### Gold Models

#### `customer_360`
Complete customer view with 50+ attributes spanning:
- Demographics
- Transaction history (lifetime + last 30 days)
- Support case metrics
- Digital engagement
- Tenure and segmentation
- Days since last activity

#### `customer_features`
ML-ready feature table with engineered features:

| Feature Category | Features |
|------------------|----------|
| **Demographic** | gender_encoded, customer_type_encoded, region_encoded |
| **Tenure** | tenure_days, tenure_months, tenure_risk_score |
| **Transaction** | total_transactions, transaction_frequency, transaction_trend_ratio |
| **Support** | total_cases, complaint_rate, has_open_complaint, avg_resolution_days |
| **Digital** | total_sessions, app_usage_frequency, is_digitally_active |
| **Composite** | churn_risk_score (0-100), engagement_health_score (0-100) |

**Churn Risk Score Calculation:**
```sql
churn_risk_score =
    -- Tenure risk (new customers = higher risk)
    CASE WHEN tenure_days <= 90 THEN 25
         WHEN tenure_days <= 180 THEN 15
         ELSE 5 END +

    -- Inactivity risk
    CASE WHEN days_since_last_transaction > 90 THEN 25
         WHEN days_since_last_transaction > 60 THEN 15
         ELSE 0 END +

    -- Complaint risk
    CASE WHEN open_support_cases > 0 THEN 25
         WHEN high_priority_cases > 0 THEN 15
         ELSE 0 END +

    -- Digital disengagement
    CASE WHEN stopped_using_app THEN 20
         ELSE 0 END
```

#### `high_risk_customers`
Actionable retention list with:
- Risk category (Critical, High, Medium)
- Risk factors (human-readable explanations)
- Recommended action
- Priority score (combines risk + customer value)

---

## CI/CD Pipeline

### Pull Request Pipeline (`.github/workflows/dbt-ci.yml`)

Triggered on every PR to `main`:

```yaml
name: dbt CI

on:
  pull_request:
    branches: [main, master]

jobs:
  dbt-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install dbt-core dbt-databricks

      - name: dbt deps
        run: dbt deps
        working-directory: ./dbt

      - name: dbt build
        run: dbt build --target ci
        working-directory: ./dbt
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
```

### Deployment Pipeline (`.github/workflows/dbt-deploy.yml`)

Triggered on merge to `main`:

```yaml
name: dbt Deploy

on:
  push:
    branches: [main, master]

jobs:
  dbt-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5

      - name: Install dependencies
        run: pip install dbt-core dbt-databricks

      - name: dbt build --target prod
        run: dbt build --target prod
        working-directory: ./dbt
        env:
          DATABRICKS_HOST: ${{ secrets.DATABRICKS_HOST }}
          DATABRICKS_TOKEN: ${{ secrets.DATABRICKS_TOKEN }}
```

---

## ML Churn Prediction Model

### Overview

The churn prediction model uses the `customer_features` table to predict which customers are likely to leave.

### Features Used

| Feature | Type | Importance |
|---------|------|------------|
| `days_since_last_transaction` | Numeric | High |
| `has_open_complaint` | Binary | High |
| `tenure_days` | Numeric | Medium |
| `complaint_rate` | Numeric | Medium |
| `transaction_frequency` | Numeric | Medium |
| `is_digitally_active` | Binary | Medium |
| `churn_risk_score` | Numeric | Reference |

### Model Training (`notebooks/ml/01_train_churn_model.py`)

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load features from Gold layer
df = spark.table("bank_proj.gold.customer_features").toPandas()

# Prepare features and target
X = df[feature_columns]
y = df['churn_label']  # Historical churn indicator

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Log to MLflow
with mlflow.start_run():
    mlflow.log_params({"n_estimators": 100})
    mlflow.log_metrics({"accuracy": accuracy, "auc": auc})
    mlflow.sklearn.log_model(model, "churn_model")
```

### Batch Scoring (`notebooks/ml/02_score_customers.py`)

```python
# Load registered model
model = mlflow.sklearn.load_model("models:/churn_model/Production")

# Score all customers
predictions = model.predict_proba(X)

# Write to ML schema
predictions_df.write.mode("overwrite").saveAsTable("bank_proj.ml.churn_predictions")
```

---

## Dashboards

### Power BI Dashboard Components

1. **Executive Summary**
   - Total customers
   - High-risk customer count
   - Average churn risk score
   - Month-over-month trends

2. **Risk Distribution**
   - Histogram of churn risk scores
   - Risk segment breakdown (pie chart)

3. **Segment Analysis**
   - Risk by territory (map)
   - Risk by customer group (bar chart)
   - Risk by tenure segment

4. **Retention Action List**
   - High-risk customers table
   - Sortable by priority score
   - Filterable by territory/segment

### Databricks SQL Dashboard

```sql
-- High Risk Customer Summary
SELECT
    risk_category,
    COUNT(*) as customer_count,
    SUM(lifetime_value) as total_value_at_risk,
    AVG(churn_risk_score) as avg_risk_score
FROM gold.high_risk_customers
GROUP BY risk_category
ORDER BY avg_risk_score DESC;
```

---

## Key Features

### 1. Customer Unification
- Matches customers across 4 systems using email
- Creates single `unified_customer_id` for 360° view
- Handles missing emails with fallback logic

### 2. Behavioral Feature Engineering
- 30+ engineered features from raw data
- Composite risk scores combining multiple signals
- Trend detection (declining engagement, inactivity)

### 3. Actionable Outputs
- Not just predictions, but recommended actions
- Priority scoring combines risk + customer value
- Human-readable risk factor explanations

### 4. Production-Ready Architecture
- Medallion architecture for data quality
- CI/CD for automated testing
- Unity Catalog for governance

---

## Learnings & Best Practices

### Data Engineering

1. **Always read before writing** - Understand source schemas before building transformations
2. **Email normalization** - Use `lower(trim(email))` for matching
3. **Defensive coding** - Use `coalesce()` for null handling
4. **Incremental where possible** - Consider incremental models for large tables

### dbt Best Practices

1. **Staging layer** - Always create staging models before transformations
2. **Ref over source** - Use `{{ ref() }}` for model dependencies
3. **Documentation** - Maintain YAML files for all models
4. **Testing** - Add unique and not_null tests for keys

### Git & CI/CD

1. **Never commit secrets** - Use `.gitignore` and GitHub secrets
2. **Branch protection** - Require PR reviews before merge
3. **Automated testing** - Let CI catch errors before production

---

## Future Enhancements

| Enhancement | Description | Priority |
|-------------|-------------|----------|
| **Real-time scoring** | Stream processing for instant churn detection | High |
| **A/B testing** | Test retention interventions | High |
| **Feature store** | Centralized feature management | Medium |
| **Model monitoring** | Track prediction drift | Medium |
| **Additional sources** | Social media sentiment, NPS surveys | Low |
| **AutoML** | Automated model retraining | Low |

---

## Author

**Sulaiman Ahmed**

- GitHub: [@sulaiman013](https://github.com/sulaiman013)
- LinkedIn: [Sulaiman Ahmed](https://linkedin.com/in/sulaimanahmed013)

---

## License

This project is for educational and portfolio demonstration purposes.

---

## Acknowledgments

- Databricks for the free tier platform
- dbt Labs for the transformation framework
- The data engineering community for best practices

---

*Built with Databricks, dbt, and determination.*
