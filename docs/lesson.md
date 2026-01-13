# End-to-End Banking Analytics: From Fragmented Systems to Unified Intelligence

## Project Overview

This project demonstrates how a mid-sized retail bank transforms fragmented, siloed data across multiple systems into a unified analytics platform that predicts customer churn and enables proactive retention.

---

# PART 1: THE BUSINESS PROBLEM

## The Bank: "Apex National Bank" (Fictional)

**Profile:**
- Mid-sized retail bank with 500,000 customers
- 50 branches across 3 regions
- Products: Savings, Current, Fixed Deposits, Credit Cards, Personal Loans
- 200 employees in operations, 15 in IT, 5 in analytics

---

## The POC Mandate

### The Proposal

You (the Analytics Engineer) presented a comprehensive Databricks-based unified data platform to the executive team. The architecture promised:
- Automated data integration from all 4 source systems
- Real-time customer 360 view
- ML-powered churn prediction
- 40 hours/month saved in manual reporting

### Stakeholder Response

> *"This looks promising, but we're not ready to commit $500K to a full implementation. Show us it works first."*
> — Chief Data Officer

### The POC Agreement

| Full Production | POC Scope |
|-----------------|-----------|
| 500,000 customers | **500 customers** (representative sample) |
| 50 branches | **10 branches** |
| 3 years of transactions | **3 years of transactions** (same timeframe) |
| All 4 source systems | **All 4 source systems** (full integration) |
| Production infrastructure | **Free/trial tiers** |

**Success Criteria for POC:**
1. Successfully integrate data from all 4 source systems
2. Demonstrate unified customer view (customer_360)
3. Build working churn prediction model (AUC > 0.75)
4. Automate with CI/CD pipeline
5. Deliver executive dashboard

**If POC succeeds:** Green light for production implementation with full data volume.

### Why This Framing Matters (For Your Portfolio)

This POC approach demonstrates you understand:
- **Enterprise decision-making**: Stakeholders need proof before big investments
- **Risk management**: Start small, prove value, then scale
- **Business acumen**: Technical skills + business awareness = senior-level thinking
- **Scalable architecture**: Design for production, prove with POC

---

## Current State: The Chaos

### System Landscape

Apex National Bank, like most banks, evolved organically over 15 years. Different departments bought different systems:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CURRENT STATE: SILOED SYSTEMS                           │
│                                                                             │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐   ┌───────────┐ │
│  │   ERPNext     │   │  Salesforce   │   │   Supabase    │   │  Google   │ │
│  │ (Core Banking)│   │    (CRM)      │   │  (Mobile App) │   │  Sheets   │ │
│  │               │   │               │   │               │   │           │ │
│  │ Owned by:     │   │ Owned by:     │   │ Owned by:     │   │ Owned by: │ │
│  │ Operations    │   │ Sales & Svc   │   │ Digital Team  │   │ Branches  │ │
│  │               │   │               │   │               │   │           │ │
│  │ Customer ID:  │   │ Customer ID:  │   │ Customer ID:  │   │ Customer: │ │
│  │ CUST_001234   │   │ SF-00001234   │   │ user_uuid     │   │ "John S." │ │
│  └───────────────┘   └───────────────┘   └───────────────┘   └───────────┘ │
│         │                   │                   │                   │      │
│         │                   │                   │                   │      │
│         ▼                   ▼                   ▼                   ▼      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    MANUAL EXCEL REPORTS                              │   │
│  │                    (Created monthly by analysts)                     │   │
│  │                    (Always outdated, often wrong)                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Four Source Systems

### 1. ERPNext (Core Banking System)

**What it stores:**
- Customer master data (name, ID, KYC info, risk rating)
- Account information (account number, type, status, balance)
- Transaction journal (every debit/credit, timestamps, references)
- Product catalog (account types, interest rates, fees)

**Who owns it:** Operations Team

**Problems:**
- Data entry errors (typos in names, wrong dates)
- Inconsistent formats (dates as DD/MM/YYYY and MM-DD-YYYY mixed)
- Duplicate customers (same person opened accounts at different branches)
- No integration with other systems

**Sample Data Issues:**
```
| customer_id | name           | phone        | created_date |
|-------------|----------------|--------------|--------------|
| CUST_001234 | John Smith     | 1234567890   | 15/03/2023   |
| CUST_001235 | john smith     | 123-456-7890 | 2023-03-15   |  ← Same person!
| CUST_001236 | JOHN SMITH JR. | +1234567890  | 03-15-2023   |  ← Maybe same?
```

---

### 2. Salesforce CRM

**What it stores:**
- Customer interactions (calls, emails, branch visits)
- Complaints and cases (issue type, status, resolution)
- Sales opportunities (cross-sell attempts, campaign responses)
- Customer satisfaction scores

**Who owns it:** Sales & Service Team

**Problems:**
- Different customer ID format than ERPNext
- Service reps don't always link cases to correct customer
- Free-text complaint descriptions (unstructured)
- Duplicate contacts (same customer, different records)

**Sample Data Issues:**
```
| sf_contact_id | email              | related_account | complaint        |
|---------------|--------------------|-----------------| -----------------|
| SF-00001234   | john@email.com     | NULL            | "ATM ate my card"|
| SF-00001235   | j.smith@email.com  | CUST_001234     | "Card stuck ATM" |  ← Same issue!
| SF-00001236   | john@email.com     | NULL            | "fee too high"   |
```

---

### 3. Supabase (Mobile Banking App)

**What it stores:**
- App login events (timestamp, device, location)
- Session data (pages viewed, time spent)
- Feature usage (transfers, bill pay, statements)
- Push notification responses

**Who owns it:** Digital Team

**Problems:**
- Uses UUID for user identification
- No direct link to Core Banking customer ID
- Event data is high volume, hard to aggregate
- Privacy concerns (location data)

**Sample Data Issues:**
```
| user_uuid                            | event_type  | timestamp           |
|--------------------------------------|-------------|---------------------|
| a1b2c3d4-e5f6-7890-abcd-ef1234567890 | login       | 2024-01-15 09:23:45 |
| a1b2c3d4-e5f6-7890-abcd-ef1234567890 | view_balance| 2024-01-15 09:23:52 |
| a1b2c3d4-e5f6-7890-abcd-ef1234567890 | logout      | 2024-01-15 09:25:01 |
```
*But which customer is this UUID? Nobody knows without manual lookup.*

---

### 4. Google Sheets (Branch Operations)

**What it stores:**
- Branch master data (location, manager, region)
- Manual reports (daily cash positions, foot traffic)
- Special customer notes ("VIP", "High Risk", "Relationship issues")
- Campaign tracking (local promotions)

**Who owns it:** Branch Managers

**Problems:**
- No validation (anyone can type anything)
- Multiple versions floating around
- Often outdated (updated monthly if at all)
- No audit trail

**Sample Data Issues:**
```
| branch_name      | manager    | region  | notes                    |
|------------------|------------|---------|--------------------------|
| Downtown Main    | Sarah J.   | North   | "Renovating Q2"          |
| downtown main    | S. Johnson | NORTH   | "Renovation done"        |  ← Duplicate?
| DT Main Branch   | Sarah      | N       | ""                       |  ← Same branch?
```

---

## The Pain Points

### 1. Manual Work: 40 Hours/Month Wasted

**Current process for monthly churn report:**

```
Week 1:
├── Analyst exports customer data from ERPNext (2 hours)
├── Analyst exports complaint data from Salesforce (1 hour)
├── Analyst requests app data from Digital team (waits 3 days)
├── Analyst collects branch sheets via email (waits 2 days)
└── Data sits in 4 different Excel files

Week 2:
├── Analyst manually matches customers across systems (8 hours)
├── Analyst finds duplicates, guesses which is correct (4 hours)
├── Analyst creates VLOOKUP formulas (2 hours)
├── VLOOKUP breaks because IDs don't match (debug: 3 hours)
└── Analyst gives up, does approximate matching

Week 3:
├── Analyst calculates churn metrics manually (4 hours)
├── Manager asks "why did customer X leave?" (no data)
├── Analyst manually looks up customer in 4 systems (1 hour per customer)
└── Creates PowerPoint with outdated numbers

Week 4:
├── Report is presented (already 3 weeks old)
├── Decisions made on stale data
├── Rinse and repeat next month
└── Total time: ~40 hours/month
```

---

### 2. Inconsistent Customer Identity

**The same customer exists differently across systems:**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ONE CUSTOMER, FOUR IDENTITIES                        │
│                                                                          │
│  ERPNext:        CUST_001234 → "John Smith" → Phone: 1234567890         │
│  Salesforce:     SF-00001234 → "J. Smith" → Email: john@email.com       │
│  Supabase:       a1b2c3d4... → NULL → Device: iPhone 14                 │
│  Sheets:         Row 47 → "John S. (VIP)" → Branch: Downtown            │
│                                                                          │
│  Question: How many complaints has this customer made?                   │
│  Answer: "We don't know. We can't link the systems."                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### 3. No Early Warning System

**Churn indicators exist but are invisible:**

| System | Has This Signal | Can We See It? |
|--------|-----------------|----------------|
| ERPNext | Balance declining over 3 months | No (would need SQL query) |
| Salesforce | 3 complaints in 30 days | No (buried in case list) |
| Supabase | Stopped logging into app | No (just raw event logs) |
| Sheets | Branch manager noted "unhappy" | No (lost in spreadsheet) |

**Result:** Customers leave without warning. By the time anyone notices, they're already gone.

---

### 4. Reactive, Not Proactive

**Current state:**
```
Customer closes account → Bank notices → "Why did they leave?" → No data → Shrug
```

**Desired state:**
```
ML model predicts churn risk → Alert to retention team → Proactive call → Customer stays
```

---

## The Cost of Doing Nothing

| Problem | Business Impact |
|---------|-----------------|
| 40 hours/month manual reporting | $50,000/year in analyst time |
| No churn prediction | 5% churn rate = 25,000 customers/year lost |
| Customer acquisition cost | $200/customer × 25,000 = $5M to replace |
| Missed cross-sell opportunities | Unknown (can't identify them) |
| Compliance risk | Manual processes = audit failures |

**Total estimated annual cost: $5-6 million**

---

# PART 2: THE SOLUTION

## Unified Data Platform on Databricks

### Target Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TARGET STATE: UNIFIED PLATFORM                       │
│                                                                              │
│  Source Systems                      Databricks Unity Catalog                │
│  ┌─────────────┐                    ┌────────────────────────────────────┐  │
│  │  ERPNext    │──── API ──────────►│  bronze.erp_customers              │  │
│  │             │                    │  bronze.erp_accounts               │  │
│  │             │                    │  bronze.erp_transactions           │  │
│  └─────────────┘                    │                                    │  │
│  ┌─────────────┐                    │  bronze.crm_contacts               │  │
│  │ Salesforce  │──── API ──────────►│  bronze.crm_cases                  │  │
│  │             │                    │  bronze.crm_interactions           │  │
│  └─────────────┘                    │                                    │  │
│  ┌─────────────┐                    │  bronze.digital_sessions           │  │
│  │  Supabase   │──── API ──────────►│  bronze.digital_events             │  │
│  │             │                    │                                    │  │
│  └─────────────┘                    │  bronze.legacy_branches            │  │
│  ┌─────────────┐                    │  bronze.legacy_notes               │  │
│  │   Sheets    │──── CSV ──────────►│                                    │  │
│  └─────────────┘                    └──────────────┬─────────────────────┘  │
│                                                    │                        │
│                                                    │ dbt transformations    │
│                                                    ▼                        │
│                                     ┌────────────────────────────────────┐  │
│                                     │  silver.dim_customer (UNIFIED!)    │  │
│                                     │  silver.dim_account                │  │
│                                     │  silver.dim_branch                 │  │
│                                     │  silver.dim_product                │  │
│                                     │  silver.fct_transactions           │  │
│                                     │  silver.fct_complaints             │  │
│                                     │  silver.fct_app_sessions           │  │
│                                     └──────────────┬─────────────────────┘  │
│                                                    │                        │
│                                                    │ dbt feature engineering│
│                                                    ▼                        │
│                                     ┌────────────────────────────────────┐  │
│                                     │  gold.customer_360                 │  │
│                                     │  gold.churn_features               │  │
│                                     │  gold.churn_predictions            │  │
│                                     │  gold.dashboard_daily_metrics      │  │
│                                     └──────────────┬─────────────────────┘  │
│                                                    │                        │
│                                                    ▼                        │
│                                     ┌────────────────────────────────────┐  │
│                                     │        ML Model (MLflow)           │  │
│                                     │     Churn Probability Score        │  │
│                                     │         0.0 ──────── 1.0           │  │
│                                     │      (stays)      (churns)         │  │
│                                     └────────────────────────────────────┘  │
│                                                    │                        │
│                                                    ▼                        │
│                                     ┌────────────────────────────────────┐  │
│                                     │  Power BI Dashboard                │  │
│                                     │  • High-risk customer list         │  │
│                                     │  • Churn trends by segment         │  │
│                                     │  • Retention campaign tracking     │  │
│                                     └────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Medallion Architecture Explained

### Bronze Layer: "Land It As-Is"

**Purpose:** Capture raw data from source systems without modification.

**Why:**
- Preserve original data (audit trail, debugging)
- If transformation logic changes, you can reprocess from raw
- Source system schema changes don't break downstream immediately

**What happens here:**
```sql
-- bronze.erp_customers
-- Exact copy from ERPNext, warts and all
SELECT
    customer_id,          -- CUST_001234
    name,                 -- "john smith" (lowercase, unclean)
    phone,                -- "123-456-7890" (inconsistent format)
    created_date,         -- "15/03/2023" (unparsed string)
    _loaded_at            -- When we ingested this record
FROM erp_api_response
```

---

### Silver Layer: "Clean It and Unify It"

**Purpose:** Create a single, consistent version of truth.

**The Magic - Customer Unification:**

```sql
-- silver.dim_customer
-- ONE ROW PER REAL CUSTOMER, regardless of source system
WITH erp_customers AS (
    SELECT
        customer_id AS source_id,
        'erp' AS source_system,
        LOWER(TRIM(name)) AS name_clean,
        REGEXP_REPLACE(phone, '[^0-9]', '') AS phone_clean,
        email
    FROM bronze.erp_customers
),
crm_contacts AS (
    SELECT
        sf_contact_id AS source_id,
        'crm' AS source_system,
        LOWER(TRIM(name)) AS name_clean,
        REGEXP_REPLACE(phone, '[^0-9]', '') AS phone_clean,
        email
    FROM bronze.crm_contacts
),
matched AS (
    -- Match on phone OR email (fuzzy matching)
    SELECT
        COALESCE(e.source_id, c.source_id) AS source_id,
        e.customer_id AS erp_customer_id,
        c.sf_contact_id AS crm_contact_id,
        -- ... matching logic
    FROM erp_customers e
    FULL OUTER JOIN crm_contacts c
        ON e.phone_clean = c.phone_clean
        OR e.email = c.email
)
SELECT
    {{ dbt_utils.generate_surrogate_key(['erp_customer_id', 'crm_contact_id']) }} AS customer_key,
    -- UNIFIED CUSTOMER!
    ...
```

**Result:**
- ERPNext's `CUST_001234`
- Salesforce's `SF-00001234`
- Supabase's `a1b2c3d4-...`

All become **ONE customer_key: `cust_abc123`**

---

### Gold Layer: "Serve It for Business"

**Purpose:** Pre-aggregated, ML-ready, dashboard-ready data.

**Customer 360 View:**

```sql
-- gold.customer_360
-- Everything we know about a customer in ONE ROW
SELECT
    c.customer_key,
    c.full_name,
    c.email,
    c.customer_since,
    c.primary_branch,

    -- From ERPNext (Accounts)
    a.total_accounts,
    a.total_balance,
    a.avg_monthly_transactions,

    -- From Salesforce (CRM)
    s.total_complaints_lifetime,
    s.complaints_last_90_days,
    s.avg_satisfaction_score,

    -- From Supabase (Digital)
    d.app_logins_last_30_days,
    d.days_since_last_login,
    d.most_used_feature,

    -- Derived
    CASE
        WHEN a.balance_trend_90d < -0.2
         AND s.complaints_last_90_days >= 2
         AND d.days_since_last_login > 14
        THEN 'HIGH_RISK'
        ELSE 'NORMAL'
    END AS churn_risk_flag

FROM silver.dim_customer c
LEFT JOIN gold.agg_customer_accounts a ON c.customer_key = a.customer_key
LEFT JOIN gold.agg_customer_service s ON c.customer_key = s.customer_key
LEFT JOIN gold.agg_customer_digital d ON c.customer_key = d.customer_key
```

---

## ML Churn Prediction

### Features We'll Engineer

| Feature | Source | Why It Predicts Churn |
|---------|--------|----------------------|
| `balance_trend_90d` | ERPNext | Declining balance = leaving |
| `transaction_frequency_change` | ERPNext | Fewer transactions = disengaged |
| `complaints_last_90_days` | Salesforce | Complaints = unhappy |
| `days_since_last_complaint_resolved` | Salesforce | Unresolved = frustrated |
| `app_login_frequency_change` | Supabase | Stopped using app = disengaged |
| `days_since_last_login` | Supabase | No engagement = risk |
| `tenure_months` | ERPNext | New customers churn more |
| `product_count` | ERPNext | More products = stickier |
| `branch_satisfaction_score` | Sheets | Branch quality matters |

### The Model

```python
# Simplified training flow
from sklearn.ensemble import GradientBoostingClassifier
import mlflow

# Load features from gold.churn_features
features = spark.table("bank_proj.gold.churn_features").toPandas()

X = features.drop(columns=['customer_key', 'churned'])
y = features['churned']

with mlflow.start_run():
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)

    # Log to MLflow
    mlflow.sklearn.log_model(model, "churn_model")
    mlflow.log_metric("auc", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
```

### Output: Actionable Predictions

```
| customer_key | churn_probability | risk_tier | recommended_action           |
|--------------|-------------------|-----------|------------------------------|
| cust_001     | 0.87              | HIGH      | Call within 48 hours         |
| cust_002     | 0.72              | HIGH      | Offer fee waiver             |
| cust_003     | 0.45              | MEDIUM    | Send engagement email        |
| cust_004     | 0.12              | LOW       | No action needed             |
```

---

# PART 3: FOUNDATION SETUP (WHAT YOU DID)

## Step 1: Databricks Free Edition + Unity Catalog

### What You Did
- Created Databricks account
- Set up catalog: `bank_proj`
- Created schemas: `bronze`, `silver`, `gold`, `ml`

### Why It Matters

**Without Unity Catalog:**
- Tables scattered everywhere
- No access control
- No lineage tracking
- Compliance audit = nightmare

**With Unity Catalog:**
- Organized: `bank_proj.silver.dim_customer`
- Permissions: Analysts read `gold`, engineers write `bronze`
- Lineage: "This column came from ERPNext via this transformation"
- Audit: "Who accessed customer PII on January 15th?"

---

## Step 2: Git Repository

### What You Did
- Created GitHub repo with structured folders
- Separated `dbt/`, `notebooks/`, `.github/workflows/`

### Why It Matters

**Without Git:**
- Code in Databricks notebooks (no history)
- "Which version is production?"
- Someone breaks something, can't undo
- No code review

**With Git:**
- Every change tracked forever
- Branch → PR → Review → Merge
- CI/CD runs tests automatically
- Rollback in seconds if needed

---

## Step 3: dbt Project

### What You Did
- Initialized dbt project with `dbt init`
- Configured connection to Databricks

### Why It Matters

**Without dbt:**
- 500 SQL scripts, no dependency management
- "Run A before B, but only after C on Tuesdays"
- No tests, no documentation
- New engineer: 3 months to understand

**With dbt:**
- `dbt run` executes in correct order
- `dbt test` catches data quality issues
- `dbt docs generate` creates documentation
- New engineer: productive in days

---

## Step 4: Secure Configuration

### What You Did
- Used environment variables for tokens
- Never hardcoded secrets in code

### Why It Matters

**Without env vars:**
- Token in code → pushed to GitHub → exposed
- Hackers scan GitHub for secrets constantly
- Breach, data loss, regulatory fines, job loss

**With env vars:**
- Secrets never in code
- Different tokens for dev/prod
- Rotate tokens without code changes
- Sleep at night

---

# PART 4: PROGRESS & WHAT'S NEXT

## Completed Steps

```
FOUNDATION (COMPLETED):
✓ Step 1: Databricks Free Edition account
✓ Step 2: Unity Catalog setup (bank_proj catalog with bronze/silver/gold/ml schemas)
✓ Step 3: GitHub repository with folder structure
✓ Step 4: dbt project initialization
✓ Step 5: dbt-databricks connection configured and tested (dbt debug passed)

SOURCE SYSTEMS (ALL COMPLETED):
✓ Step 6: ERPNext (Core Banking) - COMPLETED
  - Instance: https://erpnext-rnm-aly.m.erpnext.com
  - Data Created:
    • 500 Customers (60% active, 25% at-risk, 15% churned)
    • 10 Bank Branches (Territories)
    • 10 Banking Products (Items)
    • 8,569 Transactions (Sales Invoices) from 2023-2026
  - Key Technical Fix: set_posting_time=1 for backdated invoices
  - Files: data/raw/erp_*.csv and erp_*.json

✓ Step 7: Salesforce CRM (Customer Service) - COMPLETED
  - Developer Edition account with Connected App
  - OAuth2 authentication (SOAP API deprecated in Winter '26)
  - Data Created:
    • 500 Contacts (linked to ERPNext via email)
    • ~1,500 Cases (support tickets - churn signal!)
    • ~1,200 Tasks (customer interactions)
  - Files: data/raw/sf_*.csv and sf_*.json

✓ Step 8: Supabase (Digital Channels) - COMPLETED
  - Mobile app session/event data with churn signals
  - Data Created:
    • 15,000-20,000 App Sessions
    • 50,000+ App Events
  - Files: data/raw/sb_*.csv and sb_*.json

✓ Step 9: Google Sheets (Legacy/Branch Data) - COMPLETED
  - Branch master data and customer notes
  - Data Created:
    • 10 Branches
    • ~1,000+ Customer Notes (churn signals!)
  - Files: data/raw/gs_*.csv and gs_*.json

DATA PIPELINE:
✓ Step 10: Build Bronze layer (Databricks notebook) - COMPLETED
□ Step 11: Build Silver layer (cleaning + unification)
□ Step 12: Build Gold layer (features + aggregations)

ML & ANALYTICS:
□ Step 13: Train churn prediction model
□ Step 14: Set up automated scoring
□ Step 15: Build Power BI dashboard

CI/CD:
□ Step 16: GitHub Actions for automated testing
□ Step 17: Automated deployment pipeline
```

---

## ERPNext Implementation Details

### Data Generation Summary

| Entity | Count | Notes |
|--------|-------|-------|
| Customers | 500 | Representative sample (500K in production) |
| Branches | 10 | Territories: Downtown, Uptown, Eastside, etc. |
| Products | 10 | Savings, Current, FD, Loans, Credit Cards |
| Transactions | 8,569 | 3 years: Jan 2023 - Jan 2026 |

### Customer Segment Distribution

| Segment | Count | Percentage | Behavior Pattern |
|---------|-------|------------|------------------|
| Active | ~300 | 60% | Regular transactions throughout |
| At-Risk | ~125 | 25% | Declining activity in recent months |
| Churned | ~75 | 15% | Stopped transacting 3-12 months ago |

### Key Technical Challenges Solved

1. **Backdated Invoices**
   - Problem: ERPNext rejected historical dates
   - Solution: `set_posting_time: 1` in invoice payload
   - Reference: https://github.com/frappe/erpnext/issues/8809

2. **Linked Record Deletion**
   - Problem: Can't delete invoices/customers due to GL Entries
   - Solution: Delete GL Entries first, then cancel, then delete

### Files Generated

```
data/raw/
├── erp_customers.csv      # 500 records
├── erp_customers.json
├── erp_items.csv          # 10 records
├── erp_items.json
├── erp_territories.csv    # 13 records (10 branches + 3 system)
├── erp_territories.json
├── erp_sales_invoices.csv # 8,569 records
└── erp_sales_invoices.json
```

### Consolidated Notebook

All ERPNext scripts consolidated into: `notebooks/exploration/01_erpnext_pipeline.ipynb`

Sections:
1. Configuration - Environment setup
2. API Client - ERPNext REST wrapper
3. Exploration - Survey existing data
4. Cleanup - Delete for fresh start (optional)
5. Ingestion - Create POC data
6. Extraction - Export to CSV/JSON

---

## Salesforce CRM Implementation Details

### Data Generation Summary

| Entity | Count | Notes |
|--------|-------|-------|
| Contacts | 500 | Linked to ERPNext customers via email |
| Cases | ~1,500 | Support tickets (key churn signal) |
| Tasks | ~1,200 | Customer interactions (calls, emails) |

### Churn Signal Implementation

The number of support cases is a key predictor of churn:

| Segment | Case Count | Rationale |
|---------|------------|-----------|
| Active | 0-2 | Happy customers, few issues |
| At-Risk | 2-5 | Growing frustration |
| Churned | 3-8 | High complaints before leaving |

### Entity Resolution

Customers are linked between ERPNext and Salesforce using **email** as the join key:

```
ERPNext Customer (email_id) <---> Salesforce Contact (Email)
```

### Key Technical Challenges Solved

1. **SOAP API Deprecated**
   - Problem: Salesforce disabled SOAP API login in Winter '26
   - Solution: Use OAuth2 Resource Owner Password flow
   - Reference: https://help.salesforce.com/s/articleView?id=005132110

2. **Connected App Configuration**
   - Required: Consumer Key, Consumer Secret
   - OAuth Policies: "All users may self-authorize", "Relax IP restrictions"

### Files Generated

```
data/raw/
├── sf_contacts.csv    # 500 records (linked to ERPNext)
├── sf_contacts.json
├── sf_cases.csv       # ~1,500 records (churn signal!)
├── sf_cases.json
├── sf_tasks.csv       # ~1,200 records
└── sf_tasks.json
```

### Consolidated Notebook

Salesforce pipeline: `notebooks/exploration/02_salesforce_pipeline.ipynb`

Sections:
1. Configuration - Auto-load from docs/.env
2. API Client - OAuth2 authentication
3. Exploration - Survey existing data
4. Ingestion - Create Contacts, Cases, Tasks via API
5. Extraction - Export to CSV/JSON

---

## Supabase Digital Channels Implementation Details

### Data Generation Summary

| Entity | Count | Notes |
|--------|-------|-------|
| App Sessions | ~15,000-20,000 | Mobile app login sessions |
| App Events | ~50,000+ | User actions (view balance, transfer, etc.) |

### Churn Signal Implementation

Digital engagement is a strong predictor of churn:

| Segment | Sessions/Customer | Events/Session | Pattern |
|---------|-------------------|----------------|---------|
| Active | 30-60 | 4-10 | Regular, engaged users |
| At-Risk | 15-30 | 2-5 | Declining engagement |
| Churned | 5-15 | 1-3 | Stopped using app before closing account |

### Event Types Tracked

| Event Type | Weight | Description |
|------------|--------|-------------|
| view_balance | 30% | Most common action |
| view_transactions | 20% | Transaction history |
| transfer_internal | 15% | Internal transfers |
| bill_payment | 10% | Bill pay feature |
| transfer_external | 5% | External transfers |
| card_controls | 5% | Card management |
| view_statements | 5% | Statement downloads |
| apply_product | 5% | New product interest |
| update_profile | 3% | Profile updates |
| contact_support | 2% | Support requests |

### Entity Resolution

Customers are linked between ERPNext and Supabase using **email**:

```
ERPNext Customer (email_id) <---> Supabase Session (user_email)
```

### Supabase Tables Created

```sql
-- App Sessions Table
CREATE TABLE app_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_email TEXT NOT NULL,
    session_start TIMESTAMP WITH TIME ZONE NOT NULL,
    session_end TIMESTAMP WITH TIME ZONE,
    device_type TEXT,
    app_version TEXT,
    platform TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- App Events Table
CREATE TABLE app_events (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    session_id UUID REFERENCES app_sessions(id),
    user_email TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    event_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### Files Generated

```
data/raw/
├── sb_sessions.csv    # ~15,000-20,000 records
├── sb_sessions.json
├── sb_events.csv      # ~50,000+ records (churn signal!)
└── sb_events.json
```

### Consolidated Notebook

Supabase pipeline: `notebooks/exploration/03_supabase_pipeline.ipynb`

Sections:
1. Configuration - Auto-load from docs/.env
2. Supabase Client - Python SDK connection
3. Exploration - Survey existing data
4. Ingestion - Create sessions and events via API
5. Extraction - Export to CSV/JSON

---

## Google Sheets Legacy Data Implementation Details

### Data Generation Summary

| Entity | Count | Notes |
|--------|-------|-------|
| Branches | 10 | Branch master data |
| Customer Notes | ~1,000+ | Manual notes from relationship managers |

### Branch Data

| Branch ID | Branch Name | Region | Manager |
|-----------|-------------|--------|---------|
| BR001 | Downtown Main | Central | Sarah Johnson |
| BR002 | Westside Plaza | West | Michael Chen |
| BR003 | Northgate Mall | North | Emily Rodriguez |
| BR004 | Eastside Business | East | David Kim |
| BR005 | South Central | South | Amanda Foster |
| BR006 | University District | Central | James Wilson |
| BR007 | Tech Hub | West | Lisa Park |
| BR008 | Harbor View | South | Robert Martinez |
| BR009 | Airport Business | East | Jennifer Lee |
| BR010 | Suburban Heights | North | Thomas Brown |

### Churn Signal Implementation

Customer notes contain early warning signs:

| Segment | Notes Count | Note Types | Example Notes |
|---------|-------------|------------|---------------|
| Active | 0-3 | Positive, Neutral | "Excellent customer", "Interested in mortgage" |
| At-Risk | 2-4 | Warning | "Complained about fees", "Shopping competitors" |
| Churned | 2-5 | Negative | "Requested closure", "Moving to competitor" |

### Note Type Distribution

| Type | Segment | Sample Text |
|------|---------|-------------|
| positive | Active | "Excellent customer - always maintains high balance" |
| positive | Active | "Referred 2 new customers this month" |
| warning | At-Risk | "Complained about monthly fees - consider waiver" |
| warning | At-Risk | "Mentioned competitor rates during call" |
| negative | Churned | "Customer requested account closure" |
| negative | Churned | "Moving to competitor - cited better rates" |

### Entity Resolution

Customers are linked between ERPNext and Google Sheets using **email**:

```
ERPNext Customer (email_id) <---> Google Sheets (customer_email)
```

### Google Cloud Setup

1. Created Google Cloud project: `banking-churn-poc`
2. Enabled Google Sheets API
3. Created service account: `banking-sheets-reader@banking-churn-poc.iam.gserviceaccount.com`
4. Downloaded credentials JSON file

### Files Generated

```
data/raw/
├── gs_branches.csv         # 10 records
├── gs_branches.json
├── gs_customer_notes.csv   # ~1,000+ records (churn signal!)
└── gs_customer_notes.json
```

### Consolidated Notebook

Google Sheets pipeline: `notebooks/exploration/04_google_sheets_pipeline.ipynb`

Sections:
1. Configuration - Auto-load from docs/.env
2. Google Sheets Client - Service account auth
3. Exploration - Survey existing data
4. Ingestion - Create branches and notes via API
5. Extraction - Export to CSV/JSON

---

## All Source Systems Summary

All 4 source systems are now complete and integrated:

| System | Purpose | Records | Churn Signal |
|--------|---------|---------|--------------|
| ERPNext | Core Banking | 500 customers, 8,569 transactions | Transaction frequency decline |
| Salesforce | CRM | 500 contacts, 1,500 cases | Complaint frequency |
| Supabase | Digital | 15,000+ sessions, 50,000+ events | App usage decline |
| Google Sheets | Legacy | 10 branches, 1,000+ notes | Warning/negative notes |

### Entity Resolution Map

All systems are linked via **email** as the common key:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CUSTOMER UNIFICATION VIA EMAIL                       │
│                                                                          │
│  ERPNext         Salesforce        Supabase         Google Sheets       │
│  ────────        ──────────        ────────         ─────────────       │
│  email_id   ═══  Email        ═══  user_email  ═══  customer_email      │
│                                                                          │
│  Example:                                                                │
│  john.smith.001@email.com links:                                        │
│  - ERPNext Customer CUST_001                                            │
│  - Salesforce Contact SF-001                                            │
│  - 45 Supabase Sessions                                                 │
│  - 3 Google Sheets Notes                                                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Bronze Layer Databricks Notebook Implementation

### Overview

Created a comprehensive Databricks notebook that extracts data from all 4 source systems and loads into Unity Catalog Bronze layer as Delta tables.

**Notebook:** `notebooks/databricks/bronze_ingestion.py`

### Schedule

| Setting | Value |
|---------|-------|
| Frequency | Daily |
| Time | 1:00 PM GMT+6 (07:00 UTC) |
| Cron Expression | `0 0 7 * * ?` |
| Timezone | UTC |

### Tables Created

| Source | Bronze Table | Description |
|--------|--------------|-------------|
| ERPNext | `bronze.erp_customers` | Customer master data |
| ERPNext | `bronze.erp_items` | Banking products |
| ERPNext | `bronze.erp_territories` | Branch locations |
| ERPNext | `bronze.erp_sales_invoices` | Transactions |
| Salesforce | `bronze.sf_contacts` | CRM contacts |
| Salesforce | `bronze.sf_cases` | Support tickets (churn signal!) |
| Salesforce | `bronze.sf_tasks` | Customer interactions |
| Supabase | `bronze.sb_app_sessions` | App login sessions |
| Supabase | `bronze.sb_app_events` | User actions (churn signal!) |
| Google Sheets | `bronze.gs_branches` | Branch master data |
| Google Sheets | `bronze.gs_customer_notes` | Manual notes (churn signal!) |

### Metadata Columns Added

Every Bronze table includes:
- `_ingested_at` - Timestamp of when the record was loaded
- `_source_system` - Name of the source system (erpnext, salesforce, supabase, google_sheets)

### Key Technical Features

1. **Databricks Secrets Integration**
   - All credentials stored in Databricks Secrets scope `bank-churn-secrets`
   - Never hardcoded in notebooks

2. **Widget-Based Testing**
   - Widgets for credential input during development
   - Switch to secrets for production

3. **Delta Table Format**
   - Full overwrite mode for each run
   - Schema evolution enabled (`overwriteSchema: true`)

4. **Error Handling**
   - Graceful handling of missing credentials
   - Continue ingestion even if one source fails

### Deployment Steps

1. **Upload to Databricks**
   ```bash
   databricks workspace import notebooks/databricks/bronze_ingestion.py /Workspace/Users/your-email/notebooks/bronze_ingestion -l PYTHON
   ```

2. **Set Up Secrets**
   ```bash
   databricks secrets create-scope bank-churn-secrets
   databricks secrets put-secret bank-churn-secrets ERPNEXT_API_KEY
   # ... add all secrets
   ```

3. **Upload Google Credentials**
   ```bash
   databricks fs cp docs/banking-churn-poc-5c5c657a7df3.json dbfs:/FileStore/banking-churn-poc-credentials.json
   ```

4. **Create Scheduled Job**
   - Via UI: Workflows > Jobs > Create Job
   - Schedule: `0 0 7 * * ?` (UTC) = 1:00 PM GMT+6

### Time Zone Reference

| GMT+6 (Local) | UTC | Cron Expression |
|---------------|-----|-----------------|
| 1:00 PM | 07:00 | `0 0 7 * * ?` |
| 6:00 AM | 00:00 | `0 0 0 * * ?` |
| 12:00 AM | 18:00 (prev day) | `0 0 18 * * ?` |

### Files Created

```
notebooks/databricks/
└── bronze_ingestion.py    # Databricks notebook for all 4 sources

docs/
└── BRONZE_INGESTION_SETUP.md  # Detailed setup guide
```

---

## Success Metrics

When this project is complete, you'll demonstrate:

| Skill | Evidence |
|-------|----------|
| Multi-source integration | 4 different systems unified |
| Data modeling | Star schema with facts & dimensions |
| Data quality | dbt tests catching issues |
| ML Engineering | Churn model with MLflow tracking |
| CI/CD | Automated testing on every PR |
| Business impact | From 40hrs/month manual → automated |

---

## Interview Talking Points

> "Walk me through a data project you've built."

*"I built an end-to-end customer churn prediction platform for a simulated retail bank. The challenge was unifying data from four different systems: ERPNext for core banking, Salesforce for CRM, Supabase for mobile app events, and legacy Google Sheets from branches.*

*Each system had different customer identifiers, so I implemented a customer matching algorithm in the silver layer to create a unified customer_360 view. I used dbt for transformation logic with full test coverage, Databricks Unity Catalog for governance, and MLflow for model tracking.*

*The pipeline runs on GitHub Actions - every PR triggers dbt tests, and merges to main auto-deploy to production. The final model achieved 0.82 AUC in predicting 90-day churn, which would enable proactive retention outreach."*

---

# APPENDIX: Real-Life Examples

## Why Medallion Architecture?

**Real Story:** A fintech company loaded data directly from source to gold tables. One day, the source system changed a date format. Gold tables broke. They had no raw data to debug. They lost 3 months of transaction history.

**With Medallion:** Bronze has raw data. Even if silver/gold break, you rebuild from bronze. Nothing is lost.

---

## Why Git for Data?

**Real Story:** A bank analyst modified a SQL view that calculated interest. No code review. Wrong calculation ran for 2 weeks. Customers were overcharged $2.3 million. No audit trail of who changed what.

**With Git:** Every change is reviewed. Every change is logged. Mistakes are caught before production.

---

## Why Environment Variables?

**Real Story:** A developer at Uber committed AWS keys to GitHub. Within minutes, crypto miners spun up thousands of servers. $50,000+ bill before anyone noticed.

**With Env Vars:** Keys never touch code. Even if code is public, credentials are safe.

---

## Why dbt?

**Real Story:** A data team at a retailer had 800+ SQL scripts. When a source table changed, they spent 2 weeks finding all impacted scripts. Some were missed. Reports were wrong for a month.

**With dbt:** `dbt ls --select +changed_model+` shows all downstream impact instantly. Fix once, everything updates.


