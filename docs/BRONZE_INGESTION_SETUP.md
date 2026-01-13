# Bronze Layer Ingestion - Setup Guide

This guide walks you through deploying the Bronze ingestion notebook to Databricks and scheduling it to run daily at 1:00 PM GMT+6.

## Overview

| Setting | Value |
|---------|-------|
| **Notebook** | `notebooks/databricks/bronze_ingestion.py` |
| **Schedule** | Daily at 1:00 PM GMT+6 (07:00 UTC) |
| **Target** | `bank_proj.bronze` (Unity Catalog) |
| **Tables Created** | 11 tables from 4 source systems |

### Tables Created

| Source System | Tables | Description |
|---------------|--------|-------------|
| ERPNext | `erp_customers`, `erp_items`, `erp_territories`, `erp_sales_invoices` | Core banking data |
| Salesforce | `sf_contacts`, `sf_cases`, `sf_tasks` | CRM data |
| Supabase | `sb_app_sessions`, `sb_app_events` | Mobile app data |
| Google Sheets | `gs_branches`, `gs_customer_notes` | Legacy data |

---

## Step 1: Upload Notebook to Databricks

### Option A: Manual Upload (Recommended for First Time)

1. Go to your Databricks workspace: `https://dbc-621d2785-493f.cloud.databricks.com`
2. Navigate to **Workspace** > **Users** > **your-email**
3. Click **Import**
4. Select the file: `notebooks/databricks/bronze_ingestion.py`
5. Click **Import**

### Option B: Databricks CLI

```bash
# Install CLI if not installed
pip install databricks-cli

# Configure (use your host and token)
databricks configure --host https://dbc-621d2785-493f.cloud.databricks.com

# Import notebook
databricks workspace import notebooks/databricks/bronze_ingestion.py /Workspace/Users/your-email/notebooks/bronze_ingestion -l PYTHON
```

---

## Step 2: Set Up Databricks Secrets

Store credentials securely using Databricks Secrets instead of hardcoding.

### Create Secret Scope

```bash
databricks secrets create-scope bank-churn-secrets
```

### Add Secrets

Run each command and enter the value when prompted:

```bash
# ERPNext
databricks secrets put-secret bank-churn-secrets ERPNEXT_API_KEY
# Enter: aad27795b318fde

databricks secrets put-secret bank-churn-secrets ERPNEXT_API_SECRET
# Enter: c97e94134c7e717

# Salesforce
databricks secrets put-secret bank-churn-secrets SALESFORCE_USERNAME
# Enter: sulaimansta013.2ae597b9710e@agentforce.com

databricks secrets put-secret bank-churn-secrets SALESFORCE_PASSWORD
# Enter: popopupu1212

databricks secrets put-secret bank-churn-secrets SALESFORCE_SECURITY_TOKEN
# Enter: IZ3FRKUOG6Fs8tbuHNmCgTTFQ

databricks secrets put-secret bank-churn-secrets SALESFORCE_CONSUMER_KEY
# Enter: 3MVG97L7PWbPq6UzCp9CskUTTzUKHCV.QSDPhMHZmjIOfo43iLvNLl1suj76udagALB0Hcgz9EgiS.pTOgiPf

databricks secrets put-secret bank-churn-secrets SALESFORCE_CONSUMER_SECRET
# Enter: EE14E664DB39A0B761ECD0CB42782974F4441BB7EB01A900D1EE7459309EB367

# Supabase
databricks secrets put-secret bank-churn-secrets SUPABASE_URL
# Enter: https://wtdspfddzqkpdaokgzys.supabase.co

databricks secrets put-secret bank-churn-secrets SUPABASE_KEY
# Enter: sb_secret_Quw_we6B0UpbPoUyrTPxuA_v_AM9D8U

# Google Sheets
databricks secrets put-secret bank-churn-secrets GOOGLE_SHEETS_ID
# Enter: 1pcpF4IJqRv7aVLMmApUOsXu-gmTdWx_5lArbNNGjf6E
```

### Upload Google Credentials JSON

For Google Sheets access, upload the service account JSON to DBFS:

```bash
# Upload via CLI
databricks fs cp docs/banking-churn-poc-5c5c657a7df3.json dbfs:/FileStore/banking-churn-poc-credentials.json
```

Or via UI:
1. Go to **Data** > **DBFS** > **FileStore**
2. Click **Upload**
3. Select `docs/banking-churn-poc-5c5c657a7df3.json`
4. Rename to `banking-churn-poc-credentials.json`

---

## Step 3: Test the Notebook

Before scheduling, run the notebook manually to verify everything works.

### Using Widgets (Testing Mode)

1. Open the notebook in Databricks
2. Attach to a cluster (or create a new one)
3. Fill in the widget values at the top:

| Widget | Value |
|--------|-------|
| erpnext_url | `https://erpnext-rnm-aly.m.erpnext.com` |
| erpnext_api_key | `aad27795b318fde` |
| erpnext_api_secret | `c97e94134c7e717` |
| sf_username | `sulaimansta013.2ae597b9710e@agentforce.com` |
| sf_password | `popopupu1212` |
| sf_token | `IZ3FRKUOG6Fs8tbuHNmCgTTFQ` |
| sf_consumer_key | `3MVG97L7PWbPq...` |
| sf_consumer_secret | `EE14E664DB39...` |
| supabase_url | `https://wtdspfddzqkpdaokgzys.supabase.co` |
| supabase_key | `sb_secret_Quw_we6B0UpbPoUyrTPxuA_v_AM9D8U` |
| google_sheets_id | `1pcpF4IJqRv7aVLMmApUOsXu-gmTdWx_5lArbNNGjf6E` |

4. Click **Run All**

### Switch to Production Mode

Once testing works, edit the notebook to use secrets:

1. Comment out the widget-based configuration
2. Uncomment the secrets-based configuration (marked as "PRODUCTION MODE")

---

## Step 4: Create Scheduled Job

### Via Databricks UI (Recommended)

1. Go to **Workflows** > **Jobs**
2. Click **Create Job**
3. Configure:

| Field | Value |
|-------|-------|
| Job name | `bronze_ingestion_daily` |
| Task name | `ingest_all_sources` |
| Type | Notebook |
| Source | Workspace |
| Path | `/Workspace/Users/your-email/notebooks/bronze_ingestion` |
| Cluster | Serverless (recommended) or Job Cluster |

4. Click **Add Schedule**:

| Field | Value |
|-------|-------|
| Schedule type | Scheduled |
| Frequency | Every day |
| At | 07:00 (UTC) |
| Time zone | UTC |

**Note**: 07:00 UTC = 1:00 PM GMT+6

5. Click **Create**

### Via Databricks CLI

```bash
databricks jobs create --json '{
  "name": "bronze_ingestion_daily",
  "tasks": [{
    "task_key": "ingest_all_sources",
    "notebook_task": {
      "notebook_path": "/Workspace/Users/your-email/notebooks/bronze_ingestion"
    },
    "new_cluster": {
      "spark_version": "14.3.x-scala2.12",
      "node_type_id": "Standard_DS3_v2",
      "num_workers": 0,
      "spark_conf": {
        "spark.databricks.cluster.profile": "singleNode"
      }
    }
  }],
  "schedule": {
    "quartz_cron_expression": "0 0 7 * * ?",
    "timezone_id": "UTC",
    "pause_status": "UNPAUSED"
  },
  "email_notifications": {
    "on_failure": ["your-email@example.com"]
  },
  "max_retries": 2,
  "retry_on_timeout": true
}'
```

---

## Step 5: Verify Job

### Check Job Status

1. Go to **Workflows** > **Jobs**
2. Click on `bronze_ingestion_daily`
3. Check **Runs** tab for execution history

### Verify Data in Unity Catalog

Run these SQL queries in Databricks SQL:

```sql
-- List all bronze tables
SHOW TABLES IN bank_proj.bronze;

-- Check row counts
SELECT 'erp_customers' as table_name, COUNT(*) as rows FROM bank_proj.bronze.erp_customers
UNION ALL
SELECT 'erp_sales_invoices', COUNT(*) FROM bank_proj.bronze.erp_sales_invoices
UNION ALL
SELECT 'sf_contacts', COUNT(*) FROM bank_proj.bronze.sf_contacts
UNION ALL
SELECT 'sf_cases', COUNT(*) FROM bank_proj.bronze.sf_cases
UNION ALL
SELECT 'sb_app_sessions', COUNT(*) FROM bank_proj.bronze.sb_app_sessions
UNION ALL
SELECT 'gs_customer_notes', COUNT(*) FROM bank_proj.bronze.gs_customer_notes;

-- Check latest ingestion timestamp
SELECT _source_system, MAX(_ingested_at) as last_ingestion
FROM bank_proj.bronze.erp_customers
GROUP BY _source_system;
```

---

## Time Zone Reference

| GMT+6 (Local) | UTC | Cron Expression |
|---------------|-----|-----------------|
| 1:00 AM | 19:00 (prev day) | `0 0 19 * * ?` |
| 6:00 AM | 00:00 | `0 0 0 * * ?` |
| 12:00 PM (noon) | 06:00 | `0 0 6 * * ?` |
| **1:00 PM** | **07:00** | **`0 0 7 * * ?`** |
| 6:00 PM | 12:00 | `0 0 12 * * ?` |
| 12:00 AM (midnight) | 18:00 | `0 0 18 * * ?` |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: simple_salesforce` | Ensure `%pip install` cell runs first |
| Salesforce auth failed | Verify security token is valid (reset in Salesforce) |
| Google Sheets permission denied | Share spreadsheet with service account email |
| Supabase connection timeout | Check Supabase project is active |
| Unity Catalog permission denied | Ensure you have CREATE TABLE on bronze schema |

### Check Logs

```bash
# Via CLI
databricks jobs get-run --run-id <run_id>
databricks runs get-output --run-id <run_id>
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     BRONZE INGESTION JOB                            │
│                   (Daily at 1:00 PM GMT+6)                          │
└─────────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│   ERPNext     │       │  Salesforce   │       │   Supabase    │
│ Core Banking  │       │     CRM       │       │ Digital App   │
├───────────────┤       ├───────────────┤       ├───────────────┤
│ • customers   │       │ • contacts    │       │ • app_sessions│
│ • items       │       │ • cases       │       │ • app_events  │
│ • territories │       │ • tasks       │       │               │
│ • invoices    │       │               │       │               │
└───────────────┘       └───────────────┘       └───────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                                  ▼
                    ┌───────────────────────┐
                    │    Google Sheets      │
                    │      Legacy           │
                    ├───────────────────────┤
                    │ • branches            │
                    │ • customer_notes      │
                    └───────────────────────┘
                                  │
                                  ▼
                    ┌───────────────────────┐
                    │   Unity Catalog       │
                    │   bank_proj.bronze    │
                    ├───────────────────────┤
                    │ 11 Delta Tables       │
                    │ + _ingested_at        │
                    │ + _source_system      │
                    └───────────────────────┘
```

---

## Next Steps

After Bronze layer is working:

1. **Silver Layer**: Create dbt models for data cleaning and entity resolution
2. **Gold Layer**: Build customer_360 view and ML features
3. **ML Model**: Train churn prediction model on Gold layer
4. **Dashboard**: Create monitoring dashboard in Databricks SQL

---

*Created for Banking Customer Churn Prediction POC*
