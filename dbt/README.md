# Banking Churn Analytics - dbt Project

A production-ready dbt project for transforming raw banking data into unified customer views and ML-ready features using the **Medallion Architecture** (Bronze → Silver → Gold).

## Overview

This dbt project is part of the **Bank Customer Churn Prediction POC** that unifies data from 4 source systems:
- **ERPNext** (Core Banking): Customers, accounts, transactions
- **Salesforce CRM**: Support cases, complaints, interactions
- **Supabase** (Digital): Mobile app sessions, user events
- **Google Sheets** (Legacy): Branch data, relationship manager notes

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MEDALLION ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  BRONZE (Raw)           SILVER (Clean)           GOLD (Business)    │
│  ─────────────          ─────────────           ──────────────      │
│                                                                      │
│  erp_customers    →     dim_customer      →     customer_360        │
│  erp_transactions →     fct_transactions  →     customer_features   │
│  sf_contacts      →     fct_complaints    →     agg_churn_by_segment│
│  sf_cases         →     fct_digital       →                         │
│  sb_sessions      →     sessions                                    │
│  sb_events        →                                                 │
│  gs_branches      →                                                 │
│  gs_notes         →                                                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
dbt/
├── dbt_project.yml          # Project configuration
├── profiles.yml.example     # Connection template (copy to ~/.dbt/)
├── packages.yml             # dbt package dependencies
│
├── models/
│   ├── staging/             # Source definitions & staging models
│   │   ├── _sources.yml     # Source table definitions
│   │   ├── stg_erp_*.sql    # ERPNext staging models
│   │   ├── stg_sf_*.sql     # Salesforce staging models
│   │   ├── stg_sb_*.sql     # Supabase staging models
│   │   └── stg_gs_*.sql     # Google Sheets staging models
│   │
│   ├── intermediate/        # Silver layer - cleaned & unified
│   │   ├── _intermediate.yml
│   │   ├── dim_customer.sql      # Unified customer dimension
│   │   ├── fct_transactions.sql  # Transaction facts
│   │   ├── fct_complaints.sql    # CRM case facts
│   │   └── fct_digital_sessions.sql
│   │
│   └── marts/               # Gold layer - business ready
│       ├── _marts.yml
│       ├── customer_360.sql      # Complete customer view
│       ├── customer_features.sql # ML feature table
│       └── agg_churn_by_segment.sql
│
├── tests/                   # Custom data quality tests
├── macros/                  # Reusable SQL macros
├── seeds/                   # Static reference data (CSV)
└── snapshots/               # SCD Type 2 tables (optional)
```

## Key Models

### Silver Layer (Intermediate)

| Model | Description | Key Logic |
|-------|-------------|-----------|
| `dim_customer` | Unified customer dimension | Links all 4 systems via email |
| `fct_transactions` | Transaction facts | Standardized amounts, dates |
| `fct_complaints` | CRM case facts | Complaint severity, resolution time |
| `fct_digital_sessions` | App sessions | Session duration, event counts |

### Gold Layer (Marts)

| Model | Description | Consumers |
|-------|-------------|-----------|
| `customer_360` | Complete customer view | Dashboards, CRM |
| `customer_features` | ML feature table | Churn prediction model |
| `agg_churn_by_segment` | Aggregated metrics | Executive dashboard |

## Customer Unification

The core challenge: **Same customer exists with different IDs across systems**

```
ERPNext:   CUST_001234     ─┐
Salesforce: SF-00001234    ─┼─→  customer_key: abc123def456
Supabase:   a1b2c3d4-...   ─┤
Sheets:     Row 47         ─┘
```

**Solution:** Email-based entity resolution in `dim_customer`:

```sql
-- Simplified unification logic
SELECT
    {{ dbt_utils.generate_surrogate_key(['email']) }} AS customer_key,
    e.erp_customer_id,
    c.sf_contact_id,
    email,
    COALESCE(e.customer_name, c.contact_name) AS full_name
FROM erp_customers e
LEFT JOIN crm_contacts c ON e.email = c.email
LEFT JOIN digital_users d ON e.email = d.email
```

## Getting Started

### Prerequisites

- Python 3.9+
- dbt-core 1.6+
- dbt-databricks adapter
- Databricks workspace with Unity Catalog

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/youruser/banking-churn-databricks.git
   cd banking-churn-databricks/dbt
   ```

2. **Install dbt and dependencies:**
   ```bash
   pip install dbt-core dbt-databricks
   dbt deps
   ```

3. **Configure connection:**

   Copy the profile template and edit with your credentials:
   ```bash
   cp profiles.yml.example ~/.dbt/profiles.yml
   ```

   Edit `~/.dbt/profiles.yml`:
   ```yaml
   bank_churn:
     target: dev
     outputs:
       dev:
         type: databricks
         catalog: bank_proj
         schema: dev_{{ env_var('USER', 'default') }}
         host: "{{ env_var('DATABRICKS_HOST') }}"
         http_path: /sql/1.0/warehouses/your-warehouse-id
         token: "{{ env_var('DATABRICKS_TOKEN') }}"
   ```

4. **Set environment variables:**
   ```bash
   export DATABRICKS_HOST=adb-xxx.azuredatabricks.net
   export DATABRICKS_TOKEN=dapi_your_token_here
   ```

5. **Test connection:**
   ```bash
   dbt debug
   ```

### Running dbt

```bash
# Install packages
dbt deps

# Run all models
dbt run

# Run specific layer
dbt run --select staging
dbt run --select intermediate
dbt run --select marts

# Run specific model and all downstream
dbt run --select dim_customer+

# Run tests
dbt test

# Generate documentation
dbt docs generate
dbt docs serve
```

## Data Quality Tests

All models include comprehensive tests:

```yaml
# Example: dim_customer tests
models:
  - name: dim_customer
    columns:
      - name: customer_key
        tests:
          - unique
          - not_null
      - name: email
        tests:
          - unique
          - not_null

  - name: fct_transactions
    columns:
      - name: customer_key
        tests:
          - not_null
          - relationships:
              to: ref('dim_customer')
              field: customer_key
```

Run tests:
```bash
# All tests
dbt test

# Specific model tests
dbt test --select dim_customer

# Data freshness tests
dbt source freshness
```

## CI/CD Integration

This project integrates with GitHub Actions:

| Workflow | Trigger | Action |
|----------|---------|--------|
| `dbt-ci.yml` | Pull Request | `dbt build --target ci` |
| `dbt-deploy.yml` | Merge to main | `dbt build --target prod` |

### Target Environments

| Target | Schema | Use Case |
|--------|--------|----------|
| `dev` | `dev_<username>` | Local development |
| `ci` | `ci_test` | Automated testing in PRs |
| `prod` | `gold` | Production deployment |

## Feature Engineering

The `customer_features` model engineers 12 features for ML:

| Feature | Description | Churn Signal |
|---------|-------------|--------------|
| `tenure_days` | Days since account opened | New customers churn more |
| `total_transactions` | Lifetime transaction count | Lower = disengaged |
| `days_since_last_transaction` | Inactivity period | >60 days = high risk |
| `total_support_cases` | Support ticket count | More complaints = frustrated |
| `has_open_complaint` | Unresolved issue flag | Strong churn indicator |
| `total_digital_sessions` | App usage count | Declining = disengaging |
| `sessions_last_30d` | Recent app activity | 0 sessions = churning |
| `engagement_health_score` | Composite 0-100 score | <40 = at risk |
| `churn_risk_score` | Rule-based risk score | Baseline for ML |

## Common Commands

```bash
# Full build (run + test)
dbt build

# Rebuild Silver layer from scratch
dbt run --select intermediate --full-refresh

# See what would run without executing
dbt run --select customer_360 --dry-run

# Compile SQL to inspect (outputs to target/)
dbt compile --select customer_360

# List all models
dbt ls

# Show model dependencies
dbt ls --select +customer_360+
```

## Troubleshooting

### Common Issues

1. **Connection failed:**
   ```
   dbt debug  # Check connection settings
   ```
   Verify DATABRICKS_HOST and DATABRICKS_TOKEN are set.

2. **Schema doesn't exist:**
   ```sql
   CREATE SCHEMA IF NOT EXISTS bank_proj.silver;
   ```

3. **Source table not found:**
   Ensure Bronze ingestion notebook has run first.

4. **Test failures:**
   ```bash
   dbt test --select model_name --store-failures
   ```
   Check `target/run_results.json` for details.

## Related Resources

- [Main Project README](../README.md) - Project overview
- [Lesson Documentation](../docs/lesson.md) - Learning guide
- [ML Notebooks](../notebooks/ml/) - Churn prediction pipeline
- [Bronze Ingestion](../notebooks/databricks/bronze_ingestion.py) - Data loading

## Architecture Diagram

```
Source Systems              Bronze Layer              Silver Layer              Gold Layer
───────────────            ─────────────            ─────────────            ──────────────

┌─────────────┐         ┌────────────────┐       ┌───────────────┐       ┌─────────────────┐
│   ERPNext   │────────►│ erp_customers  │──┐    │               │       │                 │
│   (Core     │         │ erp_accounts   │  │    │               │       │   customer_360  │
│   Banking)  │         │ erp_transactions│  │    │               │       │   (Complete     │
└─────────────┘         └────────────────┘  │    │  dim_customer │──────►│    customer     │
                                            ├───►│  (Unified)    │       │    view)        │
┌─────────────┐         ┌────────────────┐  │    │               │       │                 │
│  Salesforce │────────►│ sf_contacts    │──┤    │               │       └─────────────────┘
│    (CRM)    │         │ sf_cases       │  │    └───────────────┘               │
└─────────────┘         │ sf_tasks       │  │            │                       │
                        └────────────────┘  │            │                       ▼
                                            │            ▼               ┌─────────────────┐
┌─────────────┐         ┌────────────────┐  │    ┌───────────────┐       │customer_features│
│  Supabase   │────────►│ sb_sessions    │──┤    │fct_transactions│──────►│  (ML-ready     │
│  (Digital)  │         │ sb_events      │  │    │fct_complaints  │       │   features)    │
└─────────────┘         └────────────────┘  │    │fct_digital_    │       └─────────────────┘
                                            │    │  sessions      │               │
┌─────────────┐         ┌────────────────┐  │    └───────────────┘               │
│   Sheets    │────────►│ gs_branches    │──┘                                    ▼
│  (Legacy)   │         │ gs_notes       │                               ┌─────────────────┐
└─────────────┘         └────────────────┘                               │  ML Pipeline    │
                                                                         │  (notebooks/ml/)│
                                                                         └─────────────────┘
```

## Contributing

1. Create a feature branch: `git checkout -b feature/my-model`
2. Make changes and run tests: `dbt build`
3. Create a Pull Request
4. CI will run `dbt build --target ci`
5. After approval, merge to main

## License

MIT License - See [LICENSE](../LICENSE) for details.
