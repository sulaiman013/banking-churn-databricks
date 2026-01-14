# Power BI Connection Guide for Banking Churn Analytics

Connect Power BI to Databricks Unity Catalog to build executive dashboards.

## Prerequisites

- Power BI Desktop (latest version)
- Databricks workspace access
- Personal Access Token from Databricks

---

## Step 1: Get Databricks Connection Details

From your Databricks workspace, gather:

| Setting | Where to Find | Example Value |
|---------|---------------|---------------|
| Server Hostname | SQL Warehouse > Connection Details | `adb-xxxx.azuredatabricks.net` |
| HTTP Path | SQL Warehouse > Connection Details | `/sql/1.0/warehouses/xxxxx` |
| Catalog | Unity Catalog | `bank_proj` |
| Schema | Unity Catalog | `gold` |

---

## Step 2: Connect Power BI to Databricks

1. Open **Power BI Desktop**
2. Click **Get Data** > **More...**
3. Search for **"Databricks"** and select **Azure Databricks**
4. Enter connection details:
   - **Server Hostname**: Your Databricks hostname
   - **HTTP Path**: Your SQL warehouse path
5. Choose **DirectQuery** (recommended for real-time data)
6. Authenticate with **Personal Access Token**

---

## Step 3: Select Tables

Navigate to: `bank_proj` > `gold`

Select these tables:
- `customer_360` - Master customer view
- `customer_features` - ML feature table
- `agg_churn_by_segment` - Pre-aggregated metrics
- `high_risk_customers` - Retention worklist

Click **Load** or **Transform Data** if you need to make changes.

---

## Step 4: Create Relationships

Power BI should auto-detect relationships. Verify:

```
agg_churn_by_segment (dimension_value)
    ↓ many-to-one
customer_features (risk_segment calculated)

customer_360 (unified_customer_id)
    ↓ one-to-one
customer_features (unified_customer_id)
```

---

## Step 5: Build Dashboard Visuals

### KPI Cards (Top Row)
Create cards for:
- Total Customers: `COUNTROWS(customer_360)`
- High Risk Count: `CALCULATE(COUNTROWS(customer_features), customer_features[churn_risk_score] >= 60)`
- Avg Risk Score: `AVERAGE(customer_features[churn_risk_score])`
- At-Risk Revenue: `SUMX(FILTER(customer_360, customer_360[churn_risk_score] >= 40), customer_360[total_transaction_amount])`

### Risk Distribution (Pie Chart)
- Use `agg_churn_by_segment` filtered to `dimension = "risk_segment"`
- Legend: `dimension_value`
- Values: `customer_count`

### Risk by Territory (Bar Chart)
- Use `agg_churn_by_segment` filtered to `dimension = "territory"`
- Axis: `dimension_value`
- Values: `avg_risk_score`

### High Risk Table
- Use `high_risk_customers` table
- Columns: customer_name, risk_category, churn_risk_score, lifetime_value, recommended_action
- Sort by: `priority_score` descending
- Conditional formatting: Red for Critical, Orange for High

---

## DAX Measures

Add these measures to your model:

```dax
// Risk Segment
Risk Segment =
SWITCH(
    TRUE(),
    [churn_risk_score] >= 60, "High Risk",
    [churn_risk_score] >= 40, "Medium Risk",
    [churn_risk_score] >= 20, "Low Risk",
    "Minimal Risk"
)

// At-Risk Customer Count
At Risk Customers =
CALCULATE(
    COUNTROWS(customer_features),
    customer_features[churn_risk_score] >= 40
)

// Total Value at Risk
Value at Risk =
CALCULATE(
    SUM(customer_360[total_transaction_amount]),
    customer_features[churn_risk_score] >= 40
)

// Digital Adoption Rate
Digital Adoption % =
DIVIDE(
    CALCULATE(COUNTROWS(customer_360), customer_360[is_digitally_active] = TRUE()),
    COUNTROWS(customer_360)
) * 100

// Avg Risk Score
Avg Risk Score = AVERAGE(customer_features[churn_risk_score])
```

---

## Refresh Settings

For DirectQuery:
- Data refreshes automatically on each interaction
- No scheduled refresh needed

For Import Mode:
- Set up scheduled refresh in Power BI Service
- Recommended: Daily refresh at 7 AM

---

## Sample Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│  BANKING CHURN ANALYTICS DASHBOARD                          │
│  Last Updated: [Auto]                    [Territory Filter] │
├─────────┬─────────┬─────────┬─────────┬─────────┬──────────┤
│ 1,247   │   89    │   234   │  28.5   │ $1.2M   │  67%     │
│ Total   │ High    │ Medium  │ Avg     │ At-Risk │ Digital  │
│ Custmrs │ Risk    │ Risk    │ Score   │ Revenue │ Active   │
├─────────┴─────────┴─────────┴─────────┴─────────┴──────────┤
│                                                             │
│  [Risk Distribution]          [Risk by Territory]          │
│  [Donut Chart     ]          [Horizontal Bar    ]          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Risk by Customer Group]     [Engagement vs Risk]         │
│  [Bar Chart          ]        [Scatter Plot     ]          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  HIGH RISK CUSTOMERS - ACTION REQUIRED                      │
│  ┌─────────────────────────────────────────────────────────│
│  │ Customer │ Score │ Value │ Factors    │ Action         │
│  │ ABC Corp │  78   │ $45K  │ Inactive   │ Win-back call  │
│  │ XYZ Ltd  │  72   │ $38K  │ Complaint  │ Resolve case   │
│  └─────────────────────────────────────────────────────────│
└─────────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection timeout | Ensure SQL Warehouse is running |
| Authentication failed | Generate new Personal Access Token |
| Tables not visible | Check Unity Catalog permissions |
| Slow performance | Use pre-aggregated `agg_churn_by_segment` table |
| Missing data | Verify dbt models ran successfully |

---

## Publishing to Power BI Service

1. Click **Publish** in Power BI Desktop
2. Select your workspace
3. Configure **Gateway** for DirectQuery (if on-premises)
4. Set up **Row-Level Security** if needed
5. Share dashboard with stakeholders
