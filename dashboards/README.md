# Banking Churn Analytics Dashboards

This folder contains SQL queries for creating dashboards in Databricks SQL or Power BI.

## Dashboard Overview

The dashboards provide three main views:

1. **Executive Summary** - High-level KPIs and trends
2. **Risk Analysis** - Churn risk distribution and drivers
3. **Retention Worklist** - Actionable list for retention team

## Quick Start

### Option 1: Databricks SQL Dashboard

1. Navigate to **Databricks SQL > SQL Editor**
2. Copy queries from the `.sql` files in this folder
3. Create visualizations for each query
4. Pin to a new dashboard

### Option 2: Power BI

1. Connect Power BI to Databricks (see `powerbi_setup.md`)
2. Import queries as DirectQuery sources
3. Build visuals using the provided queries

---

## Queries Reference

| File | Purpose | Refresh |
|------|---------|---------|
| `01_executive_kpis.sql` | Top-line metrics cards | Daily |
| `02_risk_distribution.sql` | Risk segment breakdown | Daily |
| `03_risk_by_dimension.sql` | Risk analysis by territory/segment | Daily |
| `04_risk_trend.sql` | Risk score trends over time | Weekly |
| `05_high_risk_worklist.sql` | Retention team action list | Daily |
| `06_engagement_metrics.sql` | Digital engagement analysis | Daily |

---

## Dashboard Layout Recommendation

```
┌────────────────────────────────────────────────────────────────┐
│                    EXECUTIVE SUMMARY                            │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────┤
│ Total    │ High     │ Medium   │ Avg Risk │ At-Risk  │ Digital │
│ Customer │ Risk     │ Risk     │ Score    │ Revenue  │ Active  │
│  1,247   │   89     │   234    │  28.5    │  $1.2M   │  67%    │
├──────────┴──────────┴──────────┴──────────┴──────────┴─────────┤
│                                                                 │
│   [Risk Distribution      ]    [Risk by Territory         ]    │
│   [Pie/Donut Chart        ]    [Bar Chart                 ]    │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   [Risk by Customer Group ]    [Risk Trend Over Time      ]    │
│   [Bar Chart              ]    [Line Chart                ]    │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    HIGH RISK WORKLIST                           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Customer   │ Risk │ Value  │ Risk Factors │ Action        │ │
│  ├────────────┼──────┼────────┼──────────────┼───────────────┤ │
│  │ ABC Corp   │  78  │ $45K   │ Inactive 90+ │ Win-back call │ │
│  │ XYZ Ltd    │  72  │ $38K   │ Open ticket  │ Resolve case  │ │
│  └────────────┴──────┴────────┴──────────────┴───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```
