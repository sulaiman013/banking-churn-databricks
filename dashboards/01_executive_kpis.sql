-- ============================================================================
-- Executive KPIs - Top-line metrics for dashboard cards
-- ============================================================================
-- Use: Create 6 counter/card visualizations with these metrics
-- Refresh: Daily
-- ============================================================================

-- Total Customers
SELECT
    customer_count AS total_customers
FROM bank_proj.gold.agg_churn_by_segment
WHERE dimension = 'overall';

-- High Risk Customers
SELECT
    high_risk_count AS high_risk_customers
FROM bank_proj.gold.agg_churn_by_segment
WHERE dimension = 'overall';

-- Medium Risk Customers
SELECT
    medium_risk_count AS medium_risk_customers
FROM bank_proj.gold.agg_churn_by_segment
WHERE dimension = 'overall';

-- Average Risk Score
SELECT
    ROUND(avg_risk_score, 1) AS avg_risk_score
FROM bank_proj.gold.agg_churn_by_segment
WHERE dimension = 'overall';

-- At-Risk Revenue (High + Medium risk customer value)
SELECT
    CONCAT('$', FORMAT_NUMBER(
        SUM(CASE WHEN dimension_value IN ('High Risk', 'Medium Risk')
            THEN total_value ELSE 0 END), 0
    )) AS at_risk_revenue
FROM bank_proj.gold.agg_churn_by_segment
WHERE dimension = 'risk_segment';

-- Digital Adoption Rate
SELECT
    CONCAT(
        ROUND(
            100.0 * SUM(CASE WHEN dimension_value = 'Digitally Active' THEN customer_count ELSE 0 END) /
            NULLIF(SUM(customer_count), 0)
        , 1), '%'
    ) AS digital_adoption_rate
FROM bank_proj.gold.agg_churn_by_segment
WHERE dimension = 'digital_status';

-- All KPIs in one query (for single data source)
SELECT
    MAX(CASE WHEN dimension = 'overall' THEN customer_count END) AS total_customers,
    MAX(CASE WHEN dimension = 'overall' THEN high_risk_count END) AS high_risk_customers,
    MAX(CASE WHEN dimension = 'overall' THEN medium_risk_count END) AS medium_risk_customers,
    MAX(CASE WHEN dimension = 'overall' THEN ROUND(avg_risk_score, 1) END) AS avg_risk_score,
    SUM(CASE WHEN dimension = 'risk_segment' AND dimension_value IN ('High Risk', 'Medium Risk')
        THEN total_value ELSE 0 END) AS at_risk_revenue,
    ROUND(100.0 *
        SUM(CASE WHEN dimension = 'digital_status' AND dimension_value = 'Digitally Active'
            THEN customer_count ELSE 0 END) /
        NULLIF(SUM(CASE WHEN dimension = 'digital_status' THEN customer_count ELSE 0 END), 0)
    , 1) AS digital_adoption_pct
FROM bank_proj.gold.agg_churn_by_segment;
