-- ============================================================================
-- Risk Trend Over Time - Line chart showing risk score evolution
-- ============================================================================
-- Note: This query uses ML model predictions if available
-- Otherwise shows a static snapshot
-- Visualization: Line chart with date on X-axis
-- ============================================================================

-- Option 1: If ML predictions table exists with historical scores
-- SELECT
--     score_date,
--     AVG(churn_probability) AS avg_churn_probability,
--     COUNT(CASE WHEN risk_tier = 'HIGH' THEN 1 END) AS high_risk_count,
--     COUNT(CASE WHEN risk_tier = 'MEDIUM' THEN 1 END) AS medium_risk_count,
--     COUNT(*) AS total_scored
-- FROM bank_proj.gold.customer_churn_predictions
-- GROUP BY score_date
-- ORDER BY score_date;

-- Option 2: Snapshot comparison (current vs historical aggregates)
-- This creates a simple trend showing risk distribution
SELECT
    'Current' AS period,
    DATE(aggregated_at) AS snapshot_date,
    dimension_value AS risk_segment,
    customer_count,
    ROUND(avg_risk_score, 1) AS avg_risk_score
FROM bank_proj.gold.agg_churn_by_segment
WHERE dimension = 'risk_segment'
ORDER BY
    CASE dimension_value
        WHEN 'High Risk' THEN 1
        WHEN 'Medium Risk' THEN 2
        WHEN 'Low Risk' THEN 3
        ELSE 4
    END;

-- Risk score distribution histogram
SELECT
    CASE
        WHEN churn_risk_score >= 80 THEN '80-100 (Critical)'
        WHEN churn_risk_score >= 60 THEN '60-79 (High)'
        WHEN churn_risk_score >= 40 THEN '40-59 (Medium)'
        WHEN churn_risk_score >= 20 THEN '20-39 (Low)'
        ELSE '0-19 (Minimal)'
    END AS risk_band,
    COUNT(*) AS customer_count
FROM bank_proj.gold.customer_features
GROUP BY
    CASE
        WHEN churn_risk_score >= 80 THEN '80-100 (Critical)'
        WHEN churn_risk_score >= 60 THEN '60-79 (High)'
        WHEN churn_risk_score >= 40 THEN '40-59 (Medium)'
        WHEN churn_risk_score >= 20 THEN '20-39 (Low)'
        ELSE '0-19 (Minimal)'
    END
ORDER BY
    CASE
        WHEN churn_risk_score >= 80 THEN 1
        WHEN churn_risk_score >= 60 THEN 2
        WHEN churn_risk_score >= 40 THEN 3
        WHEN churn_risk_score >= 20 THEN 4
        ELSE 5
    END;
