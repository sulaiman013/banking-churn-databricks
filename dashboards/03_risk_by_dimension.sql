-- ============================================================================
-- Risk Analysis by Dimension - Bar charts for different segments
-- ============================================================================
-- Create multiple bar charts:
-- 1. Risk by Territory
-- 2. Risk by Customer Group
-- 3. Risk by Tenure Segment
-- ============================================================================

-- Risk by Territory (Horizontal Bar Chart)
SELECT
    dimension_value AS territory,
    customer_count,
    ROUND(avg_risk_score, 1) AS avg_risk_score,
    high_risk_count,
    ROUND(100.0 * high_risk_count / NULLIF(customer_count, 0), 1) AS high_risk_pct
FROM bank_proj.gold.agg_churn_by_segment
WHERE dimension = 'territory'
ORDER BY avg_risk_score DESC;

-- Risk by Customer Group (Horizontal Bar Chart)
SELECT
    dimension_value AS customer_group,
    customer_count,
    ROUND(avg_risk_score, 1) AS avg_risk_score,
    high_risk_count,
    medium_risk_count,
    ROUND(100.0 * (high_risk_count + medium_risk_count) / NULLIF(customer_count, 0), 1) AS at_risk_pct
FROM bank_proj.gold.agg_churn_by_segment
WHERE dimension = 'customer_group'
ORDER BY avg_risk_score DESC;

-- Risk by Tenure Segment (Bar Chart)
SELECT
    dimension_value AS tenure_segment,
    customer_count,
    ROUND(avg_risk_score, 1) AS avg_risk_score,
    ROUND(avg_engagement_score, 1) AS avg_engagement_score,
    high_risk_count
FROM bank_proj.gold.agg_churn_by_segment
WHERE dimension = 'tenure_segment'
ORDER BY
    CASE dimension_value
        WHEN 'New' THEN 1
        WHEN 'Established' THEN 2
        WHEN 'Loyal' THEN 3
        WHEN 'Long-term' THEN 4
        ELSE 5
    END;

-- Risk by Digital Activity (Comparison Bar)
SELECT
    dimension_value AS digital_status,
    customer_count,
    ROUND(avg_risk_score, 1) AS avg_risk_score,
    ROUND(avg_engagement_score, 1) AS avg_engagement_score,
    total_value,
    high_risk_count
FROM bank_proj.gold.agg_churn_by_segment
WHERE dimension = 'digital_status'
ORDER BY avg_risk_score DESC;
