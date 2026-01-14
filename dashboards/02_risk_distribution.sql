-- ============================================================================
-- Risk Distribution - Pie/Donut chart showing customer risk segments
-- ============================================================================
-- Visualization: Pie chart or Donut chart
-- Group by: dimension_value (risk segment)
-- Value: customer_count
-- ============================================================================

SELECT
    dimension_value AS risk_segment,
    customer_count,
    ROUND(100.0 * customer_count / SUM(customer_count) OVER (), 1) AS percentage,
    total_value AS segment_value,
    ROUND(avg_risk_score, 1) AS avg_score
FROM bank_proj.gold.agg_churn_by_segment
WHERE dimension = 'risk_segment'
ORDER BY
    CASE dimension_value
        WHEN 'Critical' THEN 1
        WHEN 'High Risk' THEN 2
        WHEN 'Medium Risk' THEN 3
        WHEN 'Low Risk' THEN 4
        WHEN 'Minimal Risk' THEN 5
        ELSE 6
    END;
