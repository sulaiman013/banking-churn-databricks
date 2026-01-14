-- ============================================================================
-- Digital Engagement Analysis - Understanding customer digital behavior
-- ============================================================================
-- Visualization options:
-- - Engagement score distribution (histogram)
-- - Session metrics by segment (bar chart)
-- - Digital vs Non-digital risk comparison (dual bar)
-- ============================================================================

-- Digital Engagement Summary
SELECT
    'Digital Active' AS segment,
    COUNT(*) AS customers,
    ROUND(AVG(churn_risk_score), 1) AS avg_risk_score,
    ROUND(AVG(engagement_health_score), 1) AS avg_engagement_score,
    SUM(sessions_last_30d) AS total_sessions_30d,
    ROUND(AVG(sessions_last_30d), 1) AS avg_sessions_per_customer
FROM bank_proj.gold.customer_features
WHERE sessions_last_30d > 0
UNION ALL
SELECT
    'Not Digital Active' AS segment,
    COUNT(*) AS customers,
    ROUND(AVG(churn_risk_score), 1) AS avg_risk_score,
    ROUND(AVG(engagement_health_score), 1) AS avg_engagement_score,
    0 AS total_sessions_30d,
    0 AS avg_sessions_per_customer
FROM bank_proj.gold.customer_features
WHERE sessions_last_30d = 0 OR sessions_last_30d IS NULL;

-- Engagement Score Distribution
SELECT
    CASE
        WHEN engagement_health_score >= 80 THEN 'Highly Engaged (80-100)'
        WHEN engagement_health_score >= 60 THEN 'Engaged (60-79)'
        WHEN engagement_health_score >= 40 THEN 'Moderate (40-59)'
        WHEN engagement_health_score >= 20 THEN 'Low Engagement (20-39)'
        ELSE 'Minimal (0-19)'
    END AS engagement_band,
    COUNT(*) AS customer_count,
    ROUND(AVG(churn_risk_score), 1) AS avg_risk_score
FROM bank_proj.gold.customer_features
GROUP BY
    CASE
        WHEN engagement_health_score >= 80 THEN 'Highly Engaged (80-100)'
        WHEN engagement_health_score >= 60 THEN 'Engaged (60-79)'
        WHEN engagement_health_score >= 40 THEN 'Moderate (40-59)'
        WHEN engagement_health_score >= 20 THEN 'Low Engagement (20-39)'
        ELSE 'Minimal (0-19)'
    END
ORDER BY avg_risk_score;

-- Transaction Activity Analysis
SELECT
    CASE
        WHEN days_since_last_transaction <= 7 THEN 'Active (< 7 days)'
        WHEN days_since_last_transaction <= 30 THEN 'Recent (7-30 days)'
        WHEN days_since_last_transaction <= 60 THEN 'Cooling (30-60 days)'
        WHEN days_since_last_transaction <= 90 THEN 'At Risk (60-90 days)'
        ELSE 'Inactive (90+ days)'
    END AS activity_status,
    COUNT(*) AS customer_count,
    ROUND(AVG(churn_risk_score), 1) AS avg_risk_score,
    SUM(total_transaction_amount) AS total_value
FROM bank_proj.gold.customer_features
WHERE days_since_last_transaction IS NOT NULL
GROUP BY
    CASE
        WHEN days_since_last_transaction <= 7 THEN 'Active (< 7 days)'
        WHEN days_since_last_transaction <= 30 THEN 'Recent (7-30 days)'
        WHEN days_since_last_transaction <= 60 THEN 'Cooling (30-60 days)'
        WHEN days_since_last_transaction <= 90 THEN 'At Risk (60-90 days)'
        ELSE 'Inactive (90+ days)'
    END
ORDER BY
    CASE
        WHEN days_since_last_transaction <= 7 THEN 1
        WHEN days_since_last_transaction <= 30 THEN 2
        WHEN days_since_last_transaction <= 60 THEN 3
        WHEN days_since_last_transaction <= 90 THEN 4
        ELSE 5
    END;

-- Risk vs Engagement Scatter Data
SELECT
    unified_customer_id,
    churn_risk_score,
    engagement_health_score,
    total_transaction_amount AS value,
    CASE
        WHEN churn_risk_score >= 60 THEN 'High Risk'
        WHEN churn_risk_score >= 40 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END AS risk_category
FROM bank_proj.gold.customer_features
WHERE engagement_health_score IS NOT NULL;
