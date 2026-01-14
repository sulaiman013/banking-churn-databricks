-- ============================================================================
-- High Risk Worklist - Actionable table for retention team
-- ============================================================================
-- Visualization: Table with conditional formatting
-- - Red highlight for Critical risk
-- - Orange highlight for High risk
-- - Include action buttons/links if supported
-- ============================================================================

-- Top 50 Priority Customers for Retention
SELECT
    ROW_NUMBER() OVER (ORDER BY priority_score DESC) AS priority_rank,
    customer_name,
    email,
    territory,
    risk_category,
    ROUND(churn_risk_score, 0) AS risk_score,
    CONCAT('$', FORMAT_NUMBER(lifetime_value, 0)) AS lifetime_value,
    risk_factors,
    recommended_action,
    days_since_last_transaction AS days_inactive,
    COALESCE(DATE_FORMAT(last_transaction_date, 'MMM dd, yyyy'), 'Never') AS last_activity
FROM bank_proj.gold.high_risk_customers
ORDER BY priority_score DESC
LIMIT 50;

-- Summary by Risk Category
SELECT
    risk_category,
    COUNT(*) AS customer_count,
    SUM(lifetime_value) AS total_value_at_risk,
    ROUND(AVG(churn_risk_score), 1) AS avg_risk_score,
    ROUND(AVG(days_since_last_transaction), 0) AS avg_days_inactive
FROM bank_proj.gold.high_risk_customers
GROUP BY risk_category
ORDER BY
    CASE risk_category
        WHEN 'Critical' THEN 1
        WHEN 'High' THEN 2
        WHEN 'Medium' THEN 3
    END;

-- Action Type Distribution
SELECT
    recommended_action,
    COUNT(*) AS customer_count,
    SUM(lifetime_value) AS total_value,
    ROUND(AVG(churn_risk_score), 1) AS avg_risk_score
FROM bank_proj.gold.high_risk_customers
GROUP BY recommended_action
ORDER BY customer_count DESC;

-- Customers with Open Complaints (Urgent)
SELECT
    customer_name,
    email,
    territory,
    risk_category,
    churn_risk_score,
    lifetime_value,
    risk_factors
FROM bank_proj.gold.high_risk_customers
WHERE risk_factors LIKE '%complaint%'
ORDER BY priority_score DESC
LIMIT 20;
