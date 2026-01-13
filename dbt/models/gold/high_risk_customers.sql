-- High Risk Customers for Retention Team
-- Actionable list of customers requiring immediate attention
-- Includes risk factors and recommended actions

{{
    config(
        materialized='table'
    )
}}

with features as (
    select * from {{ ref('customer_features') }}
),

customer_360 as (
    select * from {{ ref('customer_360') }}
),

high_risk as (
    select
        -- Customer identification
        c.unified_customer_id,
        c.erp_customer_id,
        c.customer_name,
        c.email,
        c.phone,
        c.territory,
        c.customer_group,

        -- Risk metrics
        f.churn_risk_score,
        f.engagement_health_score,

        -- Risk category
        case
            when f.churn_risk_score >= 60 then 'Critical'
            when f.churn_risk_score >= 40 then 'High'
            else 'Medium'
        end as risk_category,

        -- Customer value (prioritize high-value customers)
        c.total_transaction_amount as lifetime_value,
        c.avg_transaction_amount,

        -- Risk factors (for retention team context)
        concat_ws(', ',
            case when f.has_open_complaint = 1 then 'Open complaint' end,
            case when f.has_high_priority_complaint = 1 then 'High priority complaint' end,
            case when f.days_since_last_transaction > 60 then 'Inactive 60+ days' end,
            case when f.days_since_last_transaction > 90 then 'Inactive 90+ days' end,
            case when f.tenure_days <= 90 then 'New customer' end,
            case when f.sessions_last_30d = 0 and c.total_digital_sessions > 0 then 'Stopped using app' end
        ) as risk_factors,

        -- Key dates
        c.last_transaction_date,
        c.last_engagement_date,
        c.last_case_date,
        f.days_since_last_transaction,
        f.days_since_last_engagement,

        -- Recommended action
        case
            when f.has_open_complaint = 1 then 'Resolve open complaint immediately'
            when f.days_since_last_transaction > 90 then 'Proactive outreach - win back campaign'
            when f.days_since_last_transaction > 60 then 'Check-in call from RM'
            when f.tenure_days <= 90 then 'Onboarding follow-up'
            when f.sessions_last_30d = 0 and c.total_digital_sessions > 0 then 'Re-engage with app promotion'
            else 'Schedule relationship review'
        end as recommended_action,

        -- Priority score (combines risk and value)
        round(
            f.churn_risk_score * 0.6 +
            least(40, c.total_transaction_amount / 10000) -- Cap value contribution at 40
        , 1) as priority_score,

        -- Contact info for retention team
        c.rm_note_count,
        c.rm_last_note_date,

        -- Audit
        current_timestamp() as generated_at

    from features f
    join customer_360 c on f.unified_customer_id = c.unified_customer_id
    where f.churn_risk_score >= 40  -- Only medium risk and above
)

select *
from high_risk
order by priority_score desc, lifetime_value desc
