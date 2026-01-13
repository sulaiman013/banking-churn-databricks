-- Customer Features for ML Churn Prediction
-- One row per customer with engineered features for the ML model
-- All features are numeric or encoded for model consumption

{{
    config(
        materialized='table',
        unique_key='unified_customer_id'
    )
}}

with customer_360 as (
    select * from {{ ref('customer_360') }}
),

features as (
    select
        -- Customer ID (for joining predictions back)
        unified_customer_id,
        erp_customer_id,
        email,

        -- ===================
        -- DEMOGRAPHIC FEATURES
        -- ===================

        -- Gender encoding (0=Unknown, 1=Male, 2=Female)
        case
            when lower(gender) = 'male' then 1
            when lower(gender) = 'female' then 2
            else 0
        end as gender_encoded,

        -- Customer type encoding
        case
            when lower(customer_type) = 'individual' then 1
            when lower(customer_type) = 'company' then 2
            else 0
        end as customer_type_encoded,

        -- Territory/Region encoding (for geographic risk)
        case
            when territory in ('UAE', 'Saudi Arabia', 'Kuwait') then 1  -- GCC
            when territory in ('USA', 'UK', 'Canada') then 2  -- Western
            when territory in ('India', 'Pakistan') then 3  -- South Asia
            else 0
        end as region_encoded,

        -- ===================
        -- TENURE FEATURES
        -- ===================

        tenure_days,
        tenure_months,

        -- Tenure risk (newer customers more likely to churn)
        case
            when tenure_days <= 90 then 4  -- Very High Risk
            when tenure_days <= 180 then 3  -- High Risk
            when tenure_days <= 365 then 2  -- Medium Risk
            else 1  -- Low Risk
        end as tenure_risk_score,

        -- ===================
        -- TRANSACTION FEATURES
        -- ===================

        total_transactions,
        total_transaction_amount,
        avg_transaction_amount,

        -- Transaction frequency (per month)
        case
            when tenure_months > 0
            then round(total_transactions / tenure_months, 2)
            else 0
        end as transaction_frequency,

        -- Recent activity ratio
        transactions_last_30d,
        amount_last_30d,

        -- Transaction decline indicator
        case
            when total_transactions > 0
            then round(transactions_last_30d * 12.0 / total_transactions * tenure_months, 2)
            else 0
        end as transaction_trend_ratio,

        -- Days since last transaction (inactivity signal)
        coalesce(days_since_last_transaction, 999) as days_since_last_transaction,

        -- Inactivity risk score
        case
            when days_since_last_transaction is null then 5
            when days_since_last_transaction > 90 then 4
            when days_since_last_transaction > 60 then 3
            when days_since_last_transaction > 30 then 2
            else 1
        end as inactivity_risk_score,

        -- ===================
        -- SUPPORT/COMPLAINT FEATURES
        -- ===================

        total_support_cases,
        open_support_cases,
        high_priority_cases,
        avg_case_priority,

        -- Complaint rate (cases per month of tenure)
        case
            when tenure_months > 0
            then round(total_support_cases / tenure_months, 2)
            else 0
        end as complaint_rate,

        -- Has open complaint (strong churn signal)
        case when open_support_cases > 0 then 1 else 0 end as has_open_complaint,

        -- High priority complaint flag
        case when high_priority_cases > 0 then 1 else 0 end as has_high_priority_complaint,

        -- Resolution satisfaction (lower = worse)
        coalesce(avg_resolution_days, 0) as avg_resolution_days,

        -- ===================
        -- DIGITAL ENGAGEMENT FEATURES
        -- ===================

        total_digital_sessions,
        total_digital_events,
        avg_engagement_score,
        engagement_days,

        -- Digital adoption flag
        case when is_digitally_active then 1 else 0 end as is_digitally_active,

        -- App usage frequency
        case
            when tenure_months > 0
            then round(total_digital_sessions / tenure_months, 2)
            else 0
        end as app_usage_frequency,

        -- Recent digital engagement
        sessions_last_30d,
        events_last_30d,

        -- Digital engagement decline
        coalesce(days_since_last_engagement, 999) as days_since_last_engagement,

        -- ===================
        -- RELATIONSHIP FEATURES
        -- ===================

        rm_note_count,

        -- Has RM attention (positive indicator)
        case when rm_note_count > 0 then 1 else 0 end as has_rm_attention,

        -- ===================
        -- COMPOSITE RISK SCORES
        -- ===================

        -- Overall engagement score (0-100)
        least(100, greatest(0,
            (case when total_transactions > 5 then 20 else total_transactions * 4 end) +
            (case when total_digital_sessions > 10 then 30 else total_digital_sessions * 3 end) +
            (case when transactions_last_30d > 0 then 25 else 0 end) +
            (case when sessions_last_30d > 0 then 25 else 0 end)
        )) as engagement_health_score,

        -- Churn risk composite (higher = more likely to churn)
        (
            -- Tenure risk
            case
                when tenure_days <= 90 then 25
                when tenure_days <= 180 then 15
                when tenure_days <= 365 then 10
                else 5
            end +
            -- Inactivity risk
            case
                when days_since_last_transaction is null then 30
                when days_since_last_transaction > 90 then 25
                when days_since_last_transaction > 60 then 15
                when days_since_last_transaction > 30 then 10
                else 0
            end +
            -- Complaint risk
            case
                when open_support_cases > 0 then 25
                when high_priority_cases > 0 then 15
                when total_support_cases > 3 then 10
                else 0
            end +
            -- Digital disengagement risk
            case
                when not is_digitally_active and total_digital_sessions > 0 then 20
                when total_digital_sessions = 0 then 10
                else 0
            end
        ) as churn_risk_score,

        -- Audit
        current_timestamp() as feature_generated_at

    from customer_360
)

select * from features
