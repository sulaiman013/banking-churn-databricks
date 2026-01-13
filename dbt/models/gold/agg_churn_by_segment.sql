-- Aggregated Churn Metrics by Segment
-- Pre-computed aggregations for dashboard performance
-- Refreshed with each dbt run

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

-- Churn risk segments
risk_segments as (
    select
        f.unified_customer_id,
        c.territory,
        c.customer_group,
        c.tenure_segment,
        c.is_digitally_active,
        f.churn_risk_score,
        f.engagement_health_score,
        f.total_transactions,
        f.total_transaction_amount,
        -- Segment customers by risk
        case
            when f.churn_risk_score >= 60 then 'High Risk'
            when f.churn_risk_score >= 40 then 'Medium Risk'
            when f.churn_risk_score >= 20 then 'Low Risk'
            else 'Minimal Risk'
        end as risk_segment
    from features f
    join customer_360 c on f.unified_customer_id = c.unified_customer_id
),

-- Aggregate by territory
by_territory as (
    select
        'territory' as dimension,
        territory as dimension_value,
        count(*) as customer_count,
        avg(churn_risk_score) as avg_risk_score,
        avg(engagement_health_score) as avg_engagement_score,
        sum(total_transaction_amount) as total_value,
        count(case when risk_segment = 'High Risk' then 1 end) as high_risk_count,
        count(case when risk_segment = 'Medium Risk' then 1 end) as medium_risk_count,
        count(case when risk_segment = 'Low Risk' then 1 end) as low_risk_count
    from risk_segments
    group by territory
),

-- Aggregate by customer group
by_customer_group as (
    select
        'customer_group' as dimension,
        customer_group as dimension_value,
        count(*) as customer_count,
        avg(churn_risk_score) as avg_risk_score,
        avg(engagement_health_score) as avg_engagement_score,
        sum(total_transaction_amount) as total_value,
        count(case when risk_segment = 'High Risk' then 1 end) as high_risk_count,
        count(case when risk_segment = 'Medium Risk' then 1 end) as medium_risk_count,
        count(case when risk_segment = 'Low Risk' then 1 end) as low_risk_count
    from risk_segments
    group by customer_group
),

-- Aggregate by tenure segment
by_tenure as (
    select
        'tenure_segment' as dimension,
        tenure_segment as dimension_value,
        count(*) as customer_count,
        avg(churn_risk_score) as avg_risk_score,
        avg(engagement_health_score) as avg_engagement_score,
        sum(total_transaction_amount) as total_value,
        count(case when risk_segment = 'High Risk' then 1 end) as high_risk_count,
        count(case when risk_segment = 'Medium Risk' then 1 end) as medium_risk_count,
        count(case when risk_segment = 'Low Risk' then 1 end) as low_risk_count
    from risk_segments
    group by tenure_segment
),

-- Aggregate by digital activity
by_digital_status as (
    select
        'digital_status' as dimension,
        case when is_digitally_active then 'Digitally Active' else 'Not Active' end as dimension_value,
        count(*) as customer_count,
        avg(churn_risk_score) as avg_risk_score,
        avg(engagement_health_score) as avg_engagement_score,
        sum(total_transaction_amount) as total_value,
        count(case when risk_segment = 'High Risk' then 1 end) as high_risk_count,
        count(case when risk_segment = 'Medium Risk' then 1 end) as medium_risk_count,
        count(case when risk_segment = 'Low Risk' then 1 end) as low_risk_count
    from risk_segments
    group by is_digitally_active
),

-- Aggregate by risk segment
by_risk_segment as (
    select
        'risk_segment' as dimension,
        risk_segment as dimension_value,
        count(*) as customer_count,
        avg(churn_risk_score) as avg_risk_score,
        avg(engagement_health_score) as avg_engagement_score,
        sum(total_transaction_amount) as total_value,
        count(case when risk_segment = 'High Risk' then 1 end) as high_risk_count,
        count(case when risk_segment = 'Medium Risk' then 1 end) as medium_risk_count,
        count(case when risk_segment = 'Low Risk' then 1 end) as low_risk_count
    from risk_segments
    group by risk_segment
),

-- Overall summary
overall as (
    select
        'overall' as dimension,
        'All Customers' as dimension_value,
        count(*) as customer_count,
        avg(churn_risk_score) as avg_risk_score,
        avg(engagement_health_score) as avg_engagement_score,
        sum(total_transaction_amount) as total_value,
        count(case when risk_segment = 'High Risk' then 1 end) as high_risk_count,
        count(case when risk_segment = 'Medium Risk' then 1 end) as medium_risk_count,
        count(case when risk_segment = 'Low Risk' then 1 end) as low_risk_count
    from risk_segments
),

-- Union all aggregations
final as (
    select *, current_timestamp() as aggregated_at from by_territory
    union all
    select *, current_timestamp() from by_customer_group
    union all
    select *, current_timestamp() from by_tenure
    union all
    select *, current_timestamp() from by_digital_status
    union all
    select *, current_timestamp() from by_risk_segment
    union all
    select *, current_timestamp() from overall
)

select * from final
