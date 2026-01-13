-- Customer 360 View
-- Complete unified view of each customer with all behavioral metrics
-- This is the "single source of truth" for customer analytics

{{
    config(
        materialized='table',
        unique_key='unified_customer_id'
    )
}}

with customers as (
    select * from {{ ref('dim_customer_unified') }}
),

-- Transaction aggregations per customer
transaction_metrics as (
    select
        unified_customer_id,
        count(*) as total_transactions,
        sum(transaction_amount) as total_transaction_amount,
        avg(transaction_amount) as avg_transaction_amount,
        min(transaction_date) as first_transaction_date,
        max(transaction_date) as last_transaction_date,
        count(case when transaction_status = 'Completed' then 1 end) as completed_transactions,
        count(case when transaction_status = 'Pending' then 1 end) as pending_transactions,
        count(case when transaction_status = 'Failed' then 1 end) as failed_transactions
    from {{ ref('fct_transactions') }}
    where unified_customer_id is not null
    group by unified_customer_id
),

-- Support case aggregations per customer
support_metrics as (
    select
        unified_customer_id,
        count(*) as total_cases,
        count(case when is_closed = false then 1 end) as open_cases,
        sum(priority_score) as total_priority_score,
        avg(priority_score) as avg_priority_score,
        avg(resolution_days) as avg_resolution_days,
        min(created_at) as first_case_date,
        max(created_at) as last_case_date,
        count(case when case_priority = 'High' then 1 end) as high_priority_cases
    from {{ ref('fct_support_cases') }}
    where unified_customer_id is not null
    group by unified_customer_id
),

-- Digital engagement aggregations per customer
engagement_metrics as (
    select
        unified_customer_id,
        sum(session_count) as total_sessions,
        sum(total_session_duration_seconds) as total_session_duration_seconds,
        avg(avg_session_duration_seconds) as avg_session_duration_seconds,
        sum(event_count) as total_events,
        avg(engagement_score) as avg_engagement_score,
        max(engagement_score) as max_engagement_score,
        min(engagement_date) as first_engagement_date,
        max(engagement_date) as last_engagement_date,
        count(distinct engagement_date) as engagement_days
    from {{ ref('fct_digital_engagement') }}
    where unified_customer_id is not null
    group by unified_customer_id
),

-- Recent activity (last 30 days)
recent_transactions as (
    select
        unified_customer_id,
        count(*) as transactions_last_30d,
        sum(transaction_amount) as amount_last_30d
    from {{ ref('fct_transactions') }}
    where unified_customer_id is not null
      and transaction_date >= dateadd(day, -30, current_date())
    group by unified_customer_id
),

recent_engagement as (
    select
        unified_customer_id,
        sum(session_count) as sessions_last_30d,
        sum(event_count) as events_last_30d
    from {{ ref('fct_digital_engagement') }}
    where unified_customer_id is not null
      and engagement_date >= dateadd(day, -30, current_date())
    group by unified_customer_id
),

-- Final 360 view
customer_360 as (
    select
        -- Customer identifiers
        c.unified_customer_id,
        c.erp_customer_id,
        c.sf_contact_id,
        c.app_customer_email,

        -- Demographics
        c.customer_name,
        c.email,
        c.phone,
        c.gender,
        c.customer_type,
        c.customer_group,
        c.territory,
        c.lead_source,

        -- Tenure
        c.tenure_days,
        round(c.tenure_days / 30.0, 1) as tenure_months,
        case
            when c.tenure_days <= 90 then 'New'
            when c.tenure_days <= 365 then 'Established'
            when c.tenure_days <= 730 then 'Loyal'
            else 'Long-term'
        end as tenure_segment,

        -- Transaction metrics (lifetime)
        coalesce(t.total_transactions, 0) as total_transactions,
        coalesce(t.total_transaction_amount, 0) as total_transaction_amount,
        coalesce(t.avg_transaction_amount, 0) as avg_transaction_amount,
        t.first_transaction_date,
        t.last_transaction_date,
        coalesce(t.completed_transactions, 0) as completed_transactions,
        coalesce(t.failed_transactions, 0) as failed_transactions,

        -- Transaction metrics (recent)
        coalesce(rt.transactions_last_30d, 0) as transactions_last_30d,
        coalesce(rt.amount_last_30d, 0) as amount_last_30d,

        -- Support metrics
        coalesce(s.total_cases, 0) as total_support_cases,
        coalesce(s.open_cases, 0) as open_support_cases,
        coalesce(s.high_priority_cases, 0) as high_priority_cases,
        coalesce(s.avg_priority_score, 0) as avg_case_priority,
        s.avg_resolution_days,
        s.first_case_date,
        s.last_case_date,

        -- Digital engagement (lifetime)
        c.total_app_sessions,
        c.is_digitally_active,
        coalesce(e.total_sessions, 0) as total_digital_sessions,
        coalesce(e.total_events, 0) as total_digital_events,
        coalesce(e.avg_engagement_score, 0) as avg_engagement_score,
        coalesce(e.engagement_days, 0) as engagement_days,
        e.first_engagement_date,
        e.last_engagement_date,

        -- Digital engagement (recent)
        coalesce(re.sessions_last_30d, 0) as sessions_last_30d,
        coalesce(re.events_last_30d, 0) as events_last_30d,

        -- Relationship manager notes
        c.note_count as rm_note_count,
        c.last_note_date as rm_last_note_date,

        -- Days since last activity
        datediff(day, t.last_transaction_date, current_date()) as days_since_last_transaction,
        datediff(day, e.last_engagement_date, current_date()) as days_since_last_engagement,
        datediff(day, s.last_case_date, current_date()) as days_since_last_case,

        -- Source tracking
        c.primary_source,

        -- Audit
        current_timestamp() as dbt_updated_at

    from customers c
    left join transaction_metrics t on c.unified_customer_id = t.unified_customer_id
    left join support_metrics s on c.unified_customer_id = s.unified_customer_id
    left join engagement_metrics e on c.unified_customer_id = e.unified_customer_id
    left join recent_transactions rt on c.unified_customer_id = rt.unified_customer_id
    left join recent_engagement re on c.unified_customer_id = re.unified_customer_id
)

select * from customer_360
