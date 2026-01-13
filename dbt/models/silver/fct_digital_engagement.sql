-- Digital Engagement Fact Table
-- Daily aggregated app usage metrics per customer

{{
    config(
        materialized='table',
        unique_key='engagement_key'
    )
}}

with sessions as (
    select
        session_id,
        customer_email,
        date(session_start_at) as engagement_date,
        -- Calculate session duration in seconds using timestampdiff
        timestampdiff(SECOND, to_timestamp(session_start_at), to_timestamp(session_end_at)) as session_duration_seconds,
        device_type
    from {{ ref('stg_app_sessions') }}
    where customer_email is not null
),

events as (
    select
        session_id,
        event_type,
        event_at
    from {{ ref('stg_app_events') }}
),

-- Aggregate sessions per customer per day
session_metrics as (
    select
        customer_email,
        engagement_date,
        count(distinct session_id) as session_count,
        sum(session_duration_seconds) as total_session_duration_seconds,
        avg(session_duration_seconds) as avg_session_duration_seconds,
        count(distinct device_type) as device_types_used
    from sessions
    group by customer_email, engagement_date
),

-- Aggregate events per session, then join back
event_metrics as (
    select
        s.customer_email,
        s.engagement_date,
        count(e.event_type) as event_count,
        count(distinct e.event_type) as unique_event_types
    from sessions s
    left join events e on s.session_id = e.session_id
    group by s.customer_email, s.engagement_date
),

-- Get unified customer mapping
customers as (
    select
        unified_customer_id,
        email,
        app_customer_email
    from {{ ref('dim_customer_unified') }}
),

fact as (
    select
        md5(concat(
            coalesce(c.unified_customer_id, sm.customer_email),
            ':',
            sm.engagement_date
        )) as engagement_key,

        -- Foreign keys
        c.unified_customer_id,
        sm.customer_email,

        -- Date
        sm.engagement_date,
        year(sm.engagement_date) as engagement_year,
        month(sm.engagement_date) as engagement_month,
        dayofweek(sm.engagement_date) as engagement_day_of_week,

        -- Session metrics
        sm.session_count,
        sm.total_session_duration_seconds,
        sm.avg_session_duration_seconds,
        sm.device_types_used,

        -- Event metrics
        coalesce(em.event_count, 0) as event_count,
        coalesce(em.unique_event_types, 0) as unique_event_types,

        -- Engagement score (simple weighted metric)
        (sm.session_count * 10) +
        (coalesce(sm.total_session_duration_seconds, 0) / 60) +
        (coalesce(em.event_count, 0) * 2) as engagement_score,

        -- Audit
        current_timestamp() as dbt_updated_at

    from session_metrics sm
    left join event_metrics em
        on sm.customer_email = em.customer_email
        and sm.engagement_date = em.engagement_date
    left join customers c
        on lower(trim(sm.customer_email)) = lower(trim(c.email))
        or lower(trim(sm.customer_email)) = lower(trim(c.app_customer_email))
)

select * from fact
