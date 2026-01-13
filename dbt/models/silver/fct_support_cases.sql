-- Support Cases Fact Table
-- Customer complaints and support tickets - key churn indicator

{{
    config(
        materialized='table',
        unique_key='case_key'
    )
}}

with cases as (
    select
        case_id,
        contact_id as sf_contact_id,
        case_subject,
        case_description,
        case_type,
        case_status,
        case_priority,
        case_origin,
        is_closed,
        closed_at,
        created_at
    from {{ ref('stg_sf_cases') }}
),

customers as (
    select
        unified_customer_id,
        sf_contact_id
    from {{ ref('dim_customer_unified') }}
    where sf_contact_id is not null
),

fact as (
    select
        md5(cs.case_id) as case_key,

        -- Foreign keys
        c.unified_customer_id,
        cs.sf_contact_id,

        -- Case identifiers
        cs.case_id,

        -- Case details
        cs.case_subject,
        cs.case_description,
        cs.case_type,
        cs.case_origin,

        -- Status
        cs.case_status,
        cs.case_priority,
        cs.is_closed,

        -- Priority scoring for ML (higher = more likely to churn)
        case cs.case_priority
            when 'High' then 3
            when 'Medium' then 2
            when 'Low' then 1
            else 0
        end as priority_score,

        -- Resolution
        cs.closed_at,
        case
            when cs.is_closed and cs.closed_at is not null
            then datediff(day, cs.created_at, cs.closed_at)
            else null
        end as resolution_days,

        -- Dates
        cs.created_at,
        date(cs.created_at) as case_date,
        year(cs.created_at) as case_year,
        month(cs.created_at) as case_month,

        -- Audit
        current_timestamp() as dbt_updated_at

    from cases cs
    left join customers c
        on cs.sf_contact_id = c.sf_contact_id
)

select * from fact
