-- Unified Customer Dimension
-- Combines customer data from ERPNext, Salesforce, and Supabase
-- Uses email as the primary matching key

{{
    config(
        materialized='table',
        unique_key='unified_customer_id'
    )
}}

with erp_customers as (
    select
        customer_id as erp_customer_id,
        customer_name,
        email,
        mobile_phone,
        customer_group,
        customer_type,
        territory,
        gender,
        created_at as erp_created_at
    from {{ ref('stg_erp_customers') }}
),

sf_contacts as (
    select
        contact_id as sf_contact_id,
        full_name as sf_name,
        email as sf_email,
        phone as sf_phone,
        lead_source,
        created_at as sf_created_at
    from {{ ref('stg_sf_contacts') }}
),

app_customers as (
    select
        customer_email as app_customer_email,
        min(session_start_at) as first_app_session,
        max(session_end_at) as last_app_session,
        count(*) as total_app_sessions
    from {{ ref('stg_app_sessions') }}
    where customer_email is not null
    group by customer_email
),

-- Customer notes aggregation
notes_agg as (
    select
        customer_email,
        count(*) as note_count,
        max(note_date) as last_note_date
    from {{ ref('stg_customer_notes') }}
    group by customer_email
),

-- Join all sources using email as primary key
-- ERPNext is the master system
unified as (
    select
        -- Unified key (based on email or ERP ID)
        md5(coalesce(
            lower(trim(erp.email)),
            erp.erp_customer_id
        )) as unified_customer_id,

        -- Source system IDs
        erp.erp_customer_id,
        sf.sf_contact_id,
        app.app_customer_email,

        -- Customer name (prefer ERPNext)
        coalesce(erp.customer_name, sf.sf_name) as customer_name,
        erp.customer_name as erp_name,
        sf.sf_name,

        -- Contact info (prefer ERPNext, fallback to SF)
        coalesce(erp.email, sf.sf_email) as email,
        coalesce(erp.mobile_phone, sf.sf_phone) as phone,

        -- Demographics from ERPNext
        erp.gender,
        erp.customer_type,

        -- Segmentation from ERPNext
        erp.customer_group,
        erp.territory,

        -- Lead source from Salesforce
        sf.lead_source,

        -- App engagement metrics
        coalesce(app.total_app_sessions, 0) as total_app_sessions,
        app.first_app_session,
        app.last_app_session,

        -- Digital activity flag
        case
            when app.last_app_session >= dateadd(day, -90, current_date()) then true
            else false
        end as is_digitally_active,

        -- Relationship manager notes
        coalesce(notes.note_count, 0) as note_count,
        notes.last_note_date,

        -- Tenure calculation
        datediff(day, coalesce(erp.erp_created_at, sf.sf_created_at), current_date()) as tenure_days,

        -- Source tracking
        case
            when erp.erp_customer_id is not null then 'ERPNext'
            when sf.sf_contact_id is not null then 'Salesforce'
            else 'Unknown'
        end as primary_source,

        -- Metadata
        current_timestamp() as dbt_updated_at

    from erp_customers erp
    left join sf_contacts sf
        on lower(trim(erp.email)) = lower(trim(sf.sf_email))
    left join app_customers app
        on lower(trim(erp.email)) = lower(trim(app.app_customer_email))
    left join notes_agg notes
        on lower(trim(erp.email)) = lower(trim(notes.customer_email))
)

select * from unified
where unified_customer_id is not null
