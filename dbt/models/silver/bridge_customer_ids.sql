-- Bridge table for customer ID mapping across systems
-- This enables tracing a unified customer back to each source system

{{
    config(
        materialized='table',
        unique_key='surrogate_key'
    )
}}

with erp_customers as (
    select
        customer_id as source_customer_id,
        'ERPNext' as source_system,
        customer_name,
        email
    from {{ ref('stg_erp_customers') }}
    where customer_id is not null
),

sf_contacts as (
    select
        contact_id as source_customer_id,
        'Salesforce' as source_system,
        full_name as customer_name,
        email
    from {{ ref('stg_sf_contacts') }}
    where contact_id is not null
),

app_customers as (
    select distinct
        customer_email as source_customer_id,  -- App sessions use email as customer identifier
        'Supabase' as source_system,
        null as customer_name,
        customer_email as email
    from {{ ref('stg_app_sessions') }}
    where customer_email is not null
),

-- Combine all sources
all_customers as (
    select * from erp_customers
    union all
    select * from sf_contacts
    union all
    select * from app_customers
),

-- Create unified ID based on email matching or source ID
-- Priority: ERPNext > Salesforce > Supabase
unified as (
    select
        md5(coalesce(
            lower(trim(email)),
            source_customer_id
        )) as unified_customer_id,
        source_system,
        source_customer_id,
        customer_name,
        email,
        md5(concat(source_system, ':', source_customer_id)) as surrogate_key
    from all_customers
)

select * from unified
