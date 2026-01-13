-- Geographic Dimension
-- Combines territory data from ERPNext with branch data from Google Sheets

{{
    config(
        materialized='table',
        unique_key='geography_key'
    )
}}

with territories as (
    select
        territory_code,
        territory_name
    from {{ ref('stg_erp_territories') }}
),

branches as (
    select
        branch_id,
        branch_name,
        region,
        manager_name,
        staff_count,
        opened_date,
        performance_rating
    from {{ ref('stg_branches') }}
),

-- Create geography dimension combining both sources
geography as (
    select
        md5(coalesce(t.territory_code, b.region, 'UNKNOWN')) as geography_key,

        -- Territory info (ERPNext)
        t.territory_code,
        t.territory_name,

        -- Branch info (Google Sheets)
        b.branch_id,
        b.branch_name,
        b.region,
        b.manager_name,
        b.staff_count,
        b.opened_date,
        b.performance_rating,

        -- Hierarchy
        coalesce(t.territory_name, b.region) as display_name,

        -- Metadata
        current_timestamp() as dbt_updated_at

    from territories t
    full outer join branches b
        on lower(t.territory_name) = lower(b.region)
)

select * from geography
