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

-- Aggregate branches by region first to avoid duplicates
branches_by_region as (
    select
        region,
        count(*) as branch_count,
        sum(staff_count) as total_staff,
        min(opened_date) as earliest_branch_date,
        avg(performance_rating) as avg_performance_rating
    from branches
    group by region
),

-- Create geography dimension combining both sources
geography as (
    select
        md5(coalesce(t.territory_code, br.region, 'UNKNOWN')) as geography_key,

        -- Territory info (ERPNext)
        t.territory_code,
        t.territory_name,

        -- Region info (aggregated from branches)
        br.region,
        br.branch_count,
        br.total_staff,
        br.earliest_branch_date,
        br.avg_performance_rating,

        -- Hierarchy
        coalesce(t.territory_name, br.region) as display_name,

        -- Metadata
        current_timestamp() as dbt_updated_at

    from territories t
    full outer join branches_by_region br
        on lower(t.territory_name) = lower(br.region)
)

select * from geography
