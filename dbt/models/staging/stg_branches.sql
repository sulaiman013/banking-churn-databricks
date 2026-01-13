-- Staging model for Google Sheets branch data
-- Actual columns: branch_id, branch_name, region, manager_name, staff_count, opened_date, performance_rating, _ingested_at, _source_system

with source as (
    select * from {{ source('bronze', 'gs_branches') }}
),

staged as (
    select
        -- Primary key
        branch_id,

        -- Branch attributes
        branch_name,
        region,
        manager_name,
        staff_count,
        opened_date,
        performance_rating,

        -- Metadata
        _source_system as source_system,
        _ingested_at as ingested_at

    from source
)

select * from staged
