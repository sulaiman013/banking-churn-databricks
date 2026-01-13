-- Staging model for ERPNext territories
-- Actual columns: name, creation, modified, _ingested_at, _source_system

with source as (
    select * from {{ source('bronze', 'erp_territories') }}
),

staged as (
    select
        -- Primary key
        name as territory_code,

        -- Territory attributes (name serves as both code and display name)
        name as territory_name,

        -- Metadata
        _source_system as source_system,
        _ingested_at as ingested_at,

        -- Audit
        creation as created_at,
        modified as modified_at

    from source
)

select * from staged
