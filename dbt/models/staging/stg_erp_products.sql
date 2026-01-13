-- Staging model for ERPNext items/products
-- Actual columns: name, item_code, item_name, item_group, creation, modified, _ingested_at, _source_system

with source as (
    select * from {{ source('bronze', 'erp_items') }}
),

staged as (
    select
        -- Primary key
        name as product_id,
        item_code,

        -- Product attributes
        item_name as product_name,
        item_group as product_group,

        -- Metadata
        _source_system as source_system,
        _ingested_at as ingested_at,

        -- Audit
        creation as created_at,
        modified as modified_at

    from source
)

select * from staged
