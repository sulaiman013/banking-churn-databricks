-- Staging model for ERPNext customers
-- Actual columns: name, customer_name, customer_type, customer_group, territory, gender, mobile_no, email_id, website, creation, modified, _ingested_at, _source_system

with source as (
    select * from {{ source('bronze', 'erp_customers') }}
),

staged as (
    select
        -- Primary key
        name as customer_id,

        -- Customer attributes
        customer_name,
        customer_type,
        customer_group,
        territory,
        gender,

        -- Contact info
        email_id as email,
        mobile_no as mobile_phone,
        website,

        -- Metadata
        _source_system as source_system,
        _ingested_at as ingested_at,

        -- Audit
        creation as created_at,
        modified as modified_at

    from source
)

select * from staged
