-- Staging model for Salesforce contacts
-- Actual columns: sf_contact_id, first_name, last_name, email, phone, description, lead_source, created_date, modified_date, _ingested_at, _source_system

with source as (
    select * from {{ source('bronze', 'sf_contacts') }}
),

staged as (
    select
        -- Primary key
        sf_contact_id as contact_id,

        -- Name fields
        first_name,
        last_name,
        concat(coalesce(first_name, ''), ' ', coalesce(last_name, '')) as full_name,

        -- Contact info
        email,
        phone,

        -- Additional info
        description,
        lead_source,

        -- Dates
        created_date as created_at,
        modified_date as modified_at,

        -- Metadata
        _source_system as source_system,
        _ingested_at as ingested_at

    from source
)

select * from staged
