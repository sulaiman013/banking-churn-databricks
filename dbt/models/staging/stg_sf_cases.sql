-- Staging model for Salesforce cases (complaints/support tickets)
-- Actual columns: sf_case_id, sf_contact_id, subject, description, type, status, priority, origin, closed_date, created_date, _ingested_at, _source_system

with source as (
    select * from {{ source('bronze', 'sf_cases') }}
),

staged as (
    select
        -- Primary key
        sf_case_id as case_id,

        -- Foreign keys
        sf_contact_id as contact_id,

        -- Case details
        subject as case_subject,
        description as case_description,
        type as case_type,

        -- Status and priority
        status as case_status,
        priority as case_priority,
        origin as case_origin,

        -- Resolution (derive is_closed from closed_date)
        case when closed_date is not null then true else false end as is_closed,
        closed_date as closed_at,

        -- Dates
        created_date as created_at,

        -- Metadata
        _source_system as source_system,
        _ingested_at as ingested_at

    from source
)

select * from staged
