-- Staging model for Google Sheets customer notes
-- Actual columns: note_id, customer_email, branch_id, note_date, note_text, note_type, created_by, _ingested_at, _source_system

with source as (
    select * from {{ source('bronze', 'gs_customer_notes') }}
),

staged as (
    select
        -- Primary key
        note_id,

        -- Foreign keys
        customer_email,
        branch_id,

        -- Note content
        note_date,
        note_text,
        note_type,

        -- Author
        created_by,

        -- Metadata
        _source_system as source_system,
        _ingested_at as ingested_at

    from source
)

select * from staged
