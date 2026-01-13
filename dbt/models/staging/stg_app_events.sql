-- Staging model for Supabase app events
-- Actual columns: id, session_id, event_type, event_timestamp, event_data, created_at, _ingested_at, _source_system

with source as (
    select * from {{ source('bronze', 'sb_app_events') }}
),

staged as (
    select
        -- Primary key
        id as event_id,

        -- Foreign key
        session_id,

        -- Event details
        event_type,
        event_timestamp as event_at,

        -- Event data (JSON)
        event_data,

        -- Metadata
        _source_system as source_system,
        _ingested_at as ingested_at,

        -- Audit
        created_at

    from source
)

select * from staged
