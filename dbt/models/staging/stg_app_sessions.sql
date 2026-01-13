-- Staging model for Supabase app sessions
-- Actual columns: id, user_email, session_start, session_end, device_type, app_version, platform, created_at, _ingested_at, _source_system

with source as (
    select * from {{ source('bronze', 'sb_app_sessions') }}
),

staged as (
    select
        -- Primary key
        id as session_id,

        -- Foreign key (email-based matching to customers)
        user_email as customer_email,

        -- Session timing
        session_start as session_start_at,
        session_end as session_end_at,

        -- Device info
        device_type,
        app_version,
        platform,

        -- Metadata
        _source_system as source_system,
        _ingested_at as ingested_at,

        -- Audit
        created_at

    from source
)

select * from staged
