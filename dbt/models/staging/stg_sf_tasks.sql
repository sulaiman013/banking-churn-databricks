-- Staging model for Salesforce tasks/activities
-- Actual columns: sf_task_id, sf_contact_id, subject, description, status, priority, activity_date, _ingested_at, _source_system

with source as (
    select * from {{ source('bronze', 'sf_tasks') }}
),

staged as (
    select
        -- Primary key
        sf_task_id as task_id,

        -- Foreign keys
        sf_contact_id as contact_id,

        -- Task details
        subject as task_subject,
        description as task_description,

        -- Status
        status as task_status,
        priority as task_priority,

        -- Dates
        activity_date,

        -- Metadata
        _source_system as source_system,
        _ingested_at as ingested_at

    from source
)

select * from staged
