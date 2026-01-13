-- Staging model for ERPNext sales invoices (transactions)
-- Actual columns: name, customer, territory, posting_date, due_date, grand_total, status, creation, modified, _ingested_at, _source_system

with source as (
    select * from {{ source('bronze', 'erp_sales_invoices') }}
),

staged as (
    select
        -- Primary key
        name as transaction_id,

        -- Foreign keys
        customer as customer_id,
        territory,

        -- Transaction details
        posting_date as transaction_date,
        due_date,
        grand_total as transaction_amount,

        -- Status
        status as transaction_status,

        -- Metadata
        _source_system as source_system,
        _ingested_at as ingested_at,

        -- Audit
        creation as created_at,
        modified as modified_at

    from source
)

select * from staged
