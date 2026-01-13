-- Transaction Fact Table
-- Clean transaction data linked to unified customers

{{
    config(
        materialized='table',
        unique_key='transaction_key'
    )
}}

with transactions as (
    select
        transaction_id,
        customer_id as erp_customer_id,
        territory,
        transaction_date,
        due_date,
        transaction_amount,
        transaction_status,
        created_at
    from {{ ref('stg_erp_transactions') }}
),

customers as (
    select
        unified_customer_id,
        erp_customer_id
    from {{ ref('dim_customer_unified') }}
),

fact as (
    select
        md5(t.transaction_id) as transaction_key,

        -- Foreign keys
        c.unified_customer_id,
        t.erp_customer_id,

        -- Transaction identifiers
        t.transaction_id,
        t.territory,

        -- Date (for partitioning and filtering)
        t.transaction_date,
        t.due_date,
        year(t.transaction_date) as transaction_year,
        month(t.transaction_date) as transaction_month,
        dayofweek(t.transaction_date) as transaction_day_of_week,

        -- Amounts
        t.transaction_amount,

        -- Status
        t.transaction_status,

        -- Audit
        t.created_at,
        current_timestamp() as dbt_updated_at

    from transactions t
    left join customers c
        on t.erp_customer_id = c.erp_customer_id
)

select * from fact
