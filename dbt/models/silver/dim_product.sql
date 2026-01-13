-- Product Dimension
-- Banking products from ERPNext

{{
    config(
        materialized='table',
        unique_key='product_key'
    )
}}

with products as (
    select
        product_id,
        item_code,
        product_name,
        product_group,
        created_at,
        modified_at
    from {{ ref('stg_erp_products') }}
),

enriched as (
    select
        md5(product_id) as product_key,

        -- Product identifiers
        product_id,
        item_code,
        product_name,

        -- Categorization
        product_group,
        case
            when lower(product_group) like '%loan%' then 'Lending'
            when lower(product_group) like '%deposit%' then 'Deposits'
            when lower(product_group) like '%card%' then 'Cards'
            when lower(product_group) like '%insurance%' then 'Insurance'
            when lower(product_group) like '%invest%' then 'Investment'
            else 'Other Services'
        end as product_category,

        -- Audit
        created_at,
        modified_at,
        current_timestamp() as dbt_updated_at

    from products
)

select * from enriched
