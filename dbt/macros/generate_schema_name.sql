-- Custom macro to override dbt's default schema naming behavior
-- Without this: default_schema + custom_schema = "bronze_staging"
-- With this: custom_schema = "staging" (exact name)

{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- set default_schema = target.schema -%}

    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
