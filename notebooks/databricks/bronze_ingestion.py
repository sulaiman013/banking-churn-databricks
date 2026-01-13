# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer Ingestion - All Source Systems
# MAGIC
# MAGIC This notebook extracts data from all 4 source systems and loads into Unity Catalog Bronze layer.
# MAGIC
# MAGIC ## Source Systems
# MAGIC | System | Type | Tables |
# MAGIC |--------|------|--------|
# MAGIC | ERPNext | Core Banking | customers, items, territories, sales_invoices |
# MAGIC | Salesforce | CRM | contacts, cases, tasks |
# MAGIC | Supabase | Digital Channels | app_sessions, app_events |
# MAGIC | Google Sheets | Legacy | branches, customer_notes |
# MAGIC
# MAGIC ## Schedule
# MAGIC - **Frequency**: Daily at 1:00 PM GMT+6 (07:00 UTC)
# MAGIC - **Cluster**: Serverless or smallest available
# MAGIC
# MAGIC ## Unity Catalog Target
# MAGIC - **Catalog**: `bank_proj`
# MAGIC - **Schema**: `bronze`

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration & Secrets Setup
# MAGIC
# MAGIC Store credentials in Databricks Secrets (recommended) or use widgets for testing.

# COMMAND ----------

# Install required packages (run once per cluster)
%pip install simple-salesforce supabase gspread google-auth requests --quiet

# COMMAND ----------

# Restart Python after pip install
dbutils.library.restartPython()

# COMMAND ----------

# Import libraries
import requests
import json
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import current_timestamp, lit

# Salesforce
from simple_salesforce import Salesforce

# Supabase
from supabase import create_client

# Google Sheets
import gspread
from google.oauth2.service_account import Credentials

print(f"Bronze Ingestion Started: {datetime.now()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Databricks Secrets Setup
# MAGIC
# MAGIC Run these commands in Databricks CLI to set up secrets:
# MAGIC
# MAGIC ```bash
# MAGIC # Create secret scope
# MAGIC databricks secrets create-scope bank-churn-secrets
# MAGIC
# MAGIC # ERPNext
# MAGIC databricks secrets put-secret bank-churn-secrets ERPNEXT_API_KEY
# MAGIC databricks secrets put-secret bank-churn-secrets ERPNEXT_API_SECRET
# MAGIC
# MAGIC # Salesforce
# MAGIC databricks secrets put-secret bank-churn-secrets SALESFORCE_USERNAME
# MAGIC databricks secrets put-secret bank-churn-secrets SALESFORCE_PASSWORD
# MAGIC databricks secrets put-secret bank-churn-secrets SALESFORCE_SECURITY_TOKEN
# MAGIC databricks secrets put-secret bank-churn-secrets SALESFORCE_CONSUMER_KEY
# MAGIC databricks secrets put-secret bank-churn-secrets SALESFORCE_CONSUMER_SECRET
# MAGIC
# MAGIC # Supabase
# MAGIC databricks secrets put-secret bank-churn-secrets SUPABASE_URL
# MAGIC databricks secrets put-secret bank-churn-secrets SUPABASE_KEY
# MAGIC
# MAGIC # Google Sheets
# MAGIC databricks secrets put-secret bank-churn-secrets GOOGLE_SHEETS_ID
# MAGIC databricks secrets put-secret bank-churn-secrets GOOGLE_CREDENTIALS_JSON
# MAGIC ```

# COMMAND ----------

# =============================================================================
# CONFIGURATION - Load from Databricks Secrets
# =============================================================================

# Unity Catalog target
CATALOG = "bank_proj"
SCHEMA = "bronze"

# For testing, use widgets. For production, use secrets.
# Uncomment the secrets version for production deployment.

# --- TESTING MODE (Widgets) ---
dbutils.widgets.text("erpnext_url", "https://erpnext-rnm-aly.m.erpnext.com", "ERPNext URL")
dbutils.widgets.text("erpnext_api_key", "", "ERPNext API Key")
dbutils.widgets.text("erpnext_api_secret", "", "ERPNext API Secret")
dbutils.widgets.text("sf_username", "", "Salesforce Username")
dbutils.widgets.text("sf_password", "", "Salesforce Password")
dbutils.widgets.text("sf_token", "", "Salesforce Security Token")
dbutils.widgets.text("sf_consumer_key", "", "Salesforce Consumer Key")
dbutils.widgets.text("sf_consumer_secret", "", "Salesforce Consumer Secret")
dbutils.widgets.text("supabase_url", "", "Supabase URL")
dbutils.widgets.text("supabase_key", "", "Supabase Key")
dbutils.widgets.text("google_sheets_id", "", "Google Sheets ID")

# Load widget values
ERPNEXT_URL = dbutils.widgets.get("erpnext_url")
ERPNEXT_API_KEY = dbutils.widgets.get("erpnext_api_key")
ERPNEXT_API_SECRET = dbutils.widgets.get("erpnext_api_secret")
SF_USERNAME = dbutils.widgets.get("sf_username")
SF_PASSWORD = dbutils.widgets.get("sf_password")
SF_TOKEN = dbutils.widgets.get("sf_token")
SF_CONSUMER_KEY = dbutils.widgets.get("sf_consumer_key")
SF_CONSUMER_SECRET = dbutils.widgets.get("sf_consumer_secret")
SUPABASE_URL = dbutils.widgets.get("supabase_url")
SUPABASE_KEY = dbutils.widgets.get("supabase_key")
GOOGLE_SHEETS_ID = dbutils.widgets.get("google_sheets_id")

# --- PRODUCTION MODE (Secrets) - Uncomment for production ---
# ERPNEXT_URL = "https://erpnext-rnm-aly.m.erpnext.com"
# ERPNEXT_API_KEY = dbutils.secrets.get("bank-churn-secrets", "ERPNEXT_API_KEY")
# ERPNEXT_API_SECRET = dbutils.secrets.get("bank-churn-secrets", "ERPNEXT_API_SECRET")
# SF_USERNAME = dbutils.secrets.get("bank-churn-secrets", "SALESFORCE_USERNAME")
# SF_PASSWORD = dbutils.secrets.get("bank-churn-secrets", "SALESFORCE_PASSWORD")
# SF_TOKEN = dbutils.secrets.get("bank-churn-secrets", "SALESFORCE_SECURITY_TOKEN")
# SF_CONSUMER_KEY = dbutils.secrets.get("bank-churn-secrets", "SALESFORCE_CONSUMER_KEY")
# SF_CONSUMER_SECRET = dbutils.secrets.get("bank-churn-secrets", "SALESFORCE_CONSUMER_SECRET")
# SUPABASE_URL = dbutils.secrets.get("bank-churn-secrets", "SUPABASE_URL")
# SUPABASE_KEY = dbutils.secrets.get("bank-churn-secrets", "SUPABASE_KEY")
# GOOGLE_SHEETS_ID = dbutils.secrets.get("bank-churn-secrets", "GOOGLE_SHEETS_ID")
# GOOGLE_CREDENTIALS_JSON = dbutils.secrets.get("bank-churn-secrets", "GOOGLE_CREDENTIALS_JSON")

print(f"Target: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Helper Functions

# COMMAND ----------

def write_to_bronze(df: pd.DataFrame, table_name: str, source_system: str):
    """
    Write pandas DataFrame to Unity Catalog Bronze layer as Delta table.

    Adds metadata columns:
    - _ingested_at: Timestamp of ingestion
    - _source_system: Name of source system
    """
    if df is None or df.empty:
        print(f"  SKIP: {table_name} - No data")
        return 0

    # Convert to Spark DataFrame
    spark_df = spark.createDataFrame(df)

    # Add metadata columns
    spark_df = spark_df \
        .withColumn("_ingested_at", current_timestamp()) \
        .withColumn("_source_system", lit(source_system))

    # Full table path
    full_table_name = f"{CATALOG}.{SCHEMA}.{table_name}"

    # Write as Delta table (overwrite for full refresh)
    spark_df.write \
        .format("delta") \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(full_table_name)

    row_count = spark_df.count()
    print(f"  OK: {full_table_name} - {row_count} rows")
    return row_count

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. ERPNext Extraction (Core Banking)

# COMMAND ----------

class ERPNextClient:
    """ERPNext REST API Client"""

    def __init__(self, url: str, api_key: str, api_secret: str):
        self.url = url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {api_key}:{api_secret}",
            "Accept": "application/json"
        })

    def get_list(self, doctype: str, fields: list = None, limit: int = 10000) -> list:
        """Fetch list of documents with pagination"""
        all_records = []
        offset = 0
        batch_size = 100

        while True:
            params = {"limit_page_length": batch_size, "limit_start": offset}
            if fields:
                params["fields"] = json.dumps(fields)

            try:
                response = self.session.get(f"{self.url}/api/resource/{doctype}", params=params)
                response.raise_for_status()
                data = response.json().get("data", [])

                if not data:
                    break

                all_records.extend(data)
                offset += batch_size

                if len(data) < batch_size or len(all_records) >= limit:
                    break
            except Exception as e:
                print(f"  ERPNext Error ({doctype}): {e}")
                break

        return all_records


def extract_erpnext():
    """Extract all tables from ERPNext"""
    print("\n" + "=" * 60)
    print("ERPNext Extraction (Core Banking)")
    print("=" * 60)

    if not ERPNEXT_API_KEY or not ERPNEXT_API_SECRET:
        print("  SKIP: ERPNext credentials not configured")
        return {}

    client = ERPNextClient(ERPNEXT_URL, ERPNEXT_API_KEY, ERPNEXT_API_SECRET)
    results = {}

    # Customers
    print("  Extracting customers...")
    customers = client.get_list("Customer", fields=[
        "name", "customer_name", "customer_type", "customer_group",
        "territory", "gender", "mobile_no", "email_id", "website",
        "creation", "modified"
    ])
    results["erp_customers"] = pd.DataFrame(customers) if customers else None

    # Items (Banking Products)
    print("  Extracting items...")
    items = client.get_list("Item", fields=[
        "name", "item_code", "item_name", "item_group",
        "standard_rate", "description", "creation", "modified"
    ])
    results["erp_items"] = pd.DataFrame(items) if items else None

    # Territories (Branches)
    print("  Extracting territories...")
    territories = client.get_list("Territory", fields=[
        "name", "territory_name", "parent_territory", "creation", "modified"
    ])
    results["erp_territories"] = pd.DataFrame(territories) if territories else None

    # Sales Invoices (Transactions)
    print("  Extracting sales invoices...")
    invoices = client.get_list("Sales Invoice", fields=[
        "name", "customer", "customer_name", "posting_date", "due_date",
        "territory", "grand_total", "status", "docstatus", "creation", "modified"
    ])
    results["erp_sales_invoices"] = pd.DataFrame(invoices) if invoices else None

    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Salesforce Extraction (CRM)

# COMMAND ----------

def extract_salesforce():
    """Extract all tables from Salesforce"""
    print("\n" + "=" * 60)
    print("Salesforce Extraction (CRM)")
    print("=" * 60)

    if not SF_USERNAME or not SF_PASSWORD or not SF_TOKEN:
        print("  SKIP: Salesforce credentials not configured")
        return {}

    try:
        sf = Salesforce(
            username=SF_USERNAME,
            password=SF_PASSWORD,
            security_token=SF_TOKEN,
            consumer_key=SF_CONSUMER_KEY if SF_CONSUMER_KEY else None,
            consumer_secret=SF_CONSUMER_SECRET if SF_CONSUMER_SECRET else None,
        )
        print(f"  Connected to: {sf.sf_instance}")
    except Exception as e:
        print(f"  Connection failed: {e}")
        return {}

    results = {}

    # Contacts
    print("  Extracting contacts...")
    try:
        query = """
            SELECT Id, FirstName, LastName, Email, Phone, Description,
                   LeadSource, CreatedDate, LastModifiedDate
            FROM Contact
        """
        contacts = sf.query_all(query)['records']
        # Clean Salesforce metadata
        contacts_clean = [{
            'sf_contact_id': c['Id'],
            'first_name': c['FirstName'],
            'last_name': c['LastName'],
            'email': c['Email'],
            'phone': c['Phone'],
            'description': c['Description'],
            'lead_source': c['LeadSource'],
            'created_date': c['CreatedDate'],
            'modified_date': c['LastModifiedDate'],
        } for c in contacts]
        results["sf_contacts"] = pd.DataFrame(contacts_clean)
    except Exception as e:
        print(f"    Error: {e}")
        results["sf_contacts"] = None

    # Cases (Support Tickets - Key Churn Signal!)
    print("  Extracting cases...")
    try:
        query = """
            SELECT Id, ContactId, Subject, Type, Priority, Status, Origin,
                   Description, CreatedDate, ClosedDate
            FROM Case
        """
        cases = sf.query_all(query)['records']
        cases_clean = [{
            'sf_case_id': c['Id'],
            'sf_contact_id': c['ContactId'],
            'subject': c['Subject'],
            'type': c['Type'],
            'priority': c['Priority'],
            'status': c['Status'],
            'origin': c['Origin'],
            'description': c['Description'],
            'created_date': c['CreatedDate'],
            'closed_date': c['ClosedDate'],
        } for c in cases]
        results["sf_cases"] = pd.DataFrame(cases_clean)
    except Exception as e:
        print(f"    Error: {e}")
        results["sf_cases"] = None

    # Tasks (Interactions)
    print("  Extracting tasks...")
    try:
        query = """
            SELECT Id, WhoId, Subject, Priority, Status, ActivityDate, Description
            FROM Task
        """
        tasks = sf.query_all(query)['records']
        tasks_clean = [{
            'sf_task_id': t['Id'],
            'sf_contact_id': t['WhoId'],
            'subject': t['Subject'],
            'priority': t['Priority'],
            'status': t['Status'],
            'activity_date': t['ActivityDate'],
            'description': t['Description'],
        } for t in tasks]
        results["sf_tasks"] = pd.DataFrame(tasks_clean)
    except Exception as e:
        print(f"    Error: {e}")
        results["sf_tasks"] = None

    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Supabase Extraction (Digital Channels)

# COMMAND ----------

def extract_supabase():
    """Extract all tables from Supabase"""
    print("\n" + "=" * 60)
    print("Supabase Extraction (Digital Channels)")
    print("=" * 60)

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("  SKIP: Supabase credentials not configured")
        return {}

    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print(f"  Connected to: {SUPABASE_URL}")
    except Exception as e:
        print(f"  Connection failed: {e}")
        return {}

    results = {}

    # App Sessions
    print("  Extracting app_sessions...")
    try:
        response = supabase.table("app_sessions").select("*").execute()
        results["sb_app_sessions"] = pd.DataFrame(response.data) if response.data else None
    except Exception as e:
        print(f"    Error: {e}")
        results["sb_app_sessions"] = None

    # App Events (Key Churn Signal!)
    print("  Extracting app_events...")
    try:
        response = supabase.table("app_events").select("*").execute()
        results["sb_app_events"] = pd.DataFrame(response.data) if response.data else None
    except Exception as e:
        print(f"    Error: {e}")
        results["sb_app_events"] = None

    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Google Sheets Extraction (Legacy)

# COMMAND ----------

def extract_google_sheets():
    """Extract all sheets from Google Sheets"""
    print("\n" + "=" * 60)
    print("Google Sheets Extraction (Legacy)")
    print("=" * 60)

    if not GOOGLE_SHEETS_ID:
        print("  SKIP: Google Sheets ID not configured")
        return {}

    try:
        # For production, load credentials from secret
        # credentials_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
        # credentials = Credentials.from_service_account_info(credentials_dict, scopes=[...])

        # Unity Catalog Volume path (recommended over deprecated DBFS)
        credentials_path = "/Volumes/bank_proj/bronze/credentials/banking-churn-poc-5c5c657a7df3.json"

        scopes = [
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/drive.readonly'
        ]

        try:
            credentials = Credentials.from_service_account_file(credentials_path, scopes=scopes)
            client = gspread.authorize(credentials)
            spreadsheet = client.open_by_key(GOOGLE_SHEETS_ID)
            print(f"  Connected to: {spreadsheet.title}")
        except FileNotFoundError:
            print("  SKIP: Google credentials file not found in DBFS")
            print("  Upload credentials JSON to: /dbfs/FileStore/banking-churn-poc-credentials.json")
            return {}

    except Exception as e:
        print(f"  Connection failed: {e}")
        return {}

    results = {}

    # Branches
    print("  Extracting branches...")
    try:
        ws = spreadsheet.worksheet("branches")
        records = ws.get_all_records()
        results["gs_branches"] = pd.DataFrame(records) if records else None
    except Exception as e:
        print(f"    Error: {e}")
        results["gs_branches"] = None

    # Customer Notes (Key Churn Signal!)
    print("  Extracting customer_notes...")
    try:
        ws = spreadsheet.worksheet("customer_notes")
        records = ws.get_all_records()
        results["gs_customer_notes"] = pd.DataFrame(records) if records else None
    except Exception as e:
        print(f"    Error: {e}")
        results["gs_customer_notes"] = None

    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Execute Full Ingestion

# COMMAND ----------

# Main execution
print("=" * 60)
print("BRONZE LAYER INGESTION")
print(f"Started: {datetime.now()}")
print(f"Target: {CATALOG}.{SCHEMA}")
print("=" * 60)

# Track results
ingestion_stats = {
    "source": [],
    "table": [],
    "rows": [],
    "status": []
}

# 1. ERPNext
erp_data = extract_erpnext()
for table_name, df in erp_data.items():
    rows = write_to_bronze(df, table_name, "erpnext")
    ingestion_stats["source"].append("erpnext")
    ingestion_stats["table"].append(table_name)
    ingestion_stats["rows"].append(rows)
    ingestion_stats["status"].append("OK" if rows > 0 else "EMPTY")

# 2. Salesforce
sf_data = extract_salesforce()
for table_name, df in sf_data.items():
    rows = write_to_bronze(df, table_name, "salesforce")
    ingestion_stats["source"].append("salesforce")
    ingestion_stats["table"].append(table_name)
    ingestion_stats["rows"].append(rows)
    ingestion_stats["status"].append("OK" if rows > 0 else "EMPTY")

# 3. Supabase
sb_data = extract_supabase()
for table_name, df in sb_data.items():
    rows = write_to_bronze(df, table_name, "supabase")
    ingestion_stats["source"].append("supabase")
    ingestion_stats["table"].append(table_name)
    ingestion_stats["rows"].append(rows)
    ingestion_stats["status"].append("OK" if rows > 0 else "EMPTY")

# 4. Google Sheets
gs_data = extract_google_sheets()
for table_name, df in gs_data.items():
    rows = write_to_bronze(df, table_name, "google_sheets")
    ingestion_stats["source"].append("google_sheets")
    ingestion_stats["table"].append(table_name)
    ingestion_stats["rows"].append(rows)
    ingestion_stats["status"].append("OK" if rows > 0 else "EMPTY")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Ingestion Summary

# COMMAND ----------

# Display summary
print("\n" + "=" * 60)
print("INGESTION SUMMARY")
print(f"Completed: {datetime.now()}")
print("=" * 60)

stats_df = pd.DataFrame(ingestion_stats)
display(spark.createDataFrame(stats_df))

total_rows = stats_df["rows"].sum()
total_tables = len(stats_df[stats_df["rows"] > 0])

print(f"\nTotal: {total_tables} tables, {total_rows:,} rows ingested")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Verify Bronze Tables

# COMMAND ----------

# MAGIC %sql
# MAGIC -- List all tables in bronze schema
# MAGIC SHOW TABLES IN bank_proj.bronze

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Sample: ERPNext Customers
# MAGIC SELECT * FROM bank_proj.bronze.erp_customers LIMIT 5

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Sample: Salesforce Cases (Churn Signal)
# MAGIC SELECT * FROM bank_proj.bronze.sf_cases LIMIT 5

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Sample: Supabase App Sessions
# MAGIC SELECT * FROM bank_proj.bronze.sb_app_sessions LIMIT 5

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Sample: Google Sheets Customer Notes (Churn Signal)
# MAGIC SELECT * FROM bank_proj.bronze.gs_customer_notes LIMIT 5

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Job Scheduling Instructions
# MAGIC
# MAGIC ### Schedule this notebook to run daily at 1:00 PM GMT+6 (07:00 UTC)
# MAGIC
# MAGIC #### Option 1: Databricks UI
# MAGIC 1. Go to **Workflows** > **Jobs** > **Create Job**
# MAGIC 2. Configure:
# MAGIC    - **Name**: `bronze_ingestion_daily`
# MAGIC    - **Task**: Select this notebook
# MAGIC    - **Cluster**: Serverless or Job Cluster (smallest)
# MAGIC    - **Schedule**:
# MAGIC      - Type: `Scheduled`
# MAGIC      - Cron: `0 0 7 * * ?` (07:00 UTC = 1:00 PM GMT+6)
# MAGIC      - Timezone: `UTC`
# MAGIC 3. Click **Create**
# MAGIC
# MAGIC #### Option 2: Databricks CLI
# MAGIC ```bash
# MAGIC databricks jobs create --json '{
# MAGIC   "name": "bronze_ingestion_daily",
# MAGIC   "tasks": [{
# MAGIC     "task_key": "bronze_ingestion",
# MAGIC     "notebook_task": {
# MAGIC       "notebook_path": "/Workspace/Users/your-email/notebooks/bronze_ingestion"
# MAGIC     },
# MAGIC     "new_cluster": {
# MAGIC       "spark_version": "14.3.x-scala2.12",
# MAGIC       "node_type_id": "Standard_DS3_v2",
# MAGIC       "num_workers": 0
# MAGIC     }
# MAGIC   }],
# MAGIC   "schedule": {
# MAGIC     "quartz_cron_expression": "0 0 7 * * ?",
# MAGIC     "timezone_id": "UTC"
# MAGIC   }
# MAGIC }'
# MAGIC ```
# MAGIC
# MAGIC #### Option 3: Terraform (Infrastructure as Code)
# MAGIC ```hcl
# MAGIC resource "databricks_job" "bronze_ingestion" {
# MAGIC   name = "bronze_ingestion_daily"
# MAGIC
# MAGIC   task {
# MAGIC     task_key = "bronze_ingestion"
# MAGIC     notebook_task {
# MAGIC       notebook_path = "/Workspace/Users/your-email/notebooks/bronze_ingestion"
# MAGIC     }
# MAGIC   }
# MAGIC
# MAGIC   schedule {
# MAGIC     quartz_cron_expression = "0 0 7 * * ?"
# MAGIC     timezone_id            = "UTC"
# MAGIC   }
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC ### Time Zone Reference
# MAGIC | Local Time (GMT+6) | UTC Time | Cron Expression |
# MAGIC |-------------------|----------|-----------------|
# MAGIC | 1:00 PM | 07:00 | `0 0 7 * * ?` |
# MAGIC | 6:00 AM | 00:00 | `0 0 0 * * ?` |
# MAGIC | 12:00 AM (midnight) | 18:00 (prev day) | `0 0 18 * * ?` |

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Production Checklist
# MAGIC
# MAGIC Before deploying to production:
# MAGIC
# MAGIC - [ ] **Secrets**: Move all credentials to Databricks Secrets
# MAGIC - [ ] **Google Credentials**: Upload JSON to DBFS or use Secrets
# MAGIC - [ ] **Cluster**: Configure appropriate cluster size
# MAGIC - [ ] **Alerts**: Set up job failure notifications
# MAGIC - [ ] **Monitoring**: Enable job metrics in Unity Catalog
# MAGIC - [ ] **Retry**: Configure job retry policy (e.g., 2 retries)
# MAGIC
# MAGIC ### Secrets Setup Commands
# MAGIC ```bash
# MAGIC # Create scope
# MAGIC databricks secrets create-scope bank-churn-secrets
# MAGIC
# MAGIC # Add secrets (you'll be prompted for values)
# MAGIC databricks secrets put-secret bank-churn-secrets ERPNEXT_API_KEY
# MAGIC databricks secrets put-secret bank-churn-secrets ERPNEXT_API_SECRET
# MAGIC databricks secrets put-secret bank-churn-secrets SALESFORCE_USERNAME
# MAGIC databricks secrets put-secret bank-churn-secrets SALESFORCE_PASSWORD
# MAGIC databricks secrets put-secret bank-churn-secrets SALESFORCE_SECURITY_TOKEN
# MAGIC databricks secrets put-secret bank-churn-secrets SALESFORCE_CONSUMER_KEY
# MAGIC databricks secrets put-secret bank-churn-secrets SALESFORCE_CONSUMER_SECRET
# MAGIC databricks secrets put-secret bank-churn-secrets SUPABASE_URL
# MAGIC databricks secrets put-secret bank-churn-secrets SUPABASE_KEY
# MAGIC databricks secrets put-secret bank-churn-secrets GOOGLE_SHEETS_ID
# MAGIC databricks secrets put-secret bank-churn-secrets GOOGLE_CREDENTIALS_JSON
# MAGIC ```
