# ============================================================
# orchestrator.py
# Master Pipeline Orchestrator
# Runs the full ETL pipeline in order with a single command:
#   python etl/orchestrator.py
# ============================================================

import os
import sys
import uuid
from datetime import datetime

# Add etl directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db_connection import execute_sql_file
from load_staging import load_staging
from load_bronze import load_bronze
from load_silver import load_silver
from run_gold_views import run_gold_views


SQL_DIR = os.path.join(os.path.dirname(__file__), "..", "sql")

# SQL setup scripts to run (in order) if schemas don't exist
SETUP_SCRIPTS = [
    "01_CreateDatabase.sql",
    "06_Logging.sql",
    "02_StagingLayer.sql",
    "03_BronzeLayer.sql",
    "04_SilverLayer.sql",
]


def run_setup():
    """Run DDL scripts to create schemas and tables if needed."""
    print("\n" + "=" * 60)
    print("🔧 STEP 0: Database Setup — Creating schemas & tables")
    print("=" * 60)

    for script in SETUP_SCRIPTS:
        path = os.path.join(SQL_DIR, script)
        if os.path.exists(path):
            print(f"   📄 Executing: {script}")
            execute_sql_file(path)
        else:
            print(f"   ⚠️  Skipping (not found): {script}")

    print("✅ Database setup complete.\n")


def run_pipeline():
    """Execute the full ETL pipeline."""
    batch_id = str(uuid.uuid4())
    pipeline_start = datetime.now()

    print("\n" + "=" * 60)
    print("🫀 HEART DISEASE DATA WAREHOUSE — ETL PIPELINE")
    print("=" * 60)
    print(f"🔑 Batch ID:  {batch_id}")
    print(f"🕐 Started:   {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = []

    # ── Step 0: Setup ───────────────────────────────────────
    run_setup()

    # ── Step 1: CSV → Staging ───────────────────────────────
    print("\n" + "-" * 60)
    print("📥 STEP 1: Loading CSV → Staging Layer")
    print("-" * 60)
    r1 = load_staging(batch_id)
    results.append(r1)
    if r1["status"] != "SUCCESS":
        print(f"❌ Pipeline failed at Step 1: {r1.get('error', 'Unknown error')}")
        return

    # ── Step 2: Staging → Bronze ────────────────────────────
    print("\n" + "-" * 60)
    print("📦 STEP 2: Copying Staging → Bronze Layer")
    print("-" * 60)
    r2 = load_bronze(batch_id)
    results.append(r2)
    if r2["status"] != "SUCCESS":
        print(f"❌ Pipeline failed at Step 2: {r2.get('error', 'Unknown error')}")
        return

    # ── Step 3: Bronze → Silver ─────────────────────────────
    print("\n" + "-" * 60)
    print("🧹 STEP 3: Cleaning Bronze → Silver Layer")
    print("-" * 60)
    r3 = load_silver(batch_id)
    results.append(r3)
    if r3["status"] != "SUCCESS":
        print(f"❌ Pipeline failed at Step 3: {r3.get('error', 'Unknown error')}")
        return

    # ── Step 4: Deploy Gold Views ───────────────────────────
    print("\n" + "-" * 60)
    print("🏆 STEP 4: Deploying Gold Layer Star Schema Views")
    print("-" * 60)
    r4 = run_gold_views(batch_id)
    results.append(r4)

    # ── Final Report ────────────────────────────────────────
    pipeline_end = datetime.now()
    total_duration = (pipeline_end - pipeline_start).total_seconds()

    print("\n" + "=" * 60)
    print("📊 PIPELINE EXECUTION REPORT")
    print("=" * 60)
    print(f"{'Step':<25} {'Status':<10} {'Rows':<12} {'Time (s)':<10}")
    print("-" * 60)
    for r in results:
        rows = r.get("rows_inserted", r.get("views_created", "-"))
        if isinstance(rows, list):
            rows = f"{len(rows)} views"
        else:
            rows = f"{rows:,}" if isinstance(rows, int) else rows
        dur = r.get("duration_seconds", 0)
        print(f"{r['step']:<25} {r['status']:<10} {rows:<12} {dur:<10.1f}")
    print("-" * 60)
    print(f"{'TOTAL':<25} {'':10} {'':12} {total_duration:<10.1f}")
    print("=" * 60)
    print(f"\n🎉 Pipeline completed successfully in {total_duration:.1f} seconds!")
    print(f"   Batch ID: {batch_id}")
    print(f"   Check ops.PipelineLog for detailed audit trail.\n")


if __name__ == "__main__":
    run_pipeline()
