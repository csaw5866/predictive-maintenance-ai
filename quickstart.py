#!/usr/bin/env python3
"""
Quick start script to run the complete pipeline and serve predictions
"""

import subprocess
import time
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a shell command and report status"""
    print(f"\n{'='*70}")
    print(f"🔄 {description}")
    print(f"{'='*70}")
    print(f"$ {cmd}\n")

    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"✅ {description} completed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}\n")
        return False


def main():
    """Run the complete pipeline"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  🏭 PREDICTIVE MAINTENANCE AI - QUICK START                 ║
    ║                                                              ║
    ║  This script will:                                          ║
    ║  1. Install dependencies                                   ║
    ║  2. Run the training pipeline                              ║
    ║  3. Start the API server                                   ║
    ║  4. Launch the dashboard                                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Step 1: Install dependencies
    if not run_command(
        "pip install -q -r requirements.txt",
        "Installing dependencies"
    ):
        print("⚠️  Continuing despite installation warnings...")

    # Step 2: Create directories
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    print("✅ Created required directories")

    # Step 3: Run training pipeline
    if not run_command(
        "python -m pipelines.complete_pipeline",
        "Running training pipeline"
    ):
        print("⚠️  Training pipeline encountered issues")

    # Step 4: Instructions for running services
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  🎉 SETUP COMPLETE!                                         ║
    ║                                                              ║
    ║  To start the services:                                     ║
    ║                                                              ║
    ║  Option 1: Using Docker (Recommended)                       ║
    ║  ────────────────────────────────────────────────────────   ║
    ║  $ docker compose up                                        ║
    ║                                                              ║
    ║  Option 2: Run services individually                        ║
    ║  ────────────────────────────────────────────────────────   ║
    ║  Terminal 1 - MLflow:                                       ║
    ║  $ mlflow server --backend-store-uri ./mlruns               ║
    ║                                                              ║
    ║  Terminal 2 - API:                                          ║
    ║  $ python -m uvicorn api.main:app --port 8000               ║
    ║                                                              ║
    ║  Terminal 3 - Dashboard:                                    ║
    ║  $ streamlit run dashboard/app.py                           ║
    ║                                                              ║
    ║  Access Points:                                             ║
    ║  ────────────────────────────────────────────────────────   ║
    ║  🌐 Dashboard:  http://localhost:8501                       ║
    ║  📡 API Docs:   http://localhost:8000/docs                  ║
    ║  📊 MLflow UI:  http://localhost:5000                       ║
    ║                                                              ║
    ║  Test the API:                                              ║
    ║  ────────────────────────────────────────────────────────   ║
    ║  $ curl http://localhost:8000/health                        ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏸️  Setup interrupted by user")
        sys.exit(0)
