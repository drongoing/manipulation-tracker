#!/usr/bin/env python3
"""
Quick launcher for the Manipulation Tracker Dashboard.

Usage:
    python launch_dashboard.py
"""

import sys
from pathlib import Path

# Add dashboard to path
dashboard_dir = Path(__file__).parent / "dashboard"
sys.path.insert(0, str(dashboard_dir))

# Import and run app
from app import app, data

if __name__ == "__main__":
    print("=" * 80)
    print("MANIPULATION TRACKER DASHBOARD")
    print("=" * 80)
    print(f"Loaded: {len(data.analyses)} conversations")
    print(f"Models: {', '.join(data.get_models())}")
    print(f"Data directory: {data.results_dir}")
    print("=" * 80)
    print()
    print("Starting dashboard server...")
    print("Open your browser to: http://127.0.0.1:8050")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 80)

    app.run(debug=True, host='127.0.0.1', port=8050)
