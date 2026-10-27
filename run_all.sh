#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python scripts/run_forecast.py
python -m pytest -q tests
python scripts/run_backtest.py
python scripts/make_charts.py
