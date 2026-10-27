"""Produce out-of-sample day-ahead forecasts, walk-forward, for the test period."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from caes_bt.data import load_prices  
from caes_bt.forecast import walk_forward  

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "data", "France.csv"))
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-01-19")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "outputs"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    hourly, imp = walk_forward(load_prices(a.csv), a.start, a.end)
    hourly.to_csv(os.path.join(a.outdir, "forecasts.csv"))
    imp.rename("split_count").to_csv(os.path.join(a.outdir, "feature_importance.csv"))
    print(f"wrote {len(hourly)} hourly forecasts ({hourly['forecast'].isna().sum()} missing)")
