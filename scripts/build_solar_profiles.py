"""
scripts/build_solar_profiles.py
================================
One-time script to generate solar generation parquet files for the backtest.

Run from project root ONCE before running run_phase3b_backtest.py:
    python scripts/build_solar_profiles.py --config config/backtest_config.yaml

Outputs (written to Data/Solar/):
    solar_da.parquet     : DA forecast   — one row per date, columns b01..b96 (MW)
    solar_nc.parquet     : NC nowcast    — one row per (date, block), columns nc_00..nc_11 (MW)
    solar_actuals.parquet: Actual        — one row per date, columns b01..b96 (MW)

Data sources:
    DA      : Open-Meteo Historical Forecast API (NWP model archive)
    NC      : Derived from DA with 5-10% random variance in 3-hour window
    Actuals : Open-Meteo Historical Weather API (ERA5 reanalysis)

Both APIs are completely free with no rate limits for historical data.
A bulk prefetch is performed first to minimise the number of API calls.
"""

import argparse
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.solar_loader import SolarLoader, _date_range

BLOCKS_PER_DAY = 96
NC_WINDOW      = 12
OUTPUT_DIR     = Path("Data/Solar")


def build_solar_profiles(config_path: str = "config/backtest_config.yaml",
                          start_date: str = None,
                          end_date:   str = None,
                          force:      bool = False) -> None:
    """
    Generate all three solar parquets for the backtest date range.

    Parameters
    ----------
    config_path : str   Path to backtest_config.yaml
    start_date  : str   Override start date (YYYY-MM-DD). Defaults to config backtest start.
    end_date    : str   Override end date   (YYYY-MM-DD). Defaults to config backtest end.
    force       : bool  If True, overwrite existing parquets. Default False (skip if exists).
    """
    print("=" * 60)
    print("BUILD SOLAR PROFILES")
    print("=" * 60)

    # ── Read config ───────────────────────────────────────────────────────────
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if start_date is None:
        start_date = config["splits"]["backtest"]["start"]
    if end_date is None:
        end_date = config["splits"]["backtest"]["end"]

    print(f"Date range : {start_date} → {end_date}")
    dates = _date_range(start_date, end_date)
    print(f"Total days : {len(dates)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    da_path = OUTPUT_DIR / "solar_da.parquet"
    nc_path = OUTPUT_DIR / "solar_nc.parquet"
    at_path = OUTPUT_DIR / "solar_actuals.parquet"

    if not force:
        if da_path.exists() and nc_path.exists() and at_path.exists():
            print("\nAll three solar parquets already exist.")
            print("Use --force to regenerate. Exiting.")
            return

    # ── Initialise loader and bulk prefetch ──────────────────────────────────
    print(f"\nInitialising SolarLoader (Jamnagar 22.47N 70.07E, 35 MWp) ...")
    loader = SolarLoader(cache_dir="Data/Solar/raw", seed=42)

    print("Prefetching DA forecast data from Open-Meteo Historical Forecast API ...")
    loader.prefetch_range(start_date, end_date)

    # ── Build DA parquet ──────────────────────────────────────────────────────
    print(f"\n[1/3] Building solar_da.parquet ...")
    da_rows = []
    for i, date in enumerate(dates):
        da_96 = loader.get_solar_da(date)   # (96,) MW
        row   = {"target_date": date}
        for b in range(BLOCKS_PER_DAY):
            row[f"b{b+1:02d}"] = float(da_96[b])
        da_rows.append(row)
        if (i + 1) % 10 == 0 or (i + 1) == len(dates):
            print(f"  {i+1}/{len(dates)} dates processed (DA)")

    da_df = pd.DataFrame(da_rows)
    da_df.to_parquet(da_path, index=False)
    print(f"  Saved: {da_path}  shape={da_df.shape}")

    # ── Build actuals parquet ─────────────────────────────────────────────────
    print(f"\n[2/3] Building solar_actuals.parquet ...")
    at_rows = []
    for i, date in enumerate(dates):
        at_96 = loader.get_solar_at(date)   # (96,) MW
        row   = {"target_date": date}
        for b in range(BLOCKS_PER_DAY):
            row[f"b{b+1:02d}"] = float(at_96[b])
        at_rows.append(row)
        if (i + 1) % 10 == 0 or (i + 1) == len(dates):
            print(f"  {i+1}/{len(dates)} dates processed (Actuals)")

    at_df = pd.DataFrame(at_rows)
    at_df.to_parquet(at_path, index=False)
    print(f"  Saved: {at_path}  shape={at_df.shape}")

    # ── Build NC parquet ──────────────────────────────────────────────────────
    print(f"\n[3/3] Building solar_nc.parquet ...")
    nc_rows = []
    for i, date in enumerate(dates):
        nc_matrix = loader.get_solar_nc(date)   # (96, 12) MW
        for b in range(BLOCKS_PER_DAY):
            row = {"target_date": date, "block_index": b}
            for k in range(NC_WINDOW):
                row[f"nc_{k:02d}"] = float(nc_matrix[b, k])
            nc_rows.append(row)
        if (i + 1) % 10 == 0 or (i + 1) == len(dates):
            print(f"  {i+1}/{len(dates)} dates processed (NC)")

    nc_df = pd.DataFrame(nc_rows)
    nc_df.to_parquet(nc_path, index=False)
    print(f"  Saved: {nc_path}  shape={nc_df.shape}")

    # ── Validation summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    block_cols = [f"b{b+1:02d}" for b in range(BLOCKS_PER_DAY)]

    for name, df in [("DA", da_df), ("Actuals", at_df)]:
        vals = df[block_cols].values.flatten()
        print(f"\n{name}:")
        print(f"  Shape        : {df.shape}")
        print(f"  Min MW       : {vals.min():.3f}")
        print(f"  Max MW       : {vals.max():.3f}")
        print(f"  Mean MW      : {vals.mean():.3f}")
        print(f"  Negative vals: {(vals < 0).sum()}")
        print(f"  Zero blocks  : {(vals == 0).sum()} / {len(vals)} "
              f"({100*(vals==0).mean():.1f}% — includes night)")

        # Day-level sanity check: peak generation should be between 10 and 35 MW
        day_peaks = df[block_cols].max(axis=1)
        bad_days  = (day_peaks < 1.0).sum()
        print(f"  Days with peak < 1 MW: {bad_days} "
              f"(expected ~0 for Jamnagar)")

    nc_vals = nc_df[[f"nc_{k:02d}" for k in range(NC_WINDOW)]].values.flatten()
    print(f"\nNC:")
    print(f"  Shape        : {nc_df.shape}")
    print(f"  Min MW       : {nc_vals.min():.3f}")
    print(f"  Max MW       : {nc_vals.max():.3f}")
    print(f"  Mean MW      : {nc_vals.mean():.3f}")
    print(f"  Negative vals: {(nc_vals < 0).sum()}")

    print("\nAll solar profile files built successfully.")
    print(f"Run the backtest with:  python scripts/run_phase3b_backtest.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build solar DA/NC/actuals parquets for the backtest."
    )
    parser.add_argument(
        "--config", default="config/backtest_config.yaml",
        help="Path to backtest_config.yaml"
    )
    parser.add_argument(
        "--start", default=None,
        help="Override start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", default=None,
        help="Override end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force regeneration even if parquets already exist"
    )
    args = parser.parse_args()

    t0 = datetime.now()
    build_solar_profiles(
        config_path = args.config,
        start_date  = args.start,
        end_date    = args.end,
        force       = args.force,
    )
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nTotal time: {elapsed:.1f} seconds")
