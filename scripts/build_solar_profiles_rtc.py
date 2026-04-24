"""
scripts/build_solar_profiles_rtc.py
=====================================
One-time script to generate solar parquet files for the RTC backtest.

Run from project root BEFORE run_phase3b_backtest_rtc.py:

    python scripts/build_solar_profiles_rtc.py --start 2025-08-01 --end 2025-08-07

Outputs (written to Data/Solar/rtc/):
    solar_da_rtc.parquet      — DA forecast,   one row per date, columns b01..b96 (MW)
    solar_actuals_rtc.parquet — ERA5 actuals,  one row per date, columns b01..b96 (MW)
    solar_nc_rtc.parquet      — NC nowcast,    one row per (date, block_index),
                                               columns nc_00..nc_11 (MW)

Plant: 25.4 MWp DC / 16.4 MW AC inverter / PR=0.78 at Jamnagar (22.47N, 70.07E)
Peak AC output = min(25.4 × 0.78, 16.4) = min(19.8, 16.4) = 16.4 MW

Data sources (both free, no API key):
    DA forecast : Open-Meteo Historical Forecast API
    Actuals     : Open-Meteo Historical Weather API (ERA5 reanalysis)

You can also pass --config to read start/end dates from phase3b_rtc.yaml.
"""

import argparse
import sys
import yaml
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.solar_loader_rtc import build_solar_parquets_rtc


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build RTC solar DA/NC/actuals parquets (25.4 MWp / 16.4 MW)")
    ap.add_argument(
        "--start", type=str, default=None,
        help="Start date YYYY-MM-DD  (required unless --config is given)")
    ap.add_argument(
        "--end", type=str, default=None,
        help="End date   YYYY-MM-DD  (required unless --config is given)")
    ap.add_argument(
        "--config", type=str, default=None,
        help="Path to phase3b_rtc.yaml — reads backtest_start / backtest_end keys")
    ap.add_argument(
        "--out-dir", type=str, default="Data/Solar/rtc",
        help="Output directory for parquets (default: Data/Solar/rtc)")
    ap.add_argument(
        "--cache-dir", type=str, default="Data/Solar/raw",
        help="Raw API cache directory (default: Data/Solar/raw)")
    ap.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for NC noise (default: 42)")
    ap.add_argument(
        "--force", action="store_true",
        help="Overwrite existing parquets")
    return ap.parse_args()


def main():
    args = parse_args()

    start = args.start
    end   = args.end

    # Read dates from config if not provided on CLI
    if (start is None or end is None) and args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        # Support both backtest_config.yaml style and phase3b_rtc.yaml style
        if "splits" in cfg:
            start = start or cfg["splits"]["backtest"]["start"]
            end   = end   or cfg["splits"]["backtest"]["end"]
        elif "backtest_start" in cfg:
            start = start or cfg["backtest_start"]
            end   = end   or cfg["backtest_end"]
        else:
            # Fall back: look for any key containing "start" / "end"
            for k, v in cfg.items():
                if "start" in k.lower() and start is None:
                    start = v
                if "end" in k.lower() and end is None:
                    end = v

    if start is None or end is None:
        print("ERROR: Provide --start and --end dates, or --config with date keys.")
        print("Example:")
        print("  python scripts/build_solar_profiles_rtc.py "
              "--start 2025-08-01 --end 2025-08-07")
        sys.exit(1)

    t0 = datetime.now()
    build_solar_parquets_rtc(
        start_date = start,
        end_date   = end,
        out_dir    = args.out_dir,
        cache_dir  = args.cache_dir,
        seed       = args.seed,
        force      = args.force,
    )
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nTotal time: {elapsed:.1f} s")
    print()
    print("Next step:")
    print("  Update config/phase3b_rtc.yaml solar paths to point to Data/Solar/rtc/")
    print("  Then run:  python scripts/run_phase3b_backtest_rtc.py")


if __name__ == "__main__":
    main()
