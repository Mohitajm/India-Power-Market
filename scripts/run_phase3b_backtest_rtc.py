"""
scripts/run_phase3b_backtest_rtc.py — Architecture v10 RTC
===========================================================
Solar+BESS backtest with Round-the-Clock (RTC) captive contract.

KEY DIFFERENCES from run_phase3b_backtest.py (v9_revised):
  1. Loads config/bess_rtc.yaml and config/phase3b_rtc.yaml.
  2. Uses BESSParamsRTC, TwoStageBESSRTC, evaluate_actuals_rtc.
  3. RTC_committed (scalar) committed at Stage 1 instead of variable captive_da.
  4. Block P&L includes iex_net separately (not merged into captive settlement).
  5. RTC captive penalty tracked and reported.
  6. Results saved to results/phase3b_rtc/daily/.

Usage:
    python scripts/run_phase3b_backtest_rtc.py
    python scripts/run_phase3b_backtest_rtc.py --day 2025-03-15
    python scripts/run_phase3b_backtest_rtc.py --limit 5
    python scripts/run_phase3b_backtest_rtc.py --verbose
"""

import argparse
import json
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.optimizer.bess_params_rtc import BESSParamsRTC
from src.optimizer.two_stage_bess_rtc import (
    TwoStageBESSRTC,
    evaluate_actuals_rtc,
    RESCHEDULE_BLOCKS,
    T_BLOCKS,
)
from src.optimizer.scenario_loader import ScenarioLoader

T_BLOCKS = 96


def run_backtest(args):
    print("=" * 65)
    print("PHASE 3B RTC: SOLAR+BESS BACKTEST — Architecture v10 RTC")
    print("=" * 65)

    # ── Load configs ──────────────────────────────────────────────────────
    bp     = BESSParamsRTC.from_yaml("config/bess_rtc.yaml")
    with open("config/phase3b_rtc.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    SOC_TARGET = bp.soc_terminal_min_mwh   # 40.0 MWh
    reschedule_blocks = config.get("reschedule_blocks", RESCHEDULE_BLOCKS)

    print(f"PCS: {bp.p_max_mw} MW  |  BESS: {bp.e_max_mwh} MWh  "
          f"|  Inverter: {bp.solar_inverter_mw} MW")
    print(f"RTC contract: {bp.rtc_mw} MW  |  Min: {bp.rtc_min_mw} MW  "
          f"|  ±{bp.rtc_tol_pct*100:.0f}% free band")
    print(f"SOD = EOD = {SOC_TARGET} MWh  |  "
          f"SoC band: [{bp.soc_solar_low:.0f}, {bp.soc_solar_high:.0f}] MWh")

    # ── Load scenario data ────────────────────────────────────────────────
    loader = ScenarioLoader(
        dam_path=config["paths"]["scenarios_dam"],
        rtm_path=config["paths"]["scenarios_rtm"],
        actuals_dam_path=config["paths"]["actuals_dam"],
        actuals_rtm_path=config["paths"]["actuals_rtm"],
        solar_da_path=config["paths"]["solar_da_path"],
        solar_nc_path=config["paths"]["solar_nc_path"],
        solar_at_path=config["paths"]["solar_at_path"],
        price_parquet_path=config["paths"].get("price_parquet"),
    )

    optimizer = TwoStageBESSRTC(bp, config)

    # ── RTM q50 forecast ──────────────────────────────────────────────────
    rtm_fp = Path("Data/Predictions/rtm_quantiles_backtest_recalibrated.parquet")
    if not rtm_fp.exists():
        rtm_fp = Path("Data/Predictions/rtm_quantiles_backtest.parquet")
    rtm_df = pd.read_parquet(rtm_fp)
    if rtm_df.index.name and "delivery" in str(rtm_df.index.name):
        rtm_df = rtm_df.reset_index()
    if "target_block" not in rtm_df.columns and "delivery_start_ist" in rtm_df.columns:
        ts = pd.to_datetime(rtm_df["delivery_start_ist"])
        rtm_df["target_block"] = ts.dt.hour * 4 + ts.dt.minute // 15 + 1
    if "target_date" in rtm_df.columns:
        rtm_df["target_date"] = rtm_df["target_date"].astype(str)

    # ── Output directories ────────────────────────────────────────────────
    results_dir = Path(config["paths"]["results_dir"])
    daily_dir   = results_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    # ── Date selection ────────────────────────────────────────────────────
    dates = loader.common_dates
    if getattr(args, "day", None):
        dates = [args.day]
    elif getattr(args, "limit", None):
        dates = dates[:args.limit]

    print(f"\nRunning {len(dates)} days ...\n")

    backtest_results = []

    for i, date in enumerate(dates):
        # Hard daily SOD/EOD — no chaining
        bp.soc_initial_mwh       = SOC_TARGET
        bp.soc_terminal_min_mwh  = SOC_TARGET

        # ── Load data ──────────────────────────────────────────────────────
        day  = loader.get_day_scenarios(date, n_scenarios=config["n_scenarios"])
        sol  = loader.get_day_solar(date)

        day_rtm = rtm_df[rtm_df["target_date"] == date].sort_values("target_block")
        rtm_q50 = (
            day_rtm["q50"].values[:T_BLOCKS].astype(float)
            if len(day_rtm) >= T_BLOCKS
            else np.full(T_BLOCKS, 4000.0)
        )

        print(f"[{i+1}/{len(dates)}] {date} ...", end=" ", flush=True)

        # ── Stage 1: MILP ─────────────────────────────────────────────────
        res1 = optimizer.solve(day["dam"], day["rtm"], sol["solar_da"])
        if res1["status"] != "Optimal":
            print(f"Stage 1 FAILED: {res1['status']}")
            continue

        rtc_val = res1["RTC_committed"]

        # ── Actuals settlement (Stage 2A/2B inside) ────────────────────────
        ev = evaluate_actuals_rtc(
            params=bp,
            stage1_result=res1,
            dam_actual=day["dam_actual"],
            rtm_actual=day["rtm_actual"],
            rtm_q50=rtm_q50,
            solar_da=sol["solar_da"],
            solar_nc=sol["solar_nc"],
            solar_at=sol["solar_at"],
            reschedule_blocks=reschedule_blocks,
            verbose=getattr(args, "verbose", False),
        )

        net_rev     = ev["net_revenue"]
        captive_net = ev["captive_net_total"]
        iex_net     = ev["iex_net_total"]
        cap_penalty = ev["captive_penalty_total"]
        eod_soc     = float(ev["soc_path"][-1])

        print(
            f"Net: ₹{net_rev:,.0f}  |  Cap: ₹{captive_net:,.0f}  "
            f"|  IEX: ₹{iex_net:,.0f}  |  Penalty: ₹{cap_penalty:,.0f}  "
            f"|  SoC_end: {eod_soc:.2f} MWh  |  RTC: {rtc_val:.2f} MW"
        )

        # ── Serialise DSM results ─────────────────────────────────────────
        dsm_serial = [
            {k: float(v) if isinstance(v, (float, np.floating)) else v
             for k, v in d.items()}
            for d in ev["dsm_results"]
        ]

        # ── Save daily JSON ───────────────────────────────────────────────
        daily_out = {
            "date":                     date,
            "status":                   res1["status"],
            "architecture":             "v10_rtc",
            # Hardware
            "p_max_mw":                 bp.p_max_mw,
            "solar_inverter_mw":        bp.solar_inverter_mw,
            "solar_dc_mwp":             bp.solar_capacity_mwp,
            "e_max_mwh":                bp.e_max_mwh,
            "soc_initial_mwh":          SOC_TARGET,
            "soc_terminal_actual_mwh":  eod_soc,
            # RTC contract
            "rtc_committed_mw":         rtc_val,
            "rtc_min_mw":               bp.rtc_min_mw,
            "rtc_tol_pct":              bp.rtc_tol_pct,
            # Revenue summary
            "expected_revenue":         res1["expected_revenue"],
            "net_revenue":              net_rev,
            "captive_net_revenue":      captive_net,
            "iex_net_revenue":          iex_net,
            "captive_penalty":          cap_penalty,
            # Stage 1 schedules
            "x_c":                      res1["x_c"],
            "x_d":                      res1["x_d"],
            "s_c_da":                   res1["s_c_da"],
            "s_cd_da":                  res1["s_cd_da"],
            "c_d_da":                   res1["c_d_da"],
            "captive_da":               res1["captive_da"],
            "dam_net":                  res1["dam_net"],
            "schedule_da":              res1["schedule_da"],
            "setpoint_da":              res1["setpoint_da"],
            "solar_band_mask":          res1.get("solar_band_mask", []),
            # Stage 2 RT routing
            "y_c":                      ev["y_c"].tolist(),
            "y_d":                      ev["y_d"].tolist(),
            "s_c_rt":                   ev["s_c_rt"].tolist(),
            "s_cd_rt":                  ev["s_cd_rt"].tolist(),
            "c_d_rt":                   ev["c_d_rt"].tolist(),
            # Actuals
            "s_c_actual":               ev["s_c_actual"].tolist(),
            "s_cd_actual":              ev["s_cd_actual"].tolist(),
            "c_d_actual":               ev["c_d_actual"].tolist(),
            "captive_actual":           ev["captive_actual"].tolist(),
            "captive_committed":        ev["captive_committed"].tolist(),
            "setpoint_rt":              ev["setpoint_rt"].tolist(),
            "schedule_rt":              ev["schedule_rt"].tolist(),
            "soc_realized":             ev["soc_path"].tolist(),
            # Prices and solar
            "actual_dam_prices":        day["dam_actual"].tolist(),
            "actual_rtm_prices":        day["rtm_actual"].tolist(),
            "solar_da":                 sol["solar_da"].tolist(),
            "solar_at":                 sol["solar_at"].tolist(),
            "rtm_q50_used":             rtm_q50.tolist(),
            # P&L arrays
            "block_captive_net":        ev["block_captive_net"].tolist(),
            "block_captive_penalty":    ev["block_captive_penalty"].tolist(),
            "block_iex_net":            ev["block_iex_net"].tolist(),
            "block_degradation":        ev["block_degradation"].tolist(),
            "block_net":                ev["block_net"].tolist(),
            "no_bess_dsm":              ev["no_bess_dsm"].tolist(),
            "no_bess_revenue":          ev["no_bess_revenue"].tolist(),
            # RTC notice tracking
            "rtc_notice_issued":        ev["rtc_notice_issued"].tolist(),
            # DSM details
            "dsm_results":              dsm_serial,
            "scenarios":                res1["scenarios"],
        }

        with open(daily_dir / f"result_{date}.json", "w") as f:
            json.dump(daily_out, f, indent=2,
                      default=lambda x: float(x) if isinstance(
                          x, (np.floating, np.integer)) else x)

        backtest_results.append({
            "target_date":      date,
            "net_revenue":      net_rev,
            "captive_net":      captive_net,
            "iex_net":          iex_net,
            "captive_penalty":  cap_penalty,
            "rtc_committed_mw": rtc_val,
            "soc_terminal":     eod_soc,
        })

    # ── Save summary CSV ──────────────────────────────────────────────────
    df = pd.DataFrame(backtest_results)
    df.to_csv(results_dir / "backtest_results.csv", index=False)

    if len(df):
        print(f"\n{'='*65}")
        print(f"RTC BACKTEST COMPLETE — {len(df)} days")
        print(f"  Total Net Revenue:         ₹{df['net_revenue'].sum():,.0f}")
        print(f"  Total Captive Net:          ₹{df['captive_net'].sum():,.0f}")
        print(f"  Total IEX Net:              ₹{df['iex_net'].sum():,.0f}")
        print(f"  Total Captive Penalties:    ₹{df['captive_penalty'].sum():,.0f}")
        print(f"  Avg Daily Net:              ₹{df['net_revenue'].mean():,.0f}")
        print(f"  Avg RTC Committed:          {df['rtc_committed_mw'].mean():.3f} MW")
        print(f"  Results → {results_dir}")
        print(f"{'='*65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Solar+BESS RTC backtest (Architecture v10 RTC)"
    )
    parser.add_argument("--day",     default=None, help="Single date YYYY-MM-DD")
    parser.add_argument("--limit",   type=int, default=None, help="Limit N days")
    parser.add_argument("--verbose", action="store_true")
    run_backtest(parser.parse_args())
