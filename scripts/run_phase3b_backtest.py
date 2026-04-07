"""
scripts/run_phase3b_backtest.py
================================
Phase 3B Solar+BESS Backtest — 96-block (15-min), causal Stage 2 evaluation.

STAGE EXECUTION ORDER:
  For each day:
    Stage 1 : solve stochastic LP → x_c, x_d, s_c_da, s_cd_da, c_d_da, curtail_da
    For each block B (0..95):
      if B in {34, 42, 50, 58}:
        Stage 2B : reschedule_captive()  → revised s_c_rt, s_cd_rt, c_d_rt
        Stage 2A : solve_stage2a_block() → y_c[B], y_d[B]
      else:
        Stage 2A : solve_stage2a_block() → y_c[B], y_d[B]
      Settle block B at ACTUAL prices + ACTUAL solar

NEW vs original run_phase3b_backtest.py:
  - TwoStageBESS.solve() now takes solar_da as 3rd argument
  - evaluate_actuals_solar() replaces evaluate_actuals_causal()
  - ScenarioLoader constructed with solar_da_path, solar_nc_path, solar_at_path
  - Results include captive_revenue, dsm_mwh, solar curtailment
  - SoC chaining and continuation value logic unchanged
"""

import pandas as pd
import numpy as np
import yaml
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.optimizer.bess_params     import BESSParams
from src.optimizer.scenario_loader import ScenarioLoader
from src.optimizer.two_stage_bess  import TwoStageBESS, evaluate_actuals_solar
from src.optimizer.costs           import CostModel

T_BLOCKS = 96
DT       = 0.25


def run_backtest(args):
    print("=" * 60)
    print("PHASE 3B: SOLAR+BESS TWO-STAGE STOCHASTIC BACKTEST")
    print("Stage 2: Stage 2B first at {34,42,50,58}, then Stage 2A")
    print("=" * 60)

    bess_params = BESSParams.from_yaml("config/bess.yaml")
    with open("config/phase3b.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    reschedule_blocks = config.get("reschedule_blocks", [34, 42, 50, 58])
    print(f"BESS: {bess_params.p_max_mw} MW / {bess_params.e_max_mwh} MWh")
    print(f"Solar: {bess_params.solar_capacity_mwp} MWp")
    print(f"PPA rate: Rs {bess_params.ppa_rate_rs_mwh}/MWh")
    print(f"Reschedule blocks: {reschedule_blocks}")

    # ── Loader with solar paths ───────────────────────────────────────────────
    loader = ScenarioLoader(
        dam_path          = config["paths"]["scenarios_dam"],
        rtm_path          = config["paths"]["scenarios_rtm"],
        actuals_dam_path  = config["paths"]["actuals_dam"],
        actuals_rtm_path  = config["paths"]["actuals_rtm"],
        solar_da_path     = config["paths"]["solar_da_path"],
        solar_nc_path     = config["paths"]["solar_nc_path"],
        solar_at_path     = config["paths"]["solar_at_path"],
    )

    optimizer = TwoStageBESS(bess_params, config)

    cost_model = None
    if Path("config/costs_config.yaml").exists():
        cost_model = CostModel.from_yaml("config/costs_config.yaml")
        print("Loaded CostModel from config/costs_config.yaml")

    # ── Load RTM q50 forecasts for Stage 2A ──────────────────────────────────
    rtm_forecast_path = Path("Data/Predictions/rtm_quantiles_backtest_recalibrated.parquet")
    if not rtm_forecast_path.exists():
        rtm_forecast_path = Path("Data/Predictions/rtm_quantiles_backtest.parquet")
    print(f"Loading RTM q50 forecasts from {rtm_forecast_path.name}...")
    rtm_forecasts_df = pd.read_parquet(rtm_forecast_path)

    if rtm_forecasts_df.index.name in ("delivery_start_ist",) or (
        hasattr(rtm_forecasts_df.index, "dtype")
        and str(rtm_forecasts_df.index.dtype).startswith("datetime")
    ):
        rtm_forecasts_df = rtm_forecasts_df.reset_index()

    if "target_block" not in rtm_forecasts_df.columns:
        if "delivery_start_ist" in rtm_forecasts_df.columns:
            ts = pd.to_datetime(rtm_forecasts_df["delivery_start_ist"])
            rtm_forecasts_df["target_block"] = ts.dt.hour * 4 + ts.dt.minute // 15 + 1
        elif "target_hour" in rtm_forecasts_df.columns:
            rtm_forecasts_df["target_block"] = rtm_forecasts_df["target_hour"] * 4 + 1

    if "target_date" in rtm_forecasts_df.columns:
        rtm_forecasts_df["target_date"] = rtm_forecasts_df["target_date"].astype(str)

    results_dir       = Path(config["paths"]["results_dir"])
    daily_results_dir = results_dir / "daily"
    daily_results_dir.mkdir(parents=True, exist_ok=True)

    dates = loader.common_dates
    if getattr(args, "day", None):
        dates = [args.day]
    elif getattr(args, "limit", None):
        dates = dates[:args.limit]

    verbose = getattr(args, "verbose", False)
    print(f"Running Solar+BESS backtest for {len(dates)} days...")

    backtest_results = []
    prev_soc = bess_params.soc_initial_mwh

    for i, date in enumerate(dates):
        bess_params.soc_initial_mwh = prev_soc

        # Continuation value (soft terminal)
        if bess_params.soc_terminal_mode == "soft" and i < len(dates) - 1:
            next_data = loader.get_day_scenarios(dates[i + 1], n_scenarios=20)
            spread_s  = np.max(next_data["dam"], axis=1) - np.min(next_data["dam"], axis=1)
            bess_params.soc_terminal_value_rs_mwh = max(0.0, (
                np.mean(spread_s) * bess_params.eta_charge * bess_params.eta_discharge
                - bess_params.iex_fee_rs_mwh * 2
                - bess_params.degradation_cost_rs_mwh
                - 135.0 * 2
            ))
        else:
            bess_params.soc_terminal_value_rs_mwh = 0.0

        print(f"[{i+1}/{len(dates)}] {date} (SoC₀={prev_soc:.2f} MWh)...",
              end=" ", flush=True)

        # ── Load price scenarios ──────────────────────────────────────────────
        day_data = loader.get_day_scenarios(date, n_scenarios=config["n_scenarios"])
        if day_data["dam"].shape[1] != T_BLOCKS:
            print(f"SKIP — scenario has {day_data['dam'].shape[1]} blocks")
            continue

        # ── Load solar profiles ───────────────────────────────────────────────
        try:
            solar_data = loader.get_day_solar(date)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"SKIP — solar data error: {e}")
            continue

        solar_da = solar_data["solar_da"]   # (96,) MW
        solar_nc = solar_data["solar_nc"]   # (96, 12) MW
        solar_at = solar_data["solar_at"]   # (96,) MW

        # ── RTM q50 forecast for Stage 2A ────────────────────────────────────
        day_rtm_fcst = (
            rtm_forecasts_df[rtm_forecasts_df["target_date"] == date]
            .sort_values("target_block")
        )
        if len(day_rtm_fcst) >= T_BLOCKS:
            rtm_q50 = day_rtm_fcst["q50"].values[:T_BLOCKS]
        elif len(day_rtm_fcst) == 24:
            rtm_q50 = np.repeat(day_rtm_fcst["q50"].values, 4)
        else:
            print(f"WARNING: No RTM q50 for {date} — using actuals (NOT causal).")
            rtm_q50 = day_data["rtm_actual"]

        # ── Stage 1 ───────────────────────────────────────────────────────────
        res1 = optimizer.solve(
            dam_scenarios = day_data["dam"],
            rtm_scenarios = day_data["rtm"],
            solar_da      = solar_da,
        )
        if res1["status"] != "Optimal":
            print(f"FAILED Stage 1 ({res1['status']})")
            continue

        # ── Stage 2: orchestrated loop (2B first, then 2A) ───────────────────
        eval_res = evaluate_actuals_solar(
            params           = bess_params,
            stage1_result    = res1,
            dam_actual       = day_data["dam_actual"],
            rtm_actual       = day_data["rtm_actual"],
            rtm_q50          = rtm_q50,
            solar_da         = solar_da,
            solar_nc         = solar_nc,
            solar_at         = solar_at,
            reschedule_blocks = reschedule_blocks,
            verbose          = verbose,
        )

        realized_rev = eval_res["revenue"]
        net_rev      = eval_res["net_revenue"]
        expected_rev = res1["expected_revenue"]
        captive_rev  = float(np.sum(eval_res["block_rev_captive"]))
        dsm_mwh      = eval_res["total_dsm_mwh"]

        print(f"Exp: Rs{expected_rev:,.0f} | "
              f"Realized: Rs{realized_rev:,.0f} | "
              f"Captive: Rs{captive_rev:,.0f} | "
              f"Net: Rs{net_rev:,.0f} | "
              f"DSM: {dsm_mwh:.3f} MWh")

        # ── Cost model breakdown ──────────────────────────────────────────────
        cost_breakdown = None
        if cost_model:
            y_c_arr = eval_res["y_c"]
            y_d_arr = eval_res["y_d"]
            xc_arr  = np.array(res1["x_c"])
            xd_arr  = np.array(res1["x_d"])
            cost_breakdown = cost_model.compute_costs(
                charge        = xc_arr + y_c_arr,
                discharge     = xd_arr + y_d_arr,
                dam_actual    = day_data["dam_actual"],
                rtm_actual    = day_data["rtm_actual"],
                captive_bess  = eval_res["c_d_rt"],
                captive_solar = eval_res["s_cd_rt"],
                dsm_energy_mwh = eval_res["block_dsm_energy"],
            )
            net_rev_regulated = realized_rev - cost_breakdown["total_costs"]
        else:
            net_rev_regulated = net_rev

        # ── Save daily output ─────────────────────────────────────────────────
        daily_output = {
            "date":              date,
            "status":            res1["status"],
            "expected_revenue":  expected_rev,
            "realized_revenue":  realized_rev,
            "captive_revenue":   captive_rev,
            "net_revenue":       net_rev,
            "net_revenue_regulated": net_rev_regulated,
            "total_dsm_mwh":     dsm_mwh,
            "soc_initial":       prev_soc,
            "soc_terminal":      float(eval_res["soc_path"][-1]),
            # Stage 1 schedule
            "dam_schedule":       res1["dam_schedule"],
            "x_c":               res1["x_c"],
            "x_d":               res1["x_d"],
            "s_c_da":            res1["s_c_da"],
            "s_cd_da":           res1["s_cd_da"],
            "c_d_da":            res1["c_d_da"],
            "curtail_da":        res1["curtail_da"],
            "captive_schedule_da": res1["captive_schedule_da"],
            # Stage 2 actuals
            "y_c":               eval_res["y_c"].tolist(),
            "y_d":               eval_res["y_d"].tolist(),
            "s_c_rt":            eval_res["s_c_rt"].tolist(),
            "s_cd_rt":           eval_res["s_cd_rt"].tolist(),
            "c_d_rt":            eval_res["c_d_rt"].tolist(),
            "soc_realized":      eval_res["soc_path"].tolist(),
            # Actuals
            "actual_dam_prices": day_data["dam_actual"].tolist(),
            "actual_rtm_prices": day_data["rtm_actual"].tolist(),
            "solar_da":          solar_da.tolist(),
            "solar_at":          solar_at.tolist(),
            "rtm_q50_used":      rtm_q50.tolist(),
            "cost_breakdown":    cost_breakdown,
            "scenarios":         res1["scenarios"],
        }

        prev_soc = float(eval_res["soc_path"][-1])

        with open(daily_results_dir / f"result_{date}.json", "w") as f:
            json.dump(daily_output, f, indent=2,
                      default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

        backtest_results.append({
            "target_date":           date,
            "expected_revenue":      expected_rev,
            "realized_revenue":      realized_rev,
            "captive_revenue":       captive_rev,
            "net_revenue":           net_rev,
            "net_revenue_regulated": net_rev_regulated,
            "total_dsm_mwh":         dsm_mwh,
            "soc_initial":           daily_output["soc_initial"],
            "soc_terminal":          daily_output["soc_terminal"],
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(backtest_results)
    results_df.to_csv(results_dir / "backtest_results.csv", index=False)

    if len(results_df) > 0:
        summary = {
            "evaluation_mode":           "solar_bess_stage2b_stage2a",
            "n_days":                    len(results_df),
            "total_expected_revenue":    results_df["expected_revenue"].sum(),
            "total_realized_revenue":    results_df["realized_revenue"].sum(),
            "total_captive_revenue":     results_df["captive_revenue"].sum(),
            "total_net_revenue":         results_df["net_revenue"].sum(),
            "total_net_revenue_regulated": results_df["net_revenue_regulated"].sum(),
            "total_dsm_mwh":             results_df["total_dsm_mwh"].sum(),
            "avg_daily_net_revenue":     results_df["net_revenue"].mean(),
            "worst_day_net_revenue":     results_df["net_revenue"].min(),
        }
        with open(results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2,
                      default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)
        print(f"Days run             : {summary['n_days']}")
        print(f"Total Expected Rev   : Rs {summary['total_expected_revenue']:,.0f}")
        print(f"Total Realized Rev   : Rs {summary['total_realized_revenue']:,.0f}")
        print(f"Total Captive Rev    : Rs {summary['total_captive_revenue']:,.0f}")
        print(f"Total Net Revenue    : Rs {summary['total_net_revenue']:,.0f}")
        print(f"Avg Daily Net Rev    : Rs {summary['avg_daily_net_revenue']:,.0f}")
        print(f"Worst Day Net Rev    : Rs {summary['worst_day_net_revenue']:,.0f}")
        print(f"Total DSM Energy     : {summary['total_dsm_mwh']:.3f} MWh")
        print(f"Results saved to     : {results_dir}/")
    else:
        print("No results generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3B Solar+BESS backtest."
    )
    parser.add_argument("--day",     default=None, help="Run single date YYYY-MM-DD")
    parser.add_argument("--limit",   type=int, default=None, help="Limit to N days")
    parser.add_argument("--verbose", action="store_true", help="Print per-block detail")
    args = parser.parse_args()
    run_backtest(args)
