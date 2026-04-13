"""
scripts/run_phase3b_backtest.py
================================
Phase 3B Solar+BESS Backtest -- Architecture v3.

SOC POLICY:
  soc_initial = soc_terminal = 2.5 MWh every day (hard equality, no chaining).

DISCHARGE POLICY:
  No price threshold. Stage 2A LP discharges when it maximises net revenue.

RTM BIDDING:
  Stage 2A bids y_c/y_d for block B+3 (RTM_LEAD=3, ~1 hour gate closure).
"""

import pandas as pd
import numpy as np
import yaml
import json
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.optimizer.bess_params     import BESSParams
from src.optimizer.scenario_loader import ScenarioLoader
from src.optimizer.two_stage_bess  import TwoStageBESS, evaluate_actuals_solar
from src.optimizer.costs           import CostModel

T_BLOCKS = 96
DT       = 0.25


def run_backtest(args):
    print("=" * 60)
    print("PHASE 3B: SOLAR+BESS BACKTEST -- Architecture v3")
    print("SOC: 2.5 MWh start and end every day (hard equality)")
    print("Stage 2A: RTM bid at B+3, LP-only discharge")
    print("=" * 60)

    bess_params = BESSParams.from_yaml("config/bess.yaml")
    with open("config/phase3b.yaml", "r", encoding="ascii") as f:
        config = yaml.safe_load(f)

    SOC_TARGET = bess_params.soc_terminal_min_mwh  # 2.5 MWh
    reschedule_blocks = config.get("reschedule_blocks", [34, 42, 50, 58])

    print(f"BESS PCS     : {bess_params.p_max_mw} MW")
    print(f"Battery      : {bess_params.e_max_mwh} MWh")
    print(f"Sol Inverter : {bess_params.solar_inverter_mw} MW")
    print(f"SOC start/end: {SOC_TARGET} MWh (hard equality)")
    print(f"Solar        : {bess_params.solar_capacity_mwp} MWp")
    print(f"PPA rate     : Rs {bess_params.ppa_rate_rs_mwh}/MWh")
    print(f"RTM lead     : {bess_params.rtm_lead_blocks} blocks")

    loader = ScenarioLoader(
        dam_path           = config["paths"]["scenarios_dam"],
        rtm_path           = config["paths"]["scenarios_rtm"],
        actuals_dam_path   = config["paths"]["actuals_dam"],
        actuals_rtm_path   = config["paths"]["actuals_rtm"],
        solar_da_path      = config["paths"]["solar_da_path"],
        solar_nc_path      = config["paths"]["solar_nc_path"],
        solar_at_path      = config["paths"]["solar_at_path"],
        price_parquet_path = config["paths"].get("price_parquet"),  # FIX 6
    )

    optimizer = TwoStageBESS(bess_params, config)

    cost_model = None
    if Path("config/costs_config.yaml").exists():
        cost_model = CostModel.from_yaml("config/costs_config.yaml")
        print("Loaded CostModel from config/costs_config.yaml")

    # Load RTM q50 forecasts for Stage 2A
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
    print(f"Running backtest for {len(dates)} days...\n")

    backtest_results = []

    for i, date in enumerate(dates):
        # Fixed SOC = 2.5 every day (no chaining)
        bess_params.soc_initial_mwh = SOC_TARGET

        day_data = loader.get_day_scenarios(
            date, n_scenarios=config["n_scenarios"]
        )
        solar_data = loader.get_day_solar(date)
        solar_da = solar_data["solar_da"]
        solar_nc = solar_data["solar_nc"]
        solar_at = solar_data["solar_at"]

        # RTM q50 for this day
        day_rtm = rtm_forecasts_df[
            rtm_forecasts_df["target_date"] == date
        ].sort_values("target_block")
        if len(day_rtm) >= T_BLOCKS:
            rtm_q50 = day_rtm["q50"].values[:T_BLOCKS].astype(float)
        else:
            rtm_q50 = np.full(T_BLOCKS, 4000.0)

        print(f"[{i+1}/{len(dates)}] {date}...", end=" ", flush=True)

        # Stage 1
        res1 = optimizer.solve(
            day_data["dam"], day_data["rtm"], solar_da
        )
        if res1["status"] != "Optimal":
            print(f"Stage 1 FAILED: {res1['status']}")
            continue

        # Evaluate (Stage 2B + 2A + Actuals)
        eval_res = evaluate_actuals_solar(
            params=bess_params,
            stage1_result=res1,
            dam_actual=day_data["dam_actual"],
            rtm_actual=day_data["rtm_actual"],
            rtm_q50=rtm_q50,
            solar_da=solar_da,
            solar_nc=solar_nc,
            solar_at=solar_at,
            reschedule_blocks=reschedule_blocks,
            verbose=verbose,
        )

        actual_eod_soc = float(eval_res["soc_path"][-1])
        realized_rev = eval_res["revenue"]
        net_rev      = eval_res["net_revenue"]
        expected_rev = res1["expected_revenue"]
        captive_rev  = float(np.sum(eval_res["block_rev_captive"]))
        dsm_mwh      = eval_res["total_dsm_mwh"]
        soc_deviation = actual_eod_soc - SOC_TARGET

        print(f"Exp:Rs{expected_rev:,.0f} Real:Rs{realized_rev:,.0f} "
              f"Cap:Rs{captive_rev:,.0f} Net:Rs{net_rev:,.0f} "
              f"SOC_end:{actual_eod_soc:.4f}MWh (dev:{soc_deviation:+.4f})")

        # Cost model
        cost_breakdown = None
        if cost_model:
            cost_breakdown = cost_model.compute_costs(
                charge=np.array(res1["x_c"]) + eval_res["y_c"],
                discharge=np.array(res1["x_d"]) + eval_res["y_d"],
                dam_actual=day_data["dam_actual"],
                rtm_actual=day_data["rtm_actual"],
                captive_bess=eval_res["c_d_actual"],
                captive_solar=eval_res["s_cd_actual"],
                dsm_energy_mwh=eval_res["block_dsm_energy"],
            )
            net_rev_regulated = realized_rev - cost_breakdown["total_costs"]
        else:
            net_rev_regulated = net_rev

        # Save daily JSON
        daily_output = {
            "date":                    date,
            "status":                  res1["status"],
            "p_max_mw":                bess_params.p_max_mw,
            "solar_inverter_mw":       bess_params.solar_inverter_mw,
            "soc_initial_mwh":         SOC_TARGET,
            "soc_terminal_actual_mwh": actual_eod_soc,
            "soc_terminal_target_mwh": SOC_TARGET,
            "soc_deviation_mwh":       soc_deviation,
            "expected_revenue":        expected_rev,
            "realized_revenue":        realized_rev,
            "captive_revenue":         captive_rev,
            "net_revenue":             net_rev,
            "net_revenue_regulated":   net_rev_regulated,
            "total_dsm_mwh":           dsm_mwh,
            "dam_schedule":            res1["dam_schedule"],
            "x_c":                     res1["x_c"],
            "x_d":                     res1["x_d"],
            "s_c_da":                  res1["s_c_da"],
            "s_cd_da":                 res1["s_cd_da"],
            "c_d_da":                  res1["c_d_da"],
            "curtail_da":              res1["curtail_da"],
            "captive_schedule_da":     res1["captive_schedule_da"],
            "y_c":                     eval_res["y_c"].tolist(),
            "y_d":                     eval_res["y_d"].tolist(),
            "s_c_rt":                  eval_res["s_c_rt"].tolist(),
            "s_cd_rt":                 eval_res["s_cd_rt"].tolist(),
            "c_d_rt":                  eval_res["c_d_rt"].tolist(),
            "s_c_actual":              eval_res["s_c_actual"].tolist(),
            "s_cd_actual":             eval_res["s_cd_actual"].tolist(),
            "c_d_actual":              eval_res["c_d_actual"].tolist(),
            "curtail_actual":          eval_res["curtail_actual"].tolist(),
            "captive_actual":          eval_res["captive_actual"].tolist(),
            "captive_shortfall":       eval_res["captive_shortfall"].tolist(),
            "captive_committed":       eval_res["captive_committed"].tolist(),
            "z_nc_blend":              eval_res["z_nc_blend"].tolist(),
            "soc_realized":            eval_res["soc_path"].tolist(),
            "actual_dam_prices":       day_data["dam_actual"].tolist(),
            "actual_rtm_prices":       day_data["rtm_actual"].tolist(),
            "solar_da":                solar_da.tolist(),
            "solar_at":                solar_at.tolist(),
            "rtm_q50_used":            rtm_q50.tolist(),
            "cost_breakdown":          cost_breakdown,
            "scenarios":               res1["scenarios"],
        }

        with open(daily_results_dir / f"result_{date}.json", "w") as f:
            json.dump(daily_output, f, indent=2,
                      default=lambda x: float(x) if isinstance(
                          x, (np.floating, np.integer)) else x)

        backtest_results.append({
            "target_date":              date,
            "soc_initial_mwh":          SOC_TARGET,
            "soc_terminal_actual_mwh":  actual_eod_soc,
            "soc_deviation_mwh":        soc_deviation,
            "expected_revenue":         expected_rev,
            "realized_revenue":         realized_rev,
            "captive_revenue":          captive_rev,
            "net_revenue":              net_rev,
            "net_revenue_regulated":    net_rev_regulated,
            "total_dsm_mwh":            dsm_mwh,
        })

    # Summary
    results_df = pd.DataFrame(backtest_results)
    results_df.to_csv(results_dir / "backtest_results.csv", index=False)

    if len(results_df) > 0:
        summary = {
            "architecture":                "v3_solar_bess",
            "soc_policy":                  "fixed_2.5MWh_start_end_every_day",
            "n_days":                      len(results_df),
            "total_expected_revenue":      results_df["expected_revenue"].sum(),
            "total_realized_revenue":      results_df["realized_revenue"].sum(),
            "total_captive_revenue":       results_df["captive_revenue"].sum(),
            "total_net_revenue":           results_df["net_revenue"].sum(),
            "total_net_revenue_regulated": results_df["net_revenue_regulated"].sum(),
            "total_dsm_mwh":              results_df["total_dsm_mwh"].sum(),
            "avg_daily_net_revenue":       results_df["net_revenue"].mean(),
            "worst_day_net_revenue":       results_df["net_revenue"].min(),
            "avg_soc_deviation_mwh":       results_df["soc_deviation_mwh"].abs().mean(),
            "max_soc_deviation_mwh":       results_df["soc_deviation_mwh"].abs().max(),
        }
        with open(results_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2,
                      default=lambda x: float(x) if isinstance(
                          x, (np.floating, np.integer)) else x)

        print("\n" + "=" * 60)
        print("BACKTEST SUMMARY")
        print("=" * 60)
        print(f"Days run               : {summary['n_days']}")
        print(f"SOC policy             : {summary['soc_policy']}")
        print(f"Total Expected Rev     : Rs {summary['total_expected_revenue']:,.0f}")
        print(f"Total Realized Rev     : Rs {summary['total_realized_revenue']:,.0f}")
        print(f"Total Captive Rev      : Rs {summary['total_captive_revenue']:,.0f}")
        print(f"Total Net Revenue      : Rs {summary['total_net_revenue']:,.0f}")
        print(f"Avg Daily Net Rev      : Rs {summary['avg_daily_net_revenue']:,.0f}")
        print(f"Worst Day Net Rev      : Rs {summary['worst_day_net_revenue']:,.0f}")
        print(f"Total DSM Energy       : {summary['total_dsm_mwh']:.4f} MWh")
        print(f"Avg |SOC deviation|    : {summary['avg_soc_deviation_mwh']:.4f} MWh")
        print(f"Results saved to       : {results_dir}/")
    else:
        print("No results generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3B Solar+BESS backtest -- Architecture v3"
    )
    parser.add_argument("--day", default=None, help="Single date YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=None, help="Limit N days")
    parser.add_argument("--verbose", action="store_true", help="Per-block detail")
    args = parser.parse_args()
    run_backtest(args)
