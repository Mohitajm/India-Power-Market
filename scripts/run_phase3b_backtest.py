"""
scripts/run_phase3b_backtest.py — Architecture v9_revised
"""
import pandas as pd
import numpy as np
import yaml
import json
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.optimizer.bess_params import BESSParams
from src.optimizer.scenario_loader import ScenarioLoader
from src.optimizer.two_stage_bess import TwoStageBESS, evaluate_actuals_solar
from src.optimizer.costs import CostModel

T_BLOCKS = 96


def run_backtest(args):
    print("=" * 60)
    print("PHASE 3B: SOLAR+BESS BACKTEST -- Architecture v9_revised")
    print("BESS as Deviation Hedge + IEX Arbitrage")
    print("=" * 60)

    bp = BESSParams.from_yaml("config/bess.yaml")
    with open("config/phase3b.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    SOC_TARGET = bp.soc_terminal_min_mwh
    reschedule_blocks = config.get("reschedule_blocks", [34, 42, 50, 58])

    print(f"PCS: {bp.p_max_mw} MW | Battery: {bp.e_max_mwh} MWh | "
          f"Inverter: {bp.solar_inverter_mw} MW")
    print(f"SoC solar band: [{bp.soc_solar_low:.2f}, {bp.soc_solar_high:.2f}] MWh")
    print(f"SOC start/end: {SOC_TARGET} MWh | PPA: Rs {bp.ppa_rate_rs_mwh}/MWh")

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
    optimizer = TwoStageBESS(bp, config)

    rtm_fp = Path("Data/Predictions/rtm_quantiles_backtest_recalibrated.parquet")
    if not rtm_fp.exists():
        rtm_fp = Path("Data/Predictions/rtm_quantiles_backtest.parquet")
    rtm_df = pd.read_parquet(rtm_fp)
    if rtm_df.index.name and "delivery" in str(rtm_df.index.name):
        rtm_df = rtm_df.reset_index()
    if "target_block" not in rtm_df.columns:
        if "delivery_start_ist" in rtm_df.columns:
            ts = pd.to_datetime(rtm_df["delivery_start_ist"])
            rtm_df["target_block"] = ts.dt.hour * 4 + ts.dt.minute // 15 + 1
    if "target_date" in rtm_df.columns:
        rtm_df["target_date"] = rtm_df["target_date"].astype(str)

    results_dir = Path(config["paths"]["results_dir"])
    daily_dir = results_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    dates = loader.common_dates
    if getattr(args, "day", None):
        dates = [args.day]
    elif getattr(args, "limit", None):
        dates = dates[:args.limit]

    print(f"Running {len(dates)} days...\n")
    backtest_results = []

    for i, date in enumerate(dates):
        bp.soc_initial_mwh = SOC_TARGET

        day = loader.get_day_scenarios(date, n_scenarios=config["n_scenarios"])
        sol = loader.get_day_solar(date)

        day_rtm = rtm_df[rtm_df["target_date"] == date].sort_values("target_block")
        rtm_q50 = (day_rtm["q50"].values[:T_BLOCKS].astype(float)
                   if len(day_rtm) >= T_BLOCKS else np.full(T_BLOCKS, 4000.0))

        print(f"[{i+1}/{len(dates)}] {date}...", end=" ", flush=True)

        res1 = optimizer.solve(day["dam"], day["rtm"], sol["solar_da"])
        if res1["status"] != "Optimal":
            print(f"Stage 1 FAILED: {res1['status']}")
            continue

        ev = evaluate_actuals_solar(
            bp, res1, day["dam_actual"], day["rtm_actual"], rtm_q50,
            sol["solar_da"], sol["solar_nc"], sol["solar_at"],
            reschedule_blocks=reschedule_blocks,
            verbose=getattr(args, "verbose", False))

        eod_soc = float(ev["soc_path"][-1])
        net_rev = ev["net_revenue"]
        cap_rev = float(np.sum(ev["block_captive_net"]))
        bess_val = float(np.sum(ev["no_bess_revenue"]) -
                         np.sum(ev["block_net"]))  # reversed for savings

        print(f"Net:Rs{net_rev:,.0f} Cap:Rs{cap_rev:,.0f} "
              f"SOC_end:{eod_soc:.4f}")

        # Serialize DSM results
        dsm_serial = []
        for d in ev["dsm_results"]:
            dsm_serial.append({k: float(v) if isinstance(v, (float, np.floating))
                               else v for k, v in d.items()})

        daily_out = {
            "date": date, "status": res1["status"],
            "p_max_mw": bp.p_max_mw,
            "solar_inverter_mw": bp.solar_inverter_mw,
            "soc_initial_mwh": SOC_TARGET,
            "soc_terminal_actual_mwh": eod_soc,
            "expected_revenue": res1["expected_revenue"],
            "net_revenue": net_rev,
            "captive_net_revenue": cap_rev,
            "x_c": res1["x_c"], "x_d": res1["x_d"],
            "s_c_da": res1["s_c_da"], "s_cd_da": res1["s_cd_da"],
            "c_d_da": res1["c_d_da"],
            "captive_da": res1["captive_da"],
            "schedule_da": res1["schedule_da"],
            "setpoint_da": res1["setpoint_da"],
            "dam_schedule": res1["dam_schedule"],
            "solar_band_mask": res1.get("solar_band_mask", []),
            "y_c": ev["y_c"].tolist(), "y_d": ev["y_d"].tolist(),
            "s_c_rt": ev["s_c_rt"].tolist(),
            "s_cd_rt": ev["s_cd_rt"].tolist(),
            "c_d_rt": ev["c_d_rt"].tolist(),
            "s_c_actual": ev["s_c_actual"].tolist(),
            "s_cd_actual": ev["s_cd_actual"].tolist(),
            "c_d_actual": ev["c_d_actual"].tolist(),
            "captive_actual": ev["captive_actual"].tolist(),
            "captive_committed": ev["captive_committed"].tolist(),
            "setpoint_rt": ev["setpoint_rt"].tolist(),
            "schedule_rt": ev["schedule_rt"].tolist(),
            "soc_realized": ev["soc_path"].tolist(),
            "z_nc_blend": ev["z_nc_blend"].tolist(),
            "actual_dam_prices": day["dam_actual"].tolist(),
            "actual_rtm_prices": day["rtm_actual"].tolist(),
            "solar_da": sol["solar_da"].tolist(),
            "solar_at": sol["solar_at"].tolist(),
            "rtm_q50_used": rtm_q50.tolist(),
            "dsm_results": dsm_serial,
            "iex_dam_rev": ev["iex_dam_rev"].tolist(),
            "iex_rtm_rev": ev["iex_rtm_rev"].tolist(),
            "iex_fees": ev["iex_fees"].tolist(),
            "block_captive_net": ev["block_captive_net"].tolist(),
            "block_degradation": ev["block_degradation"].tolist(),
            "block_net": ev["block_net"].tolist(),
            "no_bess_dsm": ev["no_bess_dsm"].tolist(),
            "no_bess_revenue": ev["no_bess_revenue"].tolist(),
            "scenarios": res1["scenarios"],
        }
        with open(daily_dir / f"result_{date}.json", "w") as f:
            json.dump(daily_out, f, indent=2,
                      default=lambda x: float(x) if isinstance(
                          x, (np.floating, np.integer)) else x)

        backtest_results.append({
            "target_date": date,
            "net_revenue": net_rev,
            "captive_net_revenue": cap_rev,
            "soc_terminal": eod_soc,
        })

    df = pd.DataFrame(backtest_results)
    df.to_csv(results_dir / "backtest_results.csv", index=False)
    if len(df):
        print(f"\nTotal Net Revenue: Rs {df['net_revenue'].sum():,.0f}")
        print(f"Avg Daily: Rs {df['net_revenue'].mean():,.0f}")
        print(f"Results: {results_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    run_backtest(parser.parse_args())
