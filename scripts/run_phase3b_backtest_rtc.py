"""
scripts/run_phase3b_backtest_rtc.py — Architecture v10 RTC  (FIXED)
====================================================================
Solar+BESS backtest with Round-the-Clock (RTC) captive contract.

KEY DIFFERENCES from run_phase3b_backtest.py (v9_revised):
  1. Loads config/bess_rtc.yaml and config/phase3b_rtc.yaml.
  2. Uses BESSParamsRTC, TwoStageBESSRTC, evaluate_actuals_rtc.
  3. RTC_committed (scalar) committed at Stage 1 instead of variable captive_da.
  4. Block P&L includes iex_net separately (not merged into captive settlement).
  5. RTC captive penalty tracked and reported.
  6. Results saved to results/phase3b_rtc/daily/.

BUG FIXES in this revision:
  BUG-2: Removed duplicate  T_BLOCKS = 96  line that shadowed the import
          from two_stage_bess_rtc (harmless in value but a maintenance trap).
  GAP-7: rtc_notice_issued and rtc_notice_target_block columns now written
          to the per-block CSV output.

Usage:
    python scripts/run_phase3b_backtest_rtc.py
    python scripts/run_phase3b_backtest_rtc.py --day 2025-08-01
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
    T_BLOCKS,           # ← imported from module; NOT redeclared below (BUG-2 fix)
)
from src.optimizer.scenario_loader import ScenarioLoader


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 3B RTC Backtest — Architecture v10")
    parser.add_argument("--day",     type=str, default=None,
                        help="Run single date YYYY-MM-DD")
    parser.add_argument("--limit",   type=int, default=None,
                        help="Limit number of days to process")
    parser.add_argument("--verbose", action="store_true",
                        help="Print block-level progress during actuals")
    parser.add_argument("--config",  type=str,
                        default="config/phase3b_rtc.yaml",
                        help="Path to phase3b_rtc.yaml")
    parser.add_argument("--bess",    type=str,
                        default="config/bess_rtc.yaml",
                        help="Path to bess_rtc.yaml")
    return parser.parse_args()


def build_block_df(ev: dict, date: str, res1: dict, bp: BESSParamsRTC,
                   dam_actual: np.ndarray, rtm_actual: np.ndarray,
                   solar_da: np.ndarray, solar_at: np.ndarray,
                   reschedule_blocks: list) -> pd.DataFrame:
    """
    Build a per-block (96-row) DataFrame from evaluate_actuals_rtc results.
    Includes all Architecture v10 CSV output columns plus rtc_notice columns
    (GAP-7 fix).
    """
    rtc_committed = float(res1["RTC_committed"])
    rows = []

    # cumulative accumulators
    cum_iex_net      = 0.0
    cum_captive_net  = 0.0
    cum_dsm_penalty  = 0.0
    cum_dsm_haircut  = 0.0
    cum_cap_penalty  = 0.0
    cum_degradation  = 0.0
    cum_net_revenue  = 0.0
    cum_bess_value   = 0.0
    cum_no_bess_rev  = 0.0
    cum_discharge    = 0.0   # for cycle count

    x_c_arr  = np.array(res1["x_c"])
    x_d_arr  = np.array(res1["x_d"])
    scd_da   = np.array(res1["s_cd_da"])
    cd_da    = np.array(res1["c_d_da"])
    sc_da    = np.array(res1["s_c_da"])
    cap_da   = np.array(res1["captive_da"])
    sch_da   = np.array(res1["schedule_da"])
    spt_da   = np.array(res1["setpoint_da"])
    sband    = np.array(res1["solar_band_mask"])

    for B in range(T_BLOCKS):
        dsm  = ev["dsm_results"][B]
        soc_s = ev["soc_path"][B]
        soc_e = ev["soc_path"][B + 1]
        block_time = pd.Timestamp(date) + pd.Timedelta(minutes=15 * B)

        xc_B = float(x_c_arr[B]);   xd_B = float(x_d_arr[B])
        yc_B = float(ev["y_c"][B]); yd_B = float(ev["y_d"][B])
        dam_net_B = xd_B - xc_B;    rtm_net_B = yd_B - yc_B

        cap_actual_B = float(ev["captive_actual"][B])
        block_bess_val = (float(ev["no_bess_dsm"][B]) -
                          (dsm["dsm_penalty"] + dsm["dsm_haircut"])
                          + float(ev["block_iex_net"][B])
                          - float(ev["block_degradation"][B]))

        # Cumulative updates
        cum_iex_net     += float(ev["block_iex_net"][B])
        cum_captive_net += float(ev["block_captive_net"][B])
        cum_dsm_penalty += dsm["dsm_penalty"]
        cum_dsm_haircut += dsm["dsm_haircut"]
        cum_cap_penalty += float(ev["block_captive_penalty"][B])
        cum_degradation += float(ev["block_degradation"][B])
        cum_net_revenue += float(ev["block_net"][B])
        cum_bess_value  += block_bess_val
        cum_no_bess_rev += float(ev["no_bess_revenue"][B])
        disch_B = (xd_B + float(ev["c_d_actual"][B]) + yd_B) * 0.25 / bp.eta_discharge
        cum_discharge   += disch_B
        bess_cycles     = cum_discharge / bp.usable_energy_mwh

        row = {
            # ── Identifiers ─────────────────────────────────────────────
            "date":                      date,
            "block":                     B,
            "block_time_ist":            block_time.strftime("%H:%M"),
            "is_reschedule_block":       B in reschedule_blocks,
            "stage2b_trigger_block":     B if B in reschedule_blocks else -1,

            # ── Parameters ──────────────────────────────────────────────
            "p_max_mw":                  bp.p_max_mw,
            "solar_inverter_mw":         bp.solar_inverter_mw,
            "solar_dc_mwp":              bp.solar_capacity_mwp,
            "r_ppa_rs_mwh":              bp.ppa_rate_rs_mwh,
            "avail_cap_mwh":             bp.avail_cap_mwh,

            # ── RTC Contract ─────────────────────────────────────────────
            "rtc_committed_mw":          rtc_committed,
            "rtc_min_mw":                bp.rtc_min_mw,
            "rtc_tol_pct":               bp.rtc_tol_pct,
            "rtc_notice_issued":         bool(ev["rtc_notice_issued"][B]),    # GAP-7 fix
            "rtc_notice_target_block":   int(ev["rtc_notice_target"][B]),     # GAP-7 fix

            # ── Solar ───────────────────────────────────────────────────
            "z_sol_da_mw":               float(solar_da[B]),
            "z_sol_at_mw":               float(solar_at[B]),
            "sol_forecast_error_mw":     float(solar_at[B]) - float(solar_da[B]),

            # ── Prices ──────────────────────────────────────────────────
            "actual_dam_price":          float(dam_actual[B]),
            "actual_rtm_price":          float(rtm_actual[B]),

            # ── Stage 1 ─────────────────────────────────────────────────
            "x_c_da":                    float(xc_B),
            "x_d_da":                    float(xd_B),
            "dam_net":                   float(dam_net_B),
            "s_c_da":                    float(sc_da[B]),
            "s_cd_da":                   float(scd_da[B]),
            "c_d_da":                    float(cd_da[B]),
            "captive_da":                float(cap_da[B]),
            "rtc_committed_da":          rtc_committed,
            "schedule_da":               float(sch_da[B]),
            "setpoint_da":               float(spt_da[B]),
            "soc_da_mean":               float(np.mean(
                                             [s["soc"][B] or 0.0
                                              for s in res1.get("scenarios", [])])
                                         ) if res1.get("scenarios") else 0.0,

            # ── Stage 2B ────────────────────────────────────────────────
            "s_c_rt":                    float(ev["s_c_rt"][B]),
            "s_cd_rt":                   float(ev["s_cd_rt"][B]),
            "c_d_rt":                    float(ev["c_d_rt"][B]),
            "captive_rt":                float(ev["captive_committed"][B]),
            "schedule_rt":               float(ev["schedule_rt"][B]),
            "setpoint_rt":               float(ev["setpoint"][B]),
            "captive_committed":         float(ev["captive_committed"][B]),
            "rtc_band_lo":               rtc_committed * (1.0 - bp.rtc_tol_pct),
            "rtc_band_hi":               rtc_committed * (1.0 + bp.rtc_tol_pct),

            # ── Stage 2A ────────────────────────────────────────────────
            "y_c":                       float(yc_B),
            "y_d":                       float(yd_B),
            "y_net":                     float(rtm_net_B),

            # ── Actuals ─────────────────────────────────────────────────
            "active_setpoint":           float(ev["setpoint"][B]),
            "s_c_actual":                float(ev["s_c_actual"][B]),
            "s_cd_actual":               float(ev["s_cd_actual"][B]),
            "c_d_actual":                float(ev["c_d_actual"][B]),
            "captive_actual":            float(cap_actual_B),
            "dispatch_case":             (
                                         "A" if ev["s_c_actual"][B] > 1e-4
                                         else "B" if ev["c_d_actual"][B] > 1e-4
                                         else "C"),

            # ── SoC ─────────────────────────────────────────────────────
            "soc_actual_start":          float(soc_s),
            "soc_actual_end":            float(soc_e),
            "is_solar_band":             bool(sband[B]),

            # ── DSM ─────────────────────────────────────────────────────
            "contract_rate":             float(dsm.get("charge_rate", 0.0)),
            "actual_total":              float(cap_actual_B),
            "scheduled_total":           float(ev["schedule_rt"][B]),
            "deviation_mwh":             float(dsm["dws_mwh"]),
            "deviation_pct":             float(dsm["dws_pct"]),
            "deviation_band":            str(dsm["band"]),
            "deviation_direction":       str(dsm["direction"]),
            "charge_rate":               float(dsm["charge_rate"]),
            "charge_rate_multiplier":    float(dsm["charge_rate_mult"]),

            # ── Under-injection ─────────────────────────────────────────
            "under_revenue_received":    float(dsm["under_revenue_received"]),
            "under_dsm_penalty":         float(dsm["under_dsm_penalty"]),
            "under_net_cash":            float(dsm["under_net_cash"]),
            "under_if_fully_sched":      float(dsm["under_if_fully_sched"]),
            "under_financial_damage":    float(dsm["under_damage"]),

            # ── Over-injection ──────────────────────────────────────────
            "over_revenue_sched_qty":    float(dsm["over_revenue_sched"]),
            "over_revenue_dev_qty":      float(dsm["over_revenue_dev"]),
            "over_total_received":       float(dsm["over_total_received"]),
            "over_if_all_at_cr":         float(dsm["over_if_all_cr"]),
            "over_revenue_haircut":      float(dsm["over_haircut"]),

            # ── RTC Penalty ─────────────────────────────────────────────
            "captive_shortfall_mwh":     max(0.0, (bp.rtc_min_mw - cap_actual_B) * 0.25),
            "captive_penalty_rs":        float(ev["block_captive_penalty"][B]),
            "rtc_delivery_ok":           cap_actual_B >= bp.rtc_min_mw,

            # ── IEX ─────────────────────────────────────────────────────
            "iex_dam_revenue":           float(dam_actual[B]) * dam_net_B * 0.25,
            "iex_rtm_revenue":           float(rtm_actual[B]) * rtm_net_B * 0.25,
            "iex_fees":                  bp.iex_fee_rs_mwh * (xc_B + xd_B + yc_B + yd_B) * 0.25,
            "iex_net":                   float(ev["block_iex_net"][B]),

            # ── Block P&L ───────────────────────────────────────────────
            "block_captive_net":         float(ev["block_captive_net"][B]),
            "block_iex_net":             float(ev["block_iex_net"][B]),
            "block_degradation":         float(ev["block_degradation"][B]),
            "block_net":                 float(ev["block_net"][B]),

            # ── BESS ROI ────────────────────────────────────────────────
            "no_bess_dsm_total":         float(ev["no_bess_dsm"][B]),
            "no_bess_revenue":           float(ev["no_bess_revenue"][B]),
            "bess_dsm_savings_block":    float(ev["no_bess_dsm"][B]) -
                                         (dsm["dsm_penalty"] + dsm["dsm_haircut"]),
            "bess_iex_net":              float(ev["block_iex_net"][B]),
            "bess_total_value_block":    block_bess_val,

            # ── Cumulative ──────────────────────────────────────────────
            "cum_bess_cycles":           round(bess_cycles, 4),
            "cum_iex_net":               round(cum_iex_net, 2),
            "cum_captive_net":           round(cum_captive_net, 2),
            "cum_dsm_penalty":           round(cum_dsm_penalty, 2),
            "cum_dsm_haircut":           round(cum_dsm_haircut, 2),
            "cum_captive_penalty":       round(cum_cap_penalty, 2),
            "cum_degradation":           round(cum_degradation, 2),
            "cum_net_revenue":           round(cum_net_revenue, 2),
            "cum_bess_value":            round(cum_bess_value, 2),
            "cum_no_bess_revenue":       round(cum_no_bess_rev, 2),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def run_backtest(args):
    print("=" * 65)
    print("PHASE 3B RTC: SOLAR+BESS BACKTEST — Architecture v10 RTC")
    print("=" * 65)

    # ── Load configs ──────────────────────────────────────────────────────
    bp = BESSParamsRTC.from_yaml(args.bess)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    SOC_TARGET        = bp.soc_terminal_min_mwh   # 40.0 MWh
    reschedule_blocks = config.get("reschedule_blocks", RESCHEDULE_BLOCKS)

    print(f"PCS: {bp.p_max_mw} MW  |  BESS: {bp.e_max_mwh} MWh  "
          f"|  Inverter: {bp.solar_inverter_mw} MW")
    print(f"RTC contract: {bp.rtc_mw} MW  |  Min: {bp.rtc_min_mw} MW  "
          f"|  ±{bp.rtc_tol_pct * 100:.0f}% free band")
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
    # RTM q50 — load once, keyed by (target_date, target_block)
    rtm_fp = config["paths"]["actuals_rtm"]
    rtm_q50_by_date: dict = {}
    try:
        rtm_q50_df = pd.read_csv(rtm_fp)
        # Normalise column names
        rtm_q50_df.columns = [c.lower() for c in rtm_q50_df.columns]
        date_col  = next((c for c in rtm_q50_df.columns
                          if "date" in c), None)
        block_col = next((c for c in rtm_q50_df.columns
                          if "block" in c or "hour" in c), None)
        q50_col   = next((c for c in rtm_q50_df.columns
                          if "q50" in c), None)
        if date_col and q50_col:
            for date_val, grp in rtm_q50_df.groupby(date_col):
                vals = grp.sort_values(block_col)[q50_col].values if block_col else grp[q50_col].values
                if len(vals) == 24:          # hourly → expand to 96 blocks
                    vals = np.repeat(vals, 4)
                if len(vals) >= 96:
                    rtm_q50_by_date[str(date_val)] = vals[:96].astype(float)
    except Exception as e:
        print(f"  Warning: could not parse RTM q50 CSV: {e}")

    # ── Result output dirs ────────────────────────────────────────────────
    results_dir = Path(config["paths"].get("results_dir", "results/phase3b_rtc"))
    daily_dir   = results_dir / "daily"
    csv_dir     = results_dir / "csv"
    daily_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine dates to run ─────────────────────────────────────────────
    if args.day:
        dates = [args.day] if args.day in loader.common_dates else []
        if not dates:
            print(f"ERROR: date {args.day} not in loader.common_dates. "
                  f"Available: {loader.common_dates[:3]} ...")
            return
    else:
        dates = sorted(loader.common_dates)

    if args.limit:
        dates = dates[: args.limit]

    print(f"Running {len(dates)} days ...")

    # ── Per-day previous EOD SoC (chains across days) ─────────────────────
    soc_chain = SOC_TARGET   # Day 1 fallback

    all_block_dfs = []
    day_summaries = []

    for di, date in enumerate(dates):
        print(f"[{di + 1}/{len(dates)}] {date} ...", end=" ", flush=True)

        # Load day data using correct ScenarioLoader API
        try:
            n_scen = config.get("n_scenarios", 100)
            day = loader.get_day_scenarios(date, n_scenarios=n_scen)
            sol = loader.get_day_solar(date)
        except Exception as e:
            print(f"DATA ERROR: {e}")
            continue

        # Override SOD with yesterday's EOD
        bp_day = bp   # params are immutable; soc_initial set per day in solve
        # Temporarily patch soc_initial for chaining
        import dataclasses
        bp_today = dataclasses.replace(bp, soc_initial_mwh=soc_chain)

        # RTM q50 for this date (96 blocks)
        rtm_q50 = rtm_q50_by_date.get(date, np.full(T_BLOCKS, 3000.0))

        # ── Stage 1 ───────────────────────────────────────────────────────
        opt_today = TwoStageBESSRTC(bp_today, config)
        res1 = opt_today.solve(
            dam_scenarios=day["dam"],
            rtm_scenarios=day["rtm"],
            solar_da=sol["solar_da"],
        )

        if res1["status"] != "Optimal":
            print(f"Stage 1 FAILED: {res1['status']}")
            continue

        rtc_val = float(res1["RTC_committed"])
        print(f"RTC={rtc_val:.2f} MW", end=" | ", flush=True)

        # ── Actuals settlement ────────────────────────────────────────────
        ev = evaluate_actuals_rtc(
            params=bp_today,
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
            f"|  SoC_end: {eod_soc:.2f} MWh"
        )

        # Chain SoC
        soc_chain = eod_soc

        # ── Build per-block CSV ───────────────────────────────────────────
        block_df = build_block_df(
            ev=ev, date=date, res1=res1, bp=bp_today,
            dam_actual=day["dam_actual"],
            rtm_actual=day["rtm_actual"],
            solar_da=sol["solar_da"],
            solar_at=sol["solar_at"],
            reschedule_blocks=reschedule_blocks,
        )
        all_block_dfs.append(block_df)
        csv_path = csv_dir / f"phase3b_rtc_{date}.csv"
        block_df.to_csv(csv_path, index=False)

        # ── Save daily JSON summary ───────────────────────────────────────
        dsm_serial = [
            {k: (float(v) if isinstance(v, (float, np.floating))
                 else bool(v) if isinstance(v, (bool, np.bool_))
                 else v)
             for k, v in d.items()}
            for d in ev["dsm_results"]
        ]

        daily_out = {
            "date":                       date,
            "status":                     res1["status"],
            "architecture":               "v10_rtc",
            # Hardware
            "p_max_mw":                   bp_today.p_max_mw,
            "solar_inverter_mw":          bp_today.solar_inverter_mw,
            "solar_dc_mwp":               bp_today.solar_capacity_mwp,
            "e_max_mwh":                  bp_today.e_max_mwh,
            "soc_initial_mwh":            float(bp_today.soc_initial_mwh),
            "soc_terminal_actual_mwh":    eod_soc,
            # RTC contract
            "rtc_committed_mw":           rtc_val,
            "rtc_min_mw":                 bp_today.rtc_min_mw,
            "rtc_tol_pct":                bp_today.rtc_tol_pct,
            # Revenue summary
            "expected_revenue":           res1["expected_revenue"],
            "net_revenue":                net_rev,
            "captive_net_revenue":        captive_net,
            "iex_net_revenue":            iex_net,
            "captive_penalty":            cap_penalty,
            "degradation_total":          ev["degradation_total"],
            "no_bess_revenue":            ev["no_bess_revenue_total"],
            "bess_dsm_savings":           ev["bess_dsm_savings"],
            "bess_total_value":           ev["bess_total_value"],
            # Per-block detail
            "dsm_results":                dsm_serial,
            "soc_path":                   [round(s, 4) for s in ev["soc_path"].tolist()],
            "captive_actual":             [round(v, 4) for v in ev["captive_actual"].tolist()],
            "captive_committed":          [round(v, 4) for v in ev["captive_committed"].tolist()],
            "block_net":                  [round(v, 2) for v in ev["block_net"].tolist()],
        }

        json_path = daily_dir / f"phase3b_rtc_{date}.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(daily_out, jf, indent=2, default=str)

        day_summaries.append({
            "date": date,
            "rtc_committed_mw": rtc_val,
            "net_revenue": net_rev,
            "captive_net": captive_net,
            "iex_net": iex_net,
            "captive_penalty": cap_penalty,
            "eod_soc_mwh": eod_soc,
            "bess_total_value": ev["bess_total_value"],
        })

    # ── Multi-day summary ─────────────────────────────────────────────────
    if day_summaries:
        summary_df = pd.DataFrame(day_summaries)
        summary_path = results_dir / "phase3b_rtc_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print("\n" + "=" * 65)
        print("BACKTEST COMPLETE")
        print("=" * 65)
        print(f"Days run:             {len(day_summaries)}")
        print(f"Total net revenue:    ₹{summary_df['net_revenue'].sum():,.0f}")
        print(f"Total IEX net:        ₹{summary_df['iex_net'].sum():,.0f}")
        print(f"Total captive net:    ₹{summary_df['captive_net'].sum():,.0f}")
        print(f"Total cap penalties:  ₹{summary_df['captive_penalty'].sum():,.0f}")
        print(f"Total BESS value:     ₹{summary_df['bess_total_value'].sum():,.0f}")
        print(f"Avg EOD SoC:          {summary_df['eod_soc_mwh'].mean():.2f} MWh")
        print(f"\nSummary CSV:   {summary_path}")
        print(f"Per-day CSVs:  {csv_dir}/")
        print(f"JSON files:    {daily_dir}/")

    # ── Concatenated all-days CSV ─────────────────────────────────────────
    if all_block_dfs:
        all_df = pd.concat(all_block_dfs, ignore_index=True)
        all_csv = results_dir / "phase3b_rtc_all_blocks.csv"
        all_df.to_csv(all_csv, index=False)
        print(f"All-blocks CSV:{all_csv}  ({len(all_df)} rows)")


if __name__ == "__main__":
    args = parse_args()
    run_backtest(args)
