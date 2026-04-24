"""
scripts/run_phase3b_backtest_rtc.py — Architecture v10 RTC FINAL
=================================================================
Solar+BESS backtest with RTC captive contract.

Usage:
    python scripts/run_phase3b_backtest_rtc.py
    python scripts/run_phase3b_backtest_rtc.py --day 2025-08-01
    python scripts/run_phase3b_backtest_rtc.py --limit 3 --verbose
    python scripts/run_phase3b_backtest_rtc.py --bess config/bess_rtc.yaml
"""

import argparse
import dataclasses
import json
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.optimizer.bess_params_rtc   import BESSParamsRTC
from src.optimizer.two_stage_bess_rtc import (
    TwoStageBESSRTC, evaluate_actuals_rtc,
    RESCHEDULE_BLOCKS, T_BLOCKS,
)
from src.optimizer.scenario_loader   import ScenarioLoader


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(description="Phase 3B RTC Backtest v10")
    ap.add_argument("--day",    type=str, default=None)
    ap.add_argument("--limit",  type=int, default=None)
    ap.add_argument("--verbose",action="store_true")
    ap.add_argument("--config", type=str, default="config/phase3b_rtc.yaml")
    ap.add_argument("--bess",   type=str, default="config/bess_rtc.yaml")
    return ap.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# SOLAR SCALE FACTOR
# ══════════════════════════════════════════════════════════════════════════════

def _solar_scale(bp: BESSParamsRTC, config: dict) -> float:
    """
    Return solar scale factor.
    - If solar paths point to Data/Solar/rtc/ (new RTC parquets built by
      build_solar_profiles_rtc.py), scale = 1.0 — no adjustment needed.
    - If solar paths point to Data/Solar/ (old 35MWp/25MW parquets),
      scale = 16.4/25.0 = 0.656 to correct for the different inverter size.
    """
    solar_da_path = config.get("paths", {}).get("solar_da_path", "")
    if "rtc" in solar_da_path.replace("\\", "/").replace("\\\\", "/"):
        return 1.0   # new RTC parquets — already at 16.4 MW scale
    OLD_INVERTER_MW = 25.0
    return min(1.0, bp.solar_inverter_mw / OLD_INVERTER_MW)


def _scale_solar(sol: dict, scale: float, inv_mw: float) -> dict:
    """Apply scale factor and clip to inverter MW."""
    return {
        "solar_da": np.clip(sol["solar_da"] * scale, 0, inv_mw).astype(np.float32),
        "solar_nc": np.clip(sol["solar_nc"] * scale, 0, inv_mw).astype(np.float32),
        "solar_at": np.clip(sol["solar_at"] * scale, 0, inv_mw).astype(np.float32),
    }


# ══════════════════════════════════════════════════════════════════════════════
# RTM Q50 LOADER
# ══════════════════════════════════════════════════════════════════════════════

def _load_rtm_q50(csv_path: str) -> dict:
    """Load RTM q50 forecast CSV → {date_str: np.ndarray (96,)}."""
    q50_by_date: dict = {}
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        date_col  = next((c for c in df.columns if "date"  in c), None)
        block_col = next((c for c in df.columns if "block" in c or "hour" in c), None)
        q50_col   = next((c for c in df.columns if "q50"   in c), None)
        if date_col and q50_col:
            for dv, grp in df.groupby(date_col):
                vals = (grp.sort_values(block_col)[q50_col].values
                        if block_col else grp[q50_col].values)
                if len(vals) == 24:
                    vals = np.repeat(vals, 4)   # hourly → 15-min
                if len(vals) >= 96:
                    q50_by_date[str(dv)] = vals[:96].astype(float)
    except Exception as e:
        print(f"  Warning: RTM q50 load failed: {e}")
    return q50_by_date


# ══════════════════════════════════════════════════════════════════════════════
# PER-BLOCK CSV BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def _build_block_df(ev:         dict,
                    date:       str,
                    res1:       dict,
                    bp:         BESSParamsRTC,
                    dam_actual: np.ndarray,
                    rtm_actual: np.ndarray,
                    solar_da:   np.ndarray,
                    solar_at:   np.ndarray,
                    rsched:     list) -> pd.DataFrame:

    rtc_committed = float(res1["RTC_committed"])
    rtc_threshold = bp.rtc_penalty_threshold(rtc_committed)
    rtc_lo, rtc_hi = bp.rtc_band(rtc_committed)

    x_c_arr  = np.array(res1["x_c"]);    x_d_arr  = np.array(res1["x_d"])
    sc_da    = np.array(res1["s_c_da"]); scd_da   = np.array(res1["s_cd_da"])
    cd_da    = np.array(res1["c_d_da"]); cap_da   = np.array(res1["captive_da"])
    sch_da   = np.array(res1["schedule_da"])
    spt_da   = np.array(res1["setpoint_da"])
    sband    = np.array(res1["solar_band_mask"])

    # Cumulative accumulators
    c_iex = c_cap = c_dsm_pen = c_dsm_hc = c_rtc_pen = 0.0
    c_deg = c_net = c_bess = c_no_bess = c_short_mwh = 0.0
    c_disch_mwh = 0.0

    rows = []
    for B in range(T_BLOCKS):
        dsm     = ev["dsm_results"][B]
        soc_s   = ev["soc_path"][B]
        soc_e   = ev["soc_path"][B + 1]
        bt      = pd.Timestamp(date) + pd.Timedelta(minutes=15 * B)
        xc_B    = float(x_c_arr[B]); xd_B = float(x_d_arr[B])
        yc_B    = float(ev["y_c"][B]); yd_B = float(ev["y_d"][B])
        cap_a_B = float(ev["captive_actual"][B])
        short_mw  = max(0.0, rtc_threshold - cap_a_B)
        short_mwh = short_mw * 0.25

        tot_d = xd_B + float(ev["c_d_actual"][B]) + yd_B
        c_iex       += float(ev["block_iex_net"][B])
        c_cap       += float(ev["block_captive_net"][B])
        c_dsm_pen   += dsm["dsm_penalty"]
        c_dsm_hc    += dsm["dsm_haircut"]
        c_rtc_pen   += float(ev["block_captive_penalty"][B])
        c_deg       += float(ev["block_degradation"][B])
        c_net       += float(ev["block_net"][B])
        bval         = (float(ev["no_bess_dsm"][B]) + float(ev["no_bess_rtc_penalty"][B])
                        - dsm["dsm_penalty"] - dsm["dsm_haircut"]
                        - float(ev["block_captive_penalty"][B])
                        + float(ev["block_iex_net"][B])
                        - float(ev["block_degradation"][B]))
        c_bess      += bval
        c_no_bess   += float(ev["no_bess_revenue"][B])
        c_short_mwh += short_mwh
        c_disch_mwh += tot_d * 0.25 / bp.eta_discharge
        cycles       = c_disch_mwh / bp.usable_energy_mwh

        row = {
            # Identifiers
            "date":                    date,
            "block":                   B,
            "block_time_ist":          bt.strftime("%H:%M"),
            "is_reschedule_block":     B in rsched,
            # Parameters
            "p_max_mw":                bp.p_max_mw,
            "solar_inverter_mw":       bp.solar_inverter_mw,
            "solar_dc_mwp":            bp.solar_capacity_mwp,
            "r_ppa_rs_mwh":            bp.ppa_rate_rs_mwh,
            "avail_cap_mwh":           bp.avail_cap_mwh,
            # RTC
            "rtc_committed_mw":        rtc_committed,
            "rtc_ceiling_mw":          bp.rtc_mw,
            "rtc_floor_pct":           bp.rtc_floor_pct,
            "rtc_threshold_mw":        rtc_threshold,
            "rtc_band_lo":             rtc_lo,
            "rtc_band_hi":             rtc_hi,
            "rtc_notice_issued":       bool(ev["rtc_notice_issued"][B]),
            "rtc_notice_target_block": int(ev["rtc_notice_target"][B]),
            # Solar
            "z_sol_da_mw":             float(solar_da[B]),
            "z_sol_at_mw":             float(solar_at[B]),
            "sol_forecast_error_mw":   float(solar_at[B]) - float(solar_da[B]),
            # Prices
            "actual_dam_price":        float(dam_actual[B]),
            "actual_rtm_price":        float(rtm_actual[B]),
            # Stage 1
            "x_c_da":                  float(xc_B),
            "x_d_da":                  float(xd_B),
            "dam_net":                 float(xd_B - xc_B),
            "s_c_da":                  float(sc_da[B]),
            "s_cd_da":                 float(scd_da[B]),
            "c_d_da":                  float(cd_da[B]),
            "captive_da":              float(cap_da[B]),
            "rtc_committed_da":        rtc_committed,
            "schedule_da":             float(sch_da[B]),
            "setpoint_da":             float(spt_da[B]),
            # Stage 2B
            "s_c_rt":                  float(ev["s_c_rt"][B]),
            "s_cd_rt":                 float(ev["s_cd_rt"][B]),
            "c_d_rt":                  float(ev["c_d_rt"][B]),
            "captive_rt":              float(ev["captive_committed"][B]),
            "schedule_rt":             float(ev["schedule_rt"][B]),
            "setpoint_rt":             float(ev["setpoint"][B]),
            "captive_committed":       float(ev["captive_committed"][B]),
            # Stage 2A
            "y_c":                     float(yc_B),
            "y_d":                     float(yd_B),
            "y_net":                   float(yd_B - yc_B),
            # Actuals
            "active_setpoint":         float(ev["setpoint"][B]),
            "s_c_actual":              float(ev["s_c_actual"][B]),
            "s_cd_actual":             float(ev["s_cd_actual"][B]),
            "c_d_actual":              float(ev["c_d_actual"][B]),
            "captive_actual":          float(cap_a_B),
            "dispatch_case":           ("A" if ev["s_c_actual"][B] > 1e-4
                                        else "B" if ev["c_d_actual"][B] > 1e-4
                                        else "C"),
            # SoC
            "soc_actual_start":        float(soc_s),
            "soc_actual_end":          float(soc_e),
            "is_solar_band":           bool(sband[B]),
            # DSM
            "contract_rate":           float(dsm["charge_rate"]),
            "actual_total_mw":         float(cap_a_B),
            "scheduled_total_mw":      float(ev["schedule_rt"][B]),
            "deviation_mwh":           float(dsm["dws_mwh"]),
            "deviation_pct":           float(dsm["dws_pct"]),
            "deviation_band":          str(dsm["band"]),
            "deviation_direction":     str(dsm["direction"]),
            "charge_rate":             float(dsm["charge_rate"]),
            "charge_rate_multiplier":  float(dsm["charge_rate_mult"]),
            # Under
            "under_revenue_received":  float(dsm["under_revenue_received"]),
            "under_dsm_penalty":       float(dsm["under_dsm_penalty"]),
            "under_net_cash":          float(dsm["under_net_cash"]),
            "under_if_fully_sched":    float(dsm["under_if_fully_sched"]),
            "under_financial_damage":  float(dsm["under_damage"]),
            # Over
            "over_revenue_sched_qty":  float(dsm["over_revenue_sched"]),
            "over_revenue_dev_qty":    float(dsm["over_revenue_dev"]),
            "over_total_received":     float(dsm["over_total_received"]),
            "over_if_all_at_cr":       float(dsm["over_if_all_cr"]),
            "over_revenue_haircut":    float(dsm["over_haircut"]),
            # RTC Penalty
            "rtc_shortfall_mw":        float(short_mw),
            "rtc_shortfall_mwh":       float(short_mwh),
            "rtc_penalty_rs":          float(ev["block_captive_penalty"][B]),
            "rtc_delivery_ok":         cap_a_B >= rtc_threshold,
            # IEX
            "iex_dam_revenue":         float(dam_actual[B]) * (xd_B - xc_B) * 0.25,
            "iex_rtm_revenue":         float(rtm_actual[B]) * (yd_B - yc_B) * 0.25,
            "iex_fees":                bp.iex_fee_rs_mwh * (xc_B+xd_B+yc_B+yd_B) * 0.25,
            "iex_net":                 float(ev["block_iex_net"][B]),
            # Block P&L
            "block_captive_net":       float(ev["block_captive_net"][B]),
            "block_iex_net":           float(ev["block_iex_net"][B]),
            "block_degradation":       float(ev["block_degradation"][B]),
            "block_net":               float(ev["block_net"][B]),
            # BESS ROI
            "no_bess_dsm":             float(ev["no_bess_dsm"][B]),
            "no_bess_rtc_penalty":     float(ev["no_bess_rtc_penalty"][B]),
            "no_bess_revenue":         float(ev["no_bess_revenue"][B]),
            "bess_total_value_block":  float(bval),
            # Cumulative
            "cum_bess_cycles":         round(cycles, 4),
            "cum_iex_net":             round(c_iex, 2),
            "cum_captive_net":         round(c_cap, 2),
            "cum_dsm_penalty":         round(c_dsm_pen, 2),
            "cum_dsm_haircut":         round(c_dsm_hc, 2),
            "cum_rtc_penalty":         round(c_rtc_pen, 2),
            "cum_degradation":         round(c_deg, 2),
            "cum_net_revenue":         round(c_net, 2),
            "cum_bess_value":          round(c_bess, 2),
            "cum_no_bess_revenue":     round(c_no_bess, 2),
            "cum_rtc_shortfall_mwh":   round(c_short_mwh, 4),
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(args):
    print("=" * 65)
    print("PHASE 3B RTC: SOLAR+BESS BACKTEST — Architecture v10 RTC FINAL")
    print("=" * 65)

    bp = BESSParamsRTC.from_yaml(args.bess)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    rsched = config.get("reschedule_blocks", RESCHEDULE_BLOCKS)
    n_scen = config.get("n_scenarios", 100)

    print(f"PCS: {bp.p_max_mw} MW  |  BESS: {bp.e_max_mwh} MWh  "
          f"|  Inverter: {bp.solar_inverter_mw} MW  |  DC: {bp.solar_capacity_mwp} MWp")
    print(f"RTC ceiling: {bp.rtc_mw} MW  |  Floor penalty: {bp.rtc_floor_pct*100:.0f}%  "
          f"|  PPA: Rs {bp.ppa_rate_rs_mwh:,.0f}/MWh")
    print(f"EOD terminal: {bp.soc_terminal_mode}  |  "
          f"Terminal value: Rs {bp.soc_terminal_value_rs_mwh}/MWh")

    # Load data
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

    rtm_q50_by_date = _load_rtm_q50(config["paths"]["actuals_rtm"])
    solar_scale     = _solar_scale(bp, config)
    if abs(solar_scale - 1.0) > 0.01:
        print(f"  Solar scale factor: {solar_scale:.3f} "
              f"(old 25 MW parquets → rescaled to {bp.solar_inverter_mw} MW)")
        print(f"  NOTE: Run build_solar_profiles_rtc.py to eliminate this scaling.")
    else:
        print(f"  Solar: using RTC parquets (no scaling needed, inverter = {bp.solar_inverter_mw} MW)")

    # Result directories
    results_dir = Path(config["paths"].get("results_dir", "results/phase3b_rtc"))
    daily_dir   = results_dir / "daily"
    csv_dir     = results_dir / "csv"
    daily_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # Dates
    if args.day:
        dates = [args.day] if args.day in loader.common_dates else []
        if not dates:
            print(f"ERROR: {args.day} not in loader.common_dates")
            return
    else:
        dates = sorted(loader.common_dates)
    if args.limit:
        dates = dates[:args.limit]

    print(f"Running {len(dates)} days …\n")

    # SOC chaining
    soc_chain    = bp.soc_initial_mwh   # Day 1: e_max = 80 MWh
    all_dfs      = []
    summaries    = []

    for di, date in enumerate(dates):
        print(f"[{di+1}/{len(dates)}] {date} …", end=" ", flush=True)

        # Load day data
        try:
            day = loader.get_day_scenarios(date, n_scenarios=n_scen)
            sol_raw = loader.get_day_solar(date)
        except Exception as e:
            print(f"DATA ERROR: {e}")
            continue

        # Scale solar to new plant capacity
        sol = _scale_solar(sol_raw, solar_scale, bp.solar_inverter_mw)

        # Chain SoC from prior day's actual EOD.
        # Safety clamp: if actuals drifted below LP terminal floor (60 MWh),
        # the next-day LP would be infeasible (SOD < terminal_min).
        # Clamp to max(actual_eod, soc_terminal_min_mwh) to keep LP solvable.
        soc_today = max(float(soc_chain), bp.soc_terminal_min_mwh)
        if soc_today > float(soc_chain) + 0.1:
            print(f"  [SoC clamp] actual EOD={soc_chain:.1f} MWh → LP SOD={soc_today:.1f} MWh "
                  f"(terminal floor applied)")
        bp_today = dataclasses.replace(bp, soc_initial_mwh=soc_today)

        # RTM q50 for this date
        rtm_q50 = rtm_q50_by_date.get(date, np.full(T_BLOCKS, 3000.0))

        # Stage 1
        opt   = TwoStageBESSRTC(bp_today, config)
        res1  = opt.solve(day["dam"], day["rtm"], sol["solar_da"])

        if res1["status"] != "Optimal":
            print(f"Stage 1 FAILED: {res1['status']}")
            continue

        rtc_val = float(res1["RTC_committed"])
        print(f"RTC={rtc_val:.2f} MW", end=" | ", flush=True)

        # Actuals settlement
        ev = evaluate_actuals_rtc(
            params=bp_today, stage1_result=res1,
            dam_actual=day["dam_actual"], rtm_actual=day["rtm_actual"],
            rtm_q50=rtm_q50,
            solar_da=sol["solar_da"], solar_nc=sol["solar_nc"],
            solar_at=sol["solar_at"],
            reschedule_blocks=rsched, verbose=args.verbose,
        )

        net_rev   = ev["net_revenue"]
        iex_net   = ev["iex_net_total"]
        cap_net   = ev["captive_net_total"]
        rtc_pen   = ev["rtc_penalty_total"]
        eod_soc   = float(ev["soc_path"][-1])
        soc_chain = eod_soc   # chain to tomorrow

        print(f"Net: ₹{net_rev:,.0f}  Cap: ₹{cap_net:,.0f}  "
              f"IEX: ₹{iex_net:,.0f}  Pen: ₹{rtc_pen:,.0f}  "
              f"SoC_end: {eod_soc:.1f} MWh")

        # Per-block CSV
        bdf = _build_block_df(
            ev, date, res1, bp_today,
            day["dam_actual"], day["rtm_actual"],
            sol["solar_da"], sol["solar_at"], rsched,
        )
        all_dfs.append(bdf)
        bdf.to_csv(csv_dir / f"phase3b_rtc_{date}.csv", index=False)

        # Daily JSON
        daily_out = {
            "date": date, "architecture": "v10_rtc_final",
            "status": res1["status"],
            "rtc_committed_mw": rtc_val,
            "rtc_ceiling_mw": bp_today.rtc_mw,
            "rtc_floor_pct": bp_today.rtc_floor_pct,
            "soc_initial_mwh": float(bp_today.soc_initial_mwh),
            "soc_final_mwh": eod_soc,
            "expected_revenue": res1["expected_revenue"],
            "net_revenue": net_rev,
            "captive_net": cap_net,
            "iex_net": iex_net,
            "rtc_penalty": rtc_pen,
            "degradation": ev["degradation_total"],
            "no_bess_revenue": ev["no_bess_revenue_total"],
            "bess_dsm_savings": ev["bess_dsm_savings"],
            "bess_rtc_pen_savings": ev["bess_rtc_pen_savings"],
            "bess_total_value": ev["bess_total_value"],
            "soc_path": [round(s, 3) for s in ev["soc_path"].tolist()],
            "captive_actual": [round(v, 3) for v in ev["captive_actual"].tolist()],
            "rtc_penalty_by_block": [round(v, 2) for v in ev["block_captive_penalty"].tolist()],
        }
        with open(daily_dir / f"phase3b_rtc_{date}.json", "w") as jf:
            json.dump(daily_out, jf, indent=2, default=str)

        summaries.append({
            "date": date, "rtc_committed_mw": rtc_val,
            "net_revenue": net_rev, "captive_net": cap_net,
            "iex_net": iex_net, "rtc_penalty": rtc_pen,
            "eod_soc_mwh": eod_soc, "bess_total_value": ev["bess_total_value"],
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    if summaries:
        sdf = pd.DataFrame(summaries)
        sdf.to_csv(results_dir / "phase3b_rtc_summary.csv", index=False)
        print("\n" + "=" * 65)
        print("BACKTEST COMPLETE")
        print("=" * 65)
        print(f"Days run:              {len(summaries)}")
        print(f"Avg RTC committed:     {sdf['rtc_committed_mw'].mean():.2f} MW")
        print(f"Total net revenue:     ₹{sdf['net_revenue'].sum():,.0f}")
        print(f"Total IEX net:         ₹{sdf['iex_net'].sum():,.0f}")
        print(f"Total captive net:     ₹{sdf['captive_net'].sum():,.0f}")
        print(f"Total RTC penalties:   ₹{sdf['rtc_penalty'].sum():,.0f}")
        print(f"Total BESS value:      ₹{sdf['bess_total_value'].sum():,.0f}")
        print(f"Avg EOD SoC:           {sdf['eod_soc_mwh'].mean():.1f} MWh")

    if all_dfs:
        all_df = pd.concat(all_dfs, ignore_index=True)
        all_df.to_csv(results_dir / "phase3b_rtc_all_blocks.csv", index=False)
        print(f"\nAll-blocks CSV: {results_dir}/phase3b_rtc_all_blocks.csv "
              f"({len(all_df)} rows)")


if __name__ == "__main__":
    run_backtest(parse_args())
