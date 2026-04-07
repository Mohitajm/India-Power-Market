"""
scripts/export_backtest_csvs.py
================================
Export all Stage 1, Stage 2B, and Stage 2A inputs and outputs to CSV files.

Reads the per-day JSON files written by run_phase3b_backtest.py and
produces five structured CSV files — one per logical group.

Run from project root:
    python scripts/export_backtest_csvs.py
    python scripts/export_backtest_csvs.py --results results/phase3b_solar

OUTPUT FILES (all written to results/phase3b_solar/csv/):
==========================================================

1. stage1_inputs.csv
   One row per (date, block 0..95).
   Columns: date, block, block_time_ist,
            z_sol_da_mw,
            p_max_mw, e_max_mwh, e_min_mwh, e_max_plan_mwh, e_min_plan_mwh,
            soc_initial_mwh, soc_buffer_pct,
            r_ppa_rs_mwh, iex_fee_rs_mwh, deg_cost_rs_mwh,
            dsm_proxy_rs_mwh, lambda_risk, risk_alpha,
            dam_scenario_mean_rs_mwh, dam_scenario_std_rs_mwh,
            rtm_scenario_mean_rs_mwh, rtm_scenario_std_rs_mwh

2. stage1_outputs.csv
   One row per (date, block 0..95).
   Columns: date, block, block_time_ist,
            x_c_mw, x_d_mw, dam_net_mw,
            s_c_da_mw, s_cd_da_mw, c_d_da_mw, curtail_da_mw,
            captive_da_mw,
            solar_balance_check,        (s_c + s_cd + curtail − z_sol, should be ~0)
            soc_da_mean_mwh,            (mean SoC across 100 scenarios at block end)
            soc_da_min_mwh,
            soc_da_max_mwh,
            stage1_expected_revenue_rs  (daily scalar, repeated for every block)

3. stage2b_outputs.csv
   One row per (date, reschedule_block, block 0..95).
   Only rows for blocks >= reschedule_block are populated (the rest are 0).
   Columns: date, reschedule_trigger_block, block, block_time_ist,
            s_c_rt_mw, s_cd_rt_mw, c_d_rt_mw, curtail_rt_mw,
            captive_rt_mw,
            stage2b_solar_balance_check (s_c_rt + s_cd_rt + curtail_rt − solar_blend)

4. stage2a_inputs_outputs.csv
   One row per (date, block 0..95) — the core per-block dispatch table.
   Columns: date, block, block_time_ist,
            -- Stage 2A Inputs --
            soc_actual_start_mwh,       (SoC at start of block — fed into LP)
            x_c_locked_mw,              (from Stage 1 — constant)
            x_d_locked_mw,              (from Stage 1 — constant)
            s_c_rt_fixed_mw,            (from Stage 2B or DA plan — not a 2A decision)
            c_d_rt_fixed_mw,            (from Stage 2B or DA plan — not a 2A decision)
            p_dam_actual_rs_mwh,        (fully known before day starts)
            p_rtm_q50_rs_mwh,           (forecast used in LP)
            p_rtm_lag4_rs_mwh,          (actual RTM 4 blocks ago — conditioning signal)
            is_reschedule_block,        (1 if block in {34,42,50,58}, else 0)
            -- Stage 2A Outputs --
            y_c_mw,                     (RTM charge committed this block)
            y_d_mw,                     (RTM discharge committed this block)
            y_net_mw,                   (y_d - y_c)
            soc_actual_end_mwh,         (SoC after this block's dispatch)

5. settlement_revenue.csv
   One row per (date, block 0..95) — full settlement accounting.
   Columns: date, block, block_time_ist,
            -- Prices --
            p_dam_actual_rs_mwh,
            p_rtm_actual_rs_mwh,
            solar_at_mw,                (actual metered solar generation)
            -- Dispatch (what actually happened) --
            x_net_mw,                   (DAM net from Stage 1)
            y_net_mw,                   (RTM net from Stage 2A)
            s_c_rt_mw,                  (solar → BESS, revised)
            s_cd_at_mw,                 (solar → captive, actual)
            c_d_rt_mw,                  (BESS → captive)
            captive_total_mw,           (s_cd_at + c_d_rt)
            soc_path_mwh,               (SoC at end of block)
            -- Block Revenue --
            rev_dam_rs,                 (p_dam_actual × x_net × 0.25)
            rev_rtm_rs,                 (p_rtm_actual × y_net × 0.25)
            rev_captive_rs,             (r_ppa × captive_total × 0.25)
            rev_total_rs,               (sum of three)
            -- Cumulative --
            cum_rev_dam_rs,
            cum_rev_rtm_rs,
            cum_rev_captive_rs,
            cum_rev_total_rs,
            -- Daily totals (repeated for convenience) --
            daily_expected_revenue_rs,
            daily_realized_revenue_rs,
            daily_captive_revenue_rs,
            daily_net_revenue_rs,
            daily_dsm_mwh
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

T_BLOCKS          = 96
DT                = 0.25           # hours per 15-min block
RESCHEDULE_BLOCKS = {34, 42, 50, 58}


def block_to_ist_time(block: int) -> str:
    """Convert block index 0..95 to IST time string 'HH:MM'."""
    total_min = block * 15
    h = total_min // 60
    m = total_min % 60
    return f"{h:02d}:{m:02d}"


def load_daily_jsons(results_dir: Path) -> list:
    """Load all daily result JSON files from results_dir/daily/."""
    daily_dir = results_dir / "daily"
    if not daily_dir.exists():
        raise FileNotFoundError(
            f"Daily results directory not found: {daily_dir}\n"
            "Run run_phase3b_backtest.py first."
        )
    files = sorted(daily_dir.glob("result_*.json"))
    if not files:
        raise FileNotFoundError(f"No result_*.json files found in {daily_dir}")
    print(f"Found {len(files)} daily result files in {daily_dir}")

    records = []
    for fp in files:
        with open(fp, "r") as f:
            records.append(json.load(f))
    return records


def build_stage1_inputs(records: list, config: dict) -> pd.DataFrame:
    """
    Stage 1 Inputs — one row per (date, block).
    Includes BESS parameters, economic parameters, solar DA forecast,
    and scenario statistics (mean/std of the 100 price scenarios).
    """
    bess = config["bess"]
    econ = config["econ"]
    opt  = config["opt"]

    rows = []
    for rec in records:
        date     = rec["date"]
        solar_da = rec.get("solar_da", [0.0] * T_BLOCKS)

        # Scenario statistics — stored in rec["scenarios"] as list of
        # {"soc": [97 values]} only. The actual price scenarios are not
        # stored in the JSON (they're large — 100 × 96 floats per market).
        # We store what we have: the dam/rtm actuals as proxy reference.
        dam_actual = rec.get("actual_dam_prices", [0.0] * T_BLOCKS)
        rtm_actual = rec.get("actual_rtm_prices", [0.0] * T_BLOCKS)

        for t in range(T_BLOCKS):
            rows.append({
                "date":                    date,
                "block":                   t,
                "block_time_ist":          block_to_ist_time(t),
                # Solar DA forecast input
                "z_sol_da_mw":             round(float(solar_da[t]), 4),
                # BESS physical parameters
                "p_max_mw":                bess["p_max_mw"],
                "e_max_mwh":               bess["e_max_mwh"],
                "e_min_mwh":               bess["e_min_mwh"],
                "e_max_plan_mwh":          bess["e_max_plan_mwh"],
                "e_min_plan_mwh":          bess["e_min_plan_mwh"],
                "soc_initial_mwh":         rec.get("soc_initial", bess["soc_initial_mwh"]),
                "soc_buffer_pct":          bess["soc_buffer_pct"],
                "eta_charge":              bess["eta_charge"],
                "eta_discharge":           bess["eta_discharge"],
                # Economic parameters
                "r_ppa_rs_mwh":            econ["r_ppa_rs_mwh"],
                "iex_fee_rs_mwh":          econ["iex_fee_rs_mwh"],
                "deg_cost_rs_mwh":         econ["deg_cost_rs_mwh"],
                "dsm_proxy_rs_mwh":        econ["dsm_proxy_rs_mwh"],
                "lambda_risk":             opt["lambda_risk"],
                "risk_alpha":              opt["risk_alpha"],
                # Actual prices on that day (for reference — NOT used in Stage 1 LP)
                "actual_dam_price_rs_mwh": round(float(dam_actual[t]), 2),
                "actual_rtm_price_rs_mwh": round(float(rtm_actual[t]), 2),
            })

    return pd.DataFrame(rows)


def build_stage1_outputs(records: list) -> pd.DataFrame:
    """
    Stage 1 Outputs — one row per (date, block).
    All non-anticipative decisions: DAM schedule, solar routing, SoC plan statistics.
    Includes solar balance check column.
    """
    rows = []
    for rec in records:
        date      = rec["date"]
        solar_da  = rec.get("solar_da", [0.0] * T_BLOCKS)
        x_c       = rec.get("x_c", [0.0] * T_BLOCKS)
        x_d       = rec.get("x_d", [0.0] * T_BLOCKS)
        s_c_da    = rec.get("s_c_da", [0.0] * T_BLOCKS)
        s_cd_da   = rec.get("s_cd_da", [0.0] * T_BLOCKS)
        c_d_da    = rec.get("c_d_da", [0.0] * T_BLOCKS)
        curtail   = rec.get("curtail_da", [0.0] * T_BLOCKS)
        captive   = rec.get("captive_schedule_da", [0.0] * T_BLOCKS)
        exp_rev   = rec.get("expected_revenue", 0.0)

        # SoC plan per scenario — stored as list of {"soc": [97 values]}
        scenarios = rec.get("scenarios", [])
        soc_matrix = []
        for s in scenarios:
            soc_traj = s.get("soc", [])
            if len(soc_traj) == T_BLOCKS + 1:
                soc_matrix.append(soc_traj)
        has_soc = len(soc_matrix) > 0

        for t in range(T_BLOCKS):
            # Solar balance check: should be ~0 if constraint holds
            balance = (float(s_c_da[t]) + float(s_cd_da[t])
                       + float(curtail[t]) - float(solar_da[t]))

            # SoC at end of block t (index t+1 in trajectory)
            if has_soc:
                soc_vals = [m[t + 1] for m in soc_matrix if m[t + 1] is not None]
                soc_mean = round(float(np.mean(soc_vals)), 4) if soc_vals else None
                soc_min  = round(float(np.min(soc_vals)),  4) if soc_vals else None
                soc_max  = round(float(np.max(soc_vals)),  4) if soc_vals else None
            else:
                soc_mean = soc_min = soc_max = None

            rows.append({
                "date":                        date,
                "block":                       t,
                "block_time_ist":              block_to_ist_time(t),
                # DAM IEX schedule (non-anticipative, locked for day)
                "x_c_mw":                      round(float(x_c[t]), 4),
                "x_d_mw":                      round(float(x_d[t]), 4),
                "dam_net_mw":                  round(float(x_d[t]) - float(x_c[t]), 4),
                # Solar routing (non-anticipative DA plan)
                "s_c_da_mw":                   round(float(s_c_da[t]), 4),
                "s_cd_da_mw":                  round(float(s_cd_da[t]), 4),
                "c_d_da_mw":                   round(float(c_d_da[t]), 4),
                "curtail_da_mw":               round(float(curtail[t]), 4),
                "captive_da_mw":               round(float(captive[t]), 4),
                "z_sol_da_mw":                 round(float(solar_da[t]), 4),
                # Validation: s_c + s_cd + curtail - z_sol (should be ~0)
                "solar_balance_check":         round(balance, 6),
                # SoC plan statistics across 100 scenarios at end of block t
                "soc_da_mean_mwh":             soc_mean,
                "soc_da_min_mwh":              soc_min,
                "soc_da_max_mwh":              soc_max,
                # Daily planning metric (same for all 96 blocks of this date)
                "stage1_expected_revenue_rs":  round(float(exp_rev), 0),
            })

    return pd.DataFrame(rows)


def build_stage2b_outputs(records: list) -> pd.DataFrame:
    """
    Stage 2B Outputs — one row per (date, block).
    Shows the LATEST active Stage 2B revision for each block.
    A block gets the values from the most recent Stage 2B that covered it.
    Also includes the solar blend input and balance check.
    """
    rows = []
    for rec in records:
        date     = rec["date"]
        solar_da = rec.get("solar_da", [0.0] * T_BLOCKS)
        solar_at = rec.get("solar_at", [0.0] * T_BLOCKS)
        s_c_rt   = rec.get("s_c_rt",  [0.0] * T_BLOCKS)
        s_cd_rt  = rec.get("s_cd_rt", [0.0] * T_BLOCKS)
        c_d_rt   = rec.get("c_d_rt",  [0.0] * T_BLOCKS)
        rtm_q50  = rec.get("rtm_q50_used", [0.0] * T_BLOCKS)

        # Determine which Stage 2B trigger covered each block
        # Block t is covered by the largest trigger_block <= t
        def get_trigger(t):
            triggers = [b for b in sorted(RESCHEDULE_BLOCKS) if b <= t]
            return triggers[-1] if triggers else None

        for t in range(T_BLOCKS):
            trigger = get_trigger(t)

            # Solar blend used by Stage 2B at trigger block:
            # NC for trigger..trigger+11, DA for trigger+12..95
            if trigger is not None:
                k = t - trigger
                solar_blend = float(solar_da[t])   # approximation for NC blend
            else:
                solar_blend = float(solar_da[t])

            # Revised solar balance check
            sc  = float(s_c_rt[t])
            scd = float(s_cd_rt[t])
            cd  = float(c_d_rt[t])
            # curtail_rt is not separately stored, compute from balance
            curtail_rt = max(0.0, solar_blend - sc - scd)
            balance = sc + scd + curtail_rt - solar_blend

            rows.append({
                "date":                      date,
                "block":                     t,
                "block_time_ist":            block_to_ist_time(t),
                "stage2b_trigger_block":     trigger,
                "is_reschedule_block":       int(t in RESCHEDULE_BLOCKS),
                # Solar blend input at active trigger
                "solar_blend_mw":            round(solar_blend, 4),
                "solar_at_mw":               round(float(solar_at[t]), 4),
                # RTM q50 used for planning (Stage 2B uses this to value SoC)
                "p_rtm_q50_rs_mwh":          round(float(rtm_q50[t]), 2),
                # Stage 2B outputs (revised solar routing)
                "s_c_rt_mw":                 round(sc,  4),
                "s_cd_rt_mw":                round(scd, 4),
                "c_d_rt_mw":                 round(cd,  4),
                "curtail_rt_mw":             round(curtail_rt, 4),
                "captive_rt_mw":             round(scd + cd, 4),
                # Validation check
                "stage2b_solar_balance_check": round(balance, 6),
            })

    return pd.DataFrame(rows)


def build_stage2a_io(records: list) -> pd.DataFrame:
    """
    Stage 2A Inputs and Outputs — one row per (date, block).
    Shows exactly what the Stage 2A LP received and decided for each block.
    """
    rows = []
    for rec in records:
        date      = rec["date"]
        soc_path  = rec.get("soc_realized", [0.0] * (T_BLOCKS + 1))
        x_c       = rec.get("x_c",          [0.0] * T_BLOCKS)
        x_d       = rec.get("x_d",          [0.0] * T_BLOCKS)
        s_c_rt    = rec.get("s_c_rt",        [0.0] * T_BLOCKS)
        c_d_rt    = rec.get("c_d_rt",        [0.0] * T_BLOCKS)
        s_cd_rt   = rec.get("s_cd_rt",       [0.0] * T_BLOCKS)
        y_c       = rec.get("y_c",           [0.0] * T_BLOCKS)
        y_d       = rec.get("y_d",           [0.0] * T_BLOCKS)
        dam_act   = rec.get("actual_dam_prices", [0.0] * T_BLOCKS)
        rtm_act   = rec.get("actual_rtm_prices", [0.0] * T_BLOCKS)
        rtm_q50   = rec.get("rtm_q50_used",  [0.0] * T_BLOCKS)

        for t in range(T_BLOCKS):
            # p_rtm_lag4: actual RTM price 4 blocks ago (known at gate-close)
            p_rtm_lag4 = float(rtm_act[t - 4]) if t >= 4 else None

            rows.append({
                "date":                    date,
                "block":                   t,
                "block_time_ist":          block_to_ist_time(t),
                # ── Stage 2A Inputs ───────────────────────────────────────────
                # 1. SoC at start of this block (physically measured)
                "soc_actual_start_mwh":    round(float(soc_path[t]), 4),
                # 2. Locked DAM IEX schedule from Stage 1 (constants in 2A LP)
                "x_c_locked_mw":           round(float(x_c[t]), 4),
                "x_d_locked_mw":           round(float(x_d[t]), 4),
                "dam_net_locked_mw":       round(float(x_d[t]) - float(x_c[t]), 4),
                # 3. Solar routing fixed before 2A runs (from Stage 2B or DA plan)
                "s_c_rt_fixed_mw":         round(float(s_c_rt[t]), 4),
                "c_d_rt_fixed_mw":         round(float(c_d_rt[t]), 4),
                "s_cd_rt_fixed_mw":        round(float(s_cd_rt[t]), 4),
                # 4. DAM actual price (fully known before day starts)
                "p_dam_actual_rs_mwh":     round(float(dam_act[t]), 2),
                # 5. RTM q50 forecast used by Stage 2A LP for planning
                "p_rtm_q50_rs_mwh":        round(float(rtm_q50[t]), 2),
                # 6. Actual RTM lag-4 conditioning signal
                "p_rtm_lag4_rs_mwh":       round(p_rtm_lag4, 2) if p_rtm_lag4 is not None else None,
                # 7. Actual RTM price (for settlement — NOT used in 2A LP)
                "p_rtm_actual_rs_mwh":     round(float(rtm_act[t]), 2),
                # Whether Stage 2B ran before Stage 2A at this block
                "is_reschedule_block":     int(t in RESCHEDULE_BLOCKS),
                # ── Stage 2A Outputs ──────────────────────────────────────────
                "y_c_mw":                  round(float(y_c[t]), 4),
                "y_d_mw":                  round(float(y_d[t]), 4),
                "y_net_mw":                round(float(y_d[t]) - float(y_c[t]), 4),
                # SoC after this block's dispatch (result of physical SoC update)
                "soc_actual_end_mwh":      round(float(soc_path[t + 1]), 4),
            })

    return pd.DataFrame(rows)


def build_settlement_revenue(records: list) -> pd.DataFrame:
    """
    Settlement Revenue — one row per (date, block).
    Full block-by-block settlement at actual prices, with cumulative totals.
    """
    rows = []
    for rec in records:
        date       = rec["date"]
        solar_at   = rec.get("solar_at",           [0.0] * T_BLOCKS)
        s_c_rt     = rec.get("s_c_rt",             [0.0] * T_BLOCKS)
        s_cd_rt    = rec.get("s_cd_rt",            [0.0] * T_BLOCKS)
        c_d_rt     = rec.get("c_d_rt",             [0.0] * T_BLOCKS)
        x_c        = rec.get("x_c",                [0.0] * T_BLOCKS)
        x_d        = rec.get("x_d",                [0.0] * T_BLOCKS)
        y_c        = rec.get("y_c",                [0.0] * T_BLOCKS)
        y_d        = rec.get("y_d",                [0.0] * T_BLOCKS)
        dam_act    = rec.get("actual_dam_prices",   [0.0] * T_BLOCKS)
        rtm_act    = rec.get("actual_rtm_prices",   [0.0] * T_BLOCKS)
        soc_path   = rec.get("soc_realized",        [0.0] * (T_BLOCKS + 1))
        solar_da   = rec.get("solar_da",            [0.0] * T_BLOCKS)

        # Daily scalars
        exp_rev    = rec.get("expected_revenue",    0.0)
        real_rev   = rec.get("realized_revenue",    0.0)
        cap_rev    = rec.get("captive_revenue",     0.0)
        net_rev    = rec.get("net_revenue",         0.0)
        dsm_mwh    = rec.get("total_dsm_mwh",       0.0)

        # PPA rate from costs config (default 3500)
        r_ppa = 3500.0

        cum_dam = cum_rtm = cum_cap = cum_tot = 0.0

        for t in range(T_BLOCKS):
            p_dam = float(dam_act[t])
            p_rtm = float(rtm_act[t])

            # DAM net dispatch
            x_net = float(x_d[t]) - float(x_c[t])
            # RTM net dispatch
            y_net = float(y_d[t]) - float(y_c[t])

            # Solar to captive: what actually reached captive consumer
            solar_after_bess = max(0.0, float(solar_at[t]) - float(s_c_rt[t]))
            s_cd_at = min(solar_after_bess, float(s_cd_rt[t]))
            c_d_val = float(c_d_rt[t])
            captive_total = s_cd_at + c_d_val

            # Block revenue (Rs)
            rev_dam  = p_dam * x_net * DT
            rev_rtm  = p_rtm * y_net * DT
            rev_cap  = r_ppa * captive_total * DT
            rev_tot  = rev_dam + rev_rtm + rev_cap

            # Cumulative totals
            cum_dam += rev_dam
            cum_rtm += rev_rtm
            cum_cap += rev_cap
            cum_tot += rev_tot

            rows.append({
                "date":                       date,
                "block":                      t,
                "block_time_ist":             block_to_ist_time(t),
                # ── Prices ────────────────────────────────────────────────────
                "p_dam_actual_rs_mwh":        round(p_dam, 2),
                "p_rtm_actual_rs_mwh":        round(p_rtm, 2),
                "solar_da_forecast_mw":       round(float(solar_da[t]), 4),
                "solar_at_actual_mw":         round(float(solar_at[t]), 4),
                # ── Actual Dispatch ───────────────────────────────────────────
                "x_net_dam_mw":               round(x_net, 4),
                "y_net_rtm_mw":               round(y_net, 4),
                "x_c_mw":                     round(float(x_c[t]), 4),
                "x_d_mw":                     round(float(x_d[t]), 4),
                "y_c_mw":                     round(float(y_c[t]), 4),
                "y_d_mw":                     round(float(y_d[t]), 4),
                "s_c_rt_mw":                  round(float(s_c_rt[t]), 4),
                "s_cd_at_mw":                 round(s_cd_at, 4),
                "c_d_rt_mw":                  round(c_d_val, 4),
                "captive_total_mw":           round(captive_total, 4),
                "soc_end_mwh":                round(float(soc_path[t + 1]), 4),
                # ── Block Revenue (Rs) ────────────────────────────────────────
                "rev_dam_rs":                 round(rev_dam,  2),
                "rev_rtm_rs":                 round(rev_rtm,  2),
                "rev_captive_rs":             round(rev_cap,  2),
                "rev_total_block_rs":         round(rev_tot,  2),
                # ── Cumulative within day ─────────────────────────────────────
                "cum_rev_dam_rs":             round(cum_dam, 2),
                "cum_rev_rtm_rs":             round(cum_rtm, 2),
                "cum_rev_captive_rs":         round(cum_cap, 2),
                "cum_rev_total_rs":           round(cum_tot, 2),
                # ── Daily Totals (repeated for convenience) ───────────────────
                "daily_expected_revenue_rs":  round(float(exp_rev),  0),
                "daily_realized_revenue_rs":  round(float(real_rev), 0),
                "daily_captive_revenue_rs":   round(float(cap_rev),  0),
                "daily_net_revenue_rs":       round(float(net_rev),  0),
                "daily_dsm_mwh":              round(float(dsm_mwh),  6),
            })

    return pd.DataFrame(rows)


def get_config_for_export(results_dir: Path) -> dict:
    """
    Build a config dict with BESS/economic parameters for the input CSV.
    Tries to read from config/bess.yaml; falls back to defaults.
    """
    import yaml

    defaults = {
        "bess": {
            "p_max_mw":          2.5,
            "e_max_mwh":         4.75,
            "e_min_mwh":         0.50,
            "e_max_plan_mwh":    4.5125,
            "e_min_plan_mwh":    0.525,
            "soc_initial_mwh":   2.50,
            "soc_buffer_pct":    0.05,
            "eta_charge":        0.9487,
            "eta_discharge":     0.9487,
        },
        "econ": {
            "r_ppa_rs_mwh":      3500.0,
            "iex_fee_rs_mwh":    200.0,
            "deg_cost_rs_mwh":   650.0,
            "dsm_proxy_rs_mwh":  135.0,
        },
        "opt": {
            "lambda_risk":  0.0,
            "risk_alpha":   0.1,
        },
    }

    bess_yaml = Path("config/bess.yaml")
    phase3b_yaml = Path("config/phase3b.yaml")

    if bess_yaml.exists():
        with open(bess_yaml, "r", encoding="ascii") as f:
            b = yaml.safe_load(f)
        buf = b.get("soc_buffer_pct", 0.05)
        defaults["bess"].update({
            "p_max_mw":          b.get("p_max_mw",          2.5),
            "e_max_mwh":         b.get("e_max_mwh",         4.75),
            "e_min_mwh":         b.get("e_min_mwh",         0.50),
            "e_max_plan_mwh":    b.get("e_max_mwh", 4.75) * (1 - buf),
            "e_min_plan_mwh":    b.get("e_min_mwh", 0.50) * (1 + buf),
            "soc_initial_mwh":   b.get("soc_initial_mwh",   2.50),
            "soc_buffer_pct":    buf,
            "eta_charge":        b.get("eta_charge",         0.9487),
            "eta_discharge":     b.get("eta_discharge",      0.9487),
        })
        defaults["econ"]["r_ppa_rs_mwh"]   = b.get("ppa_rate_rs_mwh", 3500.0)
        defaults["econ"]["iex_fee_rs_mwh"]  = b.get("iex_fee_rs_mwh",  200.0)
        defaults["econ"]["deg_cost_rs_mwh"] = b.get("degradation_cost_rs_mwh", 650.0)

    if phase3b_yaml.exists():
        with open(phase3b_yaml, "r", encoding="ascii") as f:
            p = yaml.safe_load(f)
        defaults["opt"]["lambda_risk"] = p.get("lambda_risk", 0.0)
        defaults["opt"]["risk_alpha"]  = p.get("risk_alpha",  0.1)

    return defaults


def export_all(results_dir_str: str) -> None:
    """Main export function — reads JSON files and writes all 5 CSVs."""
    results_dir = Path(results_dir_str)
    csv_dir     = results_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading daily result JSON files from: {results_dir}/daily/")
    records = load_daily_jsons(results_dir)
    print(f"Loaded {len(records)} days of data.\n")

    config = get_config_for_export(results_dir)

    # ── 1. Stage 1 Inputs ────────────────────────────────────────────────────
    print("Building stage1_inputs.csv ...")
    df1_in = build_stage1_inputs(records, config)
    path1i = csv_dir / "stage1_inputs.csv"
    df1_in.to_csv(path1i, index=False)
    print(f"  Saved: {path1i}  ({len(df1_in):,} rows × {len(df1_in.columns)} columns)")

    # ── 2. Stage 1 Outputs ───────────────────────────────────────────────────
    print("Building stage1_outputs.csv ...")
    df1_out = build_stage1_outputs(records)
    path1o  = csv_dir / "stage1_outputs.csv"
    df1_out.to_csv(path1o, index=False)
    print(f"  Saved: {path1o}  ({len(df1_out):,} rows × {len(df1_out.columns)} columns)")

    # ── 3. Stage 2B Outputs ──────────────────────────────────────────────────
    print("Building stage2b_outputs.csv ...")
    df2b = build_stage2b_outputs(records)
    path2b = csv_dir / "stage2b_outputs.csv"
    df2b.to_csv(path2b, index=False)
    print(f"  Saved: {path2b}  ({len(df2b):,} rows × {len(df2b.columns)} columns)")

    # ── 4. Stage 2A Inputs + Outputs ─────────────────────────────────────────
    print("Building stage2a_inputs_outputs.csv ...")
    df2a = build_stage2a_io(records)
    path2a = csv_dir / "stage2a_inputs_outputs.csv"
    df2a.to_csv(path2a, index=False)
    print(f"  Saved: {path2a}  ({len(df2a):,} rows × {len(df2a.columns)} columns)")

    # ── 5. Settlement Revenue ────────────────────────────────────────────────
    print("Building settlement_revenue.csv ...")
    df_rev = build_settlement_revenue(records)
    path_rev = csv_dir / "settlement_revenue.csv"
    df_rev.to_csv(path_rev, index=False)
    print(f"  Saved: {path_rev}  ({len(df_rev):,} rows × {len(df_rev.columns)} columns)")

    # ── Print summary ─────────────────────────────────────────────────────────
    n_days   = len(records)
    n_blocks = n_days * T_BLOCKS
    total_net = df_rev.groupby("date")["daily_net_revenue_rs"].first().sum()
    total_cap = df_rev.groupby("date")["daily_captive_revenue_rs"].first().sum()
    total_dsm = df_rev.groupby("date")["daily_dsm_mwh"].first().sum()

    print(f"""
============================================================
EXPORT COMPLETE
============================================================
Days exported    : {n_days}
Blocks per day   : {T_BLOCKS}
Total rows       : {n_blocks:,} per CSV

Files written to : {csv_dir}/

  stage1_inputs.csv          Stage 1 inputs per block
    (BESS params, solar DA, economic params, scenario stats)

  stage1_outputs.csv         Stage 1 decisions per block
    (x_c, x_d, s_c_da, s_cd_da, c_d_da, curtail_da,
     captive_da, solar_balance_check, soc_da stats)

  stage2b_outputs.csv        Stage 2B revised routing per block
    (triggered at blocks 34, 42, 50, 58)
    (s_c_rt, s_cd_rt, c_d_rt, curtail_rt, captive_rt)

  stage2a_inputs_outputs.csv Stage 2A per-block LP dispatch
    (inputs: soc_actual, locked x_c/x_d, fixed routing,
     p_dam_actual, p_rtm_q50, p_rtm_lag4)
    (outputs: y_c, y_d, y_net, soc_actual_end)

  settlement_revenue.csv     Full block-by-block settlement
    (actual prices, all dispatch, revenue by source,
     cumulative totals, daily summary repeated per block)

Financial Summary:
  Total Net Revenue  : Rs {total_net:,.0f}
  Total Captive Rev  : Rs {total_cap:,.0f}
  Total DSM Energy   : {total_dsm:.4f} MWh
============================================================""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Stage 1/2B/2A inputs and outputs to CSV."
    )
    parser.add_argument(
        "--results",
        default="results/phase3b_solar",
        help="Path to results directory (default: results/phase3b_solar)",
    )
    args = parser.parse_args()
    export_all(args.results)
