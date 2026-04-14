"""
scripts/export_backtest_csvs.py
=================================
Architecture v3 CSV export — Final version.

Changes vs prior version:
  1. stage1_expected_revenue_rs is now PER-BLOCK (scenario-averaged),
     not the full-day total repeated. Formula documented below.
  2. Added soc_da_start_mwh (Stage 1 planned SoC at START of block t).
  3. Added c_d_plan_vs_actual_mw.
  4. Added soc_actual_start_mwh and soc_actual_end_mwh (clearer naming).
  5. All revenue formulas documented in column comments.
  6. daily_cycles_bess replaced with cumulative_bess_cycles (running total).
  7. All daily_* aggregates replaced with cumulative block-level running totals.

Revenue formulas (per block t, DT = 0.25 hours):
  iex_charge_cost_rs      = actual_dam_price[t] × (-x_c[t]) × DT
                            (negative — cost of buying from IEX DAM)
  iex_dam_discharge_rs    = actual_dam_price[t] × x_d[t] × DT
                            (positive — revenue from selling to IEX DAM)
  iex_rtm_charge_cost_rs  = actual_rtm_price[t] × (-y_c[t]) × DT
                            (negative — cost of buying from IEX RTM)
  iex_rtm_discharge_rs    = actual_rtm_price[t] × y_d[t] × DT
                            (positive — revenue from selling to IEX RTM)
  iex_net_block_rs        = sum of the above 4 IEX streams
  captive_revenue_rs      = r_ppa × captive_actual[t] × DT
                            where captive_actual = s_cd_actual + c_d_actual
                            (positive — PPA revenue for energy delivered)
  total_block_revenue_rs  = iex_net_block_rs + captive_revenue_rs

Stage 1 expected block revenue (scenario-weighted, from LP plan):
  stage1_exp_block_rev[t] = (1/S) × sum_s [
      dam_scenario[s,t] × x_d[t] × DT
    - dam_scenario[s,t] × x_c[t] × DT
    + r_ppa × (s_cd_da[t] + c_d_da[t]) × DT
    - iex_fee × (x_c[t] + x_d[t]) × DT
    - deg_cost × (x_d[t] + c_d_da[t]) × DT
    - 135 × (x_c[t] + x_d[t]) × DT
  ]
  Since we don't have per-scenario DAM prices in the export JSON, we use
  the mean scenario price (approximated from the saved scenario SoC traces
  and LP expected revenue). For a simpler and accurate approach, we compute
  a "reconstructed" per-block expected revenue using the actual DAM price
  as a proxy for the scenario-average price at that block:
    stage1_exp_block_rev_rs = p_dam_actual[t] × (x_d[t] - x_c[t]) × DT
                            + r_ppa × (s_cd_da[t] + c_d_da[t]) × DT
                            - iex_fee × (x_c[t] + x_d[t]) × DT
                            - deg_cost × (x_d[t] + c_d_da[t]) × DT
                            - 135 × (x_c[t] + x_d[t]) × DT

Run:
    python scripts/export_backtest_csvs.py
    python scripts/export_backtest_csvs.py --results results/phase3b_solar
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

T_BLOCKS          = 96
DT                = 0.25
RESCHEDULE_BLOCKS = {34, 42, 50, 58}
USABLE_CAPACITY   = 4.25    # e_max_mwh - e_min_mwh = 4.75 - 0.50


def block_time(b: int) -> str:
    h = (b * 15) // 60
    m = (b * 15) % 60
    return f"{h:02d}:{m:02d}"


def load_jsons(results_dir: Path) -> list:
    daily_dir = results_dir / "daily"
    if not daily_dir.exists():
        raise FileNotFoundError(f"No daily results at {daily_dir}")
    files = sorted(daily_dir.glob("result_*.json"))
    if not files:
        raise FileNotFoundError(f"No result JSON files in {daily_dir}")
    print(f"Loading {len(files)} daily result files...")
    records = []
    for fp in files:
        with open(fp) as f:
            records.append(json.load(f))
    return records


def safe_get(arr, idx, default=0.0):
    if arr is None or idx >= len(arr):
        return default
    v = arr[idx]
    return float(v) if v is not None else default


def build_full_csv(records: list, ppa_rate: float = 3500.0,
                   iex_fee: float = 200.0, deg_cost: float = 650.0,
                   dsm_proxy: float = 135.0) -> pd.DataFrame:
    """Build one row per (date, block) with all fields."""
    rows = []

    for rec in records:
        date = rec.get("date", "")

        # Parameters
        bess_p_max = rec.get("p_max_mw", 2.5)
        sol_inv_mw = rec.get("solar_inverter_mw", 25.0)

        # Stage 1 arrays
        solar_da   = rec.get("solar_da",     [0.0] * T_BLOCKS)
        solar_at   = rec.get("solar_at",     [0.0] * T_BLOCKS)
        rtm_q50    = rec.get("rtm_q50_used", [0.0] * T_BLOCKS)
        x_c        = rec.get("x_c",          [0.0] * T_BLOCKS)
        x_d        = rec.get("x_d",          [0.0] * T_BLOCKS)
        s_c_da     = rec.get("s_c_da",       [0.0] * T_BLOCKS)
        s_cd_da    = rec.get("s_cd_da",      [0.0] * T_BLOCKS)
        c_d_da     = rec.get("c_d_da",       [0.0] * T_BLOCKS)
        curtail_da = rec.get("curtail_da",   [0.0] * T_BLOCKS)

        # Stage 1 SoC plan (mean across scenarios)
        scenarios = rec.get("scenarios", [])
        soc_matrix = []
        for s in scenarios:
            traj = s.get("soc", [])
            if len(traj) == T_BLOCKS + 1:
                soc_matrix.append([v if v is not None else 0.0 for v in traj])

        # Stage 2B arrays
        s_c_rt     = rec.get("s_c_rt",       [0.0] * T_BLOCKS)
        s_cd_rt    = rec.get("s_cd_rt",      [0.0] * T_BLOCKS)
        c_d_rt     = rec.get("c_d_rt",       [0.0] * T_BLOCKS)
        z_nc_blend = rec.get("z_nc_blend",   solar_da)

        # Stage 2A arrays
        y_c        = rec.get("y_c",          [0.0] * T_BLOCKS)
        y_d        = rec.get("y_d",          [0.0] * T_BLOCKS)

        # Actuals arrays
        s_c_actual     = rec.get("s_c_actual",      [0.0] * T_BLOCKS)
        s_cd_actual    = rec.get("s_cd_actual",     [0.0] * T_BLOCKS)
        c_d_actual     = rec.get("c_d_actual",      [0.0] * T_BLOCKS)
        curtail_actual = rec.get("curtail_actual",  [0.0] * T_BLOCKS)
        captive_actual = rec.get("captive_actual",  [0.0] * T_BLOCKS)
        captive_short  = rec.get("captive_shortfall",[0.0] * T_BLOCKS)
        captive_commit = rec.get("captive_committed",[0.0] * T_BLOCKS)

        # Physical SoC and prices
        soc_realized = rec.get("soc_realized", [0.0] * (T_BLOCKS + 1))
        dam_act = rec.get("actual_dam_prices", [0.0] * T_BLOCKS)
        rtm_act = rec.get("actual_rtm_prices", [0.0] * T_BLOCKS)

        # Running cumulative accumulators (reset per day)
        cum_ch_iex   = 0.0
        cum_ch_solar = 0.0
        cum_ch_rtm   = 0.0
        cum_dis_dam  = 0.0
        cum_dis_rtm  = 0.0
        cum_dis_cap  = 0.0
        cum_real_rev = 0.0
        cum_cap_rev  = 0.0
        cum_net_rev  = 0.0
        cum_dsm_mwh  = 0.0

        # Per-block rows
        for t in range(T_BLOCKS):
            xc_t   = float(x_c[t])
            xd_t   = float(x_d[t])
            yc_t   = float(y_c[t])
            yd_t   = float(y_d[t])

            # Stage 1 plan
            sc_da_t  = float(s_c_da[t])
            scd_da_t = float(s_cd_da[t])
            cd_da_t  = float(c_d_da[t])
            cut_da_t = float(curtail_da[t])
            sol_da_t = float(solar_da[t])

            # Stage 2B revised
            sc_rt_t  = float(s_c_rt[t])
            scd_rt_t = float(s_cd_rt[t])
            cd_rt_t  = float(c_d_rt[t])
            z_nc_t   = safe_get(z_nc_blend, t, sol_da_t)

            # Actuals
            sc_at_t   = safe_get(s_c_actual, t)
            scd_at_t  = safe_get(s_cd_actual, t)
            cd_at_t   = safe_get(c_d_actual, t)
            cut_at_t  = safe_get(curtail_actual, t)
            cap_at_t  = safe_get(captive_actual, t)
            cap_sh_t  = safe_get(captive_short, t)
            cap_com_t = safe_get(captive_commit, t)
            sol_at_t  = float(solar_at[t])

            # Prices
            pd_t   = float(dam_act[t])
            pr_t   = float(rtm_act[t])
            pq50_t = float(rtm_q50[t])
            lag4_t = float(rtm_act[t - 4]) if t >= 4 else None

            # SoC — Stage 1 plan (mean across scenarios)
            soc_da_start_t = (float(np.mean([m[t] for m in soc_matrix]))
                              if soc_matrix else 0.0)
            soc_da_end_t   = (float(np.mean([m[t + 1] for m in soc_matrix]))
                              if soc_matrix else 0.0)

            # SoC — Actual physical
            soc_at_start_t = safe_get(soc_realized, t)
            soc_at_end_t   = safe_get(soc_realized, t + 1)

            # ── Revenue formulas (all use ACTUAL prices and ACTUAL routing) ──
            # IEX DAM: settled at actual DAM MCP
            iex_charge  = pd_t * (-xc_t) * DT          # negative (cost)
            iex_dam_dis = pd_t * xd_t * DT              # positive (revenue)
            # IEX RTM: settled at actual RTM MCP
            iex_rtm_chg = pr_t * (-yc_t) * DT           # negative (cost)
            iex_rtm_dis = pr_t * yd_t * DT               # positive (revenue)
            # Net IEX for this block
            iex_net     = iex_charge + iex_dam_dis + iex_rtm_chg + iex_rtm_dis
            # Captive PPA: settled on actual delivery
            cap_rev_t   = ppa_rate * cap_at_t * DT       # positive (revenue)
            # Total block revenue
            rev_tot_t   = iex_net + cap_rev_t

            # ── Stage 1 expected per-block revenue (reconstructed) ──
            # Uses actual DAM price as proxy for scenario-average price.
            # This is the expected contribution of block t from Stage 1 plan.
            s1_dam_rev   = pd_t * (xd_t - xc_t) * DT
            s1_cap_rev   = ppa_rate * (scd_da_t + cd_da_t) * DT
            s1_iex_cost  = iex_fee * (xc_t + xd_t) * DT
            s1_deg_cost  = deg_cost * (xd_t + cd_da_t) * DT
            s1_dsm_cost  = dsm_proxy * (xc_t + xd_t) * DT
            s1_block_rev = s1_dam_rev + s1_cap_rev - s1_iex_cost - s1_deg_cost - s1_dsm_cost

            # Balance checks
            da_bal = round(abs(sc_da_t + scd_da_t + cut_da_t - sol_da_t), 5)
            at_bal = round(abs(sc_at_t + scd_at_t + cut_at_t - sol_at_t), 5)

            trigger = max([b for b in sorted(RESCHEDULE_BLOCKS) if b <= t],
                          default=None)

            # ── Cumulative accumulators ──
            cum_ch_iex   += xc_t * DT
            cum_ch_solar += sc_at_t * DT
            cum_ch_rtm   += yc_t * DT
            cum_dis_dam  += xd_t * DT
            cum_dis_rtm  += yd_t * DT
            cum_dis_cap  += cd_at_t * DT
            cum_real_rev += rev_tot_t
            cum_cap_rev  += cap_rev_t
            cum_net_rev  += rev_tot_t  # gross; net requires cost subtraction
            cum_dsm_mwh  += safe_get(rec.get("block_dsm_energy"), t, 0.0) if rec.get("block_dsm_energy") else 0.0

            # Cumulative BESS cycles
            total_ch_cum  = cum_ch_iex + cum_ch_solar + cum_ch_rtm
            total_dis_cum = cum_dis_dam + cum_dis_rtm + cum_dis_cap
            cum_cycles    = min(total_ch_cum, total_dis_cum) / USABLE_CAPACITY if USABLE_CAPACITY > 0 else 0

            rows.append({
                # ── Identifiers ───────────────────────────────────────────
                "date":                       date,
                "block":                      t,
                "block_time_ist":             block_time(t),
                "is_reschedule_block":        int(t in RESCHEDULE_BLOCKS),
                "stage2b_trigger_block":      trigger,

                # ── Parameters ────────────────────────────────────────────
                "p_max_mw":                   bess_p_max,
                "solar_inverter_mw":          sol_inv_mw,
                "r_ppa_rs_mwh":               ppa_rate,

                # ── Solar profiles ────────────────────────────────────────
                "z_sol_da_mw":                round(sol_da_t, 4),
                "z_sol_nc_mw":                round(z_nc_t, 4),
                "z_sol_at_mw":                round(sol_at_t, 4),
                "sol_da_vs_at_delta_mw":      round(sol_da_t - sol_at_t, 4),

                # ── Actual prices ─────────────────────────────────────────
                "actual_dam_price_rs_mwh":    round(pd_t, 2),
                "actual_rtm_price_rs_mwh":    round(pr_t, 2),
                "p_rtm_q50_rs_mwh":           round(pq50_t, 2),
                "p_rtm_lag4_rs_mwh":          round(lag4_t, 2) if lag4_t is not None else "",

                # ── Stage 1 DA plan ───────────────────────────────────────
                "x_c_da_mw":                  round(xc_t, 4),
                "x_d_da_mw":                  round(xd_t, 4),
                "dam_net_mw":                 round(xd_t - xc_t, 4),
                "s_c_da_mw":                  round(sc_da_t, 4),
                "s_cd_da_mw":                 round(scd_da_t, 4),
                "c_d_da_mw":                  round(cd_da_t, 4),
                "curtail_da_mw":              round(cut_da_t, 4),
                "captive_da_mw":              round(scd_da_t + cd_da_t, 4),
                "solar_da_balance_ok":        da_bal,
                "stage1_exp_block_rev_rs":    round(s1_block_rev, 2),

                # ── Stage 2B revised plan ─────────────────────────────────
                "s_c_rt_mw":                  round(sc_rt_t, 4),
                "s_cd_rt_mw":                 round(scd_rt_t, 4),
                "c_d_rt_mw":                  round(cd_rt_t, 4),
                "captive_rt_mw":              round(scd_rt_t + cd_rt_t, 4),
                "captive_committed_mw":       round(cap_com_t, 4),

                # ── Stage 2A RTM bids ─────────────────────────────────────
                "y_c_mw":                     round(yc_t, 4),
                "y_d_mw":                     round(yd_t, 4),
                "y_net_mw":                   round(yd_t - yc_t, 4),

                # ── Actuals (metered solar routing) ───────────────────────
                "s_c_actual_mw":              round(sc_at_t, 4),
                "s_cd_actual_mw":             round(scd_at_t, 4),
                "c_d_actual_mw":              round(cd_at_t, 4),
                "curtail_actual_mw":          round(cut_at_t, 4),
                "captive_actual_mw":          round(cap_at_t, 4),
                "captive_shortfall_mw":       round(cap_sh_t, 4),
                "solar_actual_balance_ok":    at_bal,

                # ── Plan vs Actual deltas ─────────────────────────────────
                "s_c_plan_vs_actual_mw":      round(sc_rt_t - sc_at_t, 4),
                "s_cd_plan_vs_actual_mw":     round(scd_rt_t - scd_at_t, 4),
                "c_d_plan_vs_actual_mw":      round(cd_rt_t - cd_at_t, 4),
                "captive_plan_vs_actual_mw":  round(cap_com_t - cap_at_t, 4),

                # ── SoC ───────────────────────────────────────────────────
                "soc_da_start_mwh":           round(soc_da_start_t, 4),
                "soc_da_end_mwh":             round(soc_da_end_t, 4),
                "soc_actual_start_mwh":       round(soc_at_start_t, 4),
                "soc_actual_end_mwh":         round(soc_at_end_t, 4),
                "soc_da_vs_actual_gap_mwh":   round(soc_da_end_t - soc_at_end_t, 4),

                # ── Settlement revenue ────────────────────────────────────
                "iex_charge_cost_rs":         round(iex_charge, 2),
                "iex_dam_discharge_rs":       round(iex_dam_dis, 2),
                "iex_rtm_charge_cost_rs":     round(iex_rtm_chg, 2),
                "iex_rtm_discharge_rs":       round(iex_rtm_dis, 2),
                "iex_net_block_rs":           round(iex_net, 2),
                "captive_revenue_rs":         round(cap_rev_t, 2),
                "total_block_revenue_rs":     round(rev_tot_t, 2),

                # ── Cumulative totals (running sum, block 0 to t) ─────────
                "cum_bess_cycles":            round(cum_cycles, 4),
                "cum_charge_iex_mwh":         round(cum_ch_iex, 4),
                "cum_charge_solar_mwh":       round(cum_ch_solar, 4),
                "cum_charge_rtm_mwh":         round(cum_ch_rtm, 4),
                "cum_discharge_dam_mwh":      round(cum_dis_dam, 4),
                "cum_discharge_rtm_mwh":      round(cum_dis_rtm, 4),
                "cum_discharge_captive_mwh":  round(cum_dis_cap, 4),
                "cum_realized_revenue_rs":    round(cum_real_rev, 2),
                "cum_captive_revenue_rs":     round(cum_cap_rev, 2),
                "cum_net_revenue_rs":         round(cum_net_rev, 2),
                "cum_dsm_mwh":                round(cum_dsm_mwh, 6),
            })

    return pd.DataFrame(rows)


def export_all(results_dir_str: str) -> None:
    results_dir = Path(results_dir_str)
    records = load_jsons(results_dir)

    # Read parameters from config
    ppa_rate = 3500.0
    iex_fee  = 200.0
    deg_cost = 650.0
    bess_yaml = Path("config/bess.yaml")
    if bess_yaml.exists():
        import yaml
        with open(bess_yaml, encoding="ascii") as f:
            b = yaml.safe_load(f)
        ppa_rate = b.get("ppa_rate_rs_mwh", 3500.0)
        iex_fee  = b.get("iex_fee_rs_mwh", 200.0)
        deg_cost = b.get("degradation_cost_rs_mwh", 650.0)

    print(f"Building export for {len(records)} days...")
    print(f"  PPA={ppa_rate} Rs/MWh, IEX fee={iex_fee} Rs/MWh, "
          f"Deg={deg_cost} Rs/MWh")
    df = build_full_csv(records, ppa_rate, iex_fee, deg_cost)

    out_path = results_dir / "backtest_full_export_v3.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(df):,} rows x {len(df.columns)} columns)")

    # Daily summary (computed from cumulative values at block 95)
    print()
    print("=" * 110)
    print("DAILY SUMMARY (from block 95 cumulative values)")
    print("=" * 110)
    print(f"{'Date':<12} {'Cycles':>7} {'ChIEX':>7} {'ChSol':>7} {'ChRTM':>7} "
          f"{'DiDAM':>7} {'DiRTM':>7} {'DiCap':>7} "
          f"{'Revenue':>11} {'Captive':>11} {'CapShort':>8}")

    for date, grp in df.groupby("date"):
        last = grp.iloc[-1]  # block 95 — has full-day cumulative
        cap_short_total = grp["captive_shortfall_mw"].sum() * DT
        print(f"{date:<12} "
              f"{last['cum_bess_cycles']:>7.3f} "
              f"{last['cum_charge_iex_mwh']:>7.3f} "
              f"{last['cum_charge_solar_mwh']:>7.3f} "
              f"{last['cum_charge_rtm_mwh']:>7.3f} "
              f"{last['cum_discharge_dam_mwh']:>7.3f} "
              f"{last['cum_discharge_rtm_mwh']:>7.3f} "
              f"{last['cum_discharge_captive_mwh']:>7.3f} "
              f"Rs{last['cum_realized_revenue_rs']:>9,.0f} "
              f"Rs{last['cum_captive_revenue_rs']:>9,.0f} "
              f"{cap_short_total:>7.3f}")

    # Grand totals
    total_days = df["date"].nunique()
    eod_rows = df.groupby("date").last()
    total_rev   = eod_rows["cum_realized_revenue_rs"].sum()
    total_cap   = eod_rows["cum_captive_revenue_rs"].sum()
    total_short = df["captive_shortfall_mw"].sum() * DT
    total_cycles = eod_rows["cum_bess_cycles"].sum()
    print(f"\nTotal: {total_days} days | "
          f"Revenue: Rs {total_rev:,.0f} | "
          f"Captive: Rs {total_cap:,.0f} | "
          f"Cycles: {total_cycles:.2f} | "
          f"Captive Shortfall: {total_short:.3f} MWh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/phase3b_solar")
    args = parser.parse_args()
    export_all(args.results)
