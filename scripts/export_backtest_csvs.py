"""
scripts/export_backtest_csvs.py
=================================
Architecture v3 CSV export.

Reads daily result JSONs from run_phase3b_backtest.py (v3) and exports
ONE clean CSV with all Stage 1, 2B, 2A, and Actuals settlement columns.

New vs v2:
  - Actual solar routing columns: s_c_actual, s_cd_actual, c_d_actual, curtail_actual
  - Captive committed vs captive actual comparison
  - Captive shortfall column (DSM risk indicator)
  - Plan vs actual solar delta columns
  - solar_inverter_mw in parameters section
  - Revenue uses ACTUAL solar routing (not planned)
  - Fixed SOC = 2.5 MWh start/end (no chaining)

Run:
    python scripts/export_backtest_csvs.py
    python scripts/export_backtest_csvs.py --results results/phase3b_solar

Output: results/phase3b_solar/backtest_full_export_v3.csv
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
    """Safely index into a list/array."""
    if arr is None or idx >= len(arr):
        return default
    v = arr[idx]
    return float(v) if v is not None else default


def build_full_csv(records: list, ppa_rate: float = 3500.0) -> pd.DataFrame:
    """Build one row per (date, block) with all Stage 1/2B/2A/Actuals fields."""
    rows = []

    for rec in records:
        date = rec.get("date", "")

        # ── Parameters ────────────────────────────────────────────────────
        bess_p_max = rec.get("p_max_mw", 2.5)
        sol_inv_mw = rec.get("solar_inverter_mw", 25.0)
        exp_rev    = rec.get("expected_revenue", 0.0)
        soc_init   = rec.get("soc_initial_mwh", 2.5)

        # ── Stage 1 arrays ────────────────────────────────────────────────
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

        # ── Stage 2B arrays (revised plan) ────────────────────────────────
        s_c_rt     = rec.get("s_c_rt",       [0.0] * T_BLOCKS)
        s_cd_rt    = rec.get("s_cd_rt",      [0.0] * T_BLOCKS)
        c_d_rt     = rec.get("c_d_rt",       [0.0] * T_BLOCKS)
        z_nc_blend = rec.get("z_nc_blend",   solar_da)

        # ── Stage 2A arrays (RTM bids) ────────────────────────────────────
        y_c        = rec.get("y_c",          [0.0] * T_BLOCKS)
        y_d        = rec.get("y_d",          [0.0] * T_BLOCKS)

        # ── Actuals arrays (NEW in v3) ────────────────────────────────────
        s_c_actual     = rec.get("s_c_actual",      [0.0] * T_BLOCKS)
        s_cd_actual    = rec.get("s_cd_actual",     [0.0] * T_BLOCKS)
        c_d_actual     = rec.get("c_d_actual",      [0.0] * T_BLOCKS)
        curtail_actual = rec.get("curtail_actual",  [0.0] * T_BLOCKS)
        captive_actual = rec.get("captive_actual",  [0.0] * T_BLOCKS)
        captive_short  = rec.get("captive_shortfall",[0.0] * T_BLOCKS)
        captive_commit = rec.get("captive_committed",[0.0] * T_BLOCKS)

        # ── Physical SoC path ─────────────────────────────────────────────
        soc_realized = rec.get("soc_realized", [0.0] * (T_BLOCKS + 1))

        # ── Actual prices ─────────────────────────────────────────────────
        dam_act = rec.get("actual_dam_prices", [0.0] * T_BLOCKS)
        rtm_act = rec.get("actual_rtm_prices", [0.0] * T_BLOCKS)

        # ── Daily aggregates ──────────────────────────────────────────────
        ch_iex   = sum(float(x_c[t]) for t in range(T_BLOCKS)) * DT
        ch_solar = sum(safe_get(s_c_actual, t) for t in range(T_BLOCKS)) * DT
        ch_rtm   = sum(float(y_c[t]) for t in range(T_BLOCKS)) * DT
        dis_dam  = sum(float(x_d[t]) for t in range(T_BLOCKS)) * DT
        dis_rtm  = sum(float(y_d[t]) for t in range(T_BLOCKS)) * DT
        dis_cap  = sum(safe_get(c_d_actual, t) for t in range(T_BLOCKS)) * DT
        total_ch  = ch_iex + ch_solar + ch_rtm
        total_dis = dis_dam + dis_rtm + dis_cap
        daily_cycles = min(total_ch, total_dis) / USABLE_CAPACITY if USABLE_CAPACITY > 0 else 0

        daily_real = rec.get("realized_revenue", 0.0)
        daily_cap  = rec.get("captive_revenue", 0.0)
        daily_net  = rec.get("net_revenue", 0.0)
        daily_dsm  = rec.get("total_dsm_mwh", 0.0)

        # ── Per-block rows ────────────────────────────────────────────────
        for t in range(T_BLOCKS):
            xc_t   = float(x_c[t])
            xd_t   = float(x_d[t])
            yc_t   = float(y_c[t])
            yd_t   = float(y_d[t])

            # Stage 1 plan
            sc_da_t   = float(s_c_da[t])
            scd_da_t  = float(s_cd_da[t])
            cd_da_t   = float(c_d_da[t])
            cut_da_t  = float(curtail_da[t])
            sol_da_t  = float(solar_da[t])

            # Stage 2B revised plan
            sc_rt_t   = float(s_c_rt[t])
            scd_rt_t  = float(s_cd_rt[t])
            cd_rt_t   = float(c_d_rt[t])
            z_nc_t    = safe_get(z_nc_blend, t, sol_da_t)

            # Actuals (metered solar routing)
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

            # SoC
            soc_da_t = (float(np.mean([m[t + 1] for m in soc_matrix]))
                        if soc_matrix else 0.0)
            soc_rt_t = safe_get(soc_realized, t + 1)
            soc_gap_t = soc_da_t - soc_rt_t

            # ── Revenue (uses ACTUAL solar routing) ───────────────────────
            iex_charge  = pd_t * (-xc_t) * DT
            iex_dam_dis = pd_t * xd_t * DT
            iex_rtm_dis = pr_t * yd_t * DT
            iex_rtm_chg = pr_t * (-yc_t) * DT
            cap_rev_t   = ppa_rate * cap_at_t * DT
            rev_tot_t   = (iex_charge + iex_dam_dis + iex_rtm_dis
                           + iex_rtm_chg + cap_rev_t)

            trigger = max([b for b in sorted(RESCHEDULE_BLOCKS) if b <= t],
                          default=None)

            # Solar balance checks
            da_bal  = round(abs(sc_da_t + scd_da_t + cut_da_t - sol_da_t), 5)
            at_bal  = round(abs(sc_at_t + scd_at_t + cut_at_t - sol_at_t), 5)

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
                "soc_da_end_t_mwh":           round(soc_da_t, 4),
                "stage1_expected_revenue_rs": round(exp_rev, 0),

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
                "captive_plan_vs_actual_mw":  round(cap_com_t - cap_at_t, 4),

                # ── Physical SoC ──────────────────────────────────────────
                "soc_rt_start_t_mwh":         round(safe_get(soc_realized, t), 4),
                "soc_rt_end_t_mwh":           round(soc_rt_t, 4),
                "soc_da_vs_rt_gap_mwh":       round(soc_gap_t, 4),

                # ── Settlement revenue ────────────────────────────────────
                "iex_charge_cost_rs":         round(iex_charge, 2),
                "iex_dam_discharge_rs":       round(iex_dam_dis, 2),
                "iex_rtm_charge_cost_rs":     round(iex_rtm_chg, 2),
                "iex_rtm_discharge_rs":       round(iex_rtm_dis, 2),
                "iex_net_block_rs":           round(iex_charge + iex_dam_dis
                                                    + iex_rtm_chg + iex_rtm_dis, 2),
                "captive_revenue_rs":         round(cap_rev_t, 2),
                "total_block_revenue_rs":     round(rev_tot_t, 2),

                # ── Daily totals (repeated per row for pivot) ─────────────
                "daily_cycles_bess":          round(daily_cycles, 4),
                "daily_charge_iex_mwh":       round(ch_iex, 4),
                "daily_charge_solar_mwh":     round(ch_solar, 4),
                "daily_charge_rtm_mwh":       round(ch_rtm, 4),
                "daily_discharge_dam_mwh":    round(dis_dam, 4),
                "daily_discharge_rtm_mwh":    round(dis_rtm, 4),
                "daily_discharge_captive_mwh":round(dis_cap, 4),
                "daily_realized_revenue_rs":  round(daily_real, 0),
                "daily_captive_revenue_rs":   round(daily_cap, 0),
                "daily_net_revenue_rs":       round(daily_net, 0),
                "daily_dsm_mwh":              round(daily_dsm, 6),
            })

    return pd.DataFrame(rows)


def export_all(results_dir_str: str) -> None:
    results_dir = Path(results_dir_str)
    records = load_jsons(results_dir)

    ppa_rate = 3500.0
    bess_yaml = Path("config/bess.yaml")
    if bess_yaml.exists():
        import yaml
        with open(bess_yaml, encoding="ascii") as f:
            b = yaml.safe_load(f)
        ppa_rate = b.get("ppa_rate_rs_mwh", 3500.0)

    print(f"Building export for {len(records)} days, PPA rate={ppa_rate}...")
    df = build_full_csv(records, ppa_rate)

    out_path = results_dir / "backtest_full_export_v3.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(df):,} rows x {len(df.columns)} columns)")

    # Daily summary
    print()
    print("=" * 100)
    print("DAILY SUMMARY")
    print("=" * 100)
    print(f"{'Date':<12} {'Cycles':>7} {'ChIEX':>7} {'ChSol':>7} {'ChRTM':>7} "
          f"{'DiDAM':>7} {'DiRTM':>7} {'DiCap':>7} "
          f"{'IEX_Net':>11} {'Captive':>11} {'Net_Rev':>11} {'CapShort':>8}")

    for date, grp in df.groupby("date"):
        r = grp.iloc[0]
        cap_short_total = grp["captive_shortfall_mw"].sum() * DT
        print(f"{date:<12} "
              f"{r['daily_cycles_bess']:>7.3f} "
              f"{r['daily_charge_iex_mwh']:>7.3f} "
              f"{r['daily_charge_solar_mwh']:>7.3f} "
              f"{r['daily_charge_rtm_mwh']:>7.3f} "
              f"{r['daily_discharge_dam_mwh']:>7.3f} "
              f"{r['daily_discharge_rtm_mwh']:>7.3f} "
              f"{r['daily_discharge_captive_mwh']:>7.3f} "
              f"Rs{grp['iex_net_block_rs'].sum():>9,.0f} "
              f"Rs{r['daily_captive_revenue_rs']:>9,.0f} "
              f"Rs{r['daily_net_revenue_rs']:>9,.0f} "
              f"{cap_short_total:>7.3f}")

    # Aggregate stats
    total_days = df["date"].nunique()
    total_net  = df.groupby("date").first()["daily_net_revenue_rs"].sum()
    total_cap  = df.groupby("date").first()["daily_captive_revenue_rs"].sum()
    total_short = df["captive_shortfall_mw"].sum() * DT
    print(f"\nTotal: {total_days} days | "
          f"Net Revenue: Rs {total_net:,.0f} | "
          f"Captive Revenue: Rs {total_cap:,.0f} | "
          f"Captive Shortfall: {total_short:.3f} MWh")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/phase3b_solar")
    args = parser.parse_args()
    export_all(args.results)
