"""
scripts/export_backtest_csvs_v2.py
====================================
Export all Stage 1, 2B, 2A inputs/outputs + settlement to ONE clean CSV.

NEW vs v1:
  - Single CSV with all sections (no merge issues)
  - Daily BESS cycle count added
  - Revenue split into 3 clear streams:
      iex_charge_cost_rs    (negative — cost of IEX buying x_c)
      iex_dam_discharge_rs  (positive — revenue from IEX selling x_d)
      iex_rtm_discharge_rs  (positive — revenue from RTM selling y_d)
      captive_revenue_rs    (positive — PPA revenue from solar/BESS to captive)
  - z_sol_nc and z_sol_at clearly named (not renamed to blend/solar_blend)
  - SoC plan vs actual gap column added
  - Daily summary row appended at block 95

Run:
    python scripts/export_backtest_csvs_v2.py
    python scripts/export_backtest_csvs_v2.py --results results/phase3b_solar

Output: results/phase3b_solar/backtest_full_export_v2.csv
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


def get_nested(rec, *keys, default=0.0):
    v = rec
    for k in keys:
        if isinstance(v, dict):
            v = v.get(k, default)
        else:
            return default
    return v if v is not None else default


def build_full_csv(records: list, ppa_rate: float = 3500.0) -> pd.DataFrame:
    """
    Build one row per (date, block) with all fields.
    """
    rows = []

    for rec in records:
        date = rec.get("date", "")

        # --- Stage 1 arrays ---
        solar_da  = rec.get("solar_da",    [0.0] * T_BLOCKS)
        solar_at  = rec.get("solar_at",    [0.0] * T_BLOCKS)
        rtm_q50   = rec.get("rtm_q50_used",[0.0] * T_BLOCKS)
        x_c       = rec.get("x_c",         [0.0] * T_BLOCKS)
        x_d       = rec.get("x_d",         [0.0] * T_BLOCKS)
        s_c_da    = rec.get("s_c_da",      [0.0] * T_BLOCKS)
        s_cd_da   = rec.get("s_cd_da",     [0.0] * T_BLOCKS)
        c_d_da    = rec.get("c_d_da",      [0.0] * T_BLOCKS)
        curtail_da= rec.get("curtail_da",  [0.0] * T_BLOCKS)
        exp_rev   = rec.get("expected_revenue", 0.0)
        soc_init  = rec.get("soc_initial",  2.5)

        # Stage 1 SoC plan from scenarios
        scenarios = rec.get("scenarios", [])
        soc_matrix = []
        for s in scenarios:
            traj = s.get("soc", [])
            if len(traj) == T_BLOCKS + 1:
                soc_matrix.append([v if v is not None else 0.0 for v in traj])

        # --- Stage 2B arrays ---
        s_c_rt    = rec.get("s_c_rt",      [0.0] * T_BLOCKS)
        s_cd_rt   = rec.get("s_cd_rt",     [0.0] * T_BLOCKS)
        c_d_rt    = rec.get("c_d_rt",      [0.0] * T_BLOCKS)

        # --- Stage 2A outputs ---
        y_c       = rec.get("y_c",         [0.0] * T_BLOCKS)
        y_d       = rec.get("y_d",         [0.0] * T_BLOCKS)
        soc_actual= rec.get("soc_realized",[0.0] * (T_BLOCKS + 1))

        # --- Actual prices ---
        dam_act   = rec.get("actual_dam_prices",[0.0] * T_BLOCKS)
        rtm_act   = rec.get("actual_rtm_prices",[0.0] * T_BLOCKS)

        # Daily BESS cycle calculation
        ch_iex   = sum(float(x_c[t]) for t in range(T_BLOCKS)) * DT
        ch_solar  = sum(float(s_c_rt[t]) for t in range(T_BLOCKS)) * DT
        dis_iex_d = sum(float(x_d[t]) for t in range(T_BLOCKS)) * DT
        dis_iex_r = sum(float(y_d[t]) for t in range(T_BLOCKS)) * DT
        dis_cap   = sum(float(c_d_rt[t]) for t in range(T_BLOCKS)) * DT
        total_ch  = ch_iex + ch_solar
        total_dis = dis_iex_d + dis_iex_r + dis_cap
        daily_cycles = min(total_ch, total_dis) / USABLE_CAPACITY

        # Daily revenue totals
        daily_real = rec.get("realized_revenue", 0.0)
        daily_cap  = rec.get("captive_revenue",  0.0)
        daily_net  = rec.get("net_revenue",       0.0)
        daily_dsm  = rec.get("total_dsm_mwh",     0.0)

        for t in range(T_BLOCKS):
            xc_t  = float(x_c[t])
            xd_t  = float(x_d[t])
            sc_t  = float(s_c_da[t])
            scd_t = float(s_cd_da[t])
            cdd_t = float(c_d_da[t])
            cut_t = float(curtail_da[t])
            sol_t = float(solar_da[t])
            srat_t= float(solar_at[t])

            # Stage 2B
            scrt_t  = float(s_c_rt[t])
            scdrt_t = float(s_cd_rt[t])
            cdrt_t  = float(c_d_rt[t])

            # Stage 2A
            yc_t  = float(y_c[t])
            yd_t  = float(y_d[t])

            # Prices
            pd_t  = float(dam_act[t])
            pr_t  = float(rtm_act[t])
            pq50_t= float(rtm_q50[t])

            # SoC
            soc_plan_t = float(soc_matrix[0][t+1]) if soc_matrix else 0.0
            soc_mean_t = float(np.mean([m[t+1] for m in soc_matrix])) if soc_matrix else 0.0
            soc_act_t  = float(soc_actual[t+1]) if len(soc_actual) > t+1 else 0.0
            soc_gap_t  = soc_plan_t - soc_act_t   # positive = plan exceeded actual

            # Lag-4 conditioning signal
            lag4_t = float(rtm_act[t-4]) if t >= 4 else None

            # Revenue breakdown — CLEAR 3-stream split
            x_net_t     = xd_t - xc_t
            iex_charge  = pd_t * (-xc_t) * DT              # negative = cost
            iex_dam_dis = pd_t * xd_t * DT                 # positive = revenue
            iex_rtm_dis = pr_t * yd_t * DT                 # positive = revenue
            iex_rtm_chg = pr_t * (-yc_t) * DT              # negative = cost (rare)

            # Actual solar to captive
            sol_after_bess = max(0.0, srat_t - scrt_t)
            scd_at_t       = min(sol_after_bess, scdrt_t)
            cap_rev_t      = ppa_rate * (scd_at_t + cdrt_t) * DT

            # Total block revenue
            rev_tot_t = iex_charge + iex_dam_dis + iex_rtm_dis + iex_rtm_chg + cap_rev_t

            # Stage 2B trigger covering this block
            trigger = max([b for b in sorted(RESCHEDULE_BLOCKS) if b <= t], default=None)

            rows.append({
                # ── Identifiers ──────────────────────────────────────────
                "date":                       date,
                "block":                      t,
                "block_time_ist":             block_time(t),
                "is_reschedule_block":        int(t in RESCHEDULE_BLOCKS),
                "stage2b_trigger_block":      trigger,

                # ── Stage 1 Inputs ────────────────────────────────────────
                "z_sol_da_mw":                round(sol_t, 4),
                "z_sol_at_mw":                round(srat_t, 4),     # actual metered
                "p_max_mw":                   2.5,
                "e_max_mwh":                  4.75,
                "e_min_mwh":                  0.50,
                "e_max_plan_mwh":             4.5125,
                "e_min_plan_mwh":             0.525,
                "soc_initial_mwh":            round(soc_init, 4),
                "r_ppa_rs_mwh":               ppa_rate,
                "actual_dam_price_rs_mwh":    round(pd_t, 2),
                "actual_rtm_price_rs_mwh":    round(pr_t, 2),
                "p_rtm_q50_rs_mwh":           round(pq50_t, 2),
                "p_rtm_lag4_rs_mwh":          round(lag4_t, 2) if lag4_t is not None else "",

                # ── Stage 1 Outputs (DA plan — non-anticipative) ──────────
                "x_c_da_mw":                  round(xc_t, 4),   # IEX buy locked
                "x_d_da_mw":                  round(xd_t, 4),   # IEX sell locked
                "dam_net_mw":                 round(x_net_t, 4),
                "s_c_da_mw":                  round(sc_t, 4),   # solar→BESS
                "s_cd_da_mw":                 round(scd_t, 4),  # solar→captive
                "c_d_da_mw":                  round(cdd_t, 4),  # BESS→captive
                "curtail_da_mw":              round(cut_t, 4),
                "captive_da_mw":              round(scd_t + cdd_t, 4),
                "solar_balance_ok":           round(abs(sc_t + scd_t + cut_t - sol_t), 5),
                "soc_da_mean_mwh":            round(soc_mean_t, 4),
                "stage1_expected_revenue_rs": round(exp_rev, 0),

                # ── Stage 2B Outputs (revised solar routing) ──────────────
                "s_c_rt_mw":                  round(scrt_t, 4),
                "s_cd_rt_mw":                 round(scdrt_t, 4),
                "c_d_rt_mw":                  round(cdrt_t, 4),
                "captive_rt_mw":              round(scdrt_t + cdrt_t, 4),
                "z_sol_nc_blend_mw":          round(sol_t, 4),  # DA blend in CSV (NC in RAM only)

                # ── Stage 2A Inputs (what LP received) ────────────────────
                "soc_actual_start_mwh":       round(float(soc_actual[t]) if len(soc_actual) > t else 0.0, 4),
                # x_c and x_d locked (from Stage 1) — constants in Stage 2A LP
                "x_c_locked_mw":              round(xc_t, 4),
                "x_d_locked_mw":              round(xd_t, 4),
                "s_c_rt_fixed_mw":            round(scrt_t, 4),  # fixed before 2A
                "c_d_rt_fixed_mw":            round(cdrt_t, 4),  # fixed before 2A

                # ── Stage 2A Outputs ──────────────────────────────────────
                "y_c_mw":                     round(yc_t, 4),  # RTM charge
                "y_d_mw":                     round(yd_t, 4),  # RTM discharge
                "y_net_mw":                   round(yd_t - yc_t, 4),
                "soc_actual_end_mwh":         round(soc_act_t, 4),
                "soc_plan_vs_actual_gap_mwh": round(soc_gap_t, 4),  # >0 = plan higher than actual

                # ── Settlement Revenue (3-stream + captive) ───────────────
                "iex_charge_cost_rs":         round(iex_charge, 2),       # negative
                "iex_dam_discharge_rs":       round(iex_dam_dis, 2),      # positive
                "iex_rtm_charge_cost_rs":     round(iex_rtm_chg, 2),      # negative (rare)
                "iex_rtm_discharge_rs":       round(iex_rtm_dis, 2),      # positive
                "iex_net_block_rs":           round(iex_charge+iex_dam_dis+iex_rtm_chg+iex_rtm_dis, 2),
                "captive_solar_to_consumer_mw": round(scd_at_t, 4),
                "captive_bess_to_consumer_mw":  round(cdrt_t, 4),
                "captive_total_mw":           round(scd_at_t + cdrt_t, 4),
                "captive_revenue_rs":         round(cap_rev_t, 2),        # positive
                "total_block_revenue_rs":     round(rev_tot_t, 2),

                # ── Daily Totals (repeated for pivot convenience) ──────────
                "daily_cycles_bess":          round(daily_cycles, 4),
                "daily_charge_from_iex_mwh":  round(ch_iex, 4),
                "daily_charge_from_solar_mwh":round(ch_solar, 4),
                "daily_discharge_iex_dam_mwh":round(dis_iex_d, 4),
                "daily_discharge_iex_rtm_mwh":round(dis_iex_r, 4),
                "daily_discharge_captive_mwh":round(dis_cap, 4),
                "daily_realized_revenue_rs":  round(daily_real, 0),
                "daily_captive_revenue_rs":   round(daily_cap, 0),
                "daily_net_revenue_rs":       round(daily_net, 0),
                "daily_dsm_mwh":              round(daily_dsm, 6),
            })

    return pd.DataFrame(rows)


def export_all(results_dir_str: str) -> None:
    results_dir = Path(results_dir_str)
    records     = load_jsons(results_dir)

    # Try to read PPA rate from config
    ppa_rate = 3500.0
    bess_yaml = Path("config/bess.yaml")
    if bess_yaml.exists():
        import yaml
        with open(bess_yaml, encoding="ascii") as f:
            b = yaml.safe_load(f)
        ppa_rate = b.get("ppa_rate_rs_mwh", 3500.0)

    print(f"Building export for {len(records)} days, PPA rate={ppa_rate}...")
    df = build_full_csv(records, ppa_rate)

    out_path = results_dir / "backtest_full_export_v2.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}  ({len(df):,} rows × {len(df.columns)} columns)")

    # Print daily summary
    print()
    print("=" * 80)
    print("DAILY SUMMARY")
    print("=" * 80)
    print(f"{'Date':<12} {'Cycles':>8} {'ChIEX':>8} {'ChSol':>8} {'DisIEX':>8} {'DisCap':>8} "
          f"{'IEX_Net':>12} {'Captive':>12} {'Net_Rev':>12}")
    for date, grp in df.groupby("date"):
        r = grp.iloc[0]
        print(f"{date:<12} {r['daily_cycles_bess']:>8.4f} "
              f"{r['daily_charge_from_iex_mwh']:>8.3f} "
              f"{r['daily_charge_from_solar_mwh']:>8.3f} "
              f"{r['daily_discharge_iex_dam_mwh']+r['daily_discharge_iex_rtm_mwh']:>8.3f} "
              f"{r['daily_discharge_captive_mwh']:>8.3f} "
              f"Rs{grp['iex_net_block_rs'].sum():>10,.0f} "
              f"Rs{r['daily_captive_revenue_rs']:>10,.0f} "
              f"Rs{r['daily_net_revenue_rs']:>10,.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/phase3b_solar")
    args = parser.parse_args()
    export_all(args.results)
