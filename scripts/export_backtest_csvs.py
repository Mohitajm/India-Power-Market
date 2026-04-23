"""
scripts/export_backtest_csvs.py — Architecture v9_revised (v2)
block_net = block_captive_net - block_degradation (no separate iex_net).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

T_BLOCKS = 96; DT = 0.25
RESCHEDULE_BLOCKS = {34, 42, 50, 58}
USABLE_CAP = 4.25


def block_time(b):
    return f"{(b*15)//60:02d}:{(b*15)%60:02d}"


def sg(arr, i, d=0.0):
    if arr is None or i >= len(arr): return d
    v = arr[i]; return float(v) if v is not None else d


def load_jsons(rdir):
    dd = Path(rdir) / "daily"
    files = sorted(dd.glob("result_*.json"))
    if not files: raise FileNotFoundError(f"No results in {dd}")
    print(f"Loading {len(files)} daily result files...")
    return [json.load(open(f)) for f in files]


def build_csv(records, ppa, deg_cost):
    avail_cap = 25.0 * DT
    rows = []
    for rec in records:
        date = rec.get("date", "")
        p_max = rec.get("p_max_mw", 2.5)
        s_inv = rec.get("solar_inverter_mw", 25.0)

        x_c = rec.get("x_c", [0]*96); x_d = rec.get("x_d", [0]*96)
        sc_da = rec.get("s_c_da", [0]*96); scd_da = rec.get("s_cd_da", [0]*96)
        cd_da = rec.get("c_d_da", [0]*96); cap_da = rec.get("captive_da", [0]*96)
        sched_da = rec.get("schedule_da", [0]*96); sp_da = rec.get("setpoint_da", [0]*96)
        sol_da = rec.get("solar_da", [0]*96); sol_at = rec.get("solar_at", [0]*96)
        band_mask = rec.get("solar_band_mask", [False]*96)

        sc_rt = rec.get("s_c_rt", [0]*96); scd_rt = rec.get("s_cd_rt", [0]*96)
        cd_rt = rec.get("c_d_rt", [0]*96)
        y_c = rec.get("y_c", [0]*96); y_d = rec.get("y_d", [0]*96)
        cap_comm = rec.get("captive_committed", [0]*96)
        sp_rt = rec.get("setpoint_rt", [0]*96); sched_rt = rec.get("schedule_rt", [0]*96)
        z_nc = rec.get("z_nc_blend", sol_da)

        sc_act = rec.get("s_c_actual", [0]*96); scd_act = rec.get("s_cd_actual", [0]*96)
        cd_act = rec.get("c_d_actual", [0]*96); cap_act = rec.get("captive_actual", [0]*96)
        soc_real = rec.get("soc_realized", [0]*97)

        dam_p = rec.get("actual_dam_prices", [0]*96); rtm_p = rec.get("actual_rtm_prices", [0]*96)
        rtm_q = rec.get("rtm_q50_used", [0]*96)

        dsm_res = rec.get("dsm_results", [{}]*96)
        b_cap_net = rec.get("block_captive_net", [0]*96)
        b_deg = rec.get("block_degradation", [0]*96)
        b_net = rec.get("block_net", [0]*96)
        nb_dsm = rec.get("no_bess_dsm", [0]*96)
        nb_rev = rec.get("no_bess_revenue", [0]*96)

        scenarios = rec.get("scenarios", [])
        soc_matrix = [[v or 0 for v in s.get("soc",[])]
                       for s in scenarios if len(s.get("soc",[])) == 97]

        cum = {k: 0.0 for k in ["cap_net","dsm_pen","dsm_hair","deg",
                                 "net_rev","bess_val","nb_rev",
                                 "ch_sol","ch_grid","dis"]}

        for t in range(96):
            xc = float(x_c[t]); xd = float(x_d[t])
            yc = float(y_c[t]); yd = float(y_d[t])
            dn = xd - xc; rn = yd - yc
            dr = dsm_res[t] if t < len(dsm_res) else {}

            cap_net_t = sg(b_cap_net, t)
            deg_t = sg(b_deg, t)
            net_t = cap_net_t - deg_t
            nb_rev_t = sg(nb_rev, t)
            bess_val_t = net_t - nb_rev_t

            cum["cap_net"] += cap_net_t; cum["deg"] += deg_t
            cum["net_rev"] += net_t; cum["bess_val"] += bess_val_t
            cum["nb_rev"] += nb_rev_t
            cum["dsm_pen"] += dr.get("dsm_penalty", 0)
            cum["dsm_hair"] += dr.get("dsm_haircut", 0)
            cum["ch_sol"] += sg(sc_act, t) * DT
            cum["ch_grid"] += (xc + yc) * DT
            cum["dis"] += sg(cd_act, t) * DT
            tc = min(cum["ch_sol"] + cum["ch_grid"], cum["dis"])
            cum_cyc = tc / USABLE_CAP if USABLE_CAP > 0 else 0

            soc_da_s = float(np.mean([m[t] for m in soc_matrix])) if soc_matrix else 0
            soc_da_e = float(np.mean([m[t+1] for m in soc_matrix])) if soc_matrix else 0
            lag4 = float(rtm_p[t-4]) if t >= 4 else None
            trig = max([b for b in sorted(RESCHEDULE_BLOCKS) if b <= t], default=None)

            rows.append({
                "date": date, "block": t, "block_time_ist": block_time(t),
                "is_reschedule_block": int(t in RESCHEDULE_BLOCKS),
                "stage2b_trigger_block": trig,
                "p_max_mw": p_max, "solar_inverter_mw": s_inv,
                "r_ppa_rs_mwh": ppa, "avail_cap_mwh": avail_cap,
                "z_sol_da_mw": round(sg(sol_da,t),4),
                "z_sol_nc_mw": round(sg(z_nc,t),4),
                "z_sol_at_mw": round(sg(sol_at,t),4),
                "sol_forecast_error_mw": round(sg(sol_da,t)-sg(sol_at,t),4),
                "actual_dam_price_rs_mwh": round(sg(dam_p,t),2),
                "actual_rtm_price_rs_mwh": round(sg(rtm_p,t),2),
                "p_rtm_q50_rs_mwh": round(sg(rtm_q,t),2),
                "p_rtm_lag4_rs_mwh": round(lag4,2) if lag4 is not None else "",
                "x_c_da_mw": round(xc,4), "x_d_da_mw": round(xd,4),
                "dam_net_mw": round(dn,4),
                "s_c_da_mw": round(sg(sc_da,t),4),
                "s_cd_da_mw": round(sg(scd_da,t),4),
                "c_d_da_mw": round(sg(cd_da,t),4),
                "captive_da_mw": round(sg(cap_da,t),4),
                "schedule_da_mw": round(sg(sched_da,t),4),
                "setpoint_da_mw": round(sg(sp_da,t),4),
                "soc_da_start_mwh": round(soc_da_s,4),
                "soc_da_end_mwh": round(soc_da_e,4),
                "s_c_rt_mw": round(sg(sc_rt,t),4),
                "s_cd_rt_mw": round(sg(scd_rt,t),4),
                "c_d_rt_mw": round(sg(cd_rt,t),4),
                "captive_rt_mw": round(sg(scd_rt,t)+sg(cd_rt,t),4),
                "schedule_rt_mw": round(sg(sched_rt,t),4),
                "setpoint_rt_mw": round(sg(sp_rt,t),4),
                "captive_committed_mw": round(sg(cap_comm,t),4),
                "y_c_mw": round(yc,4), "y_d_mw": round(yd,4),
                "y_net_mw": round(rn,4),
                "active_setpoint_mw": round(sg(sp_rt,t),4),
                "s_c_actual_mw": round(sg(sc_act,t),4),
                "s_cd_actual_mw": round(sg(scd_act,t),4),
                "c_d_actual_mw": round(sg(cd_act,t),4),
                "captive_actual_mw": round(sg(cap_act,t),4),
                "dispatch_case": ("over" if sg(sol_at,t)>sg(sp_rt,t)+0.001
                                  else "under" if sg(sol_at,t)<sg(sp_rt,t)-0.001
                                  else "exact"),
                "soc_actual_start_mwh": round(sg(soc_real,t),4),
                "soc_actual_end_mwh": round(sg(soc_real,t+1),4),
                "is_solar_band": int(bool(sg(band_mask,t,False))),
                "contract_rate_rs_mwh": round(
                    dr.get("charge_rate",ppa)/max(dr.get("charge_rate_mult",1),0.01),2
                ) if dr.get("charge_rate_mult",1) > 0 else round(ppa,2),
                "actual_total_mw": round(sg(cap_act,t),4),
                "scheduled_total_mw": round(sg(sched_rt,t),4),
                "deviation_mwh": round(dr.get("dws_mwh",0),6),
                "deviation_pct": round(dr.get("dws_pct",0),2),
                "deviation_band": dr.get("band",""),
                "deviation_direction": dr.get("direction",""),
                "charge_rate_rs_mwh": round(dr.get("charge_rate",0),2),
                "charge_rate_multiplier": round(dr.get("charge_rate_mult",0),2),
                "under_revenue_received_rs": round(dr.get("under_revenue_received",0),2),
                "under_dsm_penalty_rs": round(dr.get("under_dsm_penalty",0),2),
                "under_net_cash_flow_rs": round(dr.get("under_net_cash",0),2),
                "under_if_fully_scheduled_rs": round(dr.get("under_if_fully_sched",0),2),
                "under_financial_damage_rs": round(dr.get("under_damage",0),2),
                "over_revenue_sched_qty_rs": round(dr.get("over_revenue_sched",0),2),
                "over_revenue_dev_qty_rs": round(dr.get("over_revenue_dev",0),2),
                "over_total_received_rs": round(dr.get("over_total_received",0),2),
                "over_if_all_at_cr_rs": round(dr.get("over_if_all_cr",0),2),
                "over_revenue_haircut_rs": round(dr.get("over_haircut",0),2),
                "block_captive_net_rs": round(cap_net_t,2),
                "block_degradation_rs": round(deg_t,2),
                "block_net_rs": round(net_t,2),
                "no_bess_deviation_pct": round(
                    abs(sg(sol_at,t)-sg(cap_comm,t))*DT/avail_cap*100,2
                ) if avail_cap > 0 else 0,
                "no_bess_dsm_total_rs": round(sg(nb_dsm,t),2),
                "no_bess_revenue_rs": round(nb_rev_t,2),
                "bess_dsm_savings_rs": round(
                    sg(nb_dsm,t)-(dr.get("dsm_penalty",0)+dr.get("dsm_haircut",0)),2),
                "bess_total_value_rs": round(bess_val_t,2),
                "cum_bess_cycles": round(cum_cyc,4),
                "cum_captive_net_rs": round(cum["cap_net"],2),
                "cum_dsm_penalty_rs": round(cum["dsm_pen"],2),
                "cum_dsm_haircut_rs": round(cum["dsm_hair"],2),
                "cum_degradation_rs": round(cum["deg"],2),
                "cum_net_revenue_rs": round(cum["net_rev"],2),
                "cum_bess_value_rs": round(cum["bess_val"],2),
                "cum_no_bess_revenue_rs": round(cum["nb_rev"],2),
            })
    return pd.DataFrame(rows)


def export_all(results_dir_str):
    rdir = Path(results_dir_str)
    records = load_jsons(rdir)
    ppa, deg = 3500.0, 650.0
    by = Path("config/bess.yaml")
    if by.exists():
        import yaml
        with open(by, encoding="utf-8") as f:
            b = yaml.safe_load(f)
        ppa = b.get("ppa_rate_rs_mwh", 3500.0)
        deg = b.get("degradation_cost_rs_mwh", 650.0)
    df = build_csv(records, ppa, deg)
    out = rdir / "backtest_full_export_v9.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out} ({len(df):,} rows x {len(df.columns)} cols)")
    print(f"\n{'Date':<12} {'Net_Rev':>11} {'Cap_Net':>11} {'Deg':>9} "
          f"{'BESS_Val':>11} {'Cycles':>7}")
    for d, g in df.groupby("date"):
        r = g.iloc[-1]
        print(f"{d:<12} Rs{r['cum_net_revenue_rs']:>9,.0f} "
              f"Rs{r['cum_captive_net_rs']:>9,.0f} "
              f"Rs{r['cum_degradation_rs']:>7,.0f} "
              f"Rs{r['cum_bess_value_rs']:>9,.0f} "
              f"{r['cum_bess_cycles']:>7.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/phase3b_solar")
    export_all(parser.parse_args().results)
