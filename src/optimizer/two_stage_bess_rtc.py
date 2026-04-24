"""
src/optimizer/two_stage_bess_rtc.py — Architecture v10 RTC FINAL
=================================================================
Three-stage Solar+BESS optimizer with Round-the-Clock (RTC) captive contract.

HARDWARE: 25.4 MWp DC / 16.4 MW AC Inverter / 80 MWh BESS / 16.4 MW PCS
CONTRACT: 5 MW RTC ceiling | PPA Rs 5,000/MWh | 80% floor penalty trigger

KEY DESIGN:
  RTC_committed ∈ [0, 5.0 MW] — FREE LP variable, no hard lower bound.
  LP trades off: PPA revenue (Rs 5,000/MWh) vs IEX arbitrage revenue.
  At night BESS alone supplies captive; LP lowers RTC_committed on low-SoC days.
  80% penalty linearised via p_short[t] variable — LP avoids penalty in planning.
  SOD chained from prior day's actual EOD. Day-1 starts at e_max = 80 MWh.

STAGE 1  (D-1 10:00): Choose RTC_committed + DAM schedule
STAGE 2B (blocks 34/42/50/58): NC nowcast reschedule
STAGE 2A (every block): RTM MPC bid for B+3
SETTLEMENT: DSM + RTC shortfall penalty + IEX P&L
"""

import pulp
import numpy as np
from typing import Dict, List, Optional, Tuple

T_BLOCKS          = 96
DT                = 0.25
RESCHEDULE_BLOCKS = [34, 42, 50, 58]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_solar_band_mask_rtc(solar: np.ndarray,
                                threshold: float = 0.5,
                                buffer: int = 2) -> np.ndarray:
    """Bool mask True during solar-generation hours (±buffer blocks)."""
    mask  = np.zeros(len(solar), dtype=bool)
    solar_blocks = [t for t in range(len(solar)) if solar[t] > threshold]
    if solar_blocks:
        s = max(0, min(solar_blocks) - buffer)
        e = min(len(solar) - 1, max(solar_blocks) + buffer)
        mask[s:e + 1] = True
    return mask


def compute_setpoint_rtc(soc: float, schedule: float,
                         e_min: float, e_max: float,
                         eta_c: float, eta_d: float) -> float:
    """Bias setpoint ±10% around schedule based on SoC headroom."""
    dr = max(0.0, (soc - e_min) * eta_d)
    cr = max(0.0, (e_max - soc) / eta_c)
    br = dr / (dr + cr + 1e-9)
    return schedule * (0.9 + 0.2 * br)


def compute_contract_rate(rtc_committed: float, x_d: float, y_d: float,
                          p_dam: float, p_rtm: float, r_ppa: float) -> float:
    """Blended contract rate (CR) across PPA + DAM sell + RTM sell."""
    ppa = max(0.0, rtc_committed)
    dam = max(0.0, x_d)
    rtm = max(0.0, y_d)
    tot = ppa + dam + rtm
    return (ppa * r_ppa + dam * p_dam + rtm * p_rtm) / tot if tot > 1e-9 else r_ppa


def compute_dsm_settlement(captive_actual: float, scheduled: float,
                           cr: float, avail_cap: float) -> dict:
    """CERC DSM 2024 three-band settlement for one 15-min block."""
    act_mwh = captive_actual * DT
    sch_mwh = scheduled * DT
    dws     = (captive_actual - scheduled) * DT
    pct     = abs(dws) / avail_cap * 100.0 if avail_cap > 0 else 0.0
    over    = dws > 0

    # Charge rate table
    if pct <= 10.0:
        rate, mult, band = cr,        1.0,  "0-10%"
    elif pct <= 15.0:
        rate, mult, band = (0.90 * cr, 0.90, "10-15%") if over \
                      else (1.10 * cr, 1.10, "10-15%")
    else:
        rate, mult, band = (0.0,       0.0,  ">15%")   if over \
                      else (1.50 * cr, 1.50, ">15%")

    direction = "within" if pct <= 10 else ("over" if over else "under")
    r = {
        "dws_mwh": dws, "dws_pct": pct, "band": band,
        "direction": direction, "charge_rate": rate, "charge_rate_mult": mult,
        "net_captive_cash": 0.0, "dsm_penalty": 0.0, "dsm_haircut": 0.0,
        "financial_damage": 0.0,
        "under_revenue_received": 0.0, "under_dsm_penalty": 0.0,
        "under_net_cash": 0.0, "under_if_fully_sched": 0.0, "under_damage": 0.0,
        "over_revenue_sched": 0.0, "over_revenue_dev": 0.0,
        "over_total_received": 0.0, "over_if_all_cr": 0.0, "over_haircut": 0.0,
    }
    if pct <= 10.0:
        r["net_captive_cash"] = act_mwh * cr
    elif dws < 0:
        rev = act_mwh * cr; pen = abs(dws) * rate; net = rev - pen
        ifs = sch_mwh * cr
        r.update({"under_revenue_received": rev, "under_dsm_penalty": pen,
                  "under_net_cash": net, "under_if_fully_sched": ifs,
                  "under_damage": ifs - net, "net_captive_cash": net,
                  "dsm_penalty": pen, "financial_damage": ifs - net})
    else:
        rs  = sch_mwh * cr; rd = dws * rate; tr = rs + rd
        ia  = act_mwh * cr; hc = max(0.0, ia - tr)
        r.update({"over_revenue_sched": rs, "over_revenue_dev": rd,
                  "over_total_received": tr, "over_if_all_cr": ia,
                  "over_haircut": hc, "net_captive_cash": tr, "dsm_haircut": hc})
    return r


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1
# ══════════════════════════════════════════════════════════════════════════════

class TwoStageBESSRTC:
    """
    Stage 1 MILP — D-1 10:00 IST.

    Chooses the optimal flat RTC level and DAM schedule simultaneously,
    trading off PPA revenue (Rs 5,000/MWh) vs IEX arbitrage.

    RTC_committed ∈ [0, rtc_mw] — free variable.
    LP lowers it on low-SoC days to stay physically feasible.
    80% shortfall penalty is linearised via p_short[t].
    """

    def __init__(self, params, config: Dict):
        self.p           = params
        self.lambda_risk = config.get("lambda_risk", 0.0)
        self.risk_alpha  = config.get("risk_alpha",  0.10)

    def solve(self,
              dam_scenarios: np.ndarray,
              rtm_scenarios: np.ndarray,
              solar_da:      np.ndarray) -> Dict:
        """
        Parameters
        ----------
        dam_scenarios : (S, 96) float  — DAM price scenarios Rs/MWh
        rtm_scenarios : (S, 96) float  — RTM price scenarios Rs/MWh
        solar_da      : (96,)   float  — DA solar forecast MW
        """
        p     = self.p
        S     = dam_scenarios.shape[0]
        p_max = p.p_max_mw
        S_inv = p.solar_inverter_mw
        r_ppa = p.ppa_rate_rs_mwh
        soc_tv = float(p.soc_terminal_value_rs_mwh)

        solar_da   = np.clip(solar_da, 0.0, S_inv)
        solar_mask = compute_solar_band_mask_rtc(
            solar_da, p.solar_threshold_mw, p.solar_buffer_blocks)

        prob = pulp.LpProblem("Stage1_RTC", pulp.LpMaximize)

        # ── First-stage decision variables ────────────────────────────────
        # RTC_committed: free in [0, rtc_mw] — LP sets highest feasible level
        RTC_c   = pulp.LpVariable("RTC_c",   lowBound=p.rtc_min_mw, upBound=p.rtc_mw)
        x_c     = pulp.LpVariable.dicts("xc",  range(T_BLOCKS), 0, p_max)
        x_d     = pulp.LpVariable.dicts("xd",  range(T_BLOCKS), 0, p_max)
        s_c     = pulp.LpVariable.dicts("sc",  range(T_BLOCKS), 0, p_max)
        s_cd    = pulp.LpVariable.dicts("scd", range(T_BLOCKS), 0, S_inv)
        c_d     = pulp.LpVariable.dicts("cd",  range(T_BLOCKS), 0, p_max)
        # p_short[t]: linearised shortfall below 80% of RTC_committed
        p_short = pulp.LpVariable.dicts("psh", range(T_BLOCKS), 0)
        delta   = pulp.LpVariable.dicts("dlt", range(T_BLOCKS), cat="Binary")

        # ── Scenario variables ────────────────────────────────────────────
        soc = {si: pulp.LpVariable.dicts(
                   f"soc{si}", range(T_BLOCKS + 1), p.e_min_mwh, p.e_max_mwh)
               for si in range(S)}
        zeta = pulp.LpVariable("zeta")
        u    = pulp.LpVariable.dicts("u", range(S), lowBound=0)

        # ── First-stage constraints (added ONCE, not per scenario) ────────
        for t in range(T_BLOCKS):
            sol_t = float(solar_da[t])

            # C1: solar balance (no curtailment)
            prob += s_c[t] + s_cd[t] == sol_t,              f"C1_{t}"

            # C_RTC: flat constant delivery — the core RTC constraint
            # Night: s_cd=0 → c_d = RTC_c (BESS alone)
            # Day:   solar covers most, c_d tops up
            prob += s_cd[t] + c_d[t] == RTC_c,              f"CRTC_{t}"

            # Linearised 80% penalty: p_short[t] ≥ 0.8×RTC_c − delivery
            # Since delivery == RTC_c by C_RTC, p_short = max(0, -0.2×RTC_c) = 0
            # This is active in Stage 2 where RTC_c may not equal delivery
            prob += p_short[t] >= 0.8 * RTC_c - (s_cd[t] + c_d[t]), f"CPSH_{t}"

            # C2: PCS discharge limit
            prob += x_d[t] + c_d[t] <= p_max,               f"C2_{t}"

            # C3: AC bus mutual exclusion (TOPOLOGY CORRECTED)
            # Solar → DC bus → DC-DC converter → BESS  (s_c bypasses AC bus)
            # Solar → DC bus → Inverter → AC bus → Captive/Grid  (s_cd, x_d on AC)
            # Grid → AC bus → AC-DC rectifier → BESS  (x_c on AC bus)
            # Therefore:
            #   x_c (grid→BESS via AC): blocked when delta=0 (export mode)
            #   x_d (BESS→grid via AC): blocked when delta=1 (import mode)
            #   s_cd (solar→captive via AC): blocked when delta=1 (import mode)
            #   c_d (BESS→captive via AC): blocked when delta=1 (import mode)
            #   s_c (solar→BESS via DC): NOT on AC bus — NO delta restriction
            prob += x_c[t]  <= p_max * delta[t],                     f"C3a_{t}"
            prob += x_d[t] + c_d[t] <= p_max * (1 - delta[t]),      f"C3b_{t}"
            prob += s_cd[t] <= S_inv * (1 - delta[t]),               f"C3c_{t}"
            # s_c[t] is unrestricted by delta — DC charging path

        # ── Scenario loop: SoC dynamics + CVaR ───────────────────────────
        scen_revs = []
        for si in range(S):
            prob += soc[si][0] == p.soc_initial_mwh, f"SOD_{si}"

            if p.soc_terminal_mode == "hard":
                prob += soc[si][T_BLOCKS] == p.soc_terminal_min_mwh, f"EOD_{si}"
            else:
                # Soft floor = soc_terminal_min_mwh (40 MWh), not just e_min (8 MWh)
                # This prevents LP from draining BESS completely for IEX arb,
                # preserving next-day RTC headroom (need ~70 MWh SOD for 5 MW RTC)
                prob += soc[si][T_BLOCKS] >= p.soc_terminal_min_mwh, f"EOD_{si}"

            rev = 0
            for t in range(T_BLOCKS):
                pd_t = float(dam_scenarios[si, t])

                # C4: SoC dynamics
                prob += soc[si][t + 1] == (
                    soc[si][t]
                    + p.eta_charge    * (s_c[t] + x_c[t]) * DT
                    - (1.0 / p.eta_discharge) * (x_d[t] + c_d[t]) * DT
                ), f"C4_{si}_{t}"

                # C6: Solar band SoC constraint
                if solar_mask[t]:
                    prob += soc[si][t] >= p.soc_solar_low,  f"C6lo_{si}_{t}"
                    prob += soc[si][t] <= p.soc_solar_high, f"C6hi_{si}_{t}"

                # Revenue terms
                rev += pd_t  * x_d[t] * DT                          # DAM sell
                rev -= pd_t  * x_c[t] * DT                          # DAM buy
                rev += r_ppa * (s_cd[t] + c_d[t]) * DT              # PPA revenue
                rev -= p.iex_fee_rs_mwh * (x_c[t] + x_d[t]) * DT   # IEX fees
                rev -= r_ppa * p_short[t] * DT                       # 80% penalty

            # Terminal SoC value: reward keeping BESS charged for next day
            if soc_tv > 0:
                rev = rev + soc_tv * soc[si][T_BLOCKS] * DT

            prob += u[si] >= zeta - rev
            scen_revs.append(rev)

        # ── Objective ─────────────────────────────────────────────────────
        avg_rev = pulp.lpSum(scen_revs) / S
        cvar    = zeta - (1.0 / (S * self.risk_alpha)) * pulp.lpSum(u.values())
        prob.setObjective(avg_rev + self.lambda_risk * cvar)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        z96 = [0.0] * T_BLOCKS
        if pulp.LpStatus[prob.status] != "Optimal":
            return {
                "status": "Infeasible", "RTC_committed": 0.0,
                "x_c": z96, "x_d": z96,
                "s_c_da": z96, "s_cd_da": z96, "c_d_da": z96,
                "captive_da": z96, "dam_net": z96,
                "schedule_da": z96, "setpoint_da": z96,
                "solar_band_mask": [False] * T_BLOCKS,
                "expected_revenue": 0.0, "scenarios": [],
            }

        rtc_val = float(pulp.value(RTC_c) or 0.0)
        xc_v  = [max(0.0, pulp.value(x_c[t]) or 0.0) for t in range(T_BLOCKS)]
        xd_v  = [max(0.0, pulp.value(x_d[t]) or 0.0) for t in range(T_BLOCKS)]
        sc_v  = [max(0.0, pulp.value(s_c[t]) or 0.0) for t in range(T_BLOCKS)]
        scd_v = [max(0.0, pulp.value(s_cd[t])or 0.0) for t in range(T_BLOCKS)]
        cd_v  = [max(0.0, pulp.value(c_d[t]) or 0.0) for t in range(T_BLOCKS)]
        cap_da   = [scd_v[t] + cd_v[t] for t in range(T_BLOCKS)]
        dam_net  = [xd_v[t]  - xc_v[t]  for t in range(T_BLOCKS)]
        sched_da = [rtc_val  + dam_net[t] for t in range(T_BLOCKS)]
        soc_mean = [float(np.mean([pulp.value(soc[si][t]) or 0.0
                                   for si in range(S)]))
                    for t in range(T_BLOCKS + 1)]
        sp_da = [compute_setpoint_rtc(soc_mean[t], sched_da[t],
                                      p.e_min_mwh, p.e_max_mwh,
                                      p.eta_charge, p.eta_discharge)
                 for t in range(T_BLOCKS)]

        return {
            "status":           "Optimal",
            "expected_revenue": float(pulp.value(avg_rev) or 0.0),
            "RTC_committed":    rtc_val,
            "x_c":              xc_v,  "x_d": xd_v,
            "s_c_da":           sc_v,  "s_cd_da": scd_v, "c_d_da": cd_v,
            "captive_da":       cap_da, "dam_net": dam_net,
            "schedule_da":      sched_da, "setpoint_da": sp_da,
            "solar_band_mask":  solar_mask.tolist(),
            "scenarios":        [{"soc": [pulp.value(soc[si][t])
                                          for t in range(T_BLOCKS + 1)]}
                                 for si in range(S)],
        }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2B
# ══════════════════════════════════════════════════════════════════════════════

def reschedule_captive_rtc(params,
                            trigger_block:          int,
                            soc_actual:             float,
                            solar_nc_row:           np.ndarray,
                            solar_da:               np.ndarray,
                            rtm_q50:                np.ndarray,
                            x_c_s1:                 np.ndarray,
                            x_d_s1:                 np.ndarray,
                            y_c_committed:          np.ndarray,
                            y_d_committed:          np.ndarray,
                            rtc_committed:          float,
                            captive_committed_prev: np.ndarray,
                            rtc_notice:             np.ndarray,
                            cycle_used_so_far:      float = 0.0) -> Dict:
    """
    Stage 2B: revise solar routing with NC nowcast.
    Adjusts captive_rt within ±5% free band or wider if rtc_notice set.
    Issues rtc_notice for any block needing >5% revision.
    """
    p         = params
    B         = trigger_block
    remaining = T_BLOCKS - B
    p_max     = p.p_max_mw
    S_inv     = p.solar_inverter_mw
    r_ppa     = p.ppa_rate_rs_mwh
    RTM_LEAD  = p.rtm_lead_blocks
    CAP_BUF   = p.captive_buffer_blocks
    CAP_TOL   = p.captive_buffer_tolerance_mw

    # Solar blend: NC for next NC_WINDOW blocks, DA beyond
    solar_blend = np.zeros(remaining, dtype=float)
    for k in range(remaining):
        t_abs = B + k
        if k < len(solar_nc_row):
            solar_blend[k] = float(solar_nc_row[k])
        else:
            solar_blend[k] = float(solar_da[t_abs]) if t_abs < T_BLOCKS else 0.0
    solar_blend = np.clip(solar_blend, 0.0, S_inv)

    solar_mask = compute_solar_band_mask_rtc(
        solar_da, p.solar_threshold_mw, p.solar_buffer_blocks)

    xc_r  = np.array(x_c_s1[B:],        dtype=float)
    xd_r  = np.array(x_d_s1[B:],        dtype=float)
    yc_r  = np.array(y_c_committed[B:],  dtype=float)
    yd_r  = np.array(y_d_committed[B:],  dtype=float)
    rtm_r = np.array(rtm_q50[B:],        dtype=float)

    prob  = pulp.LpProblem(f"S2B_b{B}", pulp.LpMaximize)
    sc    = pulp.LpVariable.dicts("sc",  range(remaining), 0, p_max)
    scd   = pulp.LpVariable.dicts("scd", range(remaining), 0, S_inv)
    cd    = pulp.LpVariable.dicts("cd",  range(remaining), 0, p_max)
    psh   = pulp.LpVariable.dicts("psh", range(remaining), 0)
    soc_v = pulp.LpVariable.dicts("soc", range(remaining + 1),
                                  p.e_min_mwh, p.e_max_mwh)
    dl    = pulp.LpVariable.dicts("dl",  range(remaining), cat="Binary")

    # captive_rt bounds per block
    cap_rt = {}
    for k in range(remaining):
        t_abs = B + k
        if rtc_notice[t_abs]:
            lo, hi = p.rtc_min_mw, p.rtc_mw
        else:
            lo, hi = p.rtc_band(rtc_committed)
        cap_rt[k] = pulp.LpVariable(f"crt_{k}", lowBound=lo, upBound=hi)

    prob += soc_v[0] == float(np.clip(soc_actual, p.e_min_mwh, p.e_max_mwh))
    if p.soc_terminal_mode == "hard":
        prob += soc_v[remaining] == p.soc_terminal_min_mwh
    else:
        prob += soc_v[remaining] >= p.soc_terminal_min_mwh

    rtc_notice_out = rtc_notice.copy()
    rev = 0

    for k in range(remaining):
        t_abs = B + k
        xc_k  = float(xc_r[k])
        xd_k  = float(xd_r[k])
        yc_k  = float(yc_r[k]) if k < RTM_LEAD else 0.0
        yd_k  = float(yd_r[k]) if k < RTM_LEAD else 0.0
        pr_k  = float(rtm_r[k])
        sol_k = float(solar_blend[k])

        # Solar balance
        prob += sc[k] + scd[k] == sol_k,                     f"C1_{k}"
        # Flat delivery
        prob += scd[k] + cd[k] == cap_rt[k],                 f"CRTC_{k}"
        # 80% penalty
        prob += psh[k] >= 0.8 * cap_rt[k] - (scd[k] + cd[k]), f"PSH_{k}"
        # PCS limits
        prob += sc[k]  + xc_k + yc_k <= p_max,              f"PCS_c_{k}"
        prob += cd[k]  + xd_k + yd_k <= p_max,              f"PCS_d_{k}"
        # Mutual exclusion
        if xc_k + yc_k > 1e-6 and xd_k + yd_k < 1e-6:
            prob += dl[k] == 1
        elif xd_k + yd_k > 1e-6 and xc_k + yc_k < 1e-6:
            prob += dl[k] == 0
        prob += xc_k + yc_k      <= p_max * dl[k],          f"MEx_c_{k}"
        prob += xd_k + yd_k + cd[k] <= p_max * (1-dl[k]),  f"MEx_d_{k}"
        prob += scd[k]               <= S_inv * (1-dl[k]),  f"MEx_s_{k}"
        # sc[k] (solar→BESS via DC) not restricted by dl
        # Captive buffer smoothness
        if k < CAP_BUF:
            ct = float(captive_committed_prev[t_abs])
            prob += cap_rt[k] >= ct - CAP_TOL,               f"CBuf_lo_{k}"
            prob += cap_rt[k] <= ct + CAP_TOL,               f"CBuf_hi_{k}"
        # SoC dynamics
        prob += soc_v[k + 1] == (
            soc_v[k]
            + p.eta_charge    * (sc[k] + xc_k + yc_k) * DT
            - (1.0 / p.eta_discharge) * (cd[k] + xd_k + yd_k) * DT
        ), f"SOC_{k}"
        # Solar band
        if solar_mask[t_abs]:
            prob += soc_v[k] >= p.soc_solar_low,             f"SBlo_{k}"
            prob += soc_v[k] <= p.soc_solar_high,            f"SBhi_{k}"

        # Objective
        rev += r_ppa * (scd[k] + cd[k]) * DT
        rev += pr_k  * xd_k             * DT
        rev -= pr_k  * xc_k             * DT
        rev -= p.iex_fee_rs_mwh * (xc_k + xd_k) * DT
        rev -= r_ppa * psh[k]           * DT

    prob.setObjective(rev)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    status = pulp.LpStatus[prob.status]

    sc_out  = np.zeros(T_BLOCKS)
    scd_out = np.zeros(T_BLOCKS)
    cd_out  = np.zeros(T_BLOCKS)
    cap_out = np.full(T_BLOCKS, rtc_committed, dtype=float)

    if status == "Optimal":
        for k in range(remaining):
            t_abs = B + k
            sc_out[t_abs]  = max(0.0, pulp.value(sc[k])  or 0.0)
            scd_out[t_abs] = max(0.0, pulp.value(scd[k]) or 0.0)
            cd_out[t_abs]  = max(0.0, pulp.value(cd[k])  or 0.0)
            cap_val = float(pulp.value(cap_rt[k]) or rtc_committed)
            cap_out[t_abs] = cap_val
            # Issue rtc_notice for future blocks needing >5% revision
            dev = abs(cap_val - rtc_committed) / (rtc_committed + 1e-9)
            if dev > p.rtc_tol_pct:
                t_notice = t_abs + p.rtc_advance_blocks
                if t_notice < T_BLOCKS:
                    rtc_notice_out[t_notice] = True

    return {"status": status,
            "s_c_rt": sc_out, "s_cd_rt": scd_out, "c_d_rt": cd_out,
            "captive_rt": cap_out, "rtc_notice": rtc_notice_out}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2A
# ══════════════════════════════════════════════════════════════════════════════

def solve_stage2a_rtc(params,
                       block_B:           int,
                       soc_actual_B:      float,
                       dam_actual:        np.ndarray,
                       rtm_q50:           np.ndarray,
                       p_rtm_lag4:        float,
                       s_c_rt:            np.ndarray,
                       c_d_rt:            np.ndarray,
                       y_c_committed:     np.ndarray,
                       y_d_committed:     np.ndarray,
                       x_c_s1:            np.ndarray,
                       x_d_s1:            np.ndarray,
                       solar_da:          np.ndarray,
                       rtc_committed:     float,
                       captive_committed: np.ndarray,
                       rtc_notice:        np.ndarray,
                       cycle_used_so_far: float = 0.0) -> Tuple[float, float, np.ndarray]:
    """
    Stage 2A: receding-horizon MPC. Bids y_c/y_d for block B+RTM_LEAD.
    Issues rtc_notice[B+advance] for any >5% captive revision needed.
    """
    p         = params
    p_max     = p.p_max_mw
    r_ppa     = p.ppa_rate_rs_mwh
    RTM_LEAD  = p.rtm_lead_blocks
    bid_block = block_B + RTM_LEAD
    rtc_notice_out = rtc_notice.copy()

    if bid_block >= T_BLOCKS:
        return 0.0, 0.0, rtc_notice_out

    solar_mask = compute_solar_band_mask_rtc(
        solar_da, p.solar_threshold_mw, p.solar_buffer_blocks)

    # Lag-4 price bias adjustment
    rtm_adj = rtm_q50.copy().astype(float)
    if not np.isnan(p_rtm_lag4) and block_B >= 4:
        bias = p_rtm_lag4 - float(rtm_q50[block_B - 4])
        for t in range(bid_block, T_BLOCKS):
            rtm_adj[t] = max(0.0, rtm_adj[t] + bias * (0.85 ** (t - block_B)))

    # Roll SoC forward to bid_block using committed schedules
    soc_rf = float(np.clip(soc_actual_B, p.e_min_mwh, p.e_max_mwh))
    for t in range(block_B, bid_block):
        xc_t = float(x_c_s1[t]); xd_t = float(x_d_s1[t])
        yc_t = float(y_c_committed[t]); yd_t = float(y_d_committed[t])
        sc_t = float(s_c_rt[t]);  cd_t = float(c_d_rt[t])
        ch   = p.eta_charge    * (sc_t + xc_t + yc_t) * DT
        di   = (cd_t + xd_t + yd_t) / p.eta_discharge * DT
        soc_rf = float(np.clip(soc_rf + ch - di, p.e_min_mwh, p.e_max_mwh))

    remaining = T_BLOCKS - bid_block
    if remaining <= 0:
        return 0.0, 0.0, rtc_notice_out

    prob   = pulp.LpProblem(f"S2A_b{block_B}", pulp.LpMaximize)
    y_c    = pulp.LpVariable.dicts("yc",  range(remaining), 0, p_max)
    y_d    = pulp.LpVariable.dicts("yd",  range(remaining), 0, p_max)
    psh    = pulp.LpVariable.dicts("psh", range(remaining), 0)
    soc_lp = pulp.LpVariable.dicts("soc", range(remaining + 1),
                                   p.e_min_mwh, p.e_max_mwh)
    dl     = pulp.LpVariable.dicts("dl",  range(remaining), cat="Binary")

    cap_rt_lp = {}
    for k in range(remaining):
        t_abs = bid_block + k
        if t_abs < T_BLOCKS and rtc_notice[t_abs]:
            lo, hi = p.rtc_min_mw, p.rtc_mw
        else:
            lo, hi = p.rtc_band(rtc_committed)
        cap_rt_lp[k] = pulp.LpVariable(f"crt_{k}", lowBound=lo, upBound=hi)

    prob += soc_lp[0] == soc_rf
    if p.soc_terminal_mode == "hard":
        prob += soc_lp[remaining] == p.soc_terminal_min_mwh
    else:
        prob += soc_lp[remaining] >= p.soc_terminal_min_mwh

    rev = 0
    for k in range(remaining):
        ta   = bid_block + k
        xc_t = float(x_c_s1[ta]); xd_t = float(x_d_s1[ta])
        sc_t = float(s_c_rt[ta]); cd_t = float(c_d_rt[ta])
        pr   = float(rtm_adj[ta])

        prob += sc_t + xc_t + y_c[k] <= p_max,              f"PCS_c_{k}"
        prob += cd_t + xd_t + y_d[k] <= p_max,              f"PCS_d_{k}"
        prob += sc_t + cd_t == cap_rt_lp[k],                 f"CRTC_{k}"
        prob += psh[k] >= 0.8 * cap_rt_lp[k] - (sc_t + cd_t), f"PSH_{k}"

        imp = xc_t + sc_t; exp_ = xd_t + cd_t
        if imp > 1e-6 and exp_ < 1e-6:
            prob += dl[k] >= 1
        elif exp_ > 1e-6 and imp < 1e-6:
            prob += dl[k] <= 0
        prob += xc_t + y_c[k]        <= p_max * dl[k],     f"MEx_c_{k}"
        prob += xd_t + y_d[k] + cd_t <= p_max*(1-dl[k]),  f"MEx_d_{k}"
        # sc_t (DC solar charging) not restricted by dl

        prob += soc_lp[k + 1] == (
            soc_lp[k]
            + p.eta_charge    * (sc_t + xc_t + y_c[k]) * DT
            - (1.0 / p.eta_discharge) * (cd_t + xd_t + y_d[k]) * DT
        ), f"SOC_{k}"

        if solar_mask[ta]:
            prob += soc_lp[k] >= p.soc_solar_low,           f"SBlo_{k}"
            prob += soc_lp[k] <= p.soc_solar_high,          f"SBhi_{k}"

        rev += pr   * y_d[k]        * DT
        rev -= pr   * y_c[k]        * DT
        rev += r_ppa * cap_rt_lp[k] * DT
        rev -= p.iex_fee_rs_mwh * (y_c[k] + y_d[k]) * DT
        rev -= r_ppa * psh[k]       * DT

    prob.setObjective(rev)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] == "Optimal":
        y_c_bid = max(0.0, pulp.value(y_c[0]) or 0.0)
        y_d_bid = max(0.0, pulp.value(y_d[0]) or 0.0)
        for k in range(remaining):
            cap_val = float(pulp.value(cap_rt_lp[k]) or rtc_committed)
            dev = abs(cap_val - rtc_committed) / (rtc_committed + 1e-9)
            if dev > p.rtc_tol_pct:
                t_notice = bid_block + k + p.rtc_advance_blocks
                if t_notice < T_BLOCKS:
                    rtc_notice_out[t_notice] = True
        return y_c_bid, y_d_bid, rtc_notice_out

    return 0.0, 0.0, rtc_notice_out


# ══════════════════════════════════════════════════════════════════════════════
# ACTUALS SETTLEMENT
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_actuals_rtc(params,
                          stage1_result:    Dict,
                          dam_actual:       np.ndarray,
                          rtm_actual:       np.ndarray,
                          rtm_q50:          np.ndarray,
                          solar_da:         np.ndarray,
                          solar_nc:         np.ndarray,
                          solar_at:         np.ndarray,
                          reschedule_blocks: List[int] = RESCHEDULE_BLOCKS,
                          verbose:          bool = False) -> Dict:
    """Block-by-block settlement with DSM + RTC shortfall penalty + IEX P&L."""
    p             = params
    r_ppa         = p.ppa_rate_rs_mwh
    RTM_LEAD      = p.rtm_lead_blocks
    avail_cap     = p.avail_cap_mwh
    rtc_committed = float(stage1_result["RTC_committed"])
    rtc_threshold = p.rtc_penalty_threshold(rtc_committed)

    x_c_s1 = np.array(stage1_result["x_c"],    dtype=float)
    x_d_s1 = np.array(stage1_result["x_d"],    dtype=float)
    s_c_rt  = np.array(stage1_result["s_c_da"], dtype=float)
    s_cd_rt = np.array(stage1_result["s_cd_da"],dtype=float)
    c_d_rt  = np.array(stage1_result["c_d_da"], dtype=float)

    y_c_committed     = np.zeros(T_BLOCKS)
    y_d_committed     = np.zeros(T_BLOCKS)
    captive_committed = np.full(T_BLOCKS, rtc_committed, dtype=float)
    rtc_notice        = np.zeros(T_BLOCKS, dtype=bool)
    soc_path          = np.zeros(T_BLOCKS + 1)
    soc_path[0]       = p.soc_initial_mwh

    # Output arrays
    s_c_act   = np.zeros(T_BLOCKS); s_cd_act  = np.zeros(T_BLOCKS)
    c_d_act   = np.zeros(T_BLOCKS); cap_act   = np.zeros(T_BLOCKS)
    setpt     = np.zeros(T_BLOCKS); sch_rt    = np.zeros(T_BLOCKS)
    bl_capnet = np.zeros(T_BLOCKS); bl_cappen = np.zeros(T_BLOCKS)
    bl_iex    = np.zeros(T_BLOCKS); bl_deg    = np.zeros(T_BLOCKS)
    bl_net    = np.zeros(T_BLOCKS)
    nb_dsm    = np.zeros(T_BLOCKS); nb_rev    = np.zeros(T_BLOCKS)
    nb_pen    = np.zeros(T_BLOCKS)
    dsm_res   = []
    rtc_notice_issued = np.zeros(T_BLOCKS, dtype=bool)
    rtc_notice_target = np.full(T_BLOCKS, -1, dtype=int)
    cum_disch = 0.0

    for B in range(T_BLOCKS):
        lag4 = float(rtm_actual[B - 4]) if B >= 4 else np.nan

        # ── Stage 2B ───────────────────────────────────────────────────────
        if B in reschedule_blocks:
            nc_row = solar_nc[B] if B < len(solar_nc) else np.zeros(12)
            r2b = reschedule_captive_rtc(
                params=p, trigger_block=B, soc_actual=soc_path[B],
                solar_nc_row=nc_row, solar_da=solar_da, rtm_q50=rtm_q50,
                x_c_s1=x_c_s1, x_d_s1=x_d_s1,
                y_c_committed=y_c_committed, y_d_committed=y_d_committed,
                rtc_committed=rtc_committed,
                captive_committed_prev=captive_committed.copy(),
                rtc_notice=rtc_notice, cycle_used_so_far=cum_disch,
            )
            if r2b["status"] == "Optimal":
                s_c_rt[B:]            = r2b["s_c_rt"][B:]
                s_cd_rt[B:]           = r2b["s_cd_rt"][B:]
                c_d_rt[B:]            = r2b["c_d_rt"][B:]
                old_notice            = rtc_notice.copy()
                rtc_notice            = r2b["rtc_notice"]
                captive_committed[B:] = r2b["captive_rt"][B:]
                for tb in range(B, T_BLOCKS):
                    if rtc_notice[tb] and not old_notice[tb]:
                        rtc_notice_issued[B] = True
                        rtc_notice_target[B] = tb

        # ── Stage 2A ───────────────────────────────────────────────────────
        bid_b = B + RTM_LEAD
        if bid_b < T_BLOCKS:
            yc_b, yd_b, rtc_notice = solve_stage2a_rtc(
                params=p, block_B=B, soc_actual_B=soc_path[B],
                dam_actual=dam_actual, rtm_q50=rtm_q50, p_rtm_lag4=lag4,
                s_c_rt=s_c_rt, c_d_rt=c_d_rt,
                y_c_committed=y_c_committed, y_d_committed=y_d_committed,
                x_c_s1=x_c_s1, x_d_s1=x_d_s1, solar_da=solar_da,
                rtc_committed=rtc_committed, captive_committed=captive_committed,
                rtc_notice=rtc_notice, cycle_used_so_far=cum_disch,
            )
            y_c_committed[bid_b] = yc_b
            y_d_committed[bid_b] = yd_b

        # ── Step 1: Inputs ─────────────────────────────────────────────────
        xc_B = float(x_c_s1[B]); xd_B = float(x_d_s1[B])
        yc_B = float(y_c_committed[B]); yd_B = float(y_d_committed[B])
        cap_rt_B  = float(s_cd_rt[B] + c_d_rt[B])
        sch_rt_B  = cap_rt_B + (xd_B - xc_B) + (yd_B - yc_B)
        sch_rt[B] = sch_rt_B
        z_at      = float(solar_at[B])

        sp_B = compute_setpoint_rtc(soc_path[B], sch_rt_B,
                                    p.e_min_mwh, p.e_max_mwh,
                                    p.eta_charge, p.eta_discharge)
        setpt[B] = sp_B

        # Discharge capacity for BESS→captive (c_d only)
        # x_d is fixed from LP plan but capped by remaining SoC
        # Reserve enough SoC to survive remaining night blocks at RTC level
        blocks_remaining = T_BLOCKS - B
        rtc_reserve_soc  = min(
            rtc_committed * blocks_remaining * DT / p.eta_discharge,
            p.e_max_mwh - p.e_min_mwh
        )
        soc_after_reserve = max(p.e_min_mwh,
                                soc_path[B] - rtc_reserve_soc)

        # Cap x_d to not drain below reserve
        xd_avail   = max(0.0, (soc_path[B] - soc_after_reserve - p.e_min_mwh)
                         * p.eta_discharge / DT)
        xd_B_used  = min(xd_B, xd_avail)

        disch_cap = max(0.0, min(
            p.p_max_mw - xd_B_used - yd_B,
            (soc_path[B] - p.e_min_mwh) * p.eta_discharge / DT))
        charg_cap = max(0.0, min(
            p.p_max_mw - xc_B - yc_B,
            (p.e_max_mwh - soc_path[B]) / (p.eta_charge * DT)))
        xd_B = xd_B_used   # use capped value in SoC update and P&L

        # ── Step 2: Dispatch (Case A/B/C) ─────────────────────────────────
        if z_at > sp_B + 1e-6:                              # Case A: surplus
            sc_a  = min(charg_cap, z_at - sp_B)
            scd_a = z_at - sc_a
            cd_a  = 0.0
        elif z_at < sp_B - 1e-6:                            # Case B: deficit
            scd_a = z_at
            cd_a  = min(disch_cap, sp_B - z_at)
            sc_a  = 0.0
        else:                                                # Case C: exact
            scd_a = z_at; sc_a = 0.0; cd_a = 0.0

        cap_a = scd_a + cd_a
        s_c_act[B]  = sc_a; s_cd_act[B] = scd_a
        c_d_act[B]  = cd_a; cap_act[B]  = cap_a

        # ── Step 3: DSM ────────────────────────────────────────────────────
        cr  = compute_contract_rate(rtc_committed, xd_B, yd_B,
                                    float(dam_actual[B]), float(rtm_actual[B]),
                                    r_ppa)
        dsm = compute_dsm_settlement(cap_a, sch_rt_B, cr, avail_cap)
        dsm_res.append(dsm)

        # ── Step 4: RTC shortfall penalty ─────────────────────────────────
        short_mw  = max(0.0, rtc_threshold - cap_a)
        short_mwh = short_mw * DT
        rtc_pen   = short_mwh * r_ppa
        bl_cappen[B] = rtc_pen
        bl_capnet[B] = dsm["net_captive_cash"] - rtc_pen

        # ── Step 5: IEX ────────────────────────────────────────────────────
        iex_dam  = float(dam_actual[B]) * (xd_B - xc_B) * DT
        iex_rtm  = float(rtm_actual[B]) * (yd_B - yc_B) * DT
        iex_fees = p.iex_fee_rs_mwh * (xc_B + xd_B + yc_B + yd_B) * DT
        bl_iex[B] = iex_dam + iex_rtm - iex_fees

        # ── Step 6: SoC update ─────────────────────────────────────────────
        tot_d = xd_B + cd_a + yd_B
        ch_e  = p.eta_charge * (sc_a + xc_B + yc_B) * DT
        di_e  = tot_d / p.eta_discharge * DT
        soc_path[B + 1] = float(np.clip(
            soc_path[B] + ch_e - di_e, p.e_min_mwh, p.e_max_mwh))
        cum_disch += tot_d * DT / p.eta_discharge

        # ── Step 7: Block P&L ──────────────────────────────────────────────
        deg = p.degradation_cost_rs_mwh * tot_d * DT
        bl_deg[B] = deg
        bl_net[B] = bl_capnet[B] + bl_iex[B] - deg

        # ── Step 8: No-BESS counterfactual ────────────────────────────────
        nb_d   = compute_dsm_settlement(z_at, float(captive_committed[B]), r_ppa, avail_cap)
        nb_short = max(0.0, rtc_threshold - z_at) * DT * r_ppa
        nb_dsm[B] = nb_d["dsm_penalty"] + nb_d["dsm_haircut"]
        nb_pen[B] = nb_short
        nb_rev[B] = z_at * DT * r_ppa - nb_dsm[B] - nb_short

        if verbose and B % 16 == 0:
            print(f"  B={B:02d} soc={soc_path[B]:.1f}→{soc_path[B+1]:.1f} "
                  f"sol={z_at:.2f} cap={cap_a:.2f} rtc_pen={rtc_pen:,.0f} "
                  f"iex={bl_iex[B]:,.0f} net={bl_net[B]:,.0f}")

    # ── Aggregates ────────────────────────────────────────────────────────
    with_dsm     = sum(d["dsm_penalty"] + d["dsm_haircut"] for d in dsm_res)
    bess_dsm_sav = float(np.sum(nb_dsm)) - with_dsm
    bess_rtc_sav = float(np.sum(nb_pen)) - float(np.sum(bl_cappen))

    return {
        "net_revenue":           float(np.sum(bl_net)),
        "captive_net_total":     float(np.sum(bl_capnet)),
        "iex_net_total":         float(np.sum(bl_iex)),
        "rtc_penalty_total":     float(np.sum(bl_cappen)),
        "degradation_total":     float(np.sum(bl_deg)),
        "no_bess_revenue_total": float(np.sum(nb_rev)),
        "bess_dsm_savings":      bess_dsm_sav,
        "bess_rtc_pen_savings":  bess_rtc_sav,
        "bess_total_value":      bess_dsm_sav + bess_rtc_sav
                                 + float(np.sum(bl_iex))
                                 - float(np.sum(bl_deg)),
        "soc_path":              soc_path,
        "s_c_actual":            s_c_act,   "s_cd_actual":  s_cd_act,
        "c_d_actual":            c_d_act,   "captive_actual": cap_act,
        "setpoint":              setpt,      "schedule_rt":  sch_rt,
        "captive_committed":     captive_committed,
        "block_captive_net":     bl_capnet,  "block_captive_penalty": bl_cappen,
        "block_iex_net":         bl_iex,     "block_degradation":     bl_deg,
        "block_net":             bl_net,
        "no_bess_dsm":           nb_dsm,     "no_bess_rtc_penalty":   nb_pen,
        "no_bess_revenue":       nb_rev,
        "dsm_results":           dsm_res,
        "rtc_notice_issued":     rtc_notice_issued,
        "rtc_notice_target":     rtc_notice_target,
        "x_c":  x_c_s1,  "x_d":  x_d_s1,
        "y_c":  y_c_committed, "y_d": y_d_committed,
        "s_c_rt": s_c_rt, "s_cd_rt": s_cd_rt, "c_d_rt": c_d_rt,
    }
