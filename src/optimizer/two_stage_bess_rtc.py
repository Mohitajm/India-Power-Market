"""
src/optimizer/two_stage_bess_rtc.py — Architecture v10 RTC
===========================================================
Three-stage Solar+BESS optimizer with Round-the-Clock (RTC) captive contract.

KEY CHANGES vs two_stage_bess.py (v9_revised):
  1. Hardware resized: 80 MWh BESS, 16.4 MW PCS, 16.4 MW inverter.
  2. RTC captive replaces variable captive:
       - Stage 1 selects a scalar RTC_committed ∈ [rtc_min_mw, rtc_mw]
       - captive_da[t] = RTC_committed for ALL 96 blocks (flat constant)
       - Stage 2A/2B keep captive_rt within ±5% free band unless 16-block
         advance notice has been committed.
  3. SOD = EOD = 40 MWh (hard equality both ends).
  4. No curtailment variable: s_c + s_cd == solar_da exactly.
  5. Degradation is POST-HOC only (not in LP objective).
  6. IEX fee IS in the LP objective for x_c, x_d.
  7. RTC captive penalty computed in actuals settlement (not LP).
  8. Stage 2A: receding-horizon MPC, issues rtc_notice for >5% revisions.

STAGE 1: TwoStageBESSRTC.solve()
  Decision variables: RTC_committed (scalar), x_c[t], x_d[t], s_c_da[t],
  s_cd_da[t], c_d_da[t], delta[t], soc[s][t].
  Constraint: s_cd_da[t] + c_d_da[t] == RTC_committed ∀t (flat RTC).

STAGE 2B: reschedule_captive_rtc()
  Revises solar routing and captive_rt using NC nowcast.
  captive_rt stays within ±5% free band unless rtc_notice is pre-committed.

STAGE 2A: solve_stage2a_rtc()
  Receding-horizon per block. Bids y_c/y_d for B+RTM_LEAD.
  May commit rtc_notice[B+16] if SoC/solar conditions warrant >5% revision.

evaluate_actuals_rtc():
  Block-by-block settlement:
    Step 1: Inputs (setpoint, solar_at, SoC)
    Step 2: Dispatch (Case A/B/C)
    Step 3: DSM deviation settlement
    Step 4: RTC captive penalty (separate from DSM, if captive_actual < rtc_min_mw)
    Step 5: IEX revenue
    Step 6: SoC update
    Step 7: Block P&L
    Step 8: No-BESS counterfactual
"""

import pulp
import numpy as np
from typing import Dict, List, Optional, Tuple

T_BLOCKS = 96
DT = 0.25
RESCHEDULE_BLOCKS = [34, 42, 50, 58]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_solar_band_mask_rtc(solar_profile: np.ndarray,
                                threshold: float = 0.5,
                                buffer: int = 2) -> np.ndarray:
    """Build solar-hours boolean mask with transition buffer."""
    n = len(solar_profile)
    mask = np.zeros(n, dtype=bool)
    solar_blocks = [t for t in range(n) if solar_profile[t] > threshold]
    if solar_blocks:
        start = max(0, min(solar_blocks) - buffer)
        end   = min(n - 1, max(solar_blocks) + buffer)
        mask[start:end + 1] = True
    return mask


def compute_setpoint_rtc(soc_val: float, schedule_val: float,
                         e_min: float, e_max: float,
                         eta_c: float, eta_d: float) -> float:
    """
    Derive dispatch setpoint from SoC heuristic.
    SoC = e_min  → setpoint = 0.90 × schedule  (bias low)
    SoC = mid    → setpoint = 1.00 × schedule  (neutral)
    SoC = e_max  → setpoint = 1.10 × schedule  (bias high)
    """
    discharge_room = max(0.0, (soc_val - e_min) * eta_d)
    charge_room    = max(0.0, (e_max - soc_val) / eta_c)
    total          = discharge_room + charge_room + 1e-9
    bias_ratio     = discharge_room / total
    return schedule_val * (0.9 + 0.2 * bias_ratio)


def compute_contract_rate_rtc(rtc_committed: float, x_d: float, y_d: float,
                               p_dam: float, p_rtm: float,
                               r_ppa: float) -> float:
    """Blended contract rate across captive PPA, DAM sell, and RTM sell."""
    ppa_mw    = max(0.0, rtc_committed)
    dam_sell  = max(0.0, x_d)
    rtm_sell  = max(0.0, y_d)
    total     = ppa_mw + dam_sell + rtm_sell
    if total > 1e-9:
        return (ppa_mw * r_ppa + dam_sell * p_dam + rtm_sell * p_rtm) / total
    return r_ppa


def compute_dsm_charge_rate_rtc(dws_pct: float, is_over: bool, cr: float):
    """CERC DSM 2024 three-band charge table."""
    pct = abs(dws_pct)
    if pct <= 10.0:
        return cr, 1.0, "0-10%"
    elif pct <= 15.0:
        return (0.90 * cr, 0.90, "10-15%") if is_over else (1.10 * cr, 1.10, "10-15%")
    else:
        return (0.0, 0.0, ">15%") if is_over else (1.50 * cr, 1.50, ">15%")


def compute_dsm_settlement_rtc(captive_actual: float, scheduled_total: float,
                                cr: float, avail_cap: float) -> dict:
    """Full CERC DSM 2024 settlement for one block."""
    act_mwh   = captive_actual * DT
    sch_mwh   = scheduled_total * DT
    dws       = (captive_actual - scheduled_total) * DT
    pct       = abs(dws) / avail_cap * 100.0 if avail_cap > 0 else 0.0
    is_over   = dws > 0
    rate, mult, band = compute_dsm_charge_rate_rtc(pct, is_over, cr)
    direction = "within" if pct <= 10 else ("over" if is_over else "under")

    r = {
        "dws_mwh": dws, "dws_pct": pct, "band": band, "direction": direction,
        "charge_rate": rate, "charge_rate_mult": mult,
        "net_captive_cash": 0.0, "dsm_penalty": 0.0, "dsm_haircut": 0.0,
        "financial_damage": 0.0,
        "under_revenue_received": 0.0, "under_dsm_penalty": 0.0,
        "under_net_cash": 0.0, "under_if_fully_sched": 0.0, "under_damage": 0.0,
        "over_revenue_sched": 0.0, "over_revenue_dev": 0.0,
        "over_total_received": 0.0, "over_if_all_cr": 0.0, "over_haircut": 0.0,
    }

    if pct <= 10.0:
        r["net_captive_cash"] = act_mwh * cr
    elif dws < 0:   # under-injection
        rev = act_mwh * cr
        pen = abs(dws) * rate
        net = rev - pen
        ifs = sch_mwh * cr
        r.update({
            "under_revenue_received": rev, "under_dsm_penalty": pen,
            "under_net_cash": net, "under_if_fully_sched": ifs,
            "under_damage": ifs - net, "net_captive_cash": net,
            "dsm_penalty": pen, "financial_damage": ifs - net,
        })
    else:            # over-injection
        rs  = sch_mwh * cr
        rd  = dws * rate
        tr  = rs + rd
        ia  = act_mwh * cr
        hc  = max(0.0, ia - tr)
        r.update({
            "over_revenue_sched": rs, "over_revenue_dev": rd,
            "over_total_received": tr, "over_if_all_cr": ia,
            "over_haircut": hc, "net_captive_cash": tr, "dsm_haircut": hc,
        })
    return r


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: TwoStageBESSRTC
# ══════════════════════════════════════════════════════════════════════════════

class TwoStageBESSRTC:
    """
    Stage 1 MILP optimizer — D-1 10:00 IST.

    Selects a single scalar RTC_committed (∈ [rtc_min_mw, rtc_mw]) and
    DAM charge/discharge schedules over 96 blocks × S scenarios.

    The RTC captive constraint forces:
        s_cd_da[t] + c_d_da[t] == RTC_committed   ∀t
    making captive delivery a flat constant across the full day.

    CVaR risk management is included but degradation cost is NOT in the LP
    objective (applied post-hoc in settlement).
    """

    def __init__(self, params, config: Dict):
        self.params      = params
        self.config      = config
        self.lambda_risk = config.get("lambda_risk", 0.0)
        self.risk_alpha  = config.get("risk_alpha", 0.1)

    def solve(self, dam_scenarios: np.ndarray,
              rtm_scenarios: np.ndarray,
              solar_da: np.ndarray) -> Dict:
        """
        Parameters
        ----------
        dam_scenarios : (S, 96) — DAM price scenarios Rs/MWh
        rtm_scenarios : (S, 96) — RTM price scenarios Rs/MWh
        solar_da      : (96,)   — DA solar generation forecast MW

        Returns
        -------
        dict with keys: status, expected_revenue, RTC_committed,
             x_c, x_d, s_c_da, s_cd_da, c_d_da,
             captive_da, dam_net, schedule_da, setpoint_da,
             solar_band_mask, scenarios
        """
        S     = dam_scenarios.shape[0]
        p     = self.params
        p_max = p.p_max_mw
        S_inv = p.solar_inverter_mw
        r_ppa = p.ppa_rate_rs_mwh
        USABLE = p.usable_energy_mwh

        solar_da  = np.clip(solar_da, 0.0, S_inv)
        solar_mask = compute_solar_band_mask_rtc(
            solar_da, p.solar_threshold_mw, p.solar_buffer_blocks)

        prob = pulp.LpProblem("Stage1_RTC", pulp.LpMaximize)

        # ── Decision variables ─────────────────────────────────────────────
        # RTC_committed: single scalar in [rtc_min_mw, rtc_mw]
        RTC_c = pulp.LpVariable("RTC_committed",
                                lowBound=p.rtc_min_mw, upBound=p.rtc_mw)

        x_c   = pulp.LpVariable.dicts("xc",  range(T_BLOCKS), 0, p_max)
        x_d   = pulp.LpVariable.dicts("xd",  range(T_BLOCKS), 0, p_max)
        s_c   = pulp.LpVariable.dicts("sc",  range(T_BLOCKS), 0, p_max)
        s_cd  = pulp.LpVariable.dicts("scd", range(T_BLOCKS), 0, S_inv)
        c_d   = pulp.LpVariable.dicts("cd",  range(T_BLOCKS), 0, p_max)
        delta = pulp.LpVariable.dicts("delta", range(T_BLOCKS), cat="Binary")

        soc = {si: pulp.LpVariable.dicts(
            f"soc{si}", range(T_BLOCKS + 1), p.e_min_mwh, p.e_max_mwh)
            for si in range(S)}

        # CVaR variables
        zeta     = pulp.LpVariable("zeta")
        u        = pulp.LpVariable.dicts("u", range(S), lowBound=0)
        scen_revs = []

        # ── Per-scenario constraints ───────────────────────────────────────
        for si in range(S):
            prob += soc[si][0]         == p.soc_initial_mwh
            prob += soc[si][T_BLOCKS]  == p.soc_terminal_min_mwh   # SOD=EOD hard

            rev = 0
            for t in range(T_BLOCKS):
                pd_t = float(dam_scenarios[si, t])

                # C1: solar balance (no curtailment)
                prob += s_c[t] + s_cd[t] == float(solar_da[t])

                # C_RTC: flat captive constraint
                prob += s_cd[t] + c_d[t] == RTC_c

                # C2: PCS discharge limit
                prob += x_d[t] + c_d[t] <= p_max

                # C3: AC bus mutual exclusion (MILP)
                prob += x_c[t] + s_c[t] <= p_max * delta[t]
                prob += x_d[t] + c_d[t] <= p_max * (1 - delta[t])
                prob += s_cd[t]          <= S_inv * (1 - delta[t])

                # C4: SoC dynamics
                prob += soc[si][t + 1] == (
                    soc[si][t]
                    + p.eta_charge * (s_c[t] + x_c[t]) * DT
                    - (1.0 / p.eta_discharge) * (x_d[t] + c_d[t]) * DT
                )

                # C6: SoC solar band
                if solar_mask[t]:
                    prob += soc[si][t] >= p.soc_solar_low
                    prob += soc[si][t] <= p.soc_solar_high

                # Objective terms — only DAM IEX + captive revenue (no degradation)
                rev += pd_t * x_d[t] * DT
                rev -= pd_t * x_c[t] * DT
                rev += r_ppa * RTC_c * DT          # flat RTC revenue per block

                # IEX fees on grid flows only
                rev -= p.iex_fee_rs_mwh * (x_c[t] + x_d[t]) * DT

            # Cycle budget (1 cycle per day max)
            if p.max_cycles_per_day is not None:
                prob += pulp.lpSum(
                    [(x_d[t] + c_d[t]) * DT / p.eta_discharge
                     for t in range(T_BLOCKS)]
                ) <= p.max_cycles_per_day * USABLE, f"cycle_{si}"

            prob += u[si] >= zeta - rev
            scen_revs.append(rev)

        # ── Objective ──────────────────────────────────────────────────────
        avg_rev = pulp.lpSum(scen_revs) / S
        cvar    = zeta - (1.0 / (S * self.risk_alpha)) * pulp.lpSum(
            [u[si] for si in range(S)])
        prob.setObjective(avg_rev + self.lambda_risk * cvar)

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] != "Optimal":
            z = [0.0] * T_BLOCKS
            return {
                "status": "Infeasible",
                "RTC_committed": float(p.rtc_min_mw),
                "x_c": z, "x_d": z, "s_c_da": z, "s_cd_da": z, "c_d_da": z,
                "captive_da": z, "dam_net": z, "schedule_da": z,
                "setpoint_da": z, "expected_revenue": 0.0,
                "solar_band_mask": solar_mask.tolist(), "scenarios": [],
            }

        def v(var_dict, t):
            return max(0.0, pulp.value(var_dict[t]) or 0.0)

        rtc_val   = max(p.rtc_min_mw, min(p.rtc_mw, pulp.value(RTC_c) or p.rtc_min_mw))
        xc_v      = [v(x_c, t) for t in range(T_BLOCKS)]
        xd_v      = [v(x_d, t) for t in range(T_BLOCKS)]
        sc_v      = [v(s_c, t) for t in range(T_BLOCKS)]
        scd_v     = [v(s_cd, t) for t in range(T_BLOCKS)]
        cd_v      = [v(c_d, t) for t in range(T_BLOCKS)]
        cap_da    = [scd_v[t] + cd_v[t] for t in range(T_BLOCKS)]
        dam_net   = [xd_v[t] - xc_v[t] for t in range(T_BLOCKS)]
        sched_da  = [cap_da[t] + dam_net[t] for t in range(T_BLOCKS)]

        # Mean SoC across scenarios for setpoint derivation
        soc_mean = [
            float(np.mean([pulp.value(soc[si][t]) or 0.0 for si in range(S)]))
            for t in range(T_BLOCKS + 1)
        ]
        sp_da = [
            compute_setpoint_rtc(
                soc_mean[t], sched_da[t],
                p.e_min_mwh, p.e_max_mwh, p.eta_charge, p.eta_discharge
            )
            for t in range(T_BLOCKS)
        ]

        return {
            "status": "Optimal",
            "expected_revenue": float(pulp.value(avg_rev) or 0.0),
            "RTC_committed": float(rtc_val),
            "x_c": xc_v, "x_d": xd_v,
            "s_c_da": sc_v, "s_cd_da": scd_v, "c_d_da": cd_v,
            "captive_da": cap_da, "dam_net": dam_net,
            "schedule_da": sched_da, "setpoint_da": sp_da,
            "solar_band_mask": solar_mask.tolist(),
            "scenarios": [
                {"soc": [pulp.value(soc[si][t]) for t in range(T_BLOCKS + 1)]}
                for si in range(S)
            ],
        }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2B: reschedule_captive_rtc
# ══════════════════════════════════════════════════════════════════════════════

def reschedule_captive_rtc(params,
                            trigger_block: int,
                            soc_actual: float,
                            solar_nc_row: np.ndarray,
                            solar_da: np.ndarray,
                            rtm_q50: np.ndarray,
                            x_c_s1: np.ndarray,
                            x_d_s1: np.ndarray,
                            y_c_committed: np.ndarray,
                            y_d_committed: np.ndarray,
                            rtc_committed: float,
                            captive_committed_prev: np.ndarray,
                            rtc_notice: np.ndarray,
                            cycle_used_so_far: float = 0.0) -> Dict:
    """
    Stage 2B: revise solar routing with NC nowcast.

    captive_rt stays within ±5% free band unless rtc_notice[t] is True,
    in which case the full [rtc_min_mw, rtc_mw] range is available.

    Parameters
    ----------
    rtc_notice : (96,) bool array — True if >5% revision was committed ≥16 blocks ago
    captive_committed_prev : (96,) — most recent captive schedule (for buffer smoothness)
    cycle_used_so_far : float — MWh of discharge throughput already consumed today

    Returns
    -------
    dict: status, s_c_rt, s_cd_rt, c_d_rt, captive_rt (all shape (96,))
    """
    p         = params
    p_max     = p.p_max_mw
    S_inv     = p.solar_inverter_mw
    r_ppa     = p.ppa_rate_rs_mwh
    USABLE    = p.usable_energy_mwh
    B         = trigger_block
    remaining = T_BLOCKS - B
    RTM_LEAD  = p.rtm_lead_blocks
    CAP_BUF   = p.captive_buffer_blocks
    CAP_TOL   = p.captive_buffer_tolerance_mw

    # ── Solar blend: NC for next 12 blocks, DA beyond ──────────────────────
    solar_blend = np.zeros(remaining, dtype=float)
    for k in range(remaining):
        if k < len(solar_nc_row):
            solar_blend[k] = float(solar_nc_row[k])
        else:
            solar_blend[k] = float(solar_da[B + k])
    solar_blend = np.clip(solar_blend, 0.0, S_inv)

    # Locked schedule arrays (from Stage 1 + committed Stage 2A)
    xc_r  = np.array(x_c_s1[B:], dtype=float)
    xd_r  = np.array(x_d_s1[B:], dtype=float)
    yc_r  = np.array(y_c_committed[B:], dtype=float)
    yd_r  = np.array(y_d_committed[B:], dtype=float)
    rtm_r = np.array(rtm_q50[B:], dtype=float)

    prob = pulp.LpProblem(f"S2B_RTC_b{B}", pulp.LpMaximize)

    sc      = pulp.LpVariable.dicts("sc",  range(remaining), 0, p_max)
    scd     = pulp.LpVariable.dicts("scd", range(remaining), 0, S_inv)
    cd      = pulp.LpVariable.dicts("cd",  range(remaining), 0, p_max)
    cap_rt  = {}
    soc_v   = pulp.LpVariable.dicts("soc", range(remaining + 1),
                                    p.e_min_mwh, p.e_max_mwh)
    dl      = pulp.LpVariable.dicts("dl2b", range(remaining), cat="Binary")

    # captive_rt bounds depend on whether rtc_notice is set
    for k in range(remaining):
        t_abs = B + k
        if rtc_notice[t_abs]:
            # Full range allowed (notice was committed ≥16 blocks ago)
            cap_rt[k] = pulp.LpVariable(f"cap_rt_{k}",
                                         lowBound=p.rtc_min_mw, upBound=p.rtc_mw)
        else:
            # ±5% free band
            lo = rtc_committed * (1.0 - p.rtc_tol_pct)
            hi = rtc_committed * (1.0 + p.rtc_tol_pct)
            cap_rt[k] = pulp.LpVariable(f"cap_rt_{k}",
                                         lowBound=lo, upBound=hi)

    prob += soc_v[0] == float(np.clip(soc_actual, p.e_min_mwh, p.e_max_mwh))
    prob += soc_v[remaining] == p.soc_terminal_min_mwh  # EOD = SOC_TARGET

    rev = 0
    for k in range(remaining):
        xc_k = float(xc_r[k])
        xd_k = float(xd_r[k])
        yc_k = float(yc_r[k]) if k < RTM_LEAD else 0.0
        yd_k = float(yd_r[k]) if k < RTM_LEAD else 0.0
        pr_k = float(rtm_r[k])
        sol_k = float(solar_blend[k])

        # C2B-1: solar balance
        prob += sc[k] + scd[k] == sol_k

        # C2B-RTC: captive balance
        prob += scd[k] + cd[k] == cap_rt[k]

        # C2B-2/3: PCS limits
        prob += sc[k] + xc_k + yc_k <= p_max
        prob += cd[k] + xd_k + yd_k <= p_max

        # C2B-4: mutual exclusion
        if xc_k + yc_k > 1e-6:
            prob += dl[k] == 1
        elif xd_k + yd_k > 1e-6:
            prob += dl[k] == 0
        prob += xc_k + yc_k + sc[k] <= p_max * dl[k]
        prob += xd_k + yd_k + cd[k] <= p_max * (1 - dl[k])
        prob += scd[k]               <= S_inv * (1 - dl[k])

        # C2B-5: captive buffer smoothness (first CAP_BUF blocks)
        if k < CAP_BUF:
            prev = float(captive_committed_prev[B + k])
            prob += cap_rt[k] >= prev - CAP_TOL
            prob += cap_rt[k] <= prev + CAP_TOL

        # SoC dynamics
        prob += soc_v[k + 1] == (
            soc_v[k]
            + p.eta_charge * (sc[k] + xc_k + yc_k) * DT
            - (1.0 / p.eta_discharge) * (cd[k] + xd_k + yd_k) * DT
        )

        # Solar band SoC constraint (using solar_blend for mask)
        solar_blend_full = np.zeros(T_BLOCKS)
        solar_blend_full[B:] = solar_blend
        solar_mask = compute_solar_band_mask_rtc(
            solar_blend_full, p.solar_threshold_mw, p.solar_buffer_blocks)
        if solar_mask[B + k]:
            prob += soc_v[k] >= p.soc_solar_low
            prob += soc_v[k] <= p.soc_solar_high

        # Objective: captive PPA + IEX RTM (locked)
        rev += r_ppa * cap_rt[k] * DT
        rev += pr_k * xd_k * DT - pr_k * xc_k * DT
        rev -= p.iex_fee_rs_mwh * (xc_k + xd_k) * DT

    # Remaining cycle budget
    cycle_budget = max(0.0, (p.max_cycles_per_day or 1.0) * USABLE - cycle_used_so_far)
    prob += pulp.lpSum(
        [(cd[k] + float(xd_r[k]) + (float(yd_r[k]) if k < RTM_LEAD else 0.0))
         * DT / p.eta_discharge for k in range(remaining)]
    ) <= cycle_budget, "cycle_2b"

    prob.setObjective(rev)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]
    sc_out  = np.zeros(T_BLOCKS)
    scd_out = np.zeros(T_BLOCKS)
    cd_out  = np.zeros(T_BLOCKS)
    cap_out = np.zeros(T_BLOCKS)
    rtc_notice_out = rtc_notice.copy()

    if status == "Optimal":
        for k in range(remaining):
            t_abs = B + k
            sc_out[t_abs]  = max(0.0, pulp.value(sc[k])     or 0.0)
            scd_out[t_abs] = max(0.0, pulp.value(scd[k])    or 0.0)
            cd_out[t_abs]  = max(0.0, pulp.value(cd[k])     or 0.0)
            cap_val        = max(p.rtc_min_mw, pulp.value(cap_rt[k]) or rtc_committed)
            cap_out[t_abs] = cap_val

            # Check if >5% revision was committed for B+16
            if k + 16 < remaining:
                t_notice = t_abs + 16
                deviation_pct = abs(cap_val - rtc_committed) / (rtc_committed + 1e-9)
                if deviation_pct > p.rtc_tol_pct:
                    rtc_notice_out[t_notice] = True

    return {
        "status": status,
        "s_c_rt":    sc_out,
        "s_cd_rt":   scd_out,
        "c_d_rt":    cd_out,
        "captive_rt": cap_out,
        "rtc_notice": rtc_notice_out,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2A: solve_stage2a_rtc
# ══════════════════════════════════════════════════════════════════════════════

def solve_stage2a_rtc(params,
                       block_B: int,
                       soc_actual_B: float,
                       dam_actual: np.ndarray,
                       rtm_q50: np.ndarray,
                       p_rtm_lag4: float,
                       s_c_rt: np.ndarray,
                       c_d_rt: np.ndarray,
                       y_c_committed: np.ndarray,
                       y_d_committed: np.ndarray,
                       x_c_s1: np.ndarray,
                       x_d_s1: np.ndarray,
                       solar_da: np.ndarray,
                       rtc_committed: float,
                       captive_committed: np.ndarray,
                       rtc_notice: np.ndarray,
                       cycle_used_so_far: float = 0.0) -> Tuple[float, float, np.ndarray]:
    """
    Stage 2A: receding-horizon MPC — bids y_c/y_d for block B+RTM_LEAD.

    Also issues rtc_notice[B+16] if captive_rt would deviate >5% from
    RTC_committed for any block in the planning horizon.

    Returns
    -------
    (y_c_bid, y_d_bid, rtc_notice_updated)
    """
    p         = params
    p_max     = p.p_max_mw
    S_inv     = p.solar_inverter_mw
    r_ppa     = p.ppa_rate_rs_mwh
    USABLE    = p.usable_energy_mwh
    RTM_LEAD  = p.rtm_lead_blocks
    bid_block = block_B + RTM_LEAD

    if bid_block >= T_BLOCKS:
        return 0.0, 0.0, rtc_notice

    rtc_notice_out = rtc_notice.copy()

    # Lag-4 RTM bias adjustment
    rtm_adj = rtm_q50.copy().astype(float)
    if not np.isnan(p_rtm_lag4) and block_B >= 4:
        bias = p_rtm_lag4 - float(rtm_q50[block_B - 4])
        for t in range(bid_block, T_BLOCKS):
            rtm_adj[t] = max(0.0, rtm_adj[t] + bias * 0.85 ** (t - block_B))

    # Roll SoC forward from B to bid_block using locked schedules
    soc_rf = float(np.clip(soc_actual_B, p.e_min_mwh, p.e_max_mwh))
    for t in range(block_B, bid_block):
        xc_t  = float(x_c_s1[t])
        xd_t  = float(x_d_s1[t])
        yc_t  = float(y_c_committed[t])
        yd_t  = float(y_d_committed[t])
        sc_t  = float(s_c_rt[t])
        cd_t  = float(c_d_rt[t])
        charge = p.eta_charge * (sc_t + xc_t + yc_t) * DT
        dis    = (xd_t + cd_t + yd_t) / p.eta_discharge * DT
        soc_rf = float(np.clip(soc_rf + charge - dis, p.e_min_mwh, p.e_max_mwh))

    remaining = T_BLOCKS - bid_block
    if remaining <= 0:
        return 0.0, 0.0, rtc_notice_out

    solar_mask = compute_solar_band_mask_rtc(
        solar_da, p.solar_threshold_mw, p.solar_buffer_blocks)

    prob  = pulp.LpProblem(f"S2A_RTC_B{block_B}", pulp.LpMaximize)
    y_c   = pulp.LpVariable.dicts("yc", range(remaining), 0, p_max)
    y_d   = pulp.LpVariable.dicts("yd", range(remaining), 0, p_max)
    soc_lp = pulp.LpVariable.dicts("s", range(remaining + 1),
                                   p.e_min_mwh, p.e_max_mwh)
    dl    = pulp.LpVariable.dicts("dl2a", range(remaining), cat="Binary")

    # captive_rt variables for remaining blocks
    cap_rt_lp = {}
    for k in range(remaining):
        t_abs = bid_block + k
        if rtc_notice[t_abs]:
            cap_rt_lp[k] = pulp.LpVariable(f"crt_{k}",
                                            lowBound=p.rtc_min_mw, upBound=p.rtc_mw)
        else:
            lo = rtc_committed * (1.0 - p.rtc_tol_pct)
            hi = rtc_committed * (1.0 + p.rtc_tol_pct)
            cap_rt_lp[k] = pulp.LpVariable(f"crt_{k}", lowBound=lo, upBound=hi)

    prob += soc_lp[0]        == soc_rf
    prob += soc_lp[remaining] == p.soc_terminal_min_mwh

    rev = 0
    for k in range(remaining):
        ta   = bid_block + k
        xc_t = float(x_c_s1[ta])
        xd_t = float(x_d_s1[ta])
        sc_t = float(s_c_rt[ta])
        cd_t = float(c_d_rt[ta])
        pr   = float(rtm_adj[ta])

        prob += sc_t + xc_t + y_c[k] <= p_max
        prob += cd_t + xd_t + y_d[k] <= p_max

        # Mutual exclusion (soft hints from Stage 1)
        imp = xc_t + sc_t
        exp_= xd_t + cd_t
        if imp > 1e-6 and exp_ < 1e-6:
            prob += dl[k] >= 1
        elif exp_ > 1e-6 and imp < 1e-6:
            prob += dl[k] <= 0
        prob += xc_t + y_c[k] + sc_t <= p_max * dl[k]
        prob += xd_t + y_d[k] + cd_t <= p_max * (1 - dl[k])

        # SoC dynamics
        prob += soc_lp[k + 1] == (
            soc_lp[k]
            + p.eta_charge * (sc_t + xc_t + y_c[k]) * DT
            - (1.0 / p.eta_discharge) * (cd_t + xd_t + y_d[k]) * DT
        )

        if solar_mask[ta]:
            prob += soc_lp[k] >= p.soc_solar_low
            prob += soc_lp[k] <= p.soc_solar_high

        # Objective: RTM revenue + captive PPA
        rev += pr * y_d[k] * DT - pr * y_c[k] * DT
        rev += r_ppa * cap_rt_lp[k] * DT
        rev -= p.iex_fee_rs_mwh * (y_c[k] + y_d[k]) * DT

    # Remaining cycle budget
    cycle_budget = max(0.0, (p.max_cycles_per_day or 1.0) * USABLE - cycle_used_so_far)
    prob += pulp.lpSum(
        [(float(c_d_rt[bid_block + k]) + float(x_d_s1[bid_block + k]) + y_d[k])
         * DT / p.eta_discharge for k in range(remaining)]
    ) <= cycle_budget, "cycle_2a"

    prob.setObjective(rev)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] == "Optimal":
        y_c_bid = max(0.0, pulp.value(y_c[0]) or 0.0)
        y_d_bid = max(0.0, pulp.value(y_d[0]) or 0.0)

        # Check for >5% captive revisions and issue rtc_notice
        for k in range(remaining):
            cap_val = pulp.value(cap_rt_lp[k]) or rtc_committed
            dev_pct = abs(cap_val - rtc_committed) / (rtc_committed + 1e-9)
            if dev_pct > p.rtc_tol_pct:
                # Issue advance notice if within range
                t_notice = bid_block + k
                if t_notice - block_B >= p.rtc_advance_blocks:
                    rtc_notice_out[t_notice] = True

        return y_c_bid, y_d_bid, rtc_notice_out

    return 0.0, 0.0, rtc_notice_out


# ══════════════════════════════════════════════════════════════════════════════
# ACTUALS SETTLEMENT: evaluate_actuals_rtc
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_actuals_rtc(params,
                          stage1_result: Dict,
                          dam_actual: np.ndarray,
                          rtm_actual: np.ndarray,
                          rtm_q50: np.ndarray,
                          solar_da: np.ndarray,
                          solar_nc: np.ndarray,
                          solar_at: np.ndarray,
                          reschedule_blocks: List[int] = RESCHEDULE_BLOCKS,
                          verbose: bool = False) -> Dict:
    """
    Block-by-block settlement loop with RTC captive contract.

    Steps per block:
      1. Inputs (setpoint, solar_at, SoC)
      2. Dispatch (Case A/B/C)
      3. DSM deviation settlement
      4. RTC captive penalty (additional, if captive_actual < rtc_min_mw)
      5. IEX revenue (DAM + RTM)
      6. SoC update
      7. Block P&L
      8. No-BESS counterfactual
    """
    p       = params
    r_ppa   = p.ppa_rate_rs_mwh
    RTM_LEAD = p.rtm_lead_blocks
    avail_cap = p.avail_cap_mwh    # AVAIL_CAP = S_inv × DT = 4.1 MWh

    rtc_committed = float(stage1_result["RTC_committed"])

    x_c_s1  = np.array(stage1_result["x_c"])
    x_d_s1  = np.array(stage1_result["x_d"])
    sc_da   = np.array(stage1_result["s_c_da"])
    scd_da  = np.array(stage1_result["s_cd_da"])
    cd_da   = np.array(stage1_result["c_d_da"])

    # RT routing initialized from Stage 1 DA plan
    s_c_rt  = sc_da.copy()
    s_cd_rt = scd_da.copy()
    c_d_rt  = cd_da.copy()
    y_c_committed = np.zeros(T_BLOCKS)
    y_d_committed = np.zeros(T_BLOCKS)

    # captive_committed: starts as Stage 1 flat RTC_committed
    captive_committed = np.full(T_BLOCKS, rtc_committed, dtype=float)

    # RTC advance notice tracker: rtc_notice[t] = True means >5% revision allowed at t
    rtc_notice = np.zeros(T_BLOCKS, dtype=bool)

    soc_path = np.zeros(T_BLOCKS + 1)
    soc_path[0] = p.soc_initial_mwh

    # Output arrays
    s_c_actual_arr  = np.zeros(T_BLOCKS)
    s_cd_actual_arr = np.zeros(T_BLOCKS)
    c_d_actual_arr  = np.zeros(T_BLOCKS)
    captive_actual_arr = np.zeros(T_BLOCKS)
    setpoint_arr    = np.zeros(T_BLOCKS)
    schedule_rt_arr = np.zeros(T_BLOCKS)

    dsm_results           = []
    block_captive_net_arr = np.zeros(T_BLOCKS)
    block_captive_penalty = np.zeros(T_BLOCKS)
    block_iex_net_arr     = np.zeros(T_BLOCKS)
    block_degradation_arr = np.zeros(T_BLOCKS)
    block_net_arr         = np.zeros(T_BLOCKS)
    no_bess_dsm_arr       = np.zeros(T_BLOCKS)
    no_bess_rev_arr       = np.zeros(T_BLOCKS)

    rtc_notice_issued_arr  = np.zeros(T_BLOCKS, dtype=bool)
    rtc_notice_target_arr  = np.full(T_BLOCKS, -1, dtype=int)

    cum_discharge_mwh = 0.0   # cycle budget tracking

    for B in range(T_BLOCKS):
        lag4 = float(rtm_actual[B - 4]) if B >= 4 else np.nan

        # ── Stage 2B at reschedule blocks ─────────────────────────────────
        if B in reschedule_blocks:
            res2b = reschedule_captive_rtc(
                params=p,
                trigger_block=B,
                soc_actual=soc_path[B],
                solar_nc_row=solar_nc[B] if B < len(solar_nc) else np.zeros(12),
                solar_da=solar_da,
                rtm_q50=rtm_q50,
                x_c_s1=x_c_s1, x_d_s1=x_d_s1,
                y_c_committed=y_c_committed, y_d_committed=y_d_committed,
                rtc_committed=rtc_committed,
                captive_committed_prev=captive_committed,
                rtc_notice=rtc_notice,
                cycle_used_so_far=cum_discharge_mwh,
            )
            if res2b["status"] == "Optimal":
                s_c_rt[B:]  = res2b["s_c_rt"][B:]
                s_cd_rt[B:] = res2b["s_cd_rt"][B:]
                c_d_rt[B:]  = res2b["c_d_rt"][B:]
                captive_committed[B:] = res2b["captive_rt"][B:]
                rtc_notice = res2b["rtc_notice"]

        # ── Stage 2A: bid for B+RTM_LEAD ──────────────────────────────────
        bid_b = B + RTM_LEAD
        if bid_b < T_BLOCKS:
            yc_bid, yd_bid, rtc_notice = solve_stage2a_rtc(
                params=p, block_B=B, soc_actual_B=soc_path[B],
                dam_actual=dam_actual, rtm_q50=rtm_q50, p_rtm_lag4=lag4,
                s_c_rt=s_c_rt, c_d_rt=c_d_rt,
                y_c_committed=y_c_committed, y_d_committed=y_d_committed,
                x_c_s1=x_c_s1, x_d_s1=x_d_s1, solar_da=solar_da,
                rtc_committed=rtc_committed, captive_committed=captive_committed,
                rtc_notice=rtc_notice, cycle_used_so_far=cum_discharge_mwh,
            )
            y_c_committed[bid_b] = yc_bid
            y_d_committed[bid_b] = yd_bid

        # ── Step 1: Compute setpoint and BESS capacity ─────────────────────
        xc_B  = float(x_c_s1[B])
        xd_B  = float(x_d_s1[B])
        yc_B  = float(y_c_committed[B])
        yd_B  = float(y_d_committed[B])

        cap_rt_B   = float(s_cd_rt[B] + c_d_rt[B])
        schedule_rt_B = cap_rt_B + (xd_B - xc_B) + (yd_B - yc_B)
        schedule_rt_arr[B] = schedule_rt_B

        setpoint_B = compute_setpoint_rtc(
            soc_path[B], schedule_rt_B,
            p.e_min_mwh, p.e_max_mwh, p.eta_charge, p.eta_discharge)
        setpoint_arr[B] = setpoint_B

        z_at = float(solar_at[B])

        bess_disch_cap = max(0.0, min(
            p.p_max_mw,
            (soc_path[B] - p.e_min_mwh) * p.eta_discharge / DT
        ))
        bess_charge_cap = max(0.0, min(
            p.p_max_mw - xc_B - yc_B,
            (p.e_max_mwh - soc_path[B]) / (p.eta_charge * DT)
        ))

        # ── Step 2: Dispatch (Case A / B / C) ─────────────────────────────
        if z_at > setpoint_B + 1e-6:          # Case A — solar surplus
            s_c_actual  = min(bess_charge_cap, z_at - setpoint_B)
            s_cd_actual = z_at - s_c_actual
            c_d_actual  = 0.0
        elif z_at < setpoint_B - 1e-6:        # Case B — solar deficit
            s_cd_actual = z_at
            c_d_actual  = min(bess_disch_cap, setpoint_B - z_at)
            s_c_actual  = 0.0
        else:                                   # Case C — exact match
            s_cd_actual = z_at
            s_c_actual  = 0.0
            c_d_actual  = 0.0

        captive_actual = s_cd_actual + c_d_actual
        s_c_actual_arr[B]     = s_c_actual
        s_cd_actual_arr[B]    = s_cd_actual
        c_d_actual_arr[B]     = c_d_actual
        captive_actual_arr[B] = captive_actual

        # ── Step 3: DSM settlement ─────────────────────────────────────────
        cr = compute_contract_rate_rtc(
            rtc_committed, xd_B, yd_B,
            float(dam_actual[B]), float(rtm_actual[B]), r_ppa)
        dsm = compute_dsm_settlement_rtc(
            captive_actual, schedule_rt_B, cr, avail_cap)
        dsm_results.append(dsm)

        # ── Step 4: RTC captive penalty (additional) ───────────────────────
        if captive_actual < p.rtc_min_mw:
            shortfall_mwh  = (p.rtc_min_mw - captive_actual) * DT
            cap_penalty    = shortfall_mwh * r_ppa
        else:
            cap_penalty    = 0.0

        block_captive_net_arr[B]  = dsm["net_captive_cash"] - cap_penalty
        block_captive_penalty[B]  = cap_penalty

        # ── Step 5: IEX revenue (DAM + RTM) ───────────────────────────────
        iex_dam  = float(dam_actual[B]) * (xd_B - xc_B) * DT
        iex_rtm  = float(rtm_actual[B]) * (yd_B - yc_B) * DT
        iex_fees = p.iex_fee_rs_mwh * (xc_B + xd_B + yc_B + yd_B) * DT
        iex_net  = iex_dam + iex_rtm - iex_fees
        block_iex_net_arr[B] = iex_net

        # ── Step 6: SoC update ─────────────────────────────────────────────
        total_discharge = xd_B + c_d_actual + yd_B
        charge_e  = p.eta_charge * (s_c_actual + xc_B + yc_B) * DT
        dis_e     = total_discharge / p.eta_discharge * DT
        soc_path[B + 1] = float(np.clip(
            soc_path[B] + charge_e - dis_e, p.e_min_mwh, p.e_max_mwh))

        cum_discharge_mwh += total_discharge * DT / p.eta_discharge

        # ── Step 7: Block P&L ──────────────────────────────────────────────
        degradation        = p.degradation_cost_rs_mwh * total_discharge * DT
        block_degradation_arr[B] = degradation
        block_net_arr[B]   = block_captive_net_arr[B] + iex_net - degradation

        # ── Step 8: No-BESS counterfactual ────────────────────────────────
        nb_dsm = compute_dsm_settlement_rtc(
            z_at, float(captive_committed[B]), r_ppa, avail_cap)
        no_bess_dsm_arr[B] = nb_dsm["dsm_penalty"] + nb_dsm.get("dsm_haircut", 0.0)
        no_bess_rev_arr[B] = nb_dsm["net_captive_cash"]

        # Track RTC notice events for CSV export
        if rtc_notice[B] and not rtc_notice_issued_arr[B]:
            rtc_notice_issued_arr[B] = True

    return {
        "revenue":              float(np.sum(block_captive_net_arr) + np.sum(block_iex_net_arr)),
        "net_revenue":          float(np.sum(block_net_arr)),
        "captive_net_total":    float(np.sum(block_captive_net_arr)),
        "iex_net_total":        float(np.sum(block_iex_net_arr)),
        "captive_penalty_total": float(np.sum(block_captive_penalty)),
        "RTC_committed":        rtc_committed,
        "y_c":                  y_c_committed,
        "y_d":                  y_d_committed,
        "s_c_rt":               s_c_rt,
        "s_cd_rt":              s_cd_rt,
        "c_d_rt":               c_d_rt,
        "s_c_actual":           s_c_actual_arr,
        "s_cd_actual":          s_cd_actual_arr,
        "c_d_actual":           c_d_actual_arr,
        "captive_actual":       captive_actual_arr,
        "captive_committed":    captive_committed,
        "setpoint_rt":          setpoint_arr,
        "schedule_rt":          schedule_rt_arr,
        "soc_path":             soc_path,
        "soc":                  soc_path.tolist(),
        "block_captive_net":    block_captive_net_arr,
        "block_captive_penalty": block_captive_penalty,
        "block_iex_net":        block_iex_net_arr,
        "block_degradation":    block_degradation_arr,
        "block_net":            block_net_arr,
        "dsm_results":          dsm_results,
        "no_bess_dsm":          no_bess_dsm_arr,
        "no_bess_revenue":      no_bess_rev_arr,
        "rtc_notice_issued":    rtc_notice_issued_arr,
        "rtc_notice_target":    rtc_notice_target_arr,
    }
