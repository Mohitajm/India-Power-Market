"""
src/optimizer/two_stage_bess_rtc.py — Architecture v10 RTC  (FIXED)
=====================================================================
Three-stage Solar+BESS optimizer with Round-the-Clock (RTC) captive contract.

KEY CHANGES vs two_stage_bess.py (v9_revised):
  1. Hardware resized: 80 MWh BESS, 16.4 MW PCS, 16.4 MW inverter.
  2. RTC captive replaces variable captive:
       - Stage 1 selects a scalar RTC_committed ∈ [rtc_min_mw, rtc_mw]
       - captive_da[t] = RTC_committed for ALL 96 blocks (flat constant)
       - At night, BESS (c_d) alone supplies the captive load (no solar).
       - Stage 2A/2B keep captive_rt within ±5% free band unless 16-block
         advance notice has been committed.
  3. SOD = EOD = 40 MWh (hard equality both ends).
  4. No curtailment variable: s_c + s_cd == solar_da exactly.
  5. Degradation is POST-HOC only (not in LP objective).
  6. IEX fee IS in the LP objective for x_c, x_d.
  7. RTC captive penalty computed in actuals settlement (not LP).
  8. Stage 2A: receding-horizon MPC, issues rtc_notice for >5% revisions.

BUG FIXES in this revision
---------------------------
FIX-1 (Infeasibility — root cause):
  The original C_RTC constraint  s_cd[t] + c_d[t] == RTC_c  forces 4–5 MW
  discharge at ALL 96 blocks including nighttime when solar=0 and s_cd=0.
  With SOD=EOD=40 MWh and 72 MWh usable, sustaining 4 MW × 96 × 0.25 h =
  96 MWh of discharge is physically impossible. Fix: decouple C_RTC from
  s_cd. The captive consumer receives  s_cd + c_d  MW.  At night the BESS
  discharges (c_d) freely, and the energy balance is re-charged during the
  solar window. The RTC_committed level is therefore constrained only as
  the AVERAGE daily delivery, and at the block level as an upper bound on
  captive supply (the consumer gets at most RTC_committed; the schedule
  is filed as RTC_committed flat, deviations go through DSM). See the
  revised C_RTC formulation below.

  Revised RTC modelling:
    captive_da[t] = s_cd[t] + c_d[t]     (block-level delivery, ≤ RTC_committed)
    captive_da[t] >= RTC_committed - slack (soft lower bound via slack penalty)
    The DA Schedule filed = RTC_committed (flat). Actual delivery may deviate
    and is settled via DSM.  This is the correct physical model:
      - Day hours: solar covers most of captive, BESS tops up.
      - Night hours: BESS alone discharges to cover captive; BESS recharges
        during solar hours using surplus solar.

FIX-2 (Bug #4): C1 and C_RTC were added inside the scenario loop creating
  S×96 duplicate constraints on first-stage variables.  Moved outside.

FIX-3 (Bug #5): bess_discharge_cap in evaluate_actuals_rtc was not
  subtracting committed x_d and y_d, allowing over-dispatch.

FIX-4 (Bug #3): rtc_notice issue logic in Stage 2A/2B was checking the
  wrong block index. Now uses: notice issued at block B for delivery change
  at block B + rtc_advance_blocks.

FIX-5 (Bug #2): T_BLOCKS re-declared at module level of runner (removed
  in runner). Harmless here but documented.
"""

import pulp
import numpy as np
from typing import Dict, List, Optional, Tuple

T_BLOCKS = 96
DT       = 0.25
RESCHEDULE_BLOCKS = [34, 42, 50, 58]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def compute_solar_band_mask_rtc(solar_profile: np.ndarray,
                                threshold: float = 0.5,
                                buffer: int = 2) -> np.ndarray:
    """Build solar-hours boolean mask with transition buffer."""
    n    = len(solar_profile)
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
    ppa_mw   = max(0.0, rtc_committed)
    dam_sell = max(0.0, x_d)
    rtm_sell = max(0.0, y_d)
    total    = ppa_mw + dam_sell + rtm_sell
    if total > 1e-9:
        return (ppa_mw * r_ppa + dam_sell * p_dam + rtm_sell * p_rtm) / total
    return r_ppa


def compute_dsm_charge_rate_rtc(dws_pct: float, is_over: bool, cr: float):
    """
    CERC DSM 2024 three-band charge table.
    Band        Direction       Charge rate
    0–10%       Over/Under      100% CR
    10–15%      Over            90% CR (receive less)
    10–15%      Under           110% CR (pay extra)
    >15%        Over            0% CR (receive nothing)
    >15%        Under           150% CR (pay heavy)
    """
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
    act_mwh  = captive_actual * DT
    sch_mwh  = scheduled_total * DT
    dws      = (captive_actual - scheduled_total) * DT
    pct      = abs(dws) / avail_cap * 100.0 if avail_cap > 0 else 0.0
    is_over  = dws > 0
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

    RTC Captive Modelling (FIX-1):
    ─────────────────────────────
    The captive consumer contract is for a flat RTC_committed MW delivery.
    The DA schedule filed with SLDC is RTC_committed MW for all 96 blocks.

    Physical reality:
      - Solar hours:  s_cd[t] + c_d[t] ≈ RTC_committed  (solar + BESS top-up)
      - Night hours:  c_d[t]            ≈ RTC_committed  (BESS alone)
      - BESS recharges during solar surplus hours (s_c) and/or grid (x_c)

    LP formulation:
      captive[t] = s_cd[t] + c_d[t]          (block delivery, ≥ 0)
      captive[t] ≤ RTC_c                      (do not over-supply)
      captive[t] ≥ RTC_c - slack[t]           (soft lower bound via penalty)
      The filed schedule is RTC_c (flat). Deviations → DSM in settlement.

    This avoids the infeasibility from forcing s_cd+c_d == RTC_c at night
    (when s_cd=0 and c_d must supply 4–5 MW for all 96 blocks, draining
    the 80 MWh BESS completely — impossible to restore to 40 MWh EOD).
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
        USABLE = p.usable_energy_mwh          # 72.0 MWh

        solar_da   = np.clip(solar_da, 0.0, S_inv)
        solar_mask = compute_solar_band_mask_rtc(
            solar_da, p.solar_threshold_mw, p.solar_buffer_blocks)

        # Penalty weight for captive under-delivery in planning (Rs/MWh)
        # Set high enough to enforce delivery but below IEX premium
        SLACK_PENALTY = r_ppa * 2.0

        prob = pulp.LpProblem("Stage1_RTC", pulp.LpMaximize)

        # ── Decision variables (first-stage — same across all scenarios) ──
        RTC_c = pulp.LpVariable("RTC_committed",
                                lowBound=p.rtc_min_mw, upBound=p.rtc_mw)

        x_c   = pulp.LpVariable.dicts("xc",    range(T_BLOCKS), 0, p_max)
        x_d   = pulp.LpVariable.dicts("xd",    range(T_BLOCKS), 0, p_max)
        s_c   = pulp.LpVariable.dicts("sc",    range(T_BLOCKS), 0, p_max)
        s_cd  = pulp.LpVariable.dicts("scd",   range(T_BLOCKS), 0, S_inv)
        c_d   = pulp.LpVariable.dicts("cd",    range(T_BLOCKS), 0, p_max)
        slack = pulp.LpVariable.dicts("slack", range(T_BLOCKS), 0, p.rtc_mw)
        delta = pulp.LpVariable.dicts("delta", range(T_BLOCKS), cat="Binary")

        # ── Scenario-indexed SoC and CVaR variables ───────────────────────
        soc = {si: pulp.LpVariable.dicts(
            f"soc{si}", range(T_BLOCKS + 1), p.e_min_mwh, p.e_max_mwh)
            for si in range(S)}

        zeta      = pulp.LpVariable("zeta")
        u         = pulp.LpVariable.dicts("u", range(S), lowBound=0)
        scen_revs = []

        # ── First-stage constraints (add ONCE, outside scenario loop) ─────
        # FIX-2: moved C1, C_RTC, C2, C3 out of the scenario loop
        for t in range(T_BLOCKS):
            sol_t = float(solar_da[t])

            # C1: solar energy balance (no curtailment)
            prob += s_c[t] + s_cd[t] == sol_t,              f"C1_{t}"

            # C_RTC-upper: captive delivery at most RTC_committed
            prob += s_cd[t] + c_d[t] <= RTC_c,              f"CRTC_hi_{t}"

            # C_RTC-lower (soft via slack): push delivery to RTC_committed
            # slack[t] absorbs the gap when BESS cannot fully deliver
            prob += s_cd[t] + c_d[t] >= RTC_c - slack[t],   f"CRTC_lo_{t}"

            # C2: PCS total discharge ≤ p_max
            prob += x_d[t] + c_d[t] <= p_max,               f"C2_{t}"

            # C3: AC bus mutual exclusion (MILP)
            # delta=1 → import mode (charge); delta=0 → export mode (discharge)
            prob += x_c[t] + s_c[t]   <= p_max * delta[t],           f"C3a_{t}"
            prob += x_d[t] + c_d[t]   <= p_max * (1 - delta[t]),     f"C3b_{t}"
            prob += s_cd[t]            <= S_inv * (1 - delta[t]),     f"C3c_{t}"

        # ── Cycle budget (once per scenario — covers all blocks) ──────────
        for si in range(S):
            prob += pulp.lpSum(
                [(x_d[t] + c_d[t]) * DT / p.eta_discharge
                 for t in range(T_BLOCKS)]
            ) <= USABLE, f"cycle_{si}"

        # ── Scenario loop: SoC dynamics + CVaR ───────────────────────────
        for si in range(S):
            prob += soc[si][0]        == p.soc_initial_mwh
            prob += soc[si][T_BLOCKS] == p.soc_terminal_min_mwh   # SOD = EOD

            rev = 0
            for t in range(T_BLOCKS):
                pd_t = float(dam_scenarios[si, t])

                # C4: SoC dynamics
                prob += soc[si][t + 1] == (
                    soc[si][t]
                    + p.eta_charge    * (s_c[t] + x_c[t]) * DT
                    - (1.0 / p.eta_discharge) * (x_d[t] + c_d[t]) * DT
                ), f"C4_{si}_{t}"

                # C6: SoC solar band
                if solar_mask[t]:
                    prob += soc[si][t] >= p.soc_solar_low,  f"C6lo_{si}_{t}"
                    prob += soc[si][t] <= p.soc_solar_high, f"C6hi_{si}_{t}"

                # Objective revenue (per scenario per block)
                rev += pd_t * x_d[t] * DT           # DAM sell
                rev -= pd_t * x_c[t] * DT           # DAM buy
                rev += r_ppa * (s_cd[t] + c_d[t]) * DT   # PPA on actual delivery
                rev -= p.iex_fee_rs_mwh * (x_c[t] + x_d[t]) * DT  # IEX fees
                rev -= SLACK_PENALTY * slack[t] * DT  # penalise under-delivery

            prob += u[si] >= zeta - rev
            scen_revs.append(rev)

        # ── Objective ─────────────────────────────────────────────────────
        avg_rev = pulp.lpSum(scen_revs) / S
        cvar    = zeta - (1.0 / (S * self.risk_alpha)) * pulp.lpSum(
            [u[si] for si in range(S)])
        prob.setObjective(avg_rev + self.lambda_risk * cvar)

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] != "Optimal":
            z = [0.0] * T_BLOCKS
            return {
                "status": "Infeasible",
                "x_c": z, "x_d": z, "s_c_da": z, "s_cd_da": z, "c_d_da": z,
                "captive_da": z, "dam_net": z, "schedule_da": z,
                "setpoint_da": z, "solar_band_mask": [False] * T_BLOCKS,
                "RTC_committed": p.rtc_min_mw,
                "expected_revenue": 0.0, "scenarios": [],
            }

        rtc_val = float(pulp.value(RTC_c) or p.rtc_min_mw)

        xc_v  = [max(0.0, pulp.value(x_c[t])  or 0.0) for t in range(T_BLOCKS)]
        xd_v  = [max(0.0, pulp.value(x_d[t])  or 0.0) for t in range(T_BLOCKS)]
        sc_v  = [max(0.0, pulp.value(s_c[t])  or 0.0) for t in range(T_BLOCKS)]
        scd_v = [max(0.0, pulp.value(s_cd[t]) or 0.0) for t in range(T_BLOCKS)]
        cd_v  = [max(0.0, pulp.value(c_d[t])  or 0.0) for t in range(T_BLOCKS)]

        cap_da   = [scd_v[t] + cd_v[t] for t in range(T_BLOCKS)]
        dam_net  = [xd_v[t] - xc_v[t]  for t in range(T_BLOCKS)]
        # Filed schedule: flat RTC_committed (what SLDC sees)
        sched_da = [rtc_val + dam_net[t] for t in range(T_BLOCKS)]

        # Setpoint derivation from mean SoC across scenarios
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
            "RTC_committed": rtc_val,
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

    FIX-3 (rtc_notice): notices are issued at block B for a delivery change
    at block B + rtc_advance_blocks (not the other way round).
    FIX-1 applied: captive_rt is a soft upper-bound with slack, not equality.
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
    SLACK_PENALTY = r_ppa * 2.0

    # Solar blend: NC for next 12 blocks, DA beyond
    solar_blend = np.zeros(remaining, dtype=float)
    for k in range(remaining):
        if k < len(solar_nc_row):
            solar_blend[k] = float(solar_nc_row[k])
        else:
            solar_blend[k] = float(solar_da[B + k]) if (B + k) < T_BLOCKS else 0.0
    solar_blend = np.clip(solar_blend, 0.0, S_inv)

    solar_mask = compute_solar_band_mask_rtc(
        solar_da, p.solar_threshold_mw, p.solar_buffer_blocks)

    xc_r  = np.array(x_c_s1[B:],          dtype=float)
    xd_r  = np.array(x_d_s1[B:],          dtype=float)
    yc_r  = np.array(y_c_committed[B:],   dtype=float)
    yd_r  = np.array(y_d_committed[B:],   dtype=float)
    rtm_r = np.array(rtm_q50[B:],         dtype=float)

    prob  = pulp.LpProblem(f"S2B_RTC_b{B}", pulp.LpMaximize)

    sc    = pulp.LpVariable.dicts("sc",  range(remaining), 0, p_max)
    scd   = pulp.LpVariable.dicts("scd", range(remaining), 0, S_inv)
    cd    = pulp.LpVariable.dicts("cd",  range(remaining), 0, p_max)
    slack = pulp.LpVariable.dicts("sl",  range(remaining), 0, p.rtc_mw)
    soc_v = pulp.LpVariable.dicts("soc", range(remaining + 1),
                                  p.e_min_mwh, p.e_max_mwh)
    dl    = pulp.LpVariable.dicts("dl2b", range(remaining), cat="Binary")

    # captive_rt bound depends on whether rtc_notice is already set
    cap_rt = {}
    for k in range(remaining):
        t_abs = B + k
        if rtc_notice[t_abs]:
            cap_rt[k] = pulp.LpVariable(f"cap_rt_{k}",
                                        lowBound=p.rtc_min_mw, upBound=p.rtc_mw)
        else:
            lo = rtc_committed * (1.0 - p.rtc_tol_pct)
            hi = rtc_committed * (1.0 + p.rtc_tol_pct)
            cap_rt[k] = pulp.LpVariable(f"cap_rt_{k}", lowBound=lo, upBound=hi)

    prob += soc_v[0]        == float(np.clip(soc_actual, p.e_min_mwh, p.e_max_mwh))
    prob += soc_v[remaining] == p.soc_terminal_min_mwh   # EOD = SOC_TARGET

    # Cycle budget
    cycle_budget = max(0.0, (p.max_cycles_per_day or 1.0) * USABLE - cycle_used_so_far)
    prob += pulp.lpSum(
        [(cd[k] + float(xd_r[k]) + (float(yd_r[k]) if k < RTM_LEAD else 0.0))
         * DT / p.eta_discharge for k in range(remaining)]
    ) <= cycle_budget, "cycle_2b"

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

        # C2B-1: solar balance
        prob += sc[k] + scd[k] == sol_k,                      f"C2B1_{k}"

        # C2B-RTC: soft captive delivery (upper + lower with slack)
        prob += scd[k] + cd[k] <= cap_rt[k],                  f"CRTC2B_hi_{k}"
        prob += scd[k] + cd[k] >= cap_rt[k] - slack[k],       f"CRTC2B_lo_{k}"

        # C2B-2/3: PCS limits
        prob += sc[k] + xc_k + yc_k <= p_max,                 f"C2B2_{k}"
        prob += cd[k] + xd_k + yd_k <= p_max,                 f"C2B3_{k}"

        # C2B-4: AC bus mutual exclusion
        if xc_k + yc_k > 1e-6 and xd_k + yd_k < 1e-6:
            prob += dl[k] == 1
        elif xd_k + yd_k > 1e-6 and xc_k + yc_k < 1e-6:
            prob += dl[k] == 0
        prob += xc_k + yc_k + sc[k] <= p_max * dl[k],         f"C2B4a_{k}"
        prob += xd_k + yd_k + cd[k] <= p_max * (1 - dl[k]),   f"C2B4b_{k}"
        prob += scd[k]               <= S_inv * (1 - dl[k]),   f"C2B4c_{k}"

        # C2B-5: captive buffer smoothness (first CAP_BUF blocks)
        if k < CAP_BUF:
            ct = float(captive_committed_prev[t_abs])
            prob += scd[k] + cd[k] >= ct - CAP_TOL,           f"C2B5lo_{k}"
            prob += scd[k] + cd[k] <= ct + CAP_TOL,           f"C2B5hi_{k}"

        # C2B-6: SoC dynamics
        prob += soc_v[k + 1] == (
            soc_v[k]
            + p.eta_charge    * (sc[k] + xc_k + yc_k) * DT
            - (1.0 / p.eta_discharge) * (cd[k] + xd_k + yd_k) * DT
        ), f"C2B6_{k}"

        # C2B-7: SoC band
        if solar_mask[t_abs]:
            prob += soc_v[k] >= p.soc_solar_low,               f"C2B7lo_{k}"
            prob += soc_v[k] <= p.soc_solar_high,              f"C2B7hi_{k}"

        # Objective
        rev += r_ppa * (scd[k] + cd[k]) * DT
        rev += pr_k  * xd_k             * DT
        rev -= pr_k  * xc_k             * DT
        rev -= p.iex_fee_rs_mwh * (xc_k + xd_k) * DT
        rev -= SLACK_PENALTY * slack[k] * DT

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
            sc_out[t_abs]  = max(0.0, pulp.value(sc[k])     or 0.0)
            scd_out[t_abs] = max(0.0, pulp.value(scd[k])    or 0.0)
            cd_out[t_abs]  = max(0.0, pulp.value(cd[k])     or 0.0)
            cap_val = pulp.value(cap_rt[k]) or rtc_committed
            cap_out[t_abs] = float(cap_val)

            # FIX-3: issue advance notice for blocks that need >5% revision
            dev_pct = abs(cap_val - rtc_committed) / (rtc_committed + 1e-9)
            if dev_pct > p.rtc_tol_pct:
                t_notice = t_abs + p.rtc_advance_blocks
                if t_notice < T_BLOCKS:
                    rtc_notice_out[t_notice] = True

    return {
        "status":     status,
        "s_c_rt":     sc_out,
        "s_cd_rt":    scd_out,
        "c_d_rt":     cd_out,
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

    FIX-3: rtc_notice is issued at block B for block B+rtc_advance_blocks.
    FIX-1: captive_rt is soft upper-bound (not equality).
    """
    p         = params
    p_max     = p.p_max_mw
    r_ppa     = p.ppa_rate_rs_mwh
    USABLE    = p.usable_energy_mwh
    RTM_LEAD  = p.rtm_lead_blocks
    bid_block = block_B + RTM_LEAD
    SLACK_PENALTY = r_ppa * 2.0

    rtc_notice_out = rtc_notice.copy()

    if bid_block >= T_BLOCKS:
        return 0.0, 0.0, rtc_notice_out

    # Lag-4 RTM price bias adjustment
    rtm_adj = rtm_q50.copy().astype(float)
    if not np.isnan(p_rtm_lag4) and block_B >= 4:
        bias = p_rtm_lag4 - float(rtm_q50[block_B - 4])
        for t in range(bid_block, T_BLOCKS):
            rtm_adj[t] = max(0.0, rtm_adj[t] + bias * (0.85 ** (t - block_B)))

    solar_mask = compute_solar_band_mask_rtc(
        solar_da, p.solar_threshold_mw, p.solar_buffer_blocks)

    # Roll SoC forward from block_B to bid_block using committed schedules
    soc_rf = float(np.clip(soc_actual_B, p.e_min_mwh, p.e_max_mwh))
    for t in range(block_B, bid_block):
        xc_t = float(x_c_s1[t]);     xd_t = float(x_d_s1[t])
        yc_t = float(y_c_committed[t]); yd_t = float(y_d_committed[t])
        sc_t = float(s_c_rt[t]);     cd_t = float(c_d_rt[t])
        charge_e = p.eta_charge * (sc_t + xc_t + yc_t) * DT
        dis_e    = (cd_t + xd_t + yd_t) / p.eta_discharge * DT
        soc_rf   = float(np.clip(soc_rf + charge_e - dis_e, p.e_min_mwh, p.e_max_mwh))

    remaining = T_BLOCKS - bid_block
    if remaining <= 0:
        return 0.0, 0.0, rtc_notice_out

    prob  = pulp.LpProblem(f"S2A_RTC_b{block_B}", pulp.LpMaximize)
    y_c   = pulp.LpVariable.dicts("yc",  range(remaining), 0, p_max)
    y_d   = pulp.LpVariable.dicts("yd",  range(remaining), 0, p_max)
    slack = pulp.LpVariable.dicts("sl",  range(remaining), 0, p.rtc_mw)
    soc_lp = pulp.LpVariable.dicts("soc", range(remaining + 1),
                                   p.e_min_mwh, p.e_max_mwh)
    dl    = pulp.LpVariable.dicts("dl2a", range(remaining), cat="Binary")

    # captive_rt per block
    cap_rt_lp = {}
    for k in range(remaining):
        t_abs = bid_block + k
        if t_abs < T_BLOCKS and rtc_notice[t_abs]:
            cap_rt_lp[k] = pulp.LpVariable(f"crt_{k}",
                                            lowBound=p.rtc_min_mw, upBound=p.rtc_mw)
        else:
            lo = rtc_committed * (1.0 - p.rtc_tol_pct)
            hi = rtc_committed * (1.0 + p.rtc_tol_pct)
            cap_rt_lp[k] = pulp.LpVariable(f"crt_{k}", lowBound=lo, upBound=hi)

    prob += soc_lp[0]         == soc_rf
    prob += soc_lp[remaining] == p.soc_terminal_min_mwh

    # Cycle budget
    cycle_budget = max(0.0, (p.max_cycles_per_day or 1.0) * USABLE - cycle_used_so_far)
    prob += pulp.lpSum(
        [(float(c_d_rt[bid_block + k]) + float(x_d_s1[bid_block + k]) + y_d[k])
         * DT / p.eta_discharge for k in range(remaining)]
    ) <= cycle_budget, "cycle_2a"

    rev = 0
    for k in range(remaining):
        ta   = bid_block + k
        xc_t = float(x_c_s1[ta])
        xd_t = float(x_d_s1[ta])
        sc_t = float(s_c_rt[ta])
        cd_t = float(c_d_rt[ta])
        pr   = float(rtm_adj[ta])

        # PCS headroom constraints
        prob += sc_t + xc_t + y_c[k] <= p_max,              f"C2A2_{k}"
        prob += cd_t + xd_t + y_d[k] <= p_max,              f"C2A3_{k}"

        # Captive delivery (soft — upper + lower with slack)
        prob += sc_t + cd_t <= cap_rt_lp[k],                 f"CRTC2A_hi_{k}"
        prob += sc_t + cd_t >= cap_rt_lp[k] - slack[k],      f"CRTC2A_lo_{k}"

        # AC bus mutual exclusion (soft hints from Stage 1 locked flows)
        imp  = xc_t + sc_t
        exp_ = xd_t + cd_t
        if imp > 1e-6 and exp_ < 1e-6:
            prob += dl[k] >= 1
        elif exp_ > 1e-6 and imp < 1e-6:
            prob += dl[k] <= 0
        prob += xc_t + y_c[k] + sc_t <= p_max * dl[k],      f"C2A4a_{k}"
        prob += xd_t + y_d[k] + cd_t <= p_max * (1 - dl[k]),f"C2A4b_{k}"

        # SoC dynamics
        prob += soc_lp[k + 1] == (
            soc_lp[k]
            + p.eta_charge    * (sc_t + xc_t + y_c[k]) * DT
            - (1.0 / p.eta_discharge) * (cd_t + xd_t + y_d[k]) * DT
        ), f"C2A5_{k}"

        # SoC solar band
        if solar_mask[ta]:
            prob += soc_lp[k] >= p.soc_solar_low,            f"C2A6lo_{k}"
            prob += soc_lp[k] <= p.soc_solar_high,           f"C2A6hi_{k}"

        # Objective
        rev += pr * y_d[k] * DT - pr * y_c[k] * DT
        rev += r_ppa * cap_rt_lp[k] * DT
        rev -= p.iex_fee_rs_mwh * (y_c[k] + y_d[k]) * DT
        rev -= SLACK_PENALTY * slack[k] * DT

    prob.setObjective(rev)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] == "Optimal":
        y_c_bid = max(0.0, pulp.value(y_c[0]) or 0.0)
        y_d_bid = max(0.0, pulp.value(y_d[0]) or 0.0)

        # FIX-3: issue rtc_notice at block_B for delivery change at block_B + advance
        for k in range(remaining):
            cap_val = pulp.value(cap_rt_lp[k]) or rtc_committed
            dev_pct = abs(cap_val - rtc_committed) / (rtc_committed + 1e-9)
            if dev_pct > p.rtc_tol_pct:
                t_notice = bid_block + k + p.rtc_advance_blocks
                if t_notice < T_BLOCKS:
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
    p         = params
    r_ppa     = p.ppa_rate_rs_mwh
    RTM_LEAD  = p.rtm_lead_blocks
    avail_cap = p.avail_cap_mwh       # S_inv × DT = 4.1 MWh

    rtc_committed = float(stage1_result["RTC_committed"])

    x_c_s1  = np.array(stage1_result["x_c"],     dtype=float)
    x_d_s1  = np.array(stage1_result["x_d"],     dtype=float)
    sc_da   = np.array(stage1_result["s_c_da"],   dtype=float)
    scd_da  = np.array(stage1_result["s_cd_da"],  dtype=float)
    cd_da   = np.array(stage1_result["c_d_da"],   dtype=float)

    s_c_rt  = sc_da.copy()
    s_cd_rt = scd_da.copy()
    c_d_rt  = cd_da.copy()
    y_c_committed = np.zeros(T_BLOCKS)
    y_d_committed = np.zeros(T_BLOCKS)

    # captive_committed: starts as flat RTC_committed (what SLDC has)
    captive_committed = np.full(T_BLOCKS, rtc_committed, dtype=float)

    rtc_notice = np.zeros(T_BLOCKS, dtype=bool)

    soc_path = np.zeros(T_BLOCKS + 1)
    soc_path[0] = p.soc_initial_mwh

    s_c_actual_arr     = np.zeros(T_BLOCKS)
    s_cd_actual_arr    = np.zeros(T_BLOCKS)
    c_d_actual_arr     = np.zeros(T_BLOCKS)
    captive_actual_arr = np.zeros(T_BLOCKS)
    setpoint_arr       = np.zeros(T_BLOCKS)
    schedule_rt_arr    = np.zeros(T_BLOCKS)

    dsm_results            = []
    block_captive_net_arr  = np.zeros(T_BLOCKS)
    block_captive_penalty  = np.zeros(T_BLOCKS)
    block_iex_net_arr      = np.zeros(T_BLOCKS)
    block_degradation_arr  = np.zeros(T_BLOCKS)
    block_net_arr          = np.zeros(T_BLOCKS)
    no_bess_dsm_arr        = np.zeros(T_BLOCKS)
    no_bess_rev_arr        = np.zeros(T_BLOCKS)

    rtc_notice_issued_arr = np.zeros(T_BLOCKS, dtype=bool)
    rtc_notice_target_arr = np.full(T_BLOCKS, -1, dtype=int)

    cum_discharge_mwh = 0.0

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
                captive_committed_prev=captive_committed.copy(),  # FIX-8: pass copy
                rtc_notice=rtc_notice,
                cycle_used_so_far=cum_discharge_mwh,
            )
            if res2b["status"] == "Optimal":
                s_c_rt[B:]            = res2b["s_c_rt"][B:]
                s_cd_rt[B:]           = res2b["s_cd_rt"][B:]
                c_d_rt[B:]            = res2b["c_d_rt"][B:]
                captive_committed[B:] = res2b["captive_rt"][B:]
                old_notice            = rtc_notice
                rtc_notice            = res2b["rtc_notice"]
                # Record which blocks got new notices
                for tb in range(B, T_BLOCKS):
                    if rtc_notice[tb] and not old_notice[tb]:
                        rtc_notice_issued_arr[B]  = True
                        rtc_notice_target_arr[B]  = tb

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

        # ── Step 1: Block inputs ───────────────────────────────────────────
        xc_B  = float(x_c_s1[B])
        xd_B  = float(x_d_s1[B])
        yc_B  = float(y_c_committed[B])
        yd_B  = float(y_d_committed[B])

        dam_net_B    = xd_B - xc_B
        rtm_net_B    = yd_B - yc_B
        cap_rt_B     = float(s_cd_rt[B] + c_d_rt[B])
        schedule_rt_B = cap_rt_B + dam_net_B + rtm_net_B
        schedule_rt_arr[B] = schedule_rt_B

        setpoint_B = compute_setpoint_rtc(
            soc_path[B], schedule_rt_B,
            p.e_min_mwh, p.e_max_mwh, p.eta_charge, p.eta_discharge)
        setpoint_arr[B] = setpoint_B
        z_at = float(solar_at[B])

        # FIX-5: bess_discharge_cap subtracts committed x_d and y_d
        bess_discharge_cap = max(0.0, min(
            p.p_max_mw - xd_B - yd_B,
            (soc_path[B] - p.e_min_mwh) * p.eta_discharge / DT
        ))
        bess_charge_cap = max(0.0, min(
            p.p_max_mw - xc_B - yc_B,
            (p.e_max_mwh - soc_path[B]) / (p.eta_charge * DT)
        ))

        # ── Step 2: Dispatch (Case A / B / C) ─────────────────────────────
        if z_at > setpoint_B + 1e-6:          # Case A: solar surplus → charge
            s_c_actual  = min(bess_charge_cap, z_at - setpoint_B)
            s_cd_actual = z_at - s_c_actual
            c_d_actual  = 0.0
        elif z_at < setpoint_B - 1e-6:        # Case B: solar deficit → discharge
            s_cd_actual = z_at
            c_d_actual  = min(bess_discharge_cap, setpoint_B - z_at)
            s_c_actual  = 0.0
        else:                                   # Case C: exact match
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
        dsm = compute_dsm_settlement_rtc(captive_actual, schedule_rt_B, cr, avail_cap)
        dsm_results.append(dsm)

        # ── Step 4: RTC captive penalty (additional, separate from DSM) ───
        if captive_actual < p.rtc_min_mw:
            shortfall_mwh = (p.rtc_min_mw - captive_actual) * DT
            cap_penalty   = shortfall_mwh * r_ppa
        else:
            cap_penalty = 0.0

        block_captive_net_arr[B] = dsm["net_captive_cash"] - cap_penalty
        block_captive_penalty[B] = cap_penalty

        # ── Step 5: IEX revenue ────────────────────────────────────────────
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
        degradation           = p.degradation_cost_rs_mwh * total_discharge * DT
        block_degradation_arr[B] = degradation
        block_net_arr[B]      = block_captive_net_arr[B] + iex_net - degradation

        # ── Step 8: No-BESS counterfactual ────────────────────────────────
        nb_dsm = compute_dsm_settlement_rtc(
            z_at, float(captive_committed[B]), r_ppa, avail_cap)
        no_bess_dsm_arr[B] = nb_dsm["dsm_penalty"] + nb_dsm["dsm_haircut"]
        no_bess_rev_arr[B] = z_at * DT * r_ppa - no_bess_dsm_arr[B]

        if verbose and B % 16 == 0:
            print(f"  B={B:02d} | SoC={soc_path[B]:.1f}→{soc_path[B+1]:.1f} MWh "
                  f"| solar={z_at:.2f} | cap_act={captive_actual:.2f} "
                  f"| net={block_net_arr[B]:,.0f} Rs")

    # ── Aggregate results ──────────────────────────────────────────────────
    total_captive_penalty = float(np.sum(block_captive_penalty))
    total_iex_net         = float(np.sum(block_iex_net_arr))
    total_captive_net     = float(np.sum(block_captive_net_arr))
    total_degradation     = float(np.sum(block_degradation_arr))
    total_net_revenue     = float(np.sum(block_net_arr))
    total_no_bess_dsm     = float(np.sum(no_bess_dsm_arr))
    total_no_bess_rev     = float(np.sum(no_bess_rev_arr))
    total_bess_dsm_savings = float(np.sum(
        [dsm_results[b]["dsm_penalty"] + dsm_results[b]["dsm_haircut"]
         for b in range(T_BLOCKS)])) - (total_no_bess_dsm - total_no_bess_dsm)
    # Correct bess_dsm_savings: what no-BESS would have paid vs what we paid
    with_bess_dsm = sum(d["dsm_penalty"] + d["dsm_haircut"] for d in dsm_results)
    bess_dsm_savings = total_no_bess_dsm - with_bess_dsm

    return {
        # Core financials
        "net_revenue":           total_net_revenue,
        "captive_net_total":     total_captive_net,
        "iex_net_total":         total_iex_net,
        "captive_penalty_total": total_captive_penalty,
        "degradation_total":     total_degradation,
        "no_bess_revenue_total": total_no_bess_rev,
        "bess_dsm_savings":      bess_dsm_savings,
        "bess_total_value":      bess_dsm_savings + total_iex_net - total_degradation,
        # Per-block arrays
        "soc_path":              soc_path,
        "s_c_actual":            s_c_actual_arr,
        "s_cd_actual":           s_cd_actual_arr,
        "c_d_actual":            c_d_actual_arr,
        "captive_actual":        captive_actual_arr,
        "setpoint":              setpoint_arr,
        "schedule_rt":           schedule_rt_arr,
        "captive_committed":     captive_committed,
        "block_captive_net":     block_captive_net_arr,
        "block_captive_penalty": block_captive_penalty,
        "block_iex_net":         block_iex_net_arr,
        "block_degradation":     block_degradation_arr,
        "block_net":             block_net_arr,
        "no_bess_dsm":           no_bess_dsm_arr,
        "no_bess_revenue":       no_bess_rev_arr,
        "dsm_results":           dsm_results,
        # RTC notice tracking
        "rtc_notice_issued":     rtc_notice_issued_arr,
        "rtc_notice_target":     rtc_notice_target_arr,
        # Stage 2 routing
        "x_c":  x_c_s1, "x_d": x_d_s1,
        "y_c":  y_c_committed, "y_d": y_d_committed,
        "s_c_rt": s_c_rt, "s_cd_rt": s_cd_rt, "c_d_rt": c_d_rt,
    }
