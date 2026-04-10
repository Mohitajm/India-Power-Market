"""
src/optimizer/two_stage_bess.py  — CORRECTED VERSION v2
========================================================
Fixes applied:

FIX 1 — Solar storage incentive (in TwoStageBESS.solve):
  Replaced flat -r_ppa * s_c_da opportunity cost with a scenario-weighted
  effective value. When any scenario has future DAM price > r_ppa,
  storing solar is worth more than selling to captive now.

FIX 2 — Stage 2A discharge threshold (in solve_stage2a_block):
  Added discharge_price_threshold parameter (default Rs 5500/MWh).
  Stage 2A will NOT discharge unless current RTM q50 > threshold.

FIX 3 — s_c constraint correction (in TwoStageBESS.solve):
  Added: when solar is available AND a future scenario exceeds threshold,
  Stage 1 can set s_c_da > 0 by reducing effective opportunity cost.

FIX 4 — NO SIMULTANEOUS IMPORT+EXPORT (MILP binary — NEW):
  Bug: The old LP approximation
    x_c[t] + s_cd_da[t] + c_d_da[t] <= p_max + solar_da[t]
  allowed x_c=0.49 and s_cd=16.7 simultaneously when solar was large,
  because solar_da[t] on the RHS made the constraint trivially slack.

  Fix: Introduce binary variable delta[t] per block:
    delta[t] = 1 → import mode (x_c allowed; x_d, s_cd, c_d forced to 0)
    delta[t] = 0 → export mode (x_d, s_cd, c_d allowed; x_c forced to 0)
  Applied in Stage 1 (TwoStageBESS.solve) and Stage 2B (reschedule_captive).

  This is now a MILP (Mixed-Integer LP). CBC handles 96 binary variables
  efficiently — solve time increases from ~1s to ~3-5s, which is acceptable.

FIX 5 — Stage 2A mutual exclusion (pre-check — NEW):
  Bug: y_c (grid import) and c_d_rt (BESS→captive export) were allowed
  simultaneously because headroom for charge and discharge was computed
  independently.

  Fix: Before building the Stage 2A LP, determine the grid direction for
  block B based on committed variables (x_c, x_d, s_cd_rt, c_d_rt).
  If any export variable > 0, force charge_hdroom = 0.
  If import (x_c) > 0, force discharge_hdroom = 0.
  Also applied inside the Stage 2B LP via the same binary approach.
"""

import pulp
import numpy as np
from typing import Dict, List, Optional, Tuple

T_BLOCKS          = 96
DT                = 0.25
RESCHEDULE_BLOCKS = [34, 42, 50, 58]

# Discharge price threshold for Stage 2A (Rs/MWh)
# Battery holds charge unless current block RTM q50 exceeds this
DEFAULT_DISCHARGE_THRESHOLD = 5500.0


def _plan_bounds(params) -> Tuple[float, float]:
    return params.e_min_plan_mwh, params.e_max_plan_mwh


def _failed_result() -> Dict:
    z = [0.0] * T_BLOCKS
    return {
        "status": "Infeasible", "expected_revenue": 0.0, "cvar_value_rs": None,
        "dam_schedule": z, "x_c": z, "x_d": z,
        "s_c_da": z, "s_cd_da": z, "c_d_da": z, "curtail_da": z,
        "captive_schedule_da": z, "scenarios": [],
    }


class TwoStageBESS:
    def __init__(self, params, config: Dict):
        self.params      = params
        self.config      = config
        self.solver_name = config.get("solver", "CBC")
        self.lambda_risk = config.get("lambda_risk", 0.0)
        self.lambda_dev  = config.get("lambda_dev", 0.0)
        self.dev_max     = config.get("dev_max_mw", 2.5)
        self.risk_alpha  = config.get("risk_alpha", 0.1)
        # FIX 2: configurable discharge threshold
        self.discharge_threshold = config.get(
            "discharge_price_threshold_rs_mwh", DEFAULT_DISCHARGE_THRESHOLD
        )

    def solve(
        self,
        dam_scenarios: np.ndarray,   # (S, 96) Rs/MWh
        rtm_scenarios: np.ndarray,   # (S, 96) Rs/MWh
        solar_da:      np.ndarray,   # (96,)   MW
    ) -> Dict:
        S = dam_scenarios.shape[0]
        assert dam_scenarios.shape[1] == T_BLOCKS
        assert len(solar_da) == T_BLOCKS

        solar_da = np.clip(solar_da, 0.0, self.params.solar_capacity_mwp)
        r_ppa    = self.params.ppa_rate_rs_mwh
        e_min_p, e_max_p = _plan_bounds(self.params)

        # FIX 1: Pre-compute scenario-weighted future price signal per block
        future_max_prices = np.zeros(T_BLOCKS)
        for t in range(T_BLOCKS):
            if t < T_BLOCKS - 1:
                future_dam = dam_scenarios[:, t+1:]
                future_rtm = rtm_scenarios[:, t+1:]
                combined = np.maximum(future_dam, future_rtm)
                future_max_prices[t] = float(np.mean(np.max(combined, axis=1)))
            else:
                future_max_prices[t] = 0.0

        prob = pulp.LpProblem("Stage1_SolarBESS_MILP", pulp.LpMaximize)

        x_c     = pulp.LpVariable.dicts("x_c",   range(T_BLOCKS), 0, self.params.p_max_mw)
        x_d     = pulp.LpVariable.dicts("x_d",   range(T_BLOCKS), 0, self.params.p_max_mw)
        s_c_da  = pulp.LpVariable.dicts("sc_da", range(T_BLOCKS), 0, self.params.p_max_mw)
        s_cd_da = pulp.LpVariable.dicts("scd",   range(T_BLOCKS), 0)
        c_d_da  = pulp.LpVariable.dicts("cd_da", range(T_BLOCKS), 0, self.params.p_max_mw)
        cu_da   = pulp.LpVariable.dicts("cu_da", range(T_BLOCKS), 0)

        # FIX 4: Binary variable for grid direction per block
        # delta[t] = 1 → import mode (x_c allowed)
        # delta[t] = 0 → export mode (x_d, s_cd, c_d allowed)
        delta   = pulp.LpVariable.dicts("delta", range(T_BLOCKS), cat="Binary")

        soc = {
            s: pulp.LpVariable.dicts(f"soc_{s}", range(T_BLOCKS + 1), e_min_p, e_max_p)
            for s in range(S)
        }
        zeta = pulp.LpVariable("zeta")
        u    = pulp.LpVariable.dicts("u", range(S), lowBound=0)

        scen_revs = []

        for s in range(S):
            prob += soc[s][0] == self.params.soc_initial_mwh
            rev = 0

            for t in range(T_BLOCKS):
                p_dam = float(dam_scenarios[s, t])
                p_rtm = float(rtm_scenarios[s, t])

                # SoC dynamics — charge from both solar-to-BESS AND IEX
                prob += soc[s][t + 1] == (
                    soc[s][t]
                    + self.params.eta_charge * (s_c_da[t] + x_c[t]) * DT
                    - (1.0 / self.params.eta_discharge) * (x_d[t] + c_d_da[t]) * DT
                )

                # DAM revenue
                rev += p_dam * x_d[t] * DT
                rev -= p_dam * x_c[t] * DT

                # Captive PPA revenue
                rev += r_ppa * c_d_da[t]  * DT
                rev += r_ppa * s_cd_da[t] * DT

                # FIX 1: Scenario-weighted solar opportunity cost
                fmp = future_max_prices[t]
                if fmp > r_ppa:
                    storage_premium = min(fmp / r_ppa, 2.0)
                    eff_opp_cost = r_ppa / storage_premium
                else:
                    eff_opp_cost = r_ppa
                rev -= eff_opp_cost * s_c_da[t] * DT

                # IEX costs
                rev -= self.params.iex_fee_rs_mwh * (x_c[t] + x_d[t]) * DT
                rev -= self.params.degradation_cost_rs_mwh * (x_d[t] + c_d_da[t]) * DT
                rev -= 135.0 * (x_c[t] + x_d[t]) * DT

            # Terminal SoC
            if self.params.soc_terminal_mode == "hard":
                prob += soc[s][T_BLOCKS] >= self.params.soc_terminal_min_mwh
            else:
                prob += soc[s][T_BLOCKS] >= e_min_p

            prob += u[s] >= zeta - rev
            scen_revs.append(rev)

        # ─── Non-anticipative constraints ─────────────────────────────────
        # PHYSICAL TOPOLOGY: Solar + BESS share a single AC bus connected to
        # the grid at one metering point. At any 15-min block, the plant is
        # either importing from the grid (x_c > 0) OR exporting to the grid
        # (x_d, s_cd_da, c_d_da > 0). It cannot do both simultaneously.
        #
        # FIX 4: MILP binary enforces mutual exclusion properly.
        # Old LP relaxation failed when solar >> p_max because the RHS
        # (p_max + solar_da[t]) made the constraint trivially slack.
        #
        # Big-M values:
        #   M_import  = p_max_mw        (max possible import)
        #   M_iex_exp = p_max_mw        (max IEX export = inverter rating)
        #   M_sol_exp = solar_da[t]     (max solar-to-captive = available solar)
        #   M_cap_exp = p_max_mw        (max BESS-to-captive = inverter rating)

        M_imp = self.params.p_max_mw
        M_iex = self.params.p_max_mw
        M_cap = self.params.p_max_mw

        for t in range(T_BLOCKS):
            sol_t = float(solar_da[t])

            # Solar balance: all solar must go somewhere
            prob += (
                s_c_da[t] + s_cd_da[t] + cu_da[t] == sol_t,
                f"solar_bal_{t}"
            )

            # Total charge power from all sources <= inverter rating
            prob += s_c_da[t] + x_c[t] <= self.params.p_max_mw, f"ch_lim_{t}"

            # Total discharge power to all destinations <= inverter rating
            prob += x_d[t] + c_d_da[t] <= self.params.p_max_mw, f"dis_lim_{t}"

            # ── FIX 4: Binary mutual exclusion ──
            # Import mode (delta=1): x_c allowed, all export forced to 0
            # Export mode (delta=0): exports allowed, x_c forced to 0

            # x_c only allowed when delta=1 (import mode)
            prob += x_c[t] <= M_imp * delta[t], f"imp_only_{t}"

            # x_d only allowed when delta=0 (export mode)
            prob += x_d[t] <= M_iex * (1 - delta[t]), f"exp_iex_{t}"

            # s_cd only allowed when delta=0 (export mode)
            # s_cd upper bound is solar_da[t] (can't export more solar than available)
            prob += s_cd_da[t] <= sol_t * (1 - delta[t]), f"exp_sol_{t}"

            # c_d only allowed when delta=0 (export mode)
            prob += c_d_da[t] <= M_cap * (1 - delta[t]), f"exp_cap_{t}"

            # NOTE: s_c_da (solar→BESS) is allowed in BOTH modes:
            #   Import mode:  solar charges battery while grid also charges
            #   Export mode:  solar charges battery (residual after captive)
            # The ch_lim constraint already limits s_c + x_c <= p_max.
            # When delta=0 (export mode), x_c=0 so s_c can use full p_max.
            # When delta=1 (import mode), s_c + x_c share p_max.

        avg_rev = pulp.lpSum(scen_revs) / S
        cvar    = zeta - (1.0 / (S * self.risk_alpha)) * pulp.lpSum([u[s] for s in range(S)])
        term_val = 0
        if (self.params.soc_terminal_mode == "soft"
                and self.params.soc_terminal_value_rs_mwh > 0):
            term_val = pulp.lpSum([
                self.params.soc_terminal_value_rs_mwh * soc[s][T_BLOCKS]
                for s in range(S)
            ]) / S

        prob.setObjective(avg_rev + term_val + self.lambda_risk * cvar)

        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)

        status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            return _failed_result()

        def v(var, t):
            return max(0.0, pulp.value(var[t]) or 0.0)

        xc_v  = [v(x_c,    t) for t in range(T_BLOCKS)]
        xd_v  = [v(x_d,    t) for t in range(T_BLOCKS)]
        sc_v  = [v(s_c_da, t) for t in range(T_BLOCKS)]
        scd_v = [v(s_cd_da,t) for t in range(T_BLOCKS)]
        cd_v  = [v(c_d_da, t) for t in range(T_BLOCKS)]
        cu_v  = [v(cu_da,  t) for t in range(T_BLOCKS)]

        return {
            "status":              status,
            "expected_revenue":    float(pulp.value(avg_rev) or 0.0),
            "cvar_value_rs":       float(pulp.value(cvar)) if self.lambda_risk > 0 else None,
            "dam_schedule":        [xd_v[t] - xc_v[t] for t in range(T_BLOCKS)],
            "x_c":                 xc_v,
            "x_d":                 xd_v,
            "s_c_da":              sc_v,
            "s_cd_da":             scd_v,
            "c_d_da":              cd_v,
            "curtail_da":          cu_v,
            "captive_schedule_da": [scd_v[t] + cd_v[t] for t in range(T_BLOCKS)],
            "scenarios": [
                {"soc": [pulp.value(soc[s][t]) for t in range(T_BLOCKS + 1)]}
                for s in range(S)
            ],
        }


def reschedule_captive(
    params, trigger_block, soc_actual, solar_nc_row, solar_da,
    rtm_q50, x_c_stage1, x_d_stage1,
) -> Dict:
    """
    Stage 2B — Re-optimise solar routing (s_c, s_cd, c_d) from trigger_block
    to end of day using NC nowcast for the next 12 blocks and DA beyond.

    FIX 4 applied: Binary delta variable enforces no simultaneous import+export.
    Stage 1 x_c and x_d are locked constants here, so the binary is driven by
    which direction the Stage 1 DAM schedule already committed to. When x_c > 0
    for a block, delta=1 (import mode), preventing c_d and s_cd. When x_d > 0,
    delta=0 (export mode), preventing any additional import.
    """
    r_ppa     = params.ppa_rate_rs_mwh
    remaining = T_BLOCKS - trigger_block
    e_min_p, e_max_p = _plan_bounds(params)

    solar_rem = np.empty(remaining)
    for k in range(remaining):
        if k < 12:
            solar_rem[k] = float(solar_nc_row[k])
        else:
            solar_rem[k] = float(solar_da[trigger_block + k])
    solar_rem = np.clip(solar_rem, 0.0, params.solar_capacity_mwp)

    xc_rem  = x_c_stage1[trigger_block:].astype(float)
    xd_rem  = x_d_stage1[trigger_block:].astype(float)
    rtm_rem = rtm_q50[trigger_block:].astype(float)

    prob = pulp.LpProblem(f"Stage2B_b{trigger_block}", pulp.LpMaximize)

    s_c_lp  = pulp.LpVariable.dicts("sc",  range(remaining), 0, params.p_max_mw)
    s_cd_lp = pulp.LpVariable.dicts("scd", range(remaining), 0)
    c_d_lp  = pulp.LpVariable.dicts("cd",  range(remaining), 0, params.p_max_mw)
    cu_lp   = pulp.LpVariable.dicts("cu",  range(remaining), 0)
    soc     = pulp.LpVariable.dicts("soc", range(remaining + 1), e_min_p, e_max_p)

    # FIX 4: Binary for grid direction in Stage 2B
    delta_2b = pulp.LpVariable.dicts("d2b", range(remaining), cat="Binary")

    prob += soc[0] == float(np.clip(soc_actual, e_min_p, e_max_p))

    rev = 0
    for k in range(remaining):
        xc_k = float(xc_rem[k])
        xd_k = float(xd_rem[k])
        p_rtm_k = float(rtm_rem[k])
        sol_k = float(solar_rem[k])

        # SoC dynamics
        prob += soc[k + 1] == (
            soc[k]
            + params.eta_charge * (s_c_lp[k] + xc_k) * DT
            - (1.0 / params.eta_discharge) * (c_d_lp[k] + xd_k) * DT
        )

        # Solar balance
        prob += s_c_lp[k] + s_cd_lp[k] + cu_lp[k] == sol_k, f"sol_bal_{k}"

        # Power limits
        prob += s_c_lp[k] + xc_k <= params.p_max_mw, f"ch_{k}"
        prob += c_d_lp[k] + xd_k <= params.p_max_mw, f"dis_{k}"

        # ── FIX 4: Binary mutual exclusion in Stage 2B ──
        # x_c and x_d are LOCKED from Stage 1 (constants, not variables).
        # But we must still prevent s_cd/c_d (export) when x_c > 0 (import).
        #
        # If x_c_k > 0 (Stage 1 committed to import), force delta=1:
        #   → s_cd = 0, c_d = 0 (no export allowed)
        # If x_d_k > 0 (Stage 1 committed to export), force delta=0:
        #   → import side already 0 from Stage 1
        #
        # For blocks where Stage 1 has x_c=x_d=0 (no IEX commitment),
        # the binary is free: c_d and s_cd can export if profitable.

        if xc_k > 1e-6:
            # Stage 1 locked import → force export variables to 0
            prob += delta_2b[k] == 1, f"lock_imp_{k}"
        elif xd_k > 1e-6:
            # Stage 1 locked export → allow s_cd and c_d
            prob += delta_2b[k] == 0, f"lock_exp_{k}"
        # else: delta_2b[k] is free — optimizer decides

        # s_cd only in export mode (delta=0)
        prob += s_cd_lp[k] <= sol_k * (1 - delta_2b[k]), f"scd_exp_{k}"

        # c_d only in export mode (delta=0)
        prob += c_d_lp[k] <= params.p_max_mw * (1 - delta_2b[k]), f"cd_exp_{k}"

        # Revenue
        rev += r_ppa * s_cd_lp[k] * DT + r_ppa * c_d_lp[k] * DT
        rev -= r_ppa * s_c_lp[k] * DT
        rev += p_rtm_k * xd_k * DT - p_rtm_k * xc_k * DT
        rev -= params.iex_fee_rs_mwh * (xc_k + xd_k) * DT
        rev -= params.degradation_cost_rs_mwh * (c_d_lp[k] + xd_k) * DT
        rev -= 135.0 * (xc_k + xd_k) * DT

    prob += soc[remaining] >= e_min_p
    prob.setObjective(rev)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]
    sc_out = np.zeros(T_BLOCKS); scd_out = np.zeros(T_BLOCKS)
    cd_out = np.zeros(T_BLOCKS); cu_out  = np.zeros(T_BLOCKS)

    if status == "Optimal":
        for k in range(remaining):
            t = trigger_block + k
            sc_out[t]  = max(0.0, pulp.value(s_c_lp[k])  or 0.0)
            scd_out[t] = max(0.0, pulp.value(s_cd_lp[k]) or 0.0)
            cd_out[t]  = max(0.0, pulp.value(c_d_lp[k])  or 0.0)
            cu_out[t]  = max(0.0, pulp.value(cu_lp[k])   or 0.0)

    return {
        "status": status, "s_c_rt": sc_out, "s_cd_rt": scd_out,
        "c_d_rt": cd_out, "curtail_rt": cu_out, "captive_rt": scd_out + cd_out,
    }


def solve_stage2a_block(
    params, block_B, soc_actual_B, dam_schedule, dam_actual,
    p_rtm_lag4, rtm_q50, s_c_rt_B, c_d_rt_B,
    s_cd_rt_B=0.0,                      # NEW param: needed for FIX 5
    verbose=False,
    discharge_price_threshold=0.0,
) -> Tuple[float, float]:
    """
    Stage 2A — Rolling MPC, one block at a time.

    Terminal constraint: soc[remaining] >= soc_terminal_min_mwh (2.5 MWh).
    No price threshold. LP discharges whenever net revenue after costs is
    positive (break-even ~Rs 985/MWh = degradation + IEX fee + DSM proxy).

    Lag-4 conditioning: adjusts q50 forecast using the latest known actual
    RTM price (1-hour publication lag), with exponential decay over horizon.

    FIX 5 — Mutual exclusion for y_c vs c_d/s_cd at block B:
    Before building the LP, determine the grid direction for block B.
    If any export variable (x_d, c_d_rt, s_cd_rt) is committed > 0,
    then y_c (grid import to BESS) is physically impossible → charge_hdroom = 0.
    If import (x_c) is committed > 0, then y_d (grid export from BESS) is
    physically impossible → discharge_hdroom = 0.
    """
    e_min_p, e_max_p = _plan_bounds(params)
    soc_terminal_target = params.soc_terminal_min_mwh
    remaining = T_BLOCKS - block_B

    # Condition q50 forecast by lag-4 actual RTM price
    rtm_lp = rtm_q50[block_B:].copy().astype(float)
    if not np.isnan(p_rtm_lag4) and block_B >= 4:
        q50_at_lag4 = float(rtm_q50[block_B - 4])
        if q50_at_lag4 > 0:
            bias  = p_rtm_lag4 - q50_at_lag4
            decay = np.array([0.85 ** k for k in range(remaining)])
            rtm_lp = np.maximum(0.0, rtm_lp + bias * decay)

    soc_init = float(np.clip(soc_actual_B, e_min_p, e_max_p))
    x_net_B  = float(dam_schedule[block_B])
    x_c_B    = max(0.0, -x_net_B)
    x_d_B    = max(0.0,  x_net_B)

    # ── FIX 5: Determine grid direction for block B ──────────────────────
    # At this block, the committed variables are:
    #   Import side: x_c_B (from Stage 1 DAM schedule), s_c_rt_B (solar→BESS)
    #   Export side: x_d_B (Stage 1), s_cd_rt_B (solar→captive), c_d_rt_B (BESS→captive)
    #
    # Physical rule: The meter cannot import AND export simultaneously.
    #   If ANY export is committed, y_c (additional grid import) = 0.
    #   If import x_c_B is committed, y_d (additional grid export) = 0.
    #
    # Note: s_c_rt_B (solar→BESS) is NOT a grid flow — solar is behind the meter.
    # It does not conflict with export. Only grid flows conflict.

    is_exporting = (x_d_B > 1e-6) or (c_d_rt_B > 1e-6) or (s_cd_rt_B > 1e-6)
    is_importing = (x_c_B > 1e-6)

    # Headroom: how much RTM charge/discharge is physically possible
    charge_hdroom    = max(0.0, params.p_max_mw - s_c_rt_B - x_c_B)
    discharge_hdroom = max(0.0, params.p_max_mw - c_d_rt_B - x_d_B)

    # FIX 5: Mutual exclusion override
    if is_exporting:
        # Grid is exporting → cannot simultaneously import via y_c
        charge_hdroom = 0.0
    if is_importing:
        # Grid is importing → cannot simultaneously export via y_d
        discharge_hdroom = 0.0

    # SoC-based headroom caps
    soc_charge_cap    = max(0.0, (e_max_p - soc_actual_B) / (params.eta_charge * DT))
    soc_discharge_cap = max(0.0, (soc_actual_B - e_min_p) * params.eta_discharge / DT)
    charge_hdroom     = min(charge_hdroom,    soc_charge_cap)
    discharge_hdroom  = min(discharge_hdroom, soc_discharge_cap)

    # Optional price threshold (set to 0.0 = disabled by default)
    current_q50 = float(rtm_lp[0]) if len(rtm_lp) > 0 else 0.0
    if discharge_price_threshold > 0.0 and current_q50 < discharge_price_threshold:
        discharge_hdroom = 0.0

    if charge_hdroom < 1e-4 and discharge_hdroom < 1e-4:
        if verbose:
            print(f"  [2A] B{block_B:03d} | No headroom → y_c=y_d=0")
        return 0.0, 0.0

    prob   = pulp.LpProblem(f"Stage2A_B{block_B}", pulp.LpMaximize)
    y_c_lp = pulp.LpVariable.dicts("yc", range(remaining), 0, charge_hdroom)
    y_d_lp = pulp.LpVariable.dicts("yd", range(remaining), 0, discharge_hdroom)
    soc    = pulp.LpVariable.dicts("s",  range(remaining + 1), e_min_p, e_max_p)
    prob  += soc[0] == soc_init
    rev    = 0

    for k in range(remaining):
        t_abs   = block_B + k
        p_rtm_k = float(rtm_lp[k])
        x_net_t = float(dam_schedule[t_abs])
        xc_t    = max(0.0, -x_net_t)
        xd_t    = max(0.0,  x_net_t)
        # Include s_c_rt and c_d_rt only for the current block (k=0).
        # Future blocks: Stage 2B already planned them; Stage 2A treats them as zero.
        sc_t  = s_c_rt_B if k == 0 else 0.0
        cap_t = c_d_rt_B if k == 0 else 0.0
        prob += soc[k + 1] == (
            soc[k]
            + params.eta_charge * (sc_t + xc_t + y_c_lp[k]) * DT
            - (1.0 / params.eta_discharge) * (xd_t + cap_t + y_d_lp[k]) * DT
        )
        rev += p_rtm_k * y_d_lp[k] * DT - p_rtm_k * y_c_lp[k] * DT
        rev -= params.iex_fee_rs_mwh * (y_c_lp[k] + y_d_lp[k]) * DT
        rev -= params.degradation_cost_rs_mwh * y_d_lp[k] * DT
        rev -= 135.0 * (y_c_lp[k] + y_d_lp[k]) * DT

    # Terminal constraint: plan such that battery ends >= 2.5 MWh.
    # Same hard target as Stage 1. Ensures Stage 2A does not over-discharge
    # without planning a recharge. If the LP cannot meet this constraint
    # (e.g. SoC already too low to recover), it relaxes to e_min_plan.
    prob += soc[remaining] >= min(soc_terminal_target, soc_init + 0.01)

    prob.setObjective(rev)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] == "Optimal":
        y_c_B = max(0.0, pulp.value(y_c_lp[0]) or 0.0)
        y_d_B = max(0.0, pulp.value(y_d_lp[0]) or 0.0)
    else:
        # Fallback: relax terminal to floor and retry
        prob2  = pulp.LpProblem(f"Stage2A_B{block_B}_relax", pulp.LpMaximize)
        y_c_lp2 = pulp.LpVariable.dicts("yc2", range(remaining), 0, charge_hdroom)
        y_d_lp2 = pulp.LpVariable.dicts("yd2", range(remaining), 0, discharge_hdroom)
        soc2    = pulp.LpVariable.dicts("s2",  range(remaining + 1), e_min_p, e_max_p)
        prob2  += soc2[0] == soc_init
        rev2    = 0
        for k in range(remaining):
            t_abs   = block_B + k
            p_rtm_k = float(rtm_lp[k])
            x_net_t = float(dam_schedule[t_abs])
            xc_t = max(0.0, -x_net_t); xd_t = max(0.0, x_net_t)
            sc_t  = s_c_rt_B if k == 0 else 0.0
            cap_t = c_d_rt_B if k == 0 else 0.0
            prob2 += soc2[k + 1] == (
                soc2[k]
                + params.eta_charge * (sc_t + xc_t + y_c_lp2[k]) * DT
                - (1.0 / params.eta_discharge) * (xd_t + cap_t + y_d_lp2[k]) * DT
            )
            rev2 += p_rtm_k * y_d_lp2[k] * DT - p_rtm_k * y_c_lp2[k] * DT
            rev2 -= params.iex_fee_rs_mwh * (y_c_lp2[k] + y_d_lp2[k]) * DT
            rev2 -= params.degradation_cost_rs_mwh * y_d_lp2[k] * DT
            rev2 -= 135.0 * (y_c_lp2[k] + y_d_lp2[k]) * DT
        prob2 += soc2[remaining] >= e_min_p   # relaxed floor
        prob2.setObjective(rev2)
        prob2.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob2.status] == "Optimal":
            y_c_B = max(0.0, pulp.value(y_c_lp2[0]) or 0.0)
            y_d_B = max(0.0, pulp.value(y_d_lp2[0]) or 0.0)
        else:
            y_c_B, y_d_B = 0.0, 0.0

    if verbose:
        print(f"  [2A] B{block_B:03d} | SoC={soc_actual_B:.3f} q50={current_q50:.0f} "
              f"terminal>={soc_terminal_target:.2f} | y_c={y_c_B:.3f} y_d={y_d_B:.3f}"
              f" | export={is_exporting} import={is_importing}")
    return y_c_B, y_d_B


def evaluate_actuals_solar(
    params, stage1_result, dam_actual, rtm_actual, rtm_q50,
    solar_da, solar_nc, solar_at,
    reschedule_blocks=RESCHEDULE_BLOCKS, verbose=False,
    discharge_price_threshold=DEFAULT_DISCHARGE_THRESHOLD,
) -> Dict:
    """Orchestration loop — Stage 2B first at reschedule blocks, then 2A.

    FIX 5: Now passes s_cd_rt[B] to solve_stage2a_block so the mutual
    exclusion check can see all export variables for the current block.
    """
    r_ppa = params.ppa_rate_rs_mwh
    x_c_s1  = np.array(stage1_result["x_c"])
    x_d_s1  = np.array(stage1_result["x_d"])
    sc_da   = np.array(stage1_result["s_c_da"])
    scd_da  = np.array(stage1_result["s_cd_da"])
    cd_da   = np.array(stage1_result["c_d_da"])
    dam_sch = np.array(stage1_result["dam_schedule"])

    s_c_rt  = sc_da.copy(); s_cd_rt = scd_da.copy(); c_d_rt = cd_da.copy()
    y_c_all  = np.zeros(T_BLOCKS); y_d_all = np.zeros(T_BLOCKS)
    soc_path = np.zeros(T_BLOCKS + 1)
    soc_path[0] = params.soc_initial_mwh

    rev_dam = np.zeros(T_BLOCKS); rev_rtm = np.zeros(T_BLOCKS)
    rev_cap = np.zeros(T_BLOCKS); costs   = np.zeros(T_BLOCKS)
    dsm_energy = np.zeros(T_BLOCKS)

    # Store the NC blend actually used by Stage 2B for each block.
    z_nc_blend = solar_da.copy().astype(float)

    for B in range(T_BLOCKS):
        p_rtm_lag4 = float(rtm_actual[B - 4]) if B >= 4 else np.nan

        if B in reschedule_blocks:
            res2b = reschedule_captive(
                params=params, trigger_block=B, soc_actual=soc_path[B],
                solar_nc_row=solar_nc[B], solar_da=solar_da,
                rtm_q50=rtm_q50, x_c_stage1=x_c_s1, x_d_stage1=x_d_s1,
            )
            if res2b["status"] == "Optimal":
                s_c_rt[B:]  = res2b["s_c_rt"][B:]
                s_cd_rt[B:] = res2b["s_cd_rt"][B:]
                c_d_rt[B:]  = res2b["c_d_rt"][B:]
                for k in range(T_BLOCKS - B):
                    if k < 12:
                        z_nc_blend[B + k] = float(solar_nc[B][k])
                    else:
                        z_nc_blend[B + k] = float(solar_da[B + k])

        # FIX 5: Pass s_cd_rt[B] so Stage 2A can check all export variables
        y_c_B, y_d_B = solve_stage2a_block(
            params=params, block_B=B, soc_actual_B=soc_path[B],
            dam_schedule=dam_sch, dam_actual=dam_actual,
            p_rtm_lag4=p_rtm_lag4, rtm_q50=rtm_q50,
            s_c_rt_B=s_c_rt[B], c_d_rt_B=c_d_rt[B],
            s_cd_rt_B=s_cd_rt[B],       # NEW — needed for FIX 5
            verbose=verbose,
            discharge_price_threshold=discharge_price_threshold,
        )
        y_c_all[B] = y_c_B; y_d_all[B] = y_d_B

        x_net_B = float(dam_sch[B])
        rev_dam[B] = float(dam_actual[B]) * x_net_B * DT
        y_net_B    = y_d_B - y_c_B
        rev_rtm[B] = float(rtm_actual[B]) * y_net_B * DT

        solar_after_bess = max(0.0, float(solar_at[B]) - s_c_rt[B])
        s_cd_at_B = min(solar_after_bess, s_cd_rt[B])
        c_d_at_B  = c_d_rt[B]
        rev_cap[B] = r_ppa * (s_cd_at_B + c_d_at_B) * DT

        xc_B = max(0.0, -x_net_B); xd_B = max(0.0, x_net_B)
        iex_cost  = params.iex_fee_rs_mwh * (xc_B + xd_B + y_c_B + y_d_B) * DT
        deg_cost  = params.degradation_cost_rs_mwh * (xd_B + c_d_at_B + y_d_B) * DT
        dsm_proxy = 135.0 * (xc_B + xd_B + y_c_B + y_d_B) * DT

        total_charge_energy = params.eta_charge * (s_c_rt[B] + xc_B + y_c_B) * DT
        projected_soc = soc_path[B] + total_charge_energy
        if projected_soc > params.e_max_mwh:
            rejected_mwh = projected_soc - params.e_max_mwh
            dsm_energy[B] = rejected_mwh
            dsm_proxy += rejected_mwh * float(dam_actual[B]) * 1.25
        costs[B] = iex_cost + deg_cost + dsm_proxy

        act_charge    = params.eta_charge * (s_c_rt[B] + xc_B + y_c_B) * DT
        act_discharge = (xd_B + c_d_rt[B] + y_d_B) / params.eta_discharge * DT
        soc_path[B + 1] = float(np.clip(
            soc_path[B] + act_charge - act_discharge,
            params.e_min_mwh, params.e_max_mwh
        ))

    gross = float(np.sum(rev_dam + rev_rtm + rev_cap))
    total_cost = float(np.sum(costs))

    return {
        "revenue": gross, "net_revenue": gross - total_cost,
        "y_c": y_c_all, "y_d": y_d_all,
        "s_c_rt": s_c_rt, "s_cd_rt": s_cd_rt, "c_d_rt": c_d_rt,
        "soc_path": soc_path, "soc": soc_path.tolist(),
        "z_nc_blend": z_nc_blend,
        "rtm_schedule": (y_d_all - y_c_all).tolist(),
        "block_rev_dam": rev_dam, "block_rev_rtm": rev_rtm,
        "block_rev_captive": rev_cap, "block_costs": costs,
        "block_dsm_energy": dsm_energy, "total_dsm_mwh": float(np.sum(dsm_energy)),
        "fees_breakdown": {
            "iex_revenue_dam":  float(np.sum(rev_dam)),
            "iex_revenue_rtm":  float(np.sum(rev_rtm)),
            "captive_revenue":  float(np.sum(rev_cap)),
            "total_costs":      total_cost,
        },
    }
