"""
src/optimizer/two_stage_bess.py  -- Architecture v3
=====================================================
Solar (35 MWp, 25 MW inverter) + BESS (5 MWh, 2.5 MW PCS)

Physical topology:
  Inverter + PCS outputs connect in PARALLEL to ONE AC bus.
  AC bus has one output to Grid/Captive.

Stage 1:  D-1 planning MILP. Commits x_c, x_d, solar routing for 96 blocks.
Stage 2B: Solar reschedule at NC trigger blocks {34,42,50,58}.
Stage 2A: RTM receding-horizon MPC. Bids y_c/y_d for block B+RTM_LEAD.
Actuals:  Settlement using metered solar and actual IEX prices.

Key constraints (unified binary delta per block):
  Import mode (delta=1): x_c + s_c  <= p_max * delta        (PCS charges)
  Export mode (delta=0): x_d + c_d  <= p_max * (1-delta)    (PCS discharges)
  Export mode (delta=0): s_cd       <= S_inv * (1-delta)    (captive via bus)
  Solar balance:         s_c + s_cd + curtail == solar[t]
  Solar inverter:        s_c + s_cd <= S_inv
  BESS PCS discharge:    x_d + c_d  <= p_max
  SoC terminal:          soc[96] == 2.5 MWh (hard equality)
"""

import pulp
import numpy as np
from typing import Dict, List, Optional, Tuple

T_BLOCKS          = 96
DT                = 0.25
RESCHEDULE_BLOCKS = [34, 42, 50, 58]
DEFAULT_RTM_LEAD  = 3
DEFAULT_CAPTIVE_BUFFER = 12
DEFAULT_CAPTIVE_TOL    = 0.5


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


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: TwoStageBESS.solve()
# ══════════════════════════════════════════════════════════════════════════════

class TwoStageBESS:
    def __init__(self, params, config: Dict):
        self.params      = params
        self.config      = config
        self.solver_name = config.get("solver", "CBC")
        self.lambda_risk = config.get("lambda_risk", 0.0)
        self.lambda_dev  = config.get("lambda_dev", 0.0)
        self.dev_max     = config.get("dev_max_mw", 2.5)
        self.risk_alpha  = config.get("risk_alpha", 0.1)

    def solve(
        self,
        dam_scenarios: np.ndarray,
        rtm_scenarios: np.ndarray,
        solar_da:      np.ndarray,
    ) -> Dict:
        S = dam_scenarios.shape[0]
        assert dam_scenarios.shape[1] == T_BLOCKS
        assert len(solar_da) == T_BLOCKS

        p    = self.params
        p_max = p.p_max_mw
        S_inv = p.solar_inverter_mw
        r_ppa = p.ppa_rate_rs_mwh
        e_min_p, e_max_p = _plan_bounds(p)
        SOC_TARGET = p.soc_terminal_min_mwh

        solar_da = np.clip(solar_da, 0.0, p.solar_capacity_mwp)

        # Future max DAM price for solar opportunity cost (FIX 1)
        future_max_prices = np.zeros(T_BLOCKS)
        running_max = 0.0
        for t in range(T_BLOCKS - 1, -1, -1):
            running_max = max(running_max, float(np.mean(dam_scenarios[:, t])))
            future_max_prices[t] = running_max

        prob = pulp.LpProblem("Stage1_MILP", pulp.LpMaximize)

        # Decision variables (non-anticipative)
        x_c     = pulp.LpVariable.dicts("xc",  range(T_BLOCKS), 0, p_max)
        x_d     = pulp.LpVariable.dicts("xd",  range(T_BLOCKS), 0, p_max)
        s_c_da  = pulp.LpVariable.dicts("sc",  range(T_BLOCKS), 0, p_max)
        s_cd_da = pulp.LpVariable.dicts("scd", range(T_BLOCKS), 0, S_inv)
        c_d_da  = pulp.LpVariable.dicts("cd",  range(T_BLOCKS), 0, p_max)
        cu_da   = pulp.LpVariable.dicts("cu",  range(T_BLOCKS), 0)
        delta   = pulp.LpVariable.dicts("d",   range(T_BLOCKS), cat="Binary")

        # Per-scenario SoC + CVaR
        soc  = {s: pulp.LpVariable.dicts(f"soc{s}", range(T_BLOCKS + 1),
                                          e_min_p, e_max_p)
                for s in range(S)}
        zeta = pulp.LpVariable("zeta")
        u    = pulp.LpVariable.dicts("u", range(S), 0)

        scen_revs = []

        for s in range(S):
            prob += soc[s][0] == SOC_TARGET, f"soc_init_{s}"
            prob += soc[s][T_BLOCKS] == SOC_TARGET, f"soc_term_{s}"
            rev = 0

            for t in range(T_BLOCKS):
                p_dam = float(dam_scenarios[s, t])

                # C4 - SoC dynamics
                prob += soc[s][t + 1] == (
                    soc[s][t]
                    + p.eta_charge * (s_c_da[t] + x_c[t]) * DT
                    - (1.0 / p.eta_discharge) * (x_d[t] + c_d_da[t]) * DT
                ), f"soc_dyn_{s}_{t}"

                # Revenue
                rev += p_dam * x_d[t] * DT
                rev -= p_dam * x_c[t] * DT
                rev += r_ppa * c_d_da[t] * DT
                rev += r_ppa * s_cd_da[t] * DT

                # Solar opportunity cost (FIX 1)
                fmp = future_max_prices[t]
                if fmp > r_ppa:
                    eff_opp = r_ppa / min(fmp / r_ppa, 2.0)
                else:
                    eff_opp = r_ppa
                rev -= eff_opp * s_c_da[t] * DT

                # Costs
                rev -= p.iex_fee_rs_mwh * (x_c[t] + x_d[t]) * DT
                rev -= p.degradation_cost_rs_mwh * (x_d[t] + c_d_da[t]) * DT
                rev -= 135.0 * (x_c[t] + x_d[t]) * DT

            prob += u[s] >= zeta - rev, f"cvar_{s}"
            scen_revs.append(rev)

        # Non-anticipative constraints (per block)
        for t in range(T_BLOCKS):
            sol_t = float(solar_da[t])

            # C1 - Solar balance
            prob += s_c_da[t] + s_cd_da[t] + cu_da[t] == sol_t, f"sol_bal_{t}"

            # C2 - Solar inverter limit
            prob += s_c_da[t] + s_cd_da[t] <= S_inv, f"inv_lim_{t}"

            # C3 - PCS discharge limit
            prob += x_d[t] + c_d_da[t] <= p_max, f"pcs_dis_{t}"

            # C5 - Unified binary mutual exclusion
            # Import mode (delta=1): charge side
            prob += x_c[t] + s_c_da[t] <= p_max * delta[t], f"import_{t}"
            # Export mode (delta=0): discharge side
            prob += x_d[t] + c_d_da[t] <= p_max * (1 - delta[t]), f"export_{t}"
            # Export mode: solar to captive
            prob += s_cd_da[t] <= S_inv * (1 - delta[t]), f"scd_exp_{t}"

        # Objective
        avg_rev = pulp.lpSum(scen_revs) / S
        cvar = zeta - (1.0 / (S * self.risk_alpha)) * pulp.lpSum(
            [u[s] for s in range(S)]
        )
        prob.setObjective(avg_rev + self.lambda_risk * cvar)

        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)

        status = pulp.LpStatus[prob.status]
        if status != "Optimal":
            return _failed_result()

        def v(var, t):
            return max(0.0, pulp.value(var[t]) or 0.0)

        xc_v  = [v(x_c, t)    for t in range(T_BLOCKS)]
        xd_v  = [v(x_d, t)    for t in range(T_BLOCKS)]
        sc_v  = [v(s_c_da, t) for t in range(T_BLOCKS)]
        scd_v = [v(s_cd_da, t) for t in range(T_BLOCKS)]
        cd_v  = [v(c_d_da, t) for t in range(T_BLOCKS)]
        cu_v  = [v(cu_da, t)  for t in range(T_BLOCKS)]

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


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2B: reschedule_captive()
# ══════════════════════════════════════════════════════════════════════════════

def reschedule_captive(
    params, trigger_block, soc_actual,
    solar_nc_row, solar_da,
    rtm_q50,
    x_c_stage1, x_d_stage1,
    y_c_committed, y_d_committed,
    captive_committed,
) -> Dict:
    """
    Stage 2B: re-optimise solar routing from trigger_block to end of day
    using NC nowcast for next 12 blocks and DA forecast beyond.

    Locked inputs: x_c, x_d from Stage 1; y_c, y_d for blocks < trigger+RTM_LEAD.
    Captive buffer: blocks 0..11 relative to trigger must match captive_committed.
    No solar opportunity cost in objective.
    """
    p        = params
    p_max    = p.p_max_mw
    S_inv    = p.solar_inverter_mw
    r_ppa    = p.ppa_rate_rs_mwh
    e_min_p, e_max_p = _plan_bounds(p)
    SOC_TARGET = p.soc_terminal_min_mwh
    RTM_LEAD   = getattr(p, 'rtm_lead_blocks', DEFAULT_RTM_LEAD)
    CAP_BUF    = getattr(p, 'captive_buffer_blocks', DEFAULT_CAPTIVE_BUFFER)
    CAP_TOL    = getattr(p, 'captive_buffer_tolerance_mw', DEFAULT_CAPTIVE_TOL)

    B = trigger_block
    remaining = T_BLOCKS - B

    # Build solar blend: NC for first 12 blocks, DA beyond
    solar_rem = np.empty(remaining)
    for k in range(remaining):
        if k < len(solar_nc_row):
            solar_rem[k] = float(solar_nc_row[k])
        else:
            solar_rem[k] = float(solar_da[B + k])
    solar_rem = np.clip(solar_rem, 0.0, p.solar_capacity_mwp)

    # Locked IEX and RTM variables
    xc_rem = np.array(x_c_stage1[B:], dtype=float)
    xd_rem = np.array(x_d_stage1[B:], dtype=float)
    yc_rem = np.array(y_c_committed[B:], dtype=float)
    yd_rem = np.array(y_d_committed[B:], dtype=float)
    rtm_rem = np.array(rtm_q50[B:], dtype=float)
    cap_com = np.array(captive_committed[B:], dtype=float)

    prob = pulp.LpProblem(f"Stage2B_b{B}", pulp.LpMaximize)

    s_c_lp  = pulp.LpVariable.dicts("sc",  range(remaining), 0, p_max)
    s_cd_lp = pulp.LpVariable.dicts("scd", range(remaining), 0, S_inv)
    c_d_lp  = pulp.LpVariable.dicts("cd",  range(remaining), 0, p_max)
    cu_lp   = pulp.LpVariable.dicts("cu",  range(remaining), 0)
    soc     = pulp.LpVariable.dicts("soc", range(remaining + 1), e_min_p, e_max_p)
    delta   = pulp.LpVariable.dicts("d2b", range(remaining), cat="Binary")

    prob += soc[0] == float(np.clip(soc_actual, e_min_p, e_max_p)), "soc_init"
    prob += soc[remaining] == SOC_TARGET, "soc_term"

    rev = 0
    for k in range(remaining):
        xc_k  = float(xc_rem[k])
        xd_k  = float(xd_rem[k])
        sol_k = float(solar_rem[k])

        # y_c/y_d locked for blocks < RTM_LEAD from trigger, else 0
        yc_k = float(yc_rem[k]) if k < RTM_LEAD else 0.0
        yd_k = float(yd_rem[k]) if k < RTM_LEAD else 0.0

        # C2B-1: Solar balance
        prob += s_c_lp[k] + s_cd_lp[k] + cu_lp[k] == sol_k, f"sol_{k}"

        # C2B-2: Solar inverter limit
        prob += s_c_lp[k] + s_cd_lp[k] <= S_inv, f"inv_{k}"

        # C2B-3: PCS charge (includes locked y_c)
        prob += s_c_lp[k] + xc_k + yc_k <= p_max, f"ch_{k}"

        # C2B-4: PCS discharge (includes locked y_d)
        prob += c_d_lp[k] + xd_k + yd_k <= p_max, f"dis_{k}"

        # C2B-5: Binary mutual exclusion
        # Lock delta based on committed IEX + RTM direction
        total_import = xc_k + yc_k
        total_export = xd_k + yd_k
        if total_import > 1e-6:
            prob += delta[k] == 1, f"lock_imp_{k}"
        elif total_export > 1e-6:
            prob += delta[k] == 0, f"lock_exp_{k}"

        prob += xc_k + yc_k + s_c_lp[k] <= p_max * delta[k], f"imp_{k}"
        prob += xd_k + yd_k + c_d_lp[k] <= p_max * (1 - delta[k]), f"exp_{k}"
        prob += s_cd_lp[k] <= S_inv * (1 - delta[k]), f"scd_{k}"

        # C2B-6: SoC dynamics
        prob += soc[k + 1] == (
            soc[k]
            + p.eta_charge * (s_c_lp[k] + xc_k + yc_k) * DT
            - (1.0 / p.eta_discharge) * (c_d_lp[k] + xd_k + yd_k) * DT
        ), f"soc_{k}"

        # C2B-8: Captive buffer (first CAP_BUF blocks frozen)
        if k < CAP_BUF:
            cap_target = float(cap_com[k])
            prob += (s_cd_lp[k] + c_d_lp[k] >= cap_target - CAP_TOL,
                     f"cap_lo_{k}")
            prob += (s_cd_lp[k] + c_d_lp[k] <= cap_target + CAP_TOL,
                     f"cap_hi_{k}")

        # Objective: captive PPA revenue + locked RTM (no solar opp cost)
        p_rtm_k = float(rtm_rem[k])
        rev += r_ppa * s_cd_lp[k] * DT
        rev += r_ppa * c_d_lp[k] * DT
        rev += p_rtm_k * xd_k * DT
        rev -= p_rtm_k * xc_k * DT
        rev -= p.iex_fee_rs_mwh * (xc_k + xd_k) * DT
        rev -= p.degradation_cost_rs_mwh * (c_d_lp[k] + xd_k) * DT
        rev -= 135.0 * (xc_k + xd_k) * DT

    prob.setObjective(rev)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]
    sc_out  = np.zeros(T_BLOCKS)
    scd_out = np.zeros(T_BLOCKS)
    cd_out  = np.zeros(T_BLOCKS)
    cu_out  = np.zeros(T_BLOCKS)
    cap_out = np.zeros(T_BLOCKS)

    if status == "Optimal":
        for k in range(remaining):
            t = B + k
            sc_out[t]  = max(0.0, pulp.value(s_c_lp[k]) or 0.0)
            scd_out[t] = max(0.0, pulp.value(s_cd_lp[k]) or 0.0)
            cd_out[t]  = max(0.0, pulp.value(c_d_lp[k]) or 0.0)
            cu_out[t]  = max(0.0, pulp.value(cu_lp[k]) or 0.0)
            cap_out[t] = scd_out[t] + cd_out[t]

    return {
        "status": status,
        "s_c_rt": sc_out, "s_cd_rt": scd_out,
        "c_d_rt": cd_out, "curtail_rt": cu_out,
        "captive_rt": cap_out,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2A: solve_stage2a()
# ══════════════════════════════════════════════════════════════════════════════

def solve_stage2a(
    params, block_B, soc_actual_B,
    dam_schedule, dam_actual,
    p_rtm_lag4, rtm_q50,
    s_c_da, s_cd_da, c_d_da,
    s_c_rt, s_cd_rt, c_d_rt,
    y_c_committed, y_d_committed,
    x_c_stage1, x_d_stage1,
    verbose=False,
) -> Tuple[float, float]:
    """
    Stage 2A: Receding-horizon MPC for RTM dispatch.

    Bids y_c and y_d for block B + RTM_LEAD (default B+3).
    LP spans blocks B+RTM_LEAD..95. Blocks 0..B+RTM_LEAD-1 are used for
    SoC roll-forward only (all variables locked).

    Two regimes for solar routing:
      blocks < B:  use s_c_da, s_cd_da, c_d_da (Stage 1 plan)
      blocks >= B: use s_c_rt, s_cd_rt, c_d_rt (Stage 2B revised plan)

    Returns (y_c_bid, y_d_bid) for block B+RTM_LEAD.
    """
    p      = params
    p_max  = p.p_max_mw
    S_inv  = p.solar_inverter_mw
    e_min_p, e_max_p = _plan_bounds(p)
    SOC_TARGET = p.soc_terminal_min_mwh
    RTM_LEAD = getattr(p, 'rtm_lead_blocks', DEFAULT_RTM_LEAD)

    bid_block = block_B + RTM_LEAD
    if bid_block >= T_BLOCKS:
        return 0.0, 0.0

    # -- Lag-4 conditioning on RTM q50 --
    rtm_adj = rtm_q50.copy().astype(float)
    if not np.isnan(p_rtm_lag4) and block_B >= 4:
        q50_at_lag4 = float(rtm_q50[block_B - 4])
        if q50_at_lag4 > 0:
            bias = p_rtm_lag4 - q50_at_lag4
            for t in range(bid_block, T_BLOCKS):
                decay = 0.85 ** (t - block_B)
                rtm_adj[t] = max(0.0, rtm_adj[t] + bias * decay)

    # -- Roll forward SoC from block_B through blocks B..bid_block-1 --
    soc_rollforward = float(np.clip(soc_actual_B, e_min_p, e_max_p))
    for t in range(block_B, bid_block):
        xc_t = float(x_c_stage1[t])
        xd_t = float(x_d_stage1[t])
        yc_t = float(y_c_committed[t])
        yd_t = float(y_d_committed[t])
        if t < block_B:
            sc_t = float(s_c_da[t])
            cd_t = float(c_d_da[t])
        else:
            sc_t = float(s_c_rt[t])
            cd_t = float(c_d_rt[t])
        charge_e    = p.eta_charge * (sc_t + xc_t + yc_t) * DT
        discharge_e = (xd_t + cd_t + yd_t) / p.eta_discharge * DT
        soc_rollforward = float(np.clip(
            soc_rollforward + charge_e - discharge_e, e_min_p, e_max_p
        ))

    # -- Build LP over blocks bid_block..95 --
    remaining = T_BLOCKS - bid_block
    if remaining <= 0:
        return 0.0, 0.0

    prob = pulp.LpProblem(f"Stage2A_B{block_B}", pulp.LpMaximize)

    y_c_lp = pulp.LpVariable.dicts("yc", range(remaining), 0, p_max)
    y_d_lp = pulp.LpVariable.dicts("yd", range(remaining), 0, p_max)
    soc_lp = pulp.LpVariable.dicts("s",  range(remaining + 1), e_min_p, e_max_p)
    delta  = pulp.LpVariable.dicts("d2a", range(remaining), cat="Binary")

    prob += soc_lp[0] == soc_rollforward, "soc_init"
    prob += soc_lp[remaining] == SOC_TARGET, "soc_term"

    rev = 0
    for k in range(remaining):
        t_abs = bid_block + k
        xc_t = float(x_c_stage1[t_abs])
        xd_t = float(x_d_stage1[t_abs])

        # Solar routing: use _rt for blocks >= block_B, else _da
        if t_abs >= block_B:
            sc_t  = float(s_c_rt[t_abs])
            scd_t = float(s_cd_rt[t_abs])
            cd_t  = float(c_d_rt[t_abs])
        else:
            sc_t  = float(s_c_da[t_abs])
            scd_t = float(s_cd_da[t_abs])
            cd_t  = float(c_d_da[t_abs])

        # PCS limits
        prob += sc_t + xc_t + y_c_lp[k] <= p_max, f"ch_{k}"
        prob += cd_t + xd_t + y_d_lp[k] <= p_max, f"dis_{k}"

        # Binary mutual exclusion
        prob += xc_t + y_c_lp[k] + sc_t <= p_max * delta[k], f"imp_{k}"
        prob += xd_t + y_d_lp[k] + cd_t <= p_max * (1 - delta[k]), f"exp_{k}"
        prob += scd_t <= S_inv * (1 - delta[k]), f"scd_{k}"

        # Lock delta if direction already determined
        total_imp = xc_t + sc_t
        total_exp = xd_t + cd_t + scd_t
        if total_imp > 1e-6 and total_exp < 1e-6:
            prob += delta[k] >= 1, f"lock_imp_{k}"
        elif total_exp > 1e-6 and total_imp < 1e-6:
            prob += delta[k] <= 0, f"lock_exp_{k}"

        # SoC dynamics
        prob += soc_lp[k + 1] == (
            soc_lp[k]
            + p.eta_charge * (sc_t + xc_t + y_c_lp[k]) * DT
            - (1.0 / p.eta_discharge) * (cd_t + xd_t + y_d_lp[k]) * DT
        ), f"soc_{k}"

        # Objective: only RTM revenue/costs for y_c/y_d decisions
        p_rtm_k = float(rtm_adj[t_abs])
        rev += p_rtm_k * y_d_lp[k] * DT
        rev -= p_rtm_k * y_c_lp[k] * DT
        rev -= p.iex_fee_rs_mwh * (y_c_lp[k] + y_d_lp[k]) * DT
        rev -= p.degradation_cost_rs_mwh * y_d_lp[k] * DT
        rev -= 135.0 * (y_c_lp[k] + y_d_lp[k]) * DT

    prob.setObjective(rev)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] == "Optimal":
        y_c_bid = max(0.0, pulp.value(y_c_lp[0]) or 0.0)
        y_d_bid = max(0.0, pulp.value(y_d_lp[0]) or 0.0)
    else:
        # Fallback: relax terminal to floor
        prob2 = pulp.LpProblem(f"Stage2A_B{block_B}_relax", pulp.LpMaximize)
        y_c2 = pulp.LpVariable.dicts("yc2", range(remaining), 0, p_max)
        y_d2 = pulp.LpVariable.dicts("yd2", range(remaining), 0, p_max)
        soc2 = pulp.LpVariable.dicts("s2",  range(remaining + 1), e_min_p, e_max_p)
        del2 = pulp.LpVariable.dicts("d2r", range(remaining), cat="Binary")
        prob2 += soc2[0] == soc_rollforward
        prob2 += soc2[remaining] >= e_min_p
        rev2 = 0
        for k in range(remaining):
            t_abs = bid_block + k
            xc_t = float(x_c_stage1[t_abs])
            xd_t = float(x_d_stage1[t_abs])
            if t_abs >= block_B:
                sc_t = float(s_c_rt[t_abs]); cd_t = float(c_d_rt[t_abs])
                scd_t = float(s_cd_rt[t_abs])
            else:
                sc_t = float(s_c_da[t_abs]); cd_t = float(c_d_da[t_abs])
                scd_t = float(s_cd_da[t_abs])
            prob2 += sc_t + xc_t + y_c2[k] <= p_max
            prob2 += cd_t + xd_t + y_d2[k] <= p_max
            prob2 += xc_t + y_c2[k] + sc_t <= p_max * del2[k]
            prob2 += xd_t + y_d2[k] + cd_t <= p_max * (1 - del2[k])
            prob2 += scd_t <= S_inv * (1 - del2[k])
            prob2 += soc2[k + 1] == (
                soc2[k]
                + p.eta_charge * (sc_t + xc_t + y_c2[k]) * DT
                - (1.0 / p.eta_discharge) * (cd_t + xd_t + y_d2[k]) * DT
            )
            p_rtm_k = float(rtm_adj[t_abs])
            rev2 += p_rtm_k * y_d2[k] * DT - p_rtm_k * y_c2[k] * DT
            rev2 -= p.iex_fee_rs_mwh * (y_c2[k] + y_d2[k]) * DT
            rev2 -= p.degradation_cost_rs_mwh * y_d2[k] * DT
            rev2 -= 135.0 * (y_c2[k] + y_d2[k]) * DT
        prob2.setObjective(rev2)
        prob2.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob2.status] == "Optimal":
            y_c_bid = max(0.0, pulp.value(y_c2[0]) or 0.0)
            y_d_bid = max(0.0, pulp.value(y_d2[0]) or 0.0)
        else:
            y_c_bid, y_d_bid = 0.0, 0.0

    if verbose:
        print(f"  [2A] B{block_B:03d} bid→B{bid_block:03d} | "
              f"SoC@B3={soc_rollforward:.3f} | y_c={y_c_bid:.3f} y_d={y_d_bid:.3f}")
    return y_c_bid, y_d_bid


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE ACTUALS: Orchestration loop
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_actuals_solar(
    params, stage1_result,
    dam_actual, rtm_actual, rtm_q50,
    solar_da, solar_nc, solar_at,
    reschedule_blocks=RESCHEDULE_BLOCKS,
    verbose=False,
) -> Dict:
    """
    Run the full day simulation: Stage 2B at triggers, Stage 2A every block,
    actuals settlement using metered solar.
    """
    p      = params
    r_ppa  = p.ppa_rate_rs_mwh
    RTM_LEAD = getattr(p, 'rtm_lead_blocks', DEFAULT_RTM_LEAD)
    CAP_BUF  = getattr(p, 'captive_buffer_blocks', DEFAULT_CAPTIVE_BUFFER)

    x_c_s1  = np.array(stage1_result["x_c"])
    x_d_s1  = np.array(stage1_result["x_d"])
    dam_sch = np.array(stage1_result["dam_schedule"])
    sc_da   = np.array(stage1_result["s_c_da"])
    scd_da  = np.array(stage1_result["s_cd_da"])
    cd_da   = np.array(stage1_result["c_d_da"])

    # Initialize: start with DA plan, will be overwritten by 2B
    s_c_rt  = sc_da.copy()
    s_cd_rt = scd_da.copy()
    c_d_rt  = cd_da.copy()

    y_c_committed = np.zeros(T_BLOCKS)
    y_d_committed = np.zeros(T_BLOCKS)

    captive_committed = np.array(stage1_result["captive_schedule_da"], dtype=float)
    captive_actual_arr = np.zeros(T_BLOCKS)
    captive_shortfall  = np.zeros(T_BLOCKS)

    soc_path = np.zeros(T_BLOCKS + 1)
    soc_path[0] = p.soc_initial_mwh

    rev_dam = np.zeros(T_BLOCKS)
    rev_rtm = np.zeros(T_BLOCKS)
    rev_cap = np.zeros(T_BLOCKS)
    costs   = np.zeros(T_BLOCKS)
    dsm_energy = np.zeros(T_BLOCKS)

    z_nc_blend = solar_da.copy().astype(float)

    s_c_actual_arr  = np.zeros(T_BLOCKS)
    s_cd_actual_arr = np.zeros(T_BLOCKS)
    c_d_actual_arr  = np.zeros(T_BLOCKS)
    curtail_actual_arr = np.zeros(T_BLOCKS)

    for B in range(T_BLOCKS):
        p_rtm_lag4 = float(rtm_actual[B - 4]) if B >= 4 else np.nan

        # ── STAGE 2B (at trigger blocks) ──
        if B in reschedule_blocks:
            res2b = reschedule_captive(
                params=p, trigger_block=B, soc_actual=soc_path[B],
                solar_nc_row=solar_nc[B], solar_da=solar_da,
                rtm_q50=rtm_q50,
                x_c_stage1=x_c_s1, x_d_stage1=x_d_s1,
                y_c_committed=y_c_committed, y_d_committed=y_d_committed,
                captive_committed=captive_committed,
            )
            if res2b["status"] == "Optimal":
                s_c_rt[B:]  = res2b["s_c_rt"][B:]
                s_cd_rt[B:] = res2b["s_cd_rt"][B:]
                c_d_rt[B:]  = res2b["c_d_rt"][B:]
                # Update NC blend for diagnostics
                for k in range(T_BLOCKS - B):
                    if k < 12:
                        z_nc_blend[B + k] = float(solar_nc[B][k])
                    else:
                        z_nc_blend[B + k] = float(solar_da[B + k])
                # Update captive_committed ONLY beyond buffer
                for k in range(CAP_BUF, T_BLOCKS - B):
                    captive_committed[B + k] = float(
                        res2b["s_cd_rt"][B + k] + res2b["c_d_rt"][B + k]
                    )

        # ── STAGE 2A (every block) ──
        bid_block = B + RTM_LEAD
        if bid_block < T_BLOCKS:
            y_c_bid, y_d_bid = solve_stage2a(
                params=p, block_B=B, soc_actual_B=soc_path[B],
                dam_schedule=dam_sch, dam_actual=dam_actual,
                p_rtm_lag4=p_rtm_lag4, rtm_q50=rtm_q50,
                s_c_da=sc_da, s_cd_da=scd_da, c_d_da=cd_da,
                s_c_rt=s_c_rt, s_cd_rt=s_cd_rt, c_d_rt=c_d_rt,
                y_c_committed=y_c_committed, y_d_committed=y_d_committed,
                x_c_stage1=x_c_s1, x_d_stage1=x_d_s1,
                verbose=verbose,
            )
            y_c_committed[bid_block] = y_c_bid
            y_d_committed[bid_block] = y_d_bid

        # ── ACTUALS SETTLEMENT (block B) ──
        xc_B = float(x_c_s1[B])
        xd_B = float(x_d_s1[B])
        yc_B = float(y_c_committed[B])
        yd_B = float(y_d_committed[B])

        # Solar actual routing (priority: BESS charge → captive → curtail)
        z_at = float(solar_at[B])
        s_c_plan  = float(s_c_rt[B])
        s_cd_plan = float(s_cd_rt[B])
        c_d_plan  = float(c_d_rt[B])

        s_c_actual  = min(s_c_plan, z_at)
        solar_left  = max(0.0, z_at - s_c_actual)
        s_cd_actual = min(s_cd_plan, solar_left)
        solar_left2 = max(0.0, solar_left - s_cd_actual)
        curtail_act = solar_left2
        c_d_actual  = c_d_plan  # BESS->captive unaffected by solar forecast

        s_c_actual_arr[B]  = s_c_actual
        s_cd_actual_arr[B] = s_cd_actual
        c_d_actual_arr[B]  = c_d_actual
        curtail_actual_arr[B] = curtail_act

        captive_act = s_cd_actual + c_d_actual
        captive_actual_arr[B] = captive_act
        captive_shortfall[B] = max(0.0, captive_committed[B] - captive_act)

        # Revenue
        rev_dam[B] = float(dam_actual[B]) * (xd_B - xc_B) * DT
        rev_rtm[B] = float(rtm_actual[B]) * (yd_B - yc_B) * DT
        rev_cap[B] = r_ppa * captive_act * DT

        # Costs
        iex_cost = p.iex_fee_rs_mwh * (xc_B + xd_B + yc_B + yd_B) * DT
        deg_cost = p.degradation_cost_rs_mwh * (xd_B + c_d_actual + yd_B) * DT
        dsm_prox = 135.0 * (xc_B + xd_B + yc_B + yd_B) * DT

        # SoC update
        charge_e    = p.eta_charge * (s_c_actual + xc_B + yc_B) * DT
        discharge_e = (xd_B + c_d_actual + yd_B) / p.eta_discharge * DT

        # DSM charge failure check
        if soc_path[B] + charge_e > p.e_max_mwh:
            rejected = soc_path[B] + charge_e - p.e_max_mwh
            dsm_energy[B] = rejected
            dsm_prox += rejected * float(dam_actual[B]) * 1.25

        costs[B] = iex_cost + deg_cost + dsm_prox

        soc_path[B + 1] = float(np.clip(
            soc_path[B] + charge_e - discharge_e,
            p.e_min_mwh, p.e_max_mwh
        ))

    gross = float(np.sum(rev_dam + rev_rtm + rev_cap))
    total_cost = float(np.sum(costs))

    return {
        "revenue":           gross,
        "net_revenue":       gross - total_cost,
        "y_c":               y_c_committed,
        "y_d":               y_d_committed,
        "s_c_rt":            s_c_rt,
        "s_cd_rt":           s_cd_rt,
        "c_d_rt":            c_d_rt,
        "s_c_actual":        s_c_actual_arr,
        "s_cd_actual":       s_cd_actual_arr,
        "c_d_actual":        c_d_actual_arr,
        "curtail_actual":    curtail_actual_arr,
        "captive_actual":    captive_actual_arr,
        "captive_shortfall": captive_shortfall,
        "captive_committed": captive_committed,
        "soc_path":          soc_path,
        "soc":               soc_path.tolist(),
        "z_nc_blend":        z_nc_blend,
        "rtm_schedule":      (y_d_committed - y_c_committed).tolist(),
        "block_rev_dam":     rev_dam,
        "block_rev_rtm":     rev_rtm,
        "block_rev_captive": rev_cap,
        "block_costs":       costs,
        "block_dsm_energy":  dsm_energy,
        "total_dsm_mwh":     float(np.sum(dsm_energy)),
        "fees_breakdown": {
            "iex_revenue_dam":  float(np.sum(rev_dam)),
            "iex_revenue_rtm":  float(np.sum(rev_rtm)),
            "captive_revenue":  float(np.sum(rev_cap)),
            "total_costs":      total_cost,
        },
    }
