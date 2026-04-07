"""
src/optimizer/two_stage_bess.py
================================
Solar + BESS Two-Stage Stochastic Optimizer.  96-block (15-min) resolution.

SYSTEM:  35 MWp Solar PV  +  5 MWh BESS (2.5 MW charge/discharge)
PLANT:   Jamnagar, Gujarat, India

=============================================================================
ARCHITECTURE
=============================================================================

STAGE 1  (D-1 10:00 IST — solved ONCE, non-anticipative over 100 scenarios)
  Inputs : z_sol_da[96] MW, 100 DAM scenarios (S,96), 100 RTM scenarios (S,96)
  Decides: x_c[t], x_d[t]     IEX DAM charge/discharge   (non-anticipative)
           s_c_da[t]           Solar -> BESS   DA plan     (non-anticipative)
           s_cd_da[t]          Solar -> Captive DA plan    (non-anticipative)
           c_d_da[t]           BESS  -> Captive DA plan    (non-anticipative)
           curtail_da[t]       Solar curtailed  DA plan    (non-anticipative)
           soc_da[s,t]         SoC trajectory per scenario (scenario-specific)
  SoC bounds in LP: use 5% buffered planning bounds, NOT physical limits
  Key constraint: s_c_da + s_cd_da + curtail_da = z_sol_da  (every block)
  Opportunity cost: -ppa_rate * s_c_da forces LP to compare store vs captive

STAGE 2B (Captive reschedule — triggered FIRST at blocks 34, 42, 50, 58)
  Inputs : soc_actual, solar_nc blend, locked x_c/x_d, RTM q50
  Decides: s_c_rt[t..95], s_cd_rt[t..95], c_d_rt[t..95], curtail_rt[t..95]
  Outputs: revised captive_rt[t..95] = s_cd_rt + c_d_rt
  Must run BEFORE Stage 2A at reschedule blocks.

STAGE 2A (Block-by-block RTM dispatch — 96 times per day)
  At reschedule blocks: runs AFTER Stage 2B
  At all other blocks : runs alone
  Inputs : soc_actual[B], locked x_c/x_d, fixed s_c_rt[B], fixed c_d_rt[B],
           p_rtm_q50[B..95] adjusted by p_rtm_lag4
  Decides: y_c[B], y_d[B]  — RTM charge/discharge for block B ONLY
  s_c is NOT decided here (fixed from Stage 2B or DA plan before 2A runs).

SOC BUFFER (5%):
  e_max_plan = e_max_mwh * 0.95 = 4.75 * 0.95 = 4.5125 MWh
  e_min_plan = e_min_mwh * 1.05 = 0.50 * 1.05 = 0.5250 MWh
  All three LP functions use BUFFERED bounds.
  Physical limits (e_min/e_max) only used in settlement accounting.

IEX SETTLEMENT (per CERC IEX Regulation):
  DAM    : p_dam_actual * x_net * dt
  RTM    : p_rtm_actual * y_net * dt
  Captive: r_ppa * (s_cd_at + c_d_rt) * dt    (actual solar + BESS to captive)
=============================================================================
"""

import pulp
import numpy as np
from typing import Dict, List, Optional, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────
T_BLOCKS          = 96
DT                = 0.25   # hours per 15-min block
RESCHEDULE_BLOCKS = [34, 42, 50, 58]


def _plan_bounds(params) -> Tuple[float, float]:
    """Return (e_min_plan, e_max_plan) — buffered SoC planning bounds."""
    return params.e_min_plan_mwh, params.e_max_plan_mwh


def _failed_result() -> Dict:
    z = [0.0] * T_BLOCKS
    return {
        "status": "Infeasible", "expected_revenue": 0.0, "cvar_value_rs": None,
        "dam_schedule": z, "x_c": z, "x_d": z,
        "s_c_da": z, "s_cd_da": z, "c_d_da": z, "curtail_da": z,
        "captive_schedule_da": z, "scenarios": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — Non-Anticipative DAM + Solar Routing LP
# ─────────────────────────────────────────────────────────────────────────────

class TwoStageBESS:
    """
    Stage 1: Solar + BESS Two-Stage Stochastic LP.

    Non-anticipative variables (same value across all 100 price scenarios):
      x_c[t], x_d[t]   IEX DAM charge/discharge (MW)
      s_c_da[t]         Solar -> BESS (MW)
      s_cd_da[t]        Solar -> Captive (MW)
      c_d_da[t]         BESS  -> Captive (MW)
      curtail_da[t]     Solar curtailed (MW)

    Scenario-specific state:
      soc[s][t]         SoC per price scenario (MWh), uses BUFFERED bounds

    Key economic logic in objective:
      +r_ppa * s_cd_da  revenue for solar directly to captive
      +r_ppa * c_d_da   revenue for BESS to captive
      -r_ppa * s_c_da   OPPORTUNITY COST: solar diverted to BESS
      => LP stores solar only when expected future IEX price > PPA rate
    """

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
        dam_scenarios: np.ndarray,   # (S, 96) Rs/MWh
        rtm_scenarios: np.ndarray,   # (S, 96) Rs/MWh
        solar_da:      np.ndarray,   # (96,)   MW — DA solar generation forecast
    ) -> Dict:
        """
        Build and solve the Stage 1 stochastic LP.

        Returns
        -------
        dict with keys:
          status, expected_revenue, cvar_value_rs,
          dam_schedule (96,), x_c (96,), x_d (96,),
          s_c_da (96,), s_cd_da (96,), c_d_da (96,), curtail_da (96,),
          captive_schedule_da (96,),
          scenarios: list of {soc: list[97]}
        """
        S = dam_scenarios.shape[0]
        assert dam_scenarios.shape[1] == T_BLOCKS
        assert rtm_scenarios.shape    == dam_scenarios.shape
        assert len(solar_da)          == T_BLOCKS

        solar_da = np.clip(solar_da, 0.0, self.params.solar_capacity_mwp)
        r_ppa    = self.params.ppa_rate_rs_mwh
        e_min_p, e_max_p = _plan_bounds(self.params)

        prob = pulp.LpProblem("Stage1_SolarBESS", pulp.LpMaximize)

        # ── Non-anticipative Stage 1 decision variables ───────────────────────
        x_c      = pulp.LpVariable.dicts("x_c",   range(T_BLOCKS), 0, self.params.p_max_mw)
        x_d      = pulp.LpVariable.dicts("x_d",   range(T_BLOCKS), 0, self.params.p_max_mw)
        s_c_da   = pulp.LpVariable.dicts("sc_da", range(T_BLOCKS), 0, self.params.p_max_mw)
        s_cd_da  = pulp.LpVariable.dicts("scd",   range(T_BLOCKS), 0)
        c_d_da   = pulp.LpVariable.dicts("cd_da", range(T_BLOCKS), 0, self.params.p_max_mw)
        cu_da    = pulp.LpVariable.dicts("cu_da", range(T_BLOCKS), 0)

        # ── Scenario-specific SoC (BUFFERED bounds) ───────────────────────────
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

                # SoC dynamics:
                #   Charge from: solar-to-BESS + IEX-DAM-charge
                #   Discharge to: IEX-DAM-discharge + BESS-to-captive
                prob += soc[s][t + 1] == (
                    soc[s][t]
                    + self.params.eta_charge
                      * (s_c_da[t] + x_c[t]) * DT
                    - (1.0 / self.params.eta_discharge)
                      * (x_d[t] + c_d_da[t]) * DT
                )

                # Revenue: DAM settlement
                rev += p_dam * x_d[t] * DT
                rev -= p_dam * x_c[t] * DT

                # Revenue: Captive PPA
                rev += r_ppa * c_d_da[t]  * DT   # BESS -> Captive
                rev += r_ppa * s_cd_da[t] * DT   # Solar -> Captive (direct)

                # Opportunity cost: solar stored cannot earn PPA now
                rev -= r_ppa * s_c_da[t]  * DT

                # IEX costs (on IEX flows only — not captive)
                rev -= self.params.iex_fee_rs_mwh  * (x_c[t] + x_d[t]) * DT
                rev -= self.params.degradation_cost_rs_mwh * (x_d[t] + c_d_da[t]) * DT
                rev -= 135.0 * (x_c[t] + x_d[t]) * DT   # DSM friction proxy

            # Terminal SoC
            if self.params.soc_terminal_mode == "hard":
                prob += soc[s][T_BLOCKS] >= self.params.soc_terminal_min_mwh
            else:
                prob += soc[s][T_BLOCKS] >= e_min_p

            # Daily cycle cap
            if self.params.max_cycles_per_day is not None:
                usable = e_max_p - e_min_p
                prob += pulp.lpSum(
                    [(x_d[t] + c_d_da[t]) * DT for t in range(T_BLOCKS)]
                ) <= self.params.max_cycles_per_day * usable

            prob += u[s] >= zeta - rev
            scen_revs.append(rev)

        # ── Non-anticipative constraints (once, outside scenario loop) ─────────
        for t in range(T_BLOCKS):
            # C1: Solar balance (exact equality every block)
            prob += (
                s_c_da[t] + s_cd_da[t] + cu_da[t] == float(solar_da[t]),
                f"solar_bal_{t}"
            )
            # C2: Total charge power limit
            prob += s_c_da[t] + x_c[t] <= self.params.p_max_mw, f"ch_lim_{t}"
            # C3: Total discharge power limit
            prob += x_d[t] + c_d_da[t] <= self.params.p_max_mw, f"dis_lim_{t}"

        # ── Objective ─────────────────────────────────────────────────────────
        avg_rev = pulp.lpSum(scen_revs) / S
        cvar    = zeta - (1.0 / (S * self.risk_alpha)) * pulp.lpSum(
            [u[s] for s in range(S)]
        )
        term_val = 0
        if (self.params.soc_terminal_mode == "soft"
                and self.params.soc_terminal_value_rs_mwh > 0):
            term_val = pulp.lpSum([
                self.params.soc_terminal_value_rs_mwh * soc[s][T_BLOCKS]
                for s in range(S)
            ]) / S

        prob.setObjective(avg_rev + term_val + self.lambda_risk * cvar)

        # ── Solve ─────────────────────────────────────────────────────────────
        solver = pulp.PULP_CBC_CMD(msg=0)
        if self.solver_name.upper() == "HIGHS":
            try:
                solver = pulp.HiGHS_CMD(msg=0)
            except Exception:
                pass
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


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2B — Captive Reschedule LP
# Triggered FIRST at blocks 34, 42, 50, 58 — BEFORE Stage 2A
# ─────────────────────────────────────────────────────────────────────────────

def reschedule_captive(
    params,
    trigger_block:  int,           # One of 34, 42, 50, 58
    soc_actual:     float,         # Measured SoC at start of trigger block (MWh)
    solar_nc_row:   np.ndarray,    # (12,) NC nowcast MW for trigger..trigger+11
    solar_da:       np.ndarray,    # (96,) full DA solar forecast MW
    rtm_q50:        np.ndarray,    # (96,) RTM q50 forecast Rs/MWh
    x_c_stage1:     np.ndarray,    # (96,) locked IEX charge from Stage 1 (MW)
    x_d_stage1:     np.ndarray,    # (96,) locked IEX discharge from Stage 1 (MW)
) -> Dict:
    """
    Stage 2B: Captive reschedule LP.

    Decides for blocks trigger_block..95:
      s_c_rt[t]     Solar -> BESS (revised)
      s_cd_rt[t]    Solar -> Captive (revised)
      c_d_rt[t]     BESS  -> Captive (revised)
      curtail_rt[t] Curtailed solar (from balance)

    IEX schedule (x_c, x_d) from Stage 1 is LOCKED — treated as constants.
    RTM q50 used to value remaining BESS energy for planning.
    SoC uses BUFFERED planning bounds.

    Solar forecast blend:
      blocks trigger_block .. trigger_block+11 : NC nowcast (12 blocks)
      blocks trigger_block+12 .. 95           : DA forecast

    Returns full (96,) arrays — zeros before trigger_block.
    """
    assert trigger_block in RESCHEDULE_BLOCKS, \
        f"trigger_block must be in {RESCHEDULE_BLOCKS}, got {trigger_block}"

    r_ppa     = params.ppa_rate_rs_mwh
    remaining = T_BLOCKS - trigger_block
    e_min_p, e_max_p = _plan_bounds(params)

    # Build solar forecast for remaining blocks
    solar_rem = np.empty(remaining)
    for k in range(remaining):
        if k < 12:
            solar_rem[k] = float(solar_nc_row[k])
        else:
            solar_rem[k] = float(solar_da[trigger_block + k])
    solar_rem = np.clip(solar_rem, 0.0, params.solar_capacity_mwp)

    xc_rem = x_c_stage1[trigger_block:].astype(float)
    xd_rem = x_d_stage1[trigger_block:].astype(float)
    rtm_rem = rtm_q50[trigger_block:].astype(float)

    prob = pulp.LpProblem(f"Stage2B_b{trigger_block}", pulp.LpMaximize)

    # Decision variables for remaining blocks
    s_c_lp  = pulp.LpVariable.dicts("sc",  range(remaining), 0, params.p_max_mw)
    s_cd_lp = pulp.LpVariable.dicts("scd", range(remaining), 0)
    c_d_lp  = pulp.LpVariable.dicts("cd",  range(remaining), 0, params.p_max_mw)
    cu_lp   = pulp.LpVariable.dicts("cu",  range(remaining), 0)

    soc = pulp.LpVariable.dicts("soc", range(remaining + 1), e_min_p, e_max_p)
    prob += soc[0] == float(np.clip(soc_actual, e_min_p, e_max_p))

    rev = 0
    for k in range(remaining):
        xc_k = xc_rem[k]
        xd_k = xd_rem[k]
        p_rtm_k = rtm_rem[k]

        # SoC dynamics
        prob += soc[k + 1] == (
            soc[k]
            + params.eta_charge * (s_c_lp[k] + xc_k) * DT
            - (1.0 / params.eta_discharge) * (c_d_lp[k] + xd_k) * DT
        )

        # Solar balance constraint
        prob += (
            s_c_lp[k] + s_cd_lp[k] + cu_lp[k] == float(solar_rem[k]),
            f"sol_bal_{k}"
        )
        # Power limits (locked IEX flows count toward limit)
        prob += s_c_lp[k] + xc_k <= params.p_max_mw, f"ch_{k}"
        prob += c_d_lp[k] + xd_k <= params.p_max_mw, f"dis_{k}"

        # Revenue
        rev += r_ppa * s_cd_lp[k] * DT    # Solar -> Captive
        rev += r_ppa * c_d_lp[k]  * DT    # BESS  -> Captive
        rev -= r_ppa * s_c_lp[k]  * DT    # Opportunity cost

        # Locked IEX revenue (planning signal for SoC management)
        rev += p_rtm_k * xd_k * DT
        rev -= p_rtm_k * xc_k * DT
        rev -= params.iex_fee_rs_mwh * (xc_k + xd_k) * DT
        rev -= params.degradation_cost_rs_mwh * (c_d_lp[k] + xd_k) * DT
        rev -= 135.0 * (xc_k + xd_k) * DT

    prob += soc[remaining] >= e_min_p
    prob.setObjective(rev)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[prob.status]

    sc_out  = np.zeros(T_BLOCKS)
    scd_out = np.zeros(T_BLOCKS)
    cd_out  = np.zeros(T_BLOCKS)
    cu_out  = np.zeros(T_BLOCKS)

    if status == "Optimal":
        for k in range(remaining):
            t = trigger_block + k
            sc_out[t]  = max(0.0, pulp.value(s_c_lp[k])  or 0.0)
            scd_out[t] = max(0.0, pulp.value(s_cd_lp[k]) or 0.0)
            cd_out[t]  = max(0.0, pulp.value(c_d_lp[k])  or 0.0)
            cu_out[t]  = max(0.0, pulp.value(cu_lp[k])   or 0.0)

    return {
        "status":     status,
        "s_c_rt":     sc_out,
        "s_cd_rt":    scd_out,
        "c_d_rt":     cd_out,
        "curtail_rt": cu_out,
        "captive_rt": scd_out + cd_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2A — Block-by-Block RTM Dispatch
# Outputs y_c[B] and y_d[B] ONLY. Solar routing is FIXED before this runs.
# ─────────────────────────────────────────────────────────────────────────────

def solve_stage2a_block(
    params,
    block_B:         int,           # Current block 0..95
    soc_actual_B:    float,         # Measured SoC at start of block B (MWh)
    dam_schedule:    np.ndarray,    # (96,) x_net = x_d - x_c from Stage 1 (MW)
    dam_actual:      np.ndarray,    # (96,) actual DAM MCP Rs/MWh (fully known)
    p_rtm_lag4:      float,         # p_rtm[B-4] actual — latest known (NaN if B<4)
    rtm_q50:         np.ndarray,    # (96,) RTM q50 forecast Rs/MWh
    s_c_rt_B:        float,         # Fixed solar->BESS for block B (MW)
    c_d_rt_B:        float,         # Fixed captive BESS discharge at block B (MW)
    verbose:         bool = False,
) -> Tuple[float, float]:
    """
    Stage 2A: Solve LP for block B to get y_c[B] and y_d[B] ONLY.

    Solar routing s_c_rt[B] is FIXED (from Stage 2B or DA plan).
    Stage 2A does NOT change it.

    RTM price information at gate-close for block B:
      - p_rtm[t < B-3]  : actual (published) — already committed
      - p_rtm[B-4]      : latest known actual (4-block = 1hr lag) — p_rtm_lag4
                          Used to condition the q50 forecast for planning.
      - p_rtm[B..95]    : unknown → use q50 forecast (adjusted by lag-4 signal)

    Forecast conditioning: if p_rtm_lag4 is available, shift q50 upward/downward
    by the lag-4 bias with exponential decay (more weight on near blocks).

    SoC bounds use BUFFERED planning range. If soc_actual_B is near the
    physical ceiling (e.g. high solar day), charge headroom is reduced to 0
    and y_c = 0 automatically — no DSM charge-failure risk.

    Returns
    -------
    (y_c_B, y_d_B) : scalars in MW — committed for block B only.
    """
    e_min_p, e_max_p = _plan_bounds(params)
    remaining        = T_BLOCKS - block_B

    # Condition q50 forecast by lag-4 known actual
    rtm_lp = rtm_q50[block_B:].copy().astype(float)
    if not np.isnan(p_rtm_lag4) and block_B >= 4:
        q50_at_lag4 = float(rtm_q50[block_B - 4])
        if q50_at_lag4 > 0:
            bias  = p_rtm_lag4 - q50_at_lag4
            decay = np.array([0.85 ** k for k in range(remaining)])
            rtm_lp = np.maximum(0.0, rtm_lp + bias * decay)

    # Clamp soc to buffered range for LP initialisation
    soc_init = float(np.clip(soc_actual_B, e_min_p, e_max_p))

    # Locked Stage 1 flows for this block
    x_net_B = float(dam_schedule[block_B])
    x_c_B   = max(0.0, -x_net_B)
    x_d_B   = max(0.0,  x_net_B)

    # RTM headroom after fixed flows
    charge_hdroom    = max(0.0, params.p_max_mw - s_c_rt_B - x_c_B)
    discharge_hdroom = max(0.0, params.p_max_mw - c_d_rt_B - x_d_B)

    # SoC-based headroom (prevents hitting buffered bounds)
    soc_charge_cap    = max(0.0, (e_max_p - soc_actual_B) / (params.eta_charge * DT))
    soc_discharge_cap = max(0.0, (soc_actual_B - e_min_p) * params.eta_discharge / DT)
    charge_hdroom     = min(charge_hdroom,    soc_charge_cap)
    discharge_hdroom  = min(discharge_hdroom, soc_discharge_cap)

    # If no headroom at all — skip LP
    if charge_hdroom < 1e-4 and discharge_hdroom < 1e-4:
        if verbose:
            print(f"  [2A] B{block_B:03d} | No headroom → y_c=y_d=0")
        return 0.0, 0.0

    prob = pulp.LpProblem(f"Stage2A_B{block_B}", pulp.LpMaximize)

    y_c_lp = pulp.LpVariable.dicts("yc", range(remaining), 0, charge_hdroom)
    y_d_lp = pulp.LpVariable.dicts("yd", range(remaining), 0, discharge_hdroom)
    soc    = pulp.LpVariable.dicts("s",  range(remaining + 1), e_min_p, e_max_p)

    prob += soc[0] == soc_init
    rev = 0

    for k in range(remaining):
        t_abs  = block_B + k
        p_rtm_k = float(rtm_lp[k])
        x_net_t = float(dam_schedule[t_abs])
        xc_t    = max(0.0, -x_net_t)
        xd_t    = max(0.0,  x_net_t)

        # For block B (k=0): include fixed solar and captive in SoC dynamics
        # For future blocks (k>0): approximate as zero (Stage 2B handles them)
        sc_t  = s_c_rt_B  if k == 0 else 0.0
        cap_t = c_d_rt_B  if k == 0 else 0.0

        prob += soc[k + 1] == (
            soc[k]
            + params.eta_charge * (sc_t + xc_t + y_c_lp[k]) * DT
            - (1.0 / params.eta_discharge) * (xd_t + cap_t + y_d_lp[k]) * DT
        )

        # RTM revenue
        rev += p_rtm_k * y_d_lp[k] * DT
        rev -= p_rtm_k * y_c_lp[k] * DT
        rev -= params.iex_fee_rs_mwh * (y_c_lp[k] + y_d_lp[k]) * DT
        rev -= params.degradation_cost_rs_mwh * y_d_lp[k] * DT
        rev -= 135.0 * (y_c_lp[k] + y_d_lp[k]) * DT

    prob += soc[remaining] >= e_min_p
    prob.setObjective(rev)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] == "Optimal":
        y_c_B = max(0.0, pulp.value(y_c_lp[0]) or 0.0)
        y_d_B = max(0.0, pulp.value(y_d_lp[0]) or 0.0)
    else:
        y_c_B, y_d_B = 0.0, 0.0

    if verbose:
        print(f"  [2A] B{block_B:03d} | SoC {soc_actual_B:.3f}→ "
              f"hdCh={charge_hdroom:.2f} hdDis={discharge_hdroom:.2f} "
              f"| y_c={y_c_B:.3f} y_d={y_d_B:.3f} MW")

    return y_c_B, y_d_B


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION LOOP — orchestrates Stage 2B then 2A each block
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_actuals_solar(
    params,
    stage1_result:   Dict,          # Output of TwoStageBESS.solve()
    dam_actual:      np.ndarray,    # (96,) actual DAM MCP Rs/MWh
    rtm_actual:      np.ndarray,    # (96,) actual RTM MCP Rs/MWh (settlement)
    rtm_q50:         np.ndarray,    # (96,) RTM q50 forecast Rs/MWh (decisions)
    solar_da:        np.ndarray,    # (96,) DA solar forecast MW
    solar_nc:        np.ndarray,    # (96, 12) NC nowcast — row B = B..B+11 MW
    solar_at:        np.ndarray,    # (96,) actual metered solar MW (settlement)
    reschedule_blocks: List[int] = RESCHEDULE_BLOCKS,
    verbose:         bool = False,
) -> Dict:
    """
    Full-day settlement: Stage 2B first at reschedule blocks, then Stage 2A.

    EXECUTION EACH BLOCK B:
      if B in reschedule_blocks:
        1. Run Stage 2B → revise s_c_rt, s_cd_rt, c_d_rt for B..95
        2. Run Stage 2A → decide y_c[B], y_d[B] using updated routing
      else:
        1. Run Stage 2A only → decide y_c[B], y_d[B]

    SOLAR ROUTING between reschedule blocks:
      Fixed at values from last Stage 2B run (or DA plan before first reschedule).

    SETTLEMENT each block:
      Uses ACTUAL DAM/RTM prices and ACTUAL solar — not forecasts.
      DSM charge-failure penalty computed if physical SoC ceiling breached.

    Returns
    -------
    dict with: revenue, net_revenue, y_c (96,), y_d (96,),
               s_c_rt (96,), s_cd_rt (96,), c_d_rt (96,),
               soc_path (97,), block_rev_dam (96,), block_rev_rtm (96,),
               block_rev_captive (96,), block_costs (96,),
               block_dsm_energy (96,), total_dsm_mwh
    """
    r_ppa = params.ppa_rate_rs_mwh

    x_c_s1  = np.array(stage1_result["x_c"])
    x_d_s1  = np.array(stage1_result["x_d"])
    sc_da   = np.array(stage1_result["s_c_da"])
    scd_da  = np.array(stage1_result["s_cd_da"])
    cd_da   = np.array(stage1_result["c_d_da"])
    dam_sch = np.array(stage1_result["dam_schedule"])

    # RT solar routing — initialised from DA plan, overwritten by Stage 2B
    s_c_rt  = sc_da.copy()
    s_cd_rt = scd_da.copy()
    c_d_rt  = cd_da.copy()

    y_c_all  = np.zeros(T_BLOCKS)
    y_d_all  = np.zeros(T_BLOCKS)
    soc_path = np.zeros(T_BLOCKS + 1)
    soc_path[0] = params.soc_initial_mwh

    rev_dam     = np.zeros(T_BLOCKS)
    rev_rtm     = np.zeros(T_BLOCKS)
    rev_cap     = np.zeros(T_BLOCKS)
    costs       = np.zeros(T_BLOCKS)
    dsm_energy  = np.zeros(T_BLOCKS)

    for B in range(T_BLOCKS):

        # Latest known actual RTM price (4-block = 1hr lag)
        p_rtm_lag4 = float(rtm_actual[B - 4]) if B >= 4 else np.nan

        # ── Stage 2B: runs FIRST at reschedule blocks ─────────────────────────
        if B in reschedule_blocks:
            if verbose:
                print(f"\n[B{B:03d}] Stage 2B reschedule | SoC={soc_path[B]:.3f}")
            res2b = reschedule_captive(
                params        = params,
                trigger_block = B,
                soc_actual    = soc_path[B],
                solar_nc_row  = solar_nc[B],    # (12,) nowcast from row B
                solar_da      = solar_da,
                rtm_q50       = rtm_q50,
                x_c_stage1    = x_c_s1,
                x_d_stage1    = x_d_s1,
            )
            if res2b["status"] == "Optimal":
                s_c_rt[B:]  = res2b["s_c_rt"][B:]
                s_cd_rt[B:] = res2b["s_cd_rt"][B:]
                c_d_rt[B:]  = res2b["c_d_rt"][B:]
                if verbose:
                    print(f"  Stage 2B OK — routing revised for B..95")
            else:
                if verbose:
                    print(f"  Stage 2B FAILED ({res2b['status']}) — keeping prior plan")

        # ── Stage 2A: decide y_c[B], y_d[B] ─────────────────────────────────
        y_c_B, y_d_B = solve_stage2a_block(
            params       = params,
            block_B      = B,
            soc_actual_B = soc_path[B],
            dam_schedule = dam_sch,
            dam_actual   = dam_actual,
            p_rtm_lag4   = p_rtm_lag4,
            rtm_q50      = rtm_q50,
            s_c_rt_B     = s_c_rt[B],
            c_d_rt_B     = c_d_rt[B],
            verbose      = verbose,
        )
        y_c_all[B] = y_c_B
        y_d_all[B] = y_d_B

        # ── Settlement at ACTUAL prices and ACTUAL solar ──────────────────────

        # DAM leg
        x_net_B   = float(dam_sch[B])
        rev_dam[B] = float(dam_actual[B]) * x_net_B * DT

        # RTM leg (settled at ACTUAL price, NOT q50 forecast)
        y_net_B   = y_d_B - y_c_B
        rev_rtm[B] = float(rtm_actual[B]) * y_net_B * DT

        # Captive leg (actual solar)
        solar_after_bess = max(0.0, float(solar_at[B]) - s_c_rt[B])
        s_cd_at_B = min(solar_after_bess, s_cd_rt[B])   # actual solar to captive
        c_d_at_B  = c_d_rt[B]                            # BESS to captive
        rev_cap[B] = r_ppa * (s_cd_at_B + c_d_at_B) * DT

        # Costs
        xc_B = max(0.0, -x_net_B)
        xd_B = max(0.0,  x_net_B)
        iex_cost  = params.iex_fee_rs_mwh * (xc_B + xd_B + y_c_B + y_d_B) * DT
        deg_cost  = params.degradation_cost_rs_mwh * (xd_B + c_d_at_B + y_d_B) * DT
        dsm_proxy = 135.0 * (xc_B + xd_B + y_c_B + y_d_B) * DT

        # DSM charge-failure: physical ceiling breached despite buffer
        total_charge_energy = params.eta_charge * (s_c_rt[B] + xc_B + y_c_B) * DT
        projected_soc = soc_path[B] + total_charge_energy
        if projected_soc > params.e_max_mwh:
            rejected_mwh = projected_soc - params.e_max_mwh
            dsm_energy[B] = rejected_mwh
            dsm_proxy += rejected_mwh * float(dam_actual[B]) * 1.25

        costs[B] = iex_cost + deg_cost + dsm_proxy

        # ── Actual SoC update ─────────────────────────────────────────────────
        act_charge    = params.eta_charge * (s_c_rt[B] + xc_B + y_c_B) * DT
        act_discharge = (xd_B + c_d_rt[B] + y_d_B) / params.eta_discharge * DT
        soc_path[B + 1] = float(np.clip(
            soc_path[B] + act_charge - act_discharge,
            params.e_min_mwh, params.e_max_mwh
        ))

    gross = float(np.sum(rev_dam + rev_rtm + rev_cap))
    total_cost = float(np.sum(costs))

    return {
        "revenue":           gross,
        "net_revenue":       gross - total_cost,
        "y_c":               y_c_all,
        "y_d":               y_d_all,
        "s_c_rt":            s_c_rt,
        "s_cd_rt":           s_cd_rt,
        "c_d_rt":            c_d_rt,
        "soc_path":          soc_path,
        "soc":               soc_path.tolist(),       # compatibility alias
        "rtm_schedule":      (y_d_all - y_c_all).tolist(),  # compatibility alias
        "block_rev_dam":     rev_dam,
        "block_rev_rtm":     rev_rtm,
        "block_rev_captive": rev_cap,
        "block_costs":       costs,
        "block_dsm_energy":  dsm_energy,
        "total_dsm_mwh":     float(np.sum(dsm_energy)),
        "fees_breakdown":    {
            "iex_revenue_dam":     float(np.sum(rev_dam)),
            "iex_revenue_rtm":     float(np.sum(rev_rtm)),
            "captive_revenue":     float(np.sum(rev_cap)),
            "total_costs":         total_cost,
        },
    }
