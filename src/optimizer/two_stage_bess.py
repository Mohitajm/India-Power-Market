"""
src/optimizer/two_stage_bess.py
================================
Two-Stage Stochastic BESS Optimizer — 96-block (15-min) version.

Key differences from the 24-hour version:
  T = 96 blocks per day  (each block = 15 minutes)
  dt = 0.25 hours        (energy = power × 0.25 h per block)
  All MW × dt = MWh conversions use dt = 0.25

IEX Settlement (corrected — per IEX Regulation):
  Revenue = p_dam * x_net  +  p_rtm * y_net
  DAM settles on committed schedule x at DAM price.
  RTM settles on full physical dispatch y at RTM price.
  (NOT the old deviation formula p_rtm * (y_net - x_net))

SoC dynamics at 15-min resolution:
  SoC[t+1] = SoC[t] + eta_c * y_c[t] * dt  -  (y_d[t] / eta_d) * dt
  where dt = 0.25 h, y_c and y_d are in MW, SoC in MWh.

Cost rates applied per block (scaled by dt where needed):
  IEX fee    : Rs/MWh on physical throughput  → multiply by dt to get Rs/block
  Degradation: Rs/MWh on discharge throughput → multiply by dt
  DSM proxy  : Rs/MWh throughput              → multiply by dt
"""

import pulp
import numpy as np
from typing import Dict, List, Optional
from src.optimizer.bess_params import BESSParams

# Blocks per day and time step
T_BLOCKS = 96        # 96 × 15-min blocks per day
DT       = 0.25      # each block = 0.25 hours


class TwoStageBESS:
    """
    Two-Stage Stochastic BESS Optimizer — 96-block (15-min) version.

    Stage 1: DAM commitment x_c[b], x_d[b] for b in [0, 95] — non-anticipative.
    Stage 2: RTM recourse y_c[s][b], y_d[s][b] for b in [0, 95] — scenario-specific.
    SoC    : soc[s][b] for b in [0, 96] — 97 points per scenario.
    """

    def __init__(self, params: BESSParams, config: Dict):
        self.params      = params
        self.config      = config
        self.solver_name = config.get('solver', 'CBC')
        self.lambda_risk = config.get('lambda_risk', 0.0)
        self.lambda_dev  = config.get('lambda_dev', 0.0)
        self.dev_max     = config.get('dev_max_mw', 50.0)
        self.risk_alpha  = config.get('risk_alpha', 0.1)

    def solve(self, dam_scenarios: np.ndarray, rtm_scenarios: np.ndarray) -> Dict:
        """
        Build and solve the two-stage stochastic LP.

        Parameters
        ----------
        dam_scenarios : np.ndarray  shape (n_scenarios, 96)
            DAM price scenarios in Rs/MWh for each 15-min block.
        rtm_scenarios : np.ndarray  shape (n_scenarios, 96)
            RTM price scenarios in Rs/MWh for each 15-min block.

        Returns
        -------
        dict with keys:
            status, expected_revenue, cvar_zeta_rs, cvar_value_rs,
            dam_schedule (list of 96 net MW values),
            scenarios (list of per-scenario dicts)
        """
        n_scenarios = dam_scenarios.shape[0]
        assert dam_scenarios.shape[1] == T_BLOCKS, \
            f"Expected {T_BLOCKS} blocks, got {dam_scenarios.shape[1]}"

        prob = pulp.LpProblem("TwoStageBESS_96block", pulp.LpMaximize)

        # ── Stage 1: DAM variables (shared across all scenarios) ──────────────
        x_c = pulp.LpVariable.dicts("x_c", range(T_BLOCKS), 0, self.params.p_max_mw)
        x_d = pulp.LpVariable.dicts("x_d", range(T_BLOCKS), 0, self.params.p_max_mw)

        # ── Stage 2: RTM + SoC variables (per scenario) ──────────────────────
        y_c  = pulp.LpVariable.dicts("y_c",  (range(n_scenarios), range(T_BLOCKS)),
                                     0, self.params.p_max_mw)
        y_d  = pulp.LpVariable.dicts("y_d",  (range(n_scenarios), range(T_BLOCKS)),
                                     0, self.params.p_max_mw)
        soc  = pulp.LpVariable.dicts("soc",  (range(n_scenarios), range(T_BLOCKS + 1)),
                                     self.params.e_min_mwh, self.params.e_max_mwh)

        # Deviation auxiliaries (for stability penalty)
        dev_pos = pulp.LpVariable.dicts("dev_pos",
                                        (range(n_scenarios), range(T_BLOCKS)), lowBound=0)
        dev_neg = pulp.LpVariable.dicts("dev_neg",
                                        (range(n_scenarios), range(T_BLOCKS)), lowBound=0)

        # CVaR variables (Rockafellar-Uryasev linearisation)
        zeta = pulp.LpVariable("zeta")
        u    = pulp.LpVariable.dicts("u", range(n_scenarios), lowBound=0)

        # ── Build scenario revenues ───────────────────────────────────────────
        scenario_revenues_expr = []

        for s in range(n_scenarios):
            # Initial SoC
            prob += soc[s][0] == self.params.soc_initial_mwh

            rev = 0

            for b in range(T_BLOCKS):
                p_dam = dam_scenarios[s, b]   # Rs/MWh
                p_rtm = rtm_scenarios[s, b]   # Rs/MWh

                # ── SoC dynamics (energy = MW × DT hours) ────────────────────
                # SoC[b+1] = SoC[b] + eta_c * y_c[b] * DT - y_d[b] * DT / eta_d
                prob += soc[s][b + 1] == (
                    soc[s][b]
                    + self.params.eta_charge    * y_c[s][b] * DT
                    - (1.0 / self.params.eta_discharge) * y_d[s][b] * DT
                )

                # ── Revenue: IEX independent gross settlement ─────────────────
                # DAM leg: p_dam × x_net  (committed schedule)
                # RTM leg: p_rtm × y_net  (full physical dispatch)
                # Energy per block = MW × DT hours
                x_net_b = x_d[b] - x_c[b]
                y_net_b = y_d[s][b] - y_c[s][b]

                rev += p_dam * x_net_b * DT
                rev += p_rtm * y_net_b * DT

                # ── Costs (per block, scaled by DT where Rs/MWh) ─────────────
                # IEX transaction fee: Rs/MWh on physical throughput
                rev -= self.params.iex_fee_rs_mwh * (y_c[s][b] + y_d[s][b]) * DT

                # Battery degradation: Rs/MWh on discharge energy
                rev -= self.params.degradation_cost_rs_mwh * y_d[s][b] * DT

                # DSM friction proxy: Rs/MWh throughput
                rev -= 135.0 * (y_c[s][b] + y_d[s][b]) * DT

                # ── Deviation constraint ──────────────────────────────────────
                prob += (y_net_b - x_net_b == dev_pos[s][b] - dev_neg[s][b])
                prob += dev_pos[s][b] + dev_neg[s][b] <= self.dev_max

            # Terminal SoC constraint
            if self.params.soc_terminal_mode == "hard":
                prob += soc[s][T_BLOCKS] >= self.params.soc_terminal_min_mwh

            # Optional cycling limit
            if self.params.max_cycles_per_day is not None:
                usable = self.params.e_max_mwh - self.params.e_min_mwh
                total_discharge_energy = pulp.lpSum(
                    [y_d[s][b] * DT for b in range(T_BLOCKS)]
                )
                prob += total_discharge_energy <= self.params.max_cycles_per_day * usable

            # CVaR shortfall
            prob += u[s] >= zeta - rev

            scenario_revenues_expr.append(rev)

        # ── Objective ─────────────────────────────────────────────────────────
        avg_revenue = pulp.lpSum(scenario_revenues_expr) / n_scenarios

        cvar_expr = zeta - (1.0 / (n_scenarios * self.risk_alpha)) * pulp.lpSum(
            [u[s] for s in range(n_scenarios)]
        )

        stability_penalty = (self.lambda_dev / n_scenarios) * pulp.lpSum([
            dev_pos[s][b] + dev_neg[s][b]
            for s in range(n_scenarios)
            for b in range(T_BLOCKS)
        ])

        # Soft terminal continuation value
        terminal_value_expr = 0
        if (self.params.soc_terminal_mode == "soft"
                and self.params.soc_terminal_value_rs_mwh > 0):
            terminal_value_expr = pulp.lpSum([
                self.params.soc_terminal_value_rs_mwh * soc[s][T_BLOCKS]
                for s in range(n_scenarios)
            ]) / n_scenarios

        prob.setObjective(
            avg_revenue + terminal_value_expr
            + self.lambda_risk * cvar_expr
            - stability_penalty
        )

        # ── Solve ─────────────────────────────────────────────────────────────
        solver = pulp.PULP_CBC_CMD(msg=0)
        if self.solver_name == 'HiGHS':
            try:
                solver = pulp.HiGHS_CMD(msg=0)
                print("Using HiGHS solver...")
            except Exception as e:
                print(f"HiGHS failed ({e}), falling back to CBC.")
                solver = pulp.PULP_CBC_CMD(msg=0)
        else:
            print("Using CBC solver...")

        status = prob.solve(solver)

        if pulp.LpStatus[status] != 'Optimal':
            return {"status": pulp.LpStatus[status]}

        # ── Extract results ───────────────────────────────────────────────────
        scen_rev_vals = [pulp.value(expr) for expr in scenario_revenues_expr]

        results = {
            "status":           "Optimal",
            "expected_revenue": pulp.value(avg_revenue),
            "cvar_zeta_rs":     pulp.value(zeta),
            "cvar_value_rs":    pulp.value(cvar_expr),
            # DAM schedule: 96 net MW values (positive = discharge, negative = charge)
            "dam_schedule":     [pulp.value(x_d[b] - x_c[b]) for b in range(T_BLOCKS)],
            "scenarios":        [],
        }

        for s in range(n_scenarios):
            results["scenarios"].append({
                "id":           s,
                "rtm_dispatch": [pulp.value(y_d[s][b] - y_c[s][b])
                                 for b in range(T_BLOCKS)],
                "soc":          [pulp.value(soc[s][b])
                                 for b in range(T_BLOCKS + 1)],
                "revenue":      scen_rev_vals[s],
            })

        return results
