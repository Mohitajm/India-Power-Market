"""
src/optimizer/costs.py
=======================
Pluggable trading cost model for Solar + BESS operations.

New vs. BESS-only version:
  compute_costs() now accepts captive and solar flows to compute:
    - Captive PPA revenue (positive — BESS and solar to captive)
    - Solar opportunity cost is handled in the LP objective, not here
    - DSM charge-failure penalty (when physical SoC ceiling is breached)
"""

import numpy as np
import yaml


class CostModel:
    """Pluggable trading cost model for Solar + BESS operations."""

    def __init__(self, config: dict):
        self.config   = config.get("costs", {})
        self.iex_cfg  = self.config.get("iex_transaction_fee", {})
        self.sched_cfg = self.config.get("scheduling_charges", {})
        self.deg_cfg  = self.config.get("degradation", {})
        self.ists_cfg = self.config.get("ists_charges", {})
        self.dsm_cfg  = self.config.get("dsm_penalties", {})
        self.oa_cfg   = self.config.get("open_access", {})
        self.ppa_cfg  = self.config.get("captive_ppa", {})

    @classmethod
    def from_yaml(cls, path: str) -> "CostModel":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(data)

    def compute_costs(
        self,
        charge:           np.ndarray,              # (96,) MW — IEX charge per block
        discharge:        np.ndarray,              # (96,) MW — IEX discharge per block
        dam_actual:       np.ndarray = None,        # (96,) Rs/MWh — for DSM NR
        rtm_actual:       np.ndarray = None,        # (96,) Rs/MWh — for DSM NR
        # ── New Solar+BESS parameters ──────────────────────────────────────────
        captive_bess:     np.ndarray = None,        # (96,) MW — BESS -> Captive
        captive_solar:    np.ndarray = None,        # (96,) MW — Solar -> Captive (actual)
        dsm_energy_mwh:   np.ndarray = None,        # (96,) MWh — charge-failure energy
        dt:               float       = 0.25,        # hours per block
    ) -> dict:
        """
        Compute all trading costs and captive PPA revenue.

        Parameters
        ----------
        charge         : (96,) MW — IEX RTM + DAM charge per block
        discharge      : (96,) MW — IEX RTM + DAM discharge per block
        dam_actual     : (96,) Rs/MWh — actual DAM MCP for DSM NR
        rtm_actual     : (96,) Rs/MWh — actual RTM MCP for DSM NR
        captive_bess   : (96,) MW — BESS discharged to captive (c_d_rt)
        captive_solar  : (96,) MW — Solar delivered to captive (s_cd_at)
        dsm_energy_mwh : (96,) MWh — charge-failure rejected energy per block
        dt             : float — hours per block (0.25 for 15-min)

        Returns
        -------
        dict with all cost/revenue components and total_costs
        """
        n = len(charge)
        energy_charged    = float(np.sum(charge)    * dt)   # MWh
        energy_discharged = float(np.sum(discharge) * dt)   # MWh
        total_throughput  = energy_charged + energy_discharged

        # 1. IEX Transaction Fee
        iex_fee = 0.0
        if self.iex_cfg.get("enabled"):
            iex_fee = total_throughput * self.iex_cfg.get("fee_per_mwh_per_side", 0)

        # 2. Scheduling Charges
        scheduling = 0.0
        if self.sched_cfg.get("enabled"):
            sldc = self.sched_cfg.get("sldc_per_day", 0)
            rldc = total_throughput * self.sched_cfg.get("rldc_per_mwh", 0)
            scheduling = sldc + rldc

        # 3. Battery Degradation (on ALL discharge paths including captive)
        degradation = 0.0
        if self.deg_cfg.get("enabled"):
            total_discharge_mwh = energy_discharged
            if captive_bess is not None:
                total_discharge_mwh += float(np.sum(captive_bess) * dt)
            degradation = total_discharge_mwh * self.deg_cfg.get("cost_per_mwh_throughput", 0)

        # 4. ISTS Charges
        ists = 0.0
        if self.ists_cfg.get("enabled") and not self.ists_cfg.get("waiver"):
            ists = total_throughput * self.ists_cfg.get("charge_per_mwh", 0)

        # 5. DSM Penalties
        dsm_penalty = 0.0
        if self.dsm_cfg.get("enabled"):
            if (self.dsm_cfg.get("mode") == "block_wise_nr"
                    and dam_actual is not None and rtm_actual is not None):
                as_cost  = self.dsm_cfg.get("estimated_as_cost_rs_mwh", 5000)
                nr       = (dam_actual + rtm_actual + as_cost) / 3.0
                nr_cap   = np.minimum(nr, self.dsm_cfg.get("nr_ceiling_rs_mwh", 8000))
                err_pct  = self.dsm_cfg.get("physical_error_pct", 3.0) / 100.0
                block_tp = (charge + discharge) * dt
                dsm_penalty = float(np.sum(block_tp * err_pct * nr_cap))
            else:
                fb_rate  = self.dsm_cfg.get("fallback_nr_rs_mwh", 4500)
                err_pct  = self.dsm_cfg.get("physical_error_pct", 3.0) / 100.0
                dsm_penalty = total_throughput * err_pct * fb_rate

        # DSM charge-failure (physical SoC ceiling breach despite buffer)
        dsm_charge_fail = 0.0
        if dsm_energy_mwh is not None and dam_actual is not None:
            mult = self.dsm_cfg.get("charge_failure_multiplier", 1.25)
            dsm_charge_fail = float(
                np.sum(dsm_energy_mwh * dam_actual * mult)
            )

        # 6. Open Access
        oa = 0.0
        if self.oa_cfg.get("enabled"):
            oa = energy_charged * (
                self.oa_cfg.get("cross_subsidy_surcharge_per_mwh", 0)
                + self.oa_cfg.get("additional_surcharge_per_mwh", 0)
            )

        # 7. Captive PPA Revenue (positive — reduces net cost)
        captive_revenue = 0.0
        if self.ppa_cfg.get("enabled"):
            ppa_rate = self.ppa_cfg.get("rate_rs_mwh", 3500.0)
            bess_to_captive_mwh  = float(np.sum(captive_bess)  * dt) if captive_bess  is not None else 0.0
            solar_to_captive_mwh = float(np.sum(captive_solar) * dt) if captive_solar is not None else 0.0
            captive_revenue = ppa_rate * (bess_to_captive_mwh + solar_to_captive_mwh)

        total_costs = (iex_fee + scheduling + degradation + ists
                       + dsm_penalty + dsm_charge_fail + oa
                       - captive_revenue)  # revenue reduces net cost

        return {
            "iex_transaction_fee":    iex_fee,
            "scheduling_charges":     scheduling,
            "degradation_cost":       degradation,
            "ists_charges":           ists,
            "dsm_penalty_estimate":   dsm_penalty,
            "dsm_charge_failure":     dsm_charge_fail,
            "open_access_charges":    oa,
            "captive_ppa_revenue":    captive_revenue,
            "total_costs":            total_costs,
            "cost_breakdown_pct": {
                "iex":        (iex_fee / total_costs * 100) if total_costs > 0 else 0,
                "scheduling": (scheduling / total_costs * 100) if total_costs > 0 else 0,
                "degradation":(degradation / total_costs * 100) if total_costs > 0 else 0,
                "dsm":        ((dsm_penalty + dsm_charge_fail) / total_costs * 100)
                               if total_costs > 0 else 0,
                "captive_ppa_credit": (captive_revenue / total_costs * 100)
                               if total_costs > 0 else 0,
            },
        }
