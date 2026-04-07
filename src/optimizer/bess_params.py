"""
src/optimizer/bess_params.py
=============================
Battery + Solar parameter dataclass for the Solar + BESS optimizer.

New fields vs. BESS-only version:
  solar_capacity_mwp   : float  Solar PV nameplate capacity (MWp)
  ppa_rate_rs_mwh      : float  Captive consumer PPA rate (Rs/MWh)
                                 Also used as solar opportunity cost in LP objective.
  soc_buffer_pct       : float  SoC planning buffer (fraction, default 5%).
                                 Planning ceiling = e_max_mwh * (1 - buffer)
                                 Planning floor   = e_min_mwh * (1 + buffer)
                                 Leaves headroom for solar forecast errors,
                                 preventing DSM charge-failure penalties.
"""

from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class BESSParams:
    # ── BESS physical ─────────────────────────────────────────────────────────
    p_max_mw:              float          # Max charge / discharge power (MW)
    e_max_mwh:             float          # SoC physical ceiling (MWh)
    e_min_mwh:             float          # SoC physical floor (MWh)
    eta_charge:            float          # One-way charge efficiency (fraction)
    eta_discharge:         float          # One-way discharge efficiency (fraction)
    soc_initial_mwh:       float          # Starting SoC — chained overnight (MWh)
    soc_terminal_min_mwh:  float          # Hard terminal SoC minimum (MWh)
    degradation_cost_rs_mwh: float        # Throughput degradation cost (Rs/MWh discharged)
    iex_fee_rs_mwh:        float          # IEX transaction fee per side (Rs/MWh)

    # ── Solar PV ──────────────────────────────────────────────────────────────
    solar_capacity_mwp:    float = 35.0   # Solar PV nameplate (MWp). Jamnagar plant.

    # ── Captive consumer ──────────────────────────────────────────────────────
    ppa_rate_rs_mwh:       float = 3500.0 # PPA rate = captive revenue = solar opp cost (Rs/MWh)

    # ── SoC buffer ────────────────────────────────────────────────────────────
    soc_buffer_pct:        float = 0.05   # 5% buffer at both ceiling and floor
                                          # Planning ceiling = e_max_mwh * (1 - 0.05)
                                          # Planning floor   = e_min_mwh * (1 + 0.05)

    # ── Cycle limit ───────────────────────────────────────────────────────────
    max_cycles_per_day:    Optional[float] = None  # None = unconstrained (merchant mode)

    # ── Terminal SoC mode ─────────────────────────────────────────────────────
    soc_terminal_mode:     str   = "soft"  # "hard" | "soft" | "physical"
    soc_terminal_value_rs_mwh: float = 0.0  # Rs/MWh continuation value (soft terminal)

    # ──────────────────────────────────────────────────────────────────────────
    # Derived properties (computed, not stored in YAML)
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def e_max_plan_mwh(self) -> float:
        """SoC planning ceiling with 5% buffer. LP never exceeds this."""
        return self.e_max_mwh * (1.0 - self.soc_buffer_pct)

    @property
    def e_min_plan_mwh(self) -> float:
        """SoC planning floor with 5% buffer. LP never goes below this."""
        return self.e_min_mwh * (1.0 + self.soc_buffer_pct)

    # ──────────────────────────────────────────────────────────────────────────
    # YAML loader
    # ──────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "BESSParams":
        """Load parameters from a YAML file. File must be ASCII-encoded."""
        with open(path, "r", encoding="ascii") as f:
            data = yaml.safe_load(f)
        return cls(**data)
