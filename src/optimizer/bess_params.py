"""
src/optimizer/bess_params.py
=============================
Battery + Solar parameter dataclass — Architecture v3.

Changes vs previous version:
  solar_inverter_mw    : float  Solar inverter AC rating (MW). Default 25.0.
  rtm_lead_blocks      : int    RTM bid lead time (blocks). Default 3.
  captive_buffer_blocks: int    Captive schedule ramp buffer (blocks). Default 12.
  captive_buffer_tolerance_mw: float  Tolerance on captive buffer (MW). Default 0.5.
"""

from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class BESSParams:
    # -- BESS physical --
    p_max_mw:              float
    e_max_mwh:             float
    e_min_mwh:             float
    eta_charge:            float
    eta_discharge:         float
    soc_initial_mwh:       float
    soc_terminal_min_mwh:  float
    degradation_cost_rs_mwh: float
    iex_fee_rs_mwh:        float

    # -- Solar PV --
    solar_capacity_mwp:    float = 35.0
    solar_inverter_mw:     float = 25.0    # NEW: inverter AC rating

    # -- Captive consumer --
    ppa_rate_rs_mwh:       float = 3500.0

    # -- SoC buffer --
    soc_buffer_pct:        float = 0.05

    # -- Cycle limit --
    max_cycles_per_day:    Optional[float] = None

    # -- Terminal SoC mode --
    soc_terminal_mode:     str   = "hard"
    soc_terminal_value_rs_mwh: float = 0.0

    # -- RTM timing --
    rtm_lead_blocks:       int   = 3       # NEW: bid B+3 at block B
    captive_buffer_blocks: int   = 12      # NEW: 12-block captive ramp buffer
    captive_buffer_tolerance_mw: float = 0.5  # NEW: +/- MW tolerance

    # -- Derived --
    @property
    def e_max_plan_mwh(self) -> float:
        return self.e_max_mwh * (1.0 - self.soc_buffer_pct)

    @property
    def e_min_plan_mwh(self) -> float:
        return self.e_min_mwh * (1.0 + self.soc_buffer_pct)

    @classmethod
    def from_yaml(cls, path: str) -> "BESSParams":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
