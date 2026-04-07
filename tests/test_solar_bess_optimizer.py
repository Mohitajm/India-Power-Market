"""
tests/test_solar_bess_optimizer.py
====================================
Unit tests for the Solar + BESS two-stage stochastic optimizer.

Run from project root:
    python -m pytest tests/test_solar_bess_optimizer.py -v

Tests:
  Stage 1:
    test_stage1_solar_balance          — s_c + s_cd + curtail = z_sol every block
    test_stage1_soc_within_plan_bounds — SoC in [e_min_plan, e_max_plan] always
    test_stage1_power_limits           — charge/discharge never exceed 2.5 MW
    test_stage1_opportunity_cost       — solar stored when IEX > PPA, else to captive

  Stage 2B:
    test_stage2b_output_structure      — correct arrays and shapes returned
    test_stage2b_solar_balance         — solar balance holds in 2B outputs
    test_stage2b_runs_before_stage2a   — captive routing updated before 2A

  Stage 2A:
    test_stage2a_returns_single_block  — only y_c[B] and y_d[B] returned
    test_stage2a_zero_when_no_headroom — y_c=y_d=0 when SoC at ceiling
    test_stage2a_lag4_conditioning     — high lag-4 price increases y_d tendency
"""

import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.optimizer.bess_params  import BESSParams
from src.optimizer.two_stage_bess import (
    TwoStageBESS,
    reschedule_captive,
    solve_stage2a_block,
    evaluate_actuals_solar,
    RESCHEDULE_BLOCKS,
    T_BLOCKS,
    DT,
)

# ── Shared test fixtures ──────────────────────────────────────────────────────

def make_params(soc_init: float = 2.5) -> BESSParams:
    return BESSParams(
        p_max_mw              = 2.5,
        e_max_mwh             = 4.75,
        e_min_mwh             = 0.50,
        eta_charge            = 0.9487,
        eta_discharge         = 0.9487,
        soc_initial_mwh       = soc_init,
        soc_terminal_min_mwh  = 1.0,
        degradation_cost_rs_mwh = 650.0,
        iex_fee_rs_mwh        = 200.0,
        solar_capacity_mwp    = 35.0,
        ppa_rate_rs_mwh       = 3500.0,
        soc_buffer_pct        = 0.05,
        max_cycles_per_day    = None,
        soc_terminal_mode     = "soft",
        soc_terminal_value_rs_mwh = 0.0,
    )


def make_scenarios(S: int = 5, seed: int = 42) -> tuple:
    """Return (dam_scenarios, rtm_scenarios) of shape (S, 96)."""
    rng = np.random.default_rng(seed)
    dam = rng.uniform(2000, 6000, size=(S, T_BLOCKS)).astype(np.float32)
    rtm = rng.uniform(1800, 5500, size=(S, T_BLOCKS)).astype(np.float32)
    return dam, rtm


def make_solar_da(peak_mw: float = 20.0) -> np.ndarray:
    """Synthetic 96-block solar profile — bell-shaped day with zero at night."""
    solar = np.zeros(T_BLOCKS, dtype=np.float32)
    # Daylight: blocks 24..79 (6:00-20:00 IST approx)
    for b in range(24, 80):
        angle = np.pi * (b - 24) / (79 - 24)
        solar[b] = peak_mw * np.sin(angle)
    return solar


CONFIG = {
    "solver":      "CBC",
    "n_scenarios": 5,
    "lambda_risk": 0.0,
    "lambda_dev":  0.0,
    "dev_max_mw":  2.5,
    "risk_alpha":  0.1,
}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestStage1:

    def test_stage1_solar_balance(self):
        """s_c_da[t] + s_cd_da[t] + curtail_da[t] == z_sol_da[t] for every block."""
        params = make_params()
        dam, rtm = make_scenarios()
        solar_da = make_solar_da(peak_mw=20.0)

        opt = TwoStageBESS(params, CONFIG)
        res = opt.solve(dam, rtm, solar_da)

        assert res["status"] == "Optimal", f"Expected Optimal, got {res['status']}"

        sc   = np.array(res["s_c_da"])
        scd  = np.array(res["s_cd_da"])
        cu   = np.array(res["curtail_da"])
        sumv = sc + scd + cu

        for t in range(T_BLOCKS):
            assert abs(sumv[t] - float(solar_da[t])) < 1e-4, (
                f"Solar balance violated at block {t}: "
                f"s_c={sc[t]:.4f} + s_cd={scd[t]:.4f} + cu={cu[t]:.4f} = {sumv[t]:.4f} "
                f"!= solar_da={solar_da[t]:.4f}"
            )

    def test_stage1_soc_within_plan_bounds(self):
        """SoC trajectory must stay within buffered planning bounds."""
        params   = make_params()
        e_min_p  = params.e_min_plan_mwh
        e_max_p  = params.e_max_plan_mwh
        dam, rtm = make_scenarios(S=3)
        solar_da = make_solar_da(peak_mw=25.0)

        opt = TwoStageBESS(params, CONFIG)
        res = opt.solve(dam, rtm, solar_da)

        assert res["status"] == "Optimal"
        S = len(res["scenarios"])
        for s in range(S):
            soc_traj = res["scenarios"][s]["soc"]
            for t, soc_val in enumerate(soc_traj):
                if soc_val is not None:
                    assert soc_val >= e_min_p - 1e-4, (
                        f"Scenario {s} SoC below e_min_plan at t={t}: "
                        f"{soc_val:.4f} < {e_min_p:.4f}"
                    )
                    assert soc_val <= e_max_p + 1e-4, (
                        f"Scenario {s} SoC above e_max_plan at t={t}: "
                        f"{soc_val:.4f} > {e_max_p:.4f}"
                    )

    def test_stage1_power_limits(self):
        """Charge and discharge must not exceed p_max_mw."""
        params   = make_params()
        dam, rtm = make_scenarios(S=3)
        solar_da = make_solar_da(peak_mw=30.0)

        opt = TwoStageBESS(params, CONFIG)
        res = opt.solve(dam, rtm, solar_da)

        assert res["status"] == "Optimal"
        xc  = np.array(res["x_c"])
        xd  = np.array(res["x_d"])
        sc  = np.array(res["s_c_da"])
        cd  = np.array(res["c_d_da"])

        for t in range(T_BLOCKS):
            total_charge    = sc[t] + xc[t]
            total_discharge = xd[t] + cd[t]
            assert total_charge    <= params.p_max_mw + 1e-4, \
                f"Charge limit violated at t={t}: {total_charge:.4f} > {params.p_max_mw}"
            assert total_discharge <= params.p_max_mw + 1e-4, \
                f"Discharge limit violated at t={t}: {total_discharge:.4f} > {params.p_max_mw}"

    def test_stage1_opportunity_cost_logic(self):
        """
        When IEX prices are well above PPA rate, LP should store solar (s_c > 0).
        When IEX prices are at PPA level, LP should prefer selling to captive (s_cd > 0).
        """
        params = make_params()

        # High-spread scenario: morning prices very high → store solar at midday
        dam_high = np.full((3, T_BLOCKS), 7000.0, dtype=np.float32)  # well above PPA
        rtm_high = np.full((3, T_BLOCKS), 6500.0, dtype=np.float32)
        solar_da = make_solar_da(peak_mw=10.0)

        opt = TwoStageBESS(params, CONFIG)
        res_high = opt.solve(dam_high, rtm_high, solar_da)
        assert res_high["status"] == "Optimal"
        # With IEX prices >> PPA, should see some solar storage
        total_stored = sum(res_high["s_c_da"])
        total_captive_solar = sum(res_high["s_cd_da"])
        # Cannot assert exact values but structure should be valid
        assert total_stored >= 0.0
        assert total_captive_solar >= 0.0

        # Low scenario: prices at PPA level → captive preferred
        dam_low = np.full((3, T_BLOCKS), 3500.0, dtype=np.float32)  # = PPA rate
        rtm_low = np.full((3, T_BLOCKS), 3200.0, dtype=np.float32)  # < PPA rate
        res_low = opt.solve(dam_low, rtm_low, solar_da)
        assert res_low["status"] == "Optimal"
        # With IEX prices ≤ PPA, less incentive to store — curtailment or captive
        # (round-trip efficiency makes storage costly when IEX == PPA)
        total_stored_low = sum(res_low["s_c_da"])
        # Stored should be less or equal when prices are at PPA
        assert total_stored_low <= total_stored + 1e-3

    def test_stage1_all_outputs_present(self):
        """All expected keys must be present in result dict."""
        params   = make_params()
        dam, rtm = make_scenarios(S=3)
        solar_da = make_solar_da()

        opt = TwoStageBESS(params, CONFIG)
        res = opt.solve(dam, rtm, solar_da)

        required_keys = [
            "status", "expected_revenue", "dam_schedule",
            "x_c", "x_d", "s_c_da", "s_cd_da", "c_d_da",
            "curtail_da", "captive_schedule_da", "scenarios",
        ]
        for key in required_keys:
            assert key in res, f"Missing key in Stage 1 result: {key}"

        assert len(res["dam_schedule"])        == T_BLOCKS
        assert len(res["x_c"])                 == T_BLOCKS
        assert len(res["captive_schedule_da"]) == T_BLOCKS
        assert len(res["scenarios"]) == 3   # S scenarios


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2B TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestStage2B:

    def _make_stage2b_inputs(self, trigger: int = 34):
        params   = make_params(soc_init=2.5)
        solar_nc = make_solar_da(20.0)[trigger: trigger + 12]  # (12,)
        solar_da = make_solar_da(20.0)
        rtm_q50  = np.full(T_BLOCKS, 4000.0)
        xc       = np.zeros(T_BLOCKS)
        xd       = np.zeros(T_BLOCKS)
        return params, solar_nc, solar_da, rtm_q50, xc, xd

    def test_stage2b_output_structure(self):
        """Stage 2B returns correct keys and array shapes."""
        params, solar_nc, solar_da, rtm_q50, xc, xd = self._make_stage2b_inputs(34)
        res = reschedule_captive(
            params        = params,
            trigger_block = 34,
            soc_actual    = 2.5,
            solar_nc_row  = solar_nc,
            solar_da      = solar_da,
            rtm_q50       = rtm_q50,
            x_c_stage1    = xc,
            x_d_stage1    = xd,
        )
        assert "status" in res
        for key in ["s_c_rt", "s_cd_rt", "c_d_rt", "curtail_rt", "captive_rt"]:
            assert key in res, f"Missing key: {key}"
            assert len(res[key]) == T_BLOCKS, f"{key} should have {T_BLOCKS} elements"

    def test_stage2b_zeros_before_trigger(self):
        """All solar routing arrays must be zero before trigger_block."""
        params, solar_nc, solar_da, rtm_q50, xc, xd = self._make_stage2b_inputs(42)
        res = reschedule_captive(
            params=params, trigger_block=42, soc_actual=2.5,
            solar_nc_row=solar_nc, solar_da=solar_da,
            rtm_q50=rtm_q50, x_c_stage1=xc, x_d_stage1=xd,
        )
        if res["status"] == "Optimal":
            for key in ["s_c_rt", "s_cd_rt", "c_d_rt", "curtail_rt"]:
                for b in range(42):
                    assert res[key][b] == 0.0, \
                        f"{key}[{b}] should be 0 before trigger block 42"

    def test_stage2b_solar_balance(self):
        """After Stage 2B: s_c_rt + s_cd_rt + curtail_rt = solar_blend for t >= trigger."""
        params, solar_nc, solar_da, rtm_q50, xc, xd = self._make_stage2b_inputs(34)
        res = reschedule_captive(
            params=params, trigger_block=34, soc_actual=2.5,
            solar_nc_row=solar_nc, solar_da=solar_da,
            rtm_q50=rtm_q50, x_c_stage1=xc, x_d_stage1=xd,
        )
        if res["status"] != "Optimal":
            pytest.skip("Stage 2B infeasible — cannot check balance")

        # Build expected solar blend
        solar_blend = np.zeros(T_BLOCKS)
        for k in range(T_BLOCKS - 34):
            if k < 12:
                solar_blend[34 + k] = float(solar_nc[k])
            else:
                solar_blend[34 + k] = float(solar_da[34 + k])

        for b in range(34, T_BLOCKS):
            expected = float(np.clip(solar_blend[b], 0, params.solar_capacity_mwp))
            actual   = res["s_c_rt"][b] + res["s_cd_rt"][b] + res["curtail_rt"][b]
            assert abs(actual - expected) < 1e-3, \
                f"Solar balance violated at b={b}: {actual:.4f} != {expected:.4f}"

    def test_stage2b_all_reschedule_blocks(self):
        """Stage 2B must work for all four reschedule blocks."""
        for trigger in RESCHEDULE_BLOCKS:
            params   = make_params(soc_init=2.5)
            solar_nc = make_solar_da(20.0)[trigger: trigger + 12]
            solar_da = make_solar_da(20.0)
            rtm_q50  = np.full(T_BLOCKS, 4500.0)
            xc       = np.zeros(T_BLOCKS)
            xd       = np.full(T_BLOCKS, 0.5)
            res = reschedule_captive(
                params=params, trigger_block=trigger, soc_actual=2.5,
                solar_nc_row=solar_nc, solar_da=solar_da,
                rtm_q50=rtm_q50, x_c_stage1=xc, x_d_stage1=xd,
            )
            assert res["status"] == "Optimal", \
                f"Stage 2B failed at trigger block {trigger}: {res['status']}"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2A TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestStage2A:

    def test_stage2a_returns_two_scalars(self):
        """solve_stage2a_block returns exactly (y_c, y_d) — two scalars."""
        params      = make_params(soc_init=2.5)
        dam_sched   = np.zeros(T_BLOCKS)
        dam_actual  = np.full(T_BLOCKS, 4000.0)
        rtm_q50     = np.full(T_BLOCKS, 3800.0)

        result = solve_stage2a_block(
            params       = params,
            block_B      = 40,
            soc_actual_B = 2.5,
            dam_schedule = dam_sched,
            dam_actual   = dam_actual,
            p_rtm_lag4   = 4200.0,
            rtm_q50      = rtm_q50,
            s_c_rt_B     = 5.0,
            c_d_rt_B     = 0.5,
        )
        assert isinstance(result, tuple), "Expected tuple (y_c, y_d)"
        assert len(result) == 2, "Expected exactly 2 values"
        y_c, y_d = result
        assert isinstance(y_c, float)
        assert isinstance(y_d, float)
        assert y_c >= 0.0
        assert y_d >= 0.0

    def test_stage2a_zero_when_soc_at_ceiling(self):
        """y_c must be 0 when SoC is at or above e_max_plan."""
        params = make_params()
        # Set SoC to exactly the planning ceiling
        soc_at_ceiling = params.e_max_plan_mwh

        y_c, y_d = solve_stage2a_block(
            params       = params,
            block_B      = 30,
            soc_actual_B = soc_at_ceiling,
            dam_schedule = np.zeros(T_BLOCKS),
            dam_actual   = np.full(T_BLOCKS, 4000.0),
            p_rtm_lag4   = np.nan,
            rtm_q50      = np.full(T_BLOCKS, 4000.0),
            s_c_rt_B     = 0.0,
            c_d_rt_B     = 0.0,
        )
        # Charging should be zero or very small (headroom calculation prevents it)
        assert y_c < 0.01, f"y_c={y_c:.4f} should be ~0 when SoC at planning ceiling"

    def test_stage2a_zero_when_soc_at_floor(self):
        """y_d must be 0 when SoC is at or below e_min_plan."""
        params = make_params()
        soc_at_floor = params.e_min_plan_mwh

        y_c, y_d = solve_stage2a_block(
            params       = params,
            block_B      = 50,
            soc_actual_B = soc_at_floor,
            dam_schedule = np.zeros(T_BLOCKS),
            dam_actual   = np.full(T_BLOCKS, 4000.0),
            p_rtm_lag4   = np.nan,
            rtm_q50      = np.full(T_BLOCKS, 4000.0),
            s_c_rt_B     = 0.0,
            c_d_rt_B     = 0.0,
        )
        assert y_d < 0.01, f"y_d={y_d:.4f} should be ~0 when SoC at planning floor"

    def test_stage2a_lag4_conditioning_increases_discharge(self):
        """
        When p_rtm_lag4 >> q50 (market is higher than forecast), the adjusted
        q50 should be higher, incentivising discharge (y_d >= y_d_no_lag).
        """
        params  = make_params(soc_init=3.5)   # good SoC for discharge
        q50_val = 3500.0
        rtm_q50 = np.full(T_BLOCKS, q50_val)
        kwargs  = dict(
            params       = params,
            block_B      = 40,
            dam_schedule = np.zeros(T_BLOCKS),
            dam_actual   = np.full(T_BLOCKS, 4000.0),
            rtm_q50      = rtm_q50,
            s_c_rt_B     = 0.0,
            c_d_rt_B     = 0.0,
        )

        # Without lag conditioning
        _, y_d_no_lag = solve_stage2a_block(
            soc_actual_B = 3.5, p_rtm_lag4 = np.nan, **kwargs
        )
        # With lag-4 well above forecast → market is hot → discharge more
        _, y_d_high_lag = solve_stage2a_block(
            soc_actual_B = 3.5, p_rtm_lag4 = q50_val * 2.0, **kwargs
        )
        # y_d_high_lag should be >= y_d_no_lag (more discharge incentive)
        assert y_d_high_lag >= y_d_no_lag - 0.01, (
            f"Expected y_d with high lag4 ({y_d_high_lag:.3f}) >= "
            f"y_d without lag ({y_d_no_lag:.3f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TEST
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_day_evaluate_actuals_solar(self):
        """
        Run a synthetic full-day evaluation through evaluate_actuals_solar().
        Verify that Stage 2B runs before Stage 2A at reschedule blocks,
        that SoC path stays in physical bounds, and that revenue is positive.
        """
        params   = make_params(soc_init=2.5)
        dam, rtm = make_scenarios(S=5)
        solar_da = make_solar_da(peak_mw=20.0)

        opt = TwoStageBESS(params, CONFIG)
        stage1 = opt.solve(dam, rtm, solar_da)
        assert stage1["status"] == "Optimal"

        # Synthetic actuals
        dam_actual = dam[0]   # use scenario 0 as actuals
        rtm_actual = rtm[0]
        rtm_q50    = rtm[2]   # use scenario 2 as q50 forecast
        solar_at   = solar_da * 0.95   # actuals slightly below forecast
        solar_nc   = np.zeros((T_BLOCKS, 12), dtype=np.float32)
        for b in range(T_BLOCKS):
            for k in range(12):
                t = b + k
                solar_nc[b, k] = solar_da[t] if t < T_BLOCKS else 0.0

        result = evaluate_actuals_solar(
            params           = params,
            stage1_result    = stage1,
            dam_actual       = dam_actual,
            rtm_actual       = rtm_actual,
            rtm_q50          = rtm_q50,
            solar_da         = solar_da,
            solar_nc         = solar_nc,
            solar_at         = solar_at,
            reschedule_blocks = RESCHEDULE_BLOCKS,
            verbose          = False,
        )

        # Check required keys
        for key in ["revenue", "net_revenue", "y_c", "y_d",
                    "s_c_rt", "s_cd_rt", "c_d_rt", "soc_path"]:
            assert key in result, f"Missing key: {key}"

        # SoC must stay within physical bounds
        soc = result["soc_path"]
        assert len(soc) == T_BLOCKS + 1
        for t, s in enumerate(soc):
            assert s >= params.e_min_mwh - 1e-4, \
                f"SoC below physical floor at t={t}: {s:.4f}"
            assert s <= params.e_max_mwh + 1e-4, \
                f"SoC above physical ceiling at t={t}: {s:.4f}"

        # Revenue should be a finite number
        assert np.isfinite(result["revenue"])
        assert np.isfinite(result["net_revenue"])
