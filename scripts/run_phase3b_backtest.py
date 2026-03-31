"""
scripts/run_phase3b_backtest.py
================================
Phase 3B backtest — 96-block (15-min), CAUSAL Stage 2 evaluation.

THE FIX: evaluate_actuals_causal()
────────────────────────────────────
The original evaluate_actuals() was a look-ahead oracle:
  it solved ONE LP over all 96 blocks at once using ACTUAL RTM prices
  for every block including future ones.  In reality you can never know
  block B's RTM price before gate-close for block B.

Information available at gate-close for block B (B = 0..95):
  ✓  DAM actual prices p_dam[0..95]   — released D-1 14:00, fully known
  ✓  RTM actual prices p_rtm[0..B-1]  — past cleared blocks
  ✓  Current SoC after dispatching blocks 0..B-1
  ✓  x_net[B]                         — locked from Stage 1
  ✗  p_rtm[B]     — gate just closed, price not yet published
  ✗  p_rtm[B+1..95] — future blocks

What we use instead for the unknown RTM prices:
  • p_rtm[B]      → RTM quantile model q50 forecast for block B
                    (already computed in train_models.py, stored in
                     rtm_quantiles_backtest_recalibrated.parquet)
  • p_rtm[B+1..95] → q50 forecasts from the same parquet

Algorithm — sequential rolling LP:
  For B = 0, 1, ..., 95:
    1. Build a look-ahead mini-LP over blocks B..95
    2. RTM price inputs:
         p_rtm_known[b]    = actual price  for b < B   (past)
         p_rtm_known[b]    = q50 forecast  for b >= B  (future incl. current)
    3. Solve → get optimal y_c[B], y_d[B]
    4. COMMIT only block B dispatch (y_c_committed[B], y_d_committed[B])
    5. Observe ACTUAL RTM price for block B, compute block revenue
    6. Update SoC, advance to B+1

This produces a y schedule where each block's decision was made with
only causally available information, exactly as a real operator would.

Revenue formula (IEX corrected):
  Revenue_B = p_dam_actual[B] * x_net[B] * DT
            + p_rtm_actual[B] * y_net[B]  * DT

The RTM settlement uses the ACTUAL price once revealed (not the forecast).
The forecast is only used to DECIDE y, not to settle it.
"""

import pandas as pd
import numpy as np
import yaml
import json
import sys
from pathlib import Path
import argparse
import pulp

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.optimizer.bess_params     import BESSParams
from src.optimizer.scenario_loader import ScenarioLoader
from src.optimizer.two_stage_bess   import TwoStageBESS
from src.optimizer.costs            import CostModel

# ── Constants ─────────────────────────────────────────────────────────────────
T_BLOCKS = 96      # 15-min blocks per day
DT       = 0.25    # hours per block


# ─────────────────────────────────────────────────────────────────────────────
# CAUSAL Stage 2 evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_actuals_causal(
    bess_params,
    dam_schedule,          # (96,)  x_net from Stage 1 — MW
    dam_actual,            # (96,)  actual DAM MCP — Rs/MWh  (fully known)
    rtm_actual,            # (96,)  actual RTM MCP — Rs/MWh  (revealed block-by-block)
    rtm_forecast_q50,      # (96,)  RTM q50 forecast — Rs/MWh  (known at gate-close)
    cost_model: CostModel = None,
    lambda_dev: float = 0.0,
    force_terminal_mode: str = None,
    verbose: bool = False,
) -> dict:
    """
    Causal sequential rolling LP for Stage 2 RTM dispatch.

    For each block B we solve a look-ahead LP over blocks B..95 where:
      • Past RTM prices  (b < B)  : actual (already revealed) — only affect SoC carry-in
      • Current RTM price (b = B) : q50 forecast  ← gate just closed, actual not yet known
      • Future RTM prices (b > B) : q50 forecast

    Only the dispatch for block B is committed.  The LP is re-solved from
    scratch at block B+1 with the updated SoC.

    IEX settlement uses the ACTUAL price once it is revealed — the forecast
    is purely a decision input, never a settlement input.

    Parameters
    ----------
    dam_schedule      : array (96,)   x_net[b] = x_d[b] - x_c[b]  (Stage 1 output)
    dam_actual        : array (96,)   actual DAM clearing price
    rtm_actual        : array (96,)   actual RTM clearing price  (used for settlement only)
    rtm_forecast_q50  : array (96,)   RTM q50 forecast at D-1 08:00 snapshot
                                       (causal: available before the day starts)
    """
    assert len(dam_schedule)     == T_BLOCKS
    assert len(dam_actual)       == T_BLOCKS
    assert len(rtm_actual)       == T_BLOCKS
    assert len(rtm_forecast_q50) == T_BLOCKS

    effective_mode = force_terminal_mode or bess_params.soc_terminal_mode

    # ── Track committed dispatch block by block ───────────────────────────────
    y_net_committed   = np.zeros(T_BLOCKS)   # net RTM dispatch MW  (+ = discharge)
    y_c_committed     = np.zeros(T_BLOCKS)   # charge MW
    y_d_committed     = np.zeros(T_BLOCKS)   # discharge MW
    soc_path          = np.zeros(T_BLOCKS + 1)
    soc_path[0]       = bess_params.soc_initial_mwh

    total_revenue     = 0.0   # gross (DAM + RTM settlement)
    total_proxy_costs = 0.0   # in-LP proxy costs for net revenue estimate

    # ── Sequential block loop ─────────────────────────────────────────────────
    for B in range(T_BLOCKS):
        remaining = T_BLOCKS - B    # number of blocks left including B

        # ── Build RTM price vector seen by the LP at gate-close for block B ───
        #
        # Blocks 0..B-1 : actual prices already revealed  (but we've already
        #                  committed those blocks — we only need them here if
        #                  they appear in a lag feature, which the LP doesn't
        #                  use directly.  For the LP objective we just need
        #                  prices for blocks B..95.)
        # Block B       : q50 forecast  (actual not yet released)
        # Blocks B+1..95: q50 forecast  (future, unknown)
        #
        # So the LP price array for blocks B..95 is simply the forecast.
        rtm_price_lp = rtm_forecast_q50[B:]          # (remaining,)  all forecast
        dam_price_lp = dam_actual[B:]                 # (remaining,)  fully known
        x_net_lp     = np.array(dam_schedule[B:])     # (remaining,)

        # Current SoC (updated after each committed block)
        soc_now = soc_path[B]

        # ── LP: optimise over blocks B..95 ───────────────────────────────────
        prob = pulp.LpProblem(f"RTM_causal_B{B}", pulp.LpMaximize)

        y_c_lp = pulp.LpVariable.dicts("yc", range(remaining),
                                        lowBound=0,
                                        upBound=bess_params.p_max_mw)
        y_d_lp = pulp.LpVariable.dicts("yd", range(remaining),
                                        lowBound=0,
                                        upBound=bess_params.p_max_mw)
        soc_lp = pulp.LpVariable.dicts("s",  range(remaining + 1),
                                        lowBound=bess_params.e_min_mwh,
                                        upBound=bess_params.e_max_mwh)

        # Initial SoC for this sub-problem
        prob += soc_lp[0] == soc_now

        # SoC dynamics
        for k in range(remaining):
            prob += soc_lp[k + 1] == (
                soc_lp[k]
                + bess_params.eta_charge    * y_c_lp[k] * DT
                - (1.0 / bess_params.eta_discharge) * y_d_lp[k] * DT
            )

        # Terminal SoC — only apply at end of full day (B=0 horizon = 96 blocks)
        # For mid-day sub-problems: always use physical floor only
        # (hard terminal at every step would make most sub-problems infeasible
        #  because the battery cannot guarantee 100 MWh by end-of-day from
        #  every intermediate SoC state)
        if effective_mode == "hard" and B == 0:
            prob += soc_lp[remaining] >= bess_params.soc_terminal_min_mwh

        # ── Objective: maximise expected revenue over remaining blocks ─────────
        # Revenue uses forecast RTM price (causal), not actual
        # Settlement (outside this LP) will use actual prices
        rev = pulp.lpSum([
            dam_price_lp[k] * x_net_lp[k] * DT            # DAM leg (known)
            + rtm_price_lp[k] * (y_d_lp[k] - y_c_lp[k]) * DT  # RTM leg (forecast)
            - bess_params.iex_fee_rs_mwh        * (y_c_lp[k] + y_d_lp[k]) * DT
            - bess_params.degradation_cost_rs_mwh * y_d_lp[k] * DT
            - 135.0                               * (y_c_lp[k] + y_d_lp[k]) * DT
            for k in range(remaining)
        ])

        # Deviation penalty |y_net - x_net| for first block only
        # (to match Stage 1 planning cost structure)
        if lambda_dev > 0:
            dev0 = pulp.LpVariable("dev0", lowBound=0)
            y_net_0 = y_d_lp[0] - y_c_lp[0]
            prob += dev0 >= y_net_0 - x_net_lp[0]
            prob += dev0 >= x_net_lp[0] - y_net_0
            rev = rev - lambda_dev * dev0

        prob.objective = rev
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if pulp.LpStatus[prob.status] != 'Optimal':
            # Fallback: match DAM schedule exactly (safe default)
            y_c_b = max(-dam_schedule[B], 0)
            y_d_b = max( dam_schedule[B], 0)
            if verbose:
                print(f"  Block {B:2d}: LP infeasible, falling back to DAM schedule")
        else:
            y_c_b = pulp.value(y_c_lp[0])
            y_d_b = pulp.value(y_d_lp[0])

        # ── COMMIT block B dispatch ───────────────────────────────────────────
        y_c_committed[B] = y_c_b
        y_d_committed[B] = y_d_b
        y_net_committed[B] = y_d_b - y_c_b

        # ── SETTLE block B using ACTUAL prices (not forecast) ─────────────────
        dam_rev_b  = dam_actual[B]  * dam_schedule[B]    * DT
        rtm_rev_b  = rtm_actual[B]  * y_net_committed[B] * DT
        total_revenue += dam_rev_b + rtm_rev_b

        # Proxy costs for the committed block
        proxy_b = (
            bess_params.iex_fee_rs_mwh          * (y_c_b + y_d_b) * DT
            + bess_params.degradation_cost_rs_mwh * y_d_b           * DT
            + 135.0                                * (y_c_b + y_d_b) * DT
        )
        total_proxy_costs += proxy_b

        # ── Advance SoC using ACTUAL dispatch ─────────────────────────────────
        soc_path[B + 1] = (
            soc_path[B]
            + bess_params.eta_charge * y_c_b * DT
            - (1.0 / bess_params.eta_discharge) * y_d_b * DT
        )
        # Clip to physical bounds (rounding errors)
        soc_path[B + 1] = np.clip(
            soc_path[B + 1],
            bess_params.e_min_mwh,
            bess_params.e_max_mwh,
        )

        if verbose and B % 16 == 0:
            h = B // 4
            m = (B % 4) * 15
            print(f"  B{B:02d} {h:02d}:{m:02d}  "
                  f"SoC={soc_path[B]:.1f}  "
                  f"y_net={y_net_committed[B]:+.1f}  "
                  f"rtm_fcst={rtm_forecast_q50[B]:.0f}  "
                  f"rtm_actual={rtm_actual[B]:.0f}")

    # ── Net revenue via CostModel ─────────────────────────────────────────────
    net_revenue = total_revenue - total_proxy_costs
    cost_breakdown = None

    if cost_model:
        cost_breakdown = cost_model.compute_costs(
            charge    = y_c_committed * DT,
            discharge = y_d_committed * DT,
            dam_actual= dam_actual,
            rtm_actual= rtm_actual,
        )
        net_revenue = total_revenue - cost_breakdown['total_costs']

    return {
        "revenue":        total_revenue,
        "net_revenue":    net_revenue,
        "fees_breakdown": cost_breakdown,
        "rtm_schedule":   y_net_committed.tolist(),
        "y_c":            y_c_committed.tolist(),
        "y_d":            y_d_committed.tolist(),
        "soc":            soc_path.tolist(),
        "forecast_used":  rtm_forecast_q50.tolist(),   # for diagnostics
    }


# ── Oracle (look-ahead) version — kept for benchmark comparison ───────────────

def evaluate_actuals_oracle(
    bess_params,
    dam_schedule,
    dam_actual,
    rtm_actual,
    cost_model: CostModel = None,
    lambda_dev: float = 0.0,
    force_terminal_mode: str = None,
):
    """
    ORACLE benchmark: solve ONE LP over all 96 blocks with ACTUAL RTM prices.

    This is NOT production-safe (requires future RTM prices) but provides a
    realistic upper bound on causal Stage 2 performance.  Keep this to measure
    the 'forecast cost' = oracle_revenue - causal_revenue.
    """
    effective_mode = force_terminal_mode or bess_params.soc_terminal_mode

    prob = pulp.LpProblem("RTM_Oracle_96block", pulp.LpMaximize)

    y_c = pulp.LpVariable.dicts("yc", range(T_BLOCKS), 0, bess_params.p_max_mw)
    y_d = pulp.LpVariable.dicts("yd", range(T_BLOCKS), 0, bess_params.p_max_mw)
    soc = pulp.LpVariable.dicts("s",  range(T_BLOCKS + 1),
                                 bess_params.e_min_mwh, bess_params.e_max_mwh)
    dev = pulp.LpVariable.dicts("dev", range(T_BLOCKS), lowBound=0)

    prob += soc[0] == bess_params.soc_initial_mwh
    for b in range(T_BLOCKS):
        prob += soc[b + 1] == (
            soc[b]
            + bess_params.eta_charge * y_c[b] * DT
            - (1.0 / bess_params.eta_discharge) * y_d[b] * DT
        )
        prob += dev[b] >= y_d[b] - y_c[b] - dam_schedule[b]
        prob += dev[b] >= dam_schedule[b] - (y_d[b] - y_c[b])

    if effective_mode == "hard":
        prob += soc[T_BLOCKS] >= bess_params.soc_terminal_min_mwh

    revenue = pulp.lpSum([
        dam_actual[b] * dam_schedule[b] * DT
        + rtm_actual[b] * (y_d[b] - y_c[b]) * DT
        for b in range(T_BLOCKS)
    ])
    fees        = bess_params.iex_fee_rs_mwh * pulp.lpSum(
                      [(y_c[b] + y_d[b]) * DT for b in range(T_BLOCKS)])
    degradation = bess_params.degradation_cost_rs_mwh * pulp.lpSum(
                      [y_d[b] * DT for b in range(T_BLOCKS)])
    dsm_friction = 135.0 * pulp.lpSum(
                      [(y_c[b] + y_d[b]) * DT for b in range(T_BLOCKS)])

    prob.objective = (revenue - fees - degradation - dsm_friction
                      - lambda_dev * pulp.lpSum([dev[b] for b in range(T_BLOCKS)]))
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] != 'Optimal':
        return None

    y_net = [pulp.value(y_d[b] - y_c[b]) for b in range(T_BLOCKS)]
    y_c_arr = np.array([max(0, -v) for v in y_net])
    y_d_arr = np.array([max(0,  v) for v in y_net])
    realized_rev = pulp.value(revenue)

    net_revenue = realized_rev - pulp.value(fees + degradation + dsm_friction)
    cost_breakdown = None
    if cost_model:
        cost_breakdown = cost_model.compute_costs(
            charge    = y_c_arr * DT,
            discharge = y_d_arr * DT,
            dam_actual= dam_actual,
            rtm_actual= rtm_actual,
        )
        net_revenue = realized_rev - cost_breakdown['total_costs']

    return {
        "revenue":      realized_rev,
        "net_revenue":  net_revenue,
        "fees_breakdown": cost_breakdown,
        "rtm_schedule": y_net,
        "soc":          [pulp.value(soc[b]) for b in range(T_BLOCKS + 1)],
    }


# ── Main backtest loop ────────────────────────────────────────────────────────

def run_backtest(args):
    print("============================================================")
    print("PHASE 3B: TWO-STAGE STOCHASTIC BESS BACKTEST (96-block)")
    print("Stage 2 evaluation: CAUSAL sequential rolling LP")
    print("============================================================")

    bess_params = BESSParams.from_yaml("config/bess.yaml")
    with open("config/phase3b.yaml", 'r') as f:
        config = yaml.safe_load(f)

    loader = ScenarioLoader(
        dam_path         = config['paths']['scenarios_dam'],
        rtm_path         = config['paths']['scenarios_rtm'],
        actuals_dam_path = config['paths']['actuals_dam'],
        actuals_rtm_path = config['paths']['actuals_rtm'],
    )

    optimizer = TwoStageBESS(bess_params, config)

    cost_model = None
    if Path("config/costs_config.yaml").exists():
        cost_model = CostModel.from_yaml("config/costs_config.yaml")
        print("Loaded CostModel from config/costs_config.yaml")

    # ── Load RTM q50 forecasts for causal evaluation ──────────────────────────
    # These are the RTM price forecasts known at D-1 08:00 snapshot.
    # Causal rule: for block B, we use rtm_forecast_q50[B] as our best
    # estimate of p_rtm[B] at gate-close.
    rtm_forecast_path = Path("Data/Predictions/rtm_quantiles_backtest_recalibrated.parquet")
    if not rtm_forecast_path.exists():
        rtm_forecast_path = Path("Data/Predictions/rtm_quantiles_backtest.parquet")
    print(f"Loading RTM q50 forecasts from {rtm_forecast_path.name}...")
    rtm_forecasts_df = pd.read_parquet(rtm_forecast_path)

    # Normalise: ensure target_date and target_block columns
    if rtm_forecasts_df.index.name == 'delivery_start_ist' or (
        hasattr(rtm_forecasts_df.index, 'dtype')
        and str(rtm_forecasts_df.index.dtype).startswith('datetime')
    ):
        rtm_forecasts_df = rtm_forecasts_df.reset_index()

    if 'target_block' not in rtm_forecasts_df.columns:
        if 'delivery_start_ist' in rtm_forecasts_df.columns:
            ts = pd.to_datetime(rtm_forecasts_df['delivery_start_ist'])
            rtm_forecasts_df['target_block'] = ts.dt.hour * 4 + ts.dt.minute // 15 + 1
        elif 'target_hour' in rtm_forecasts_df.columns:
            rtm_forecasts_df['target_block'] = rtm_forecasts_df['target_hour'] * 4 + 1

    if 'target_date' in rtm_forecasts_df.columns:
        rtm_forecasts_df['target_date'] = rtm_forecasts_df['target_date'].astype(str)

    results_dir       = Path(config['paths']['results_dir'])
    daily_results_dir = results_dir / "daily"
    daily_results_dir.mkdir(parents=True, exist_ok=True)

    dates = loader.common_dates
    if args.day:
        dates = [args.day]
    elif args.limit:
        dates = dates[:args.limit]

    run_oracle = getattr(args, 'oracle', False)
    print(f"Running causal backtest for {len(dates)} days...")
    if run_oracle:
        print("Oracle benchmark ENABLED — also running look-ahead LP per day.")

    backtest_results = []
    prev_soc = bess_params.soc_initial_mwh

    for i, date in enumerate(dates):
        bess_params.soc_initial_mwh = prev_soc

        # Continuation value for soft terminal
        if bess_params.soc_terminal_mode == "soft" and i < len(dates) - 1:
            next_data = loader.get_day_scenarios(dates[i + 1], n_scenarios=20)
            spread_s  = (np.max(next_data['dam'], axis=1)
                         - np.min(next_data['dam'], axis=1))
            bess_params.soc_terminal_value_rs_mwh = max(0, (
                np.mean(spread_s)
                * bess_params.eta_charge
                * bess_params.eta_discharge
                - bess_params.iex_fee_rs_mwh * 2
                - bess_params.degradation_cost_rs_mwh
                - 135.0 * 2
            ))
        else:
            bess_params.soc_terminal_value_rs_mwh = 0.0

        print(f"[{i+1}/{len(dates)}] {date}...", end=" ", flush=True)

        day_data = loader.get_day_scenarios(date, n_scenarios=config['n_scenarios'])

        if day_data['dam'].shape[1] != T_BLOCKS:
            print(f"SKIP — scenario has {day_data['dam'].shape[1]} blocks")
            continue

        # ── Stage 1: solve stochastic LP ─────────────────────────────────────
        res = optimizer.solve(day_data['dam'], day_data['rtm'])
        if res['status'] != 'Optimal':
            print(f"FAILED Stage 1 ({res['status']})")
            continue

        # ── Get RTM q50 forecast for this date (96 blocks) ───────────────────
        day_rtm_fcst = (
            rtm_forecasts_df[rtm_forecasts_df['target_date'] == date]
            .sort_values('target_block')
        )

        if len(day_rtm_fcst) >= T_BLOCKS:
            rtm_q50 = day_rtm_fcst['q50'].values[:T_BLOCKS]
        elif len(day_rtm_fcst) == 24:
            # Hourly forecasts — expand to 96 blocks (repeat each 4×)
            rtm_q50 = np.repeat(day_rtm_fcst['q50'].values, 4)
        else:
            # No forecast available for this date — fall back to oracle
            print(f"WARNING: No RTM q50 forecast for {date}. "
                  f"Using oracle (look-ahead). This day is NOT causal.")
            rtm_q50 = day_data['rtm_actual']

        # ── Stage 2: CAUSAL sequential evaluation ────────────────────────────
        eval_res = evaluate_actuals_causal(
            bess_params,
            res['dam_schedule'],
            day_data['dam_actual'],
            day_data['rtm_actual'],
            rtm_forecast_q50 = rtm_q50,
            cost_model        = cost_model,
            lambda_dev        = config['lambda_dev'],
            verbose           = getattr(args, 'verbose', False),
        )

        if eval_res is None:
            print("EVAL FAILED")
            continue

        realized_rev = eval_res['revenue']
        net_rev      = eval_res['net_revenue']
        expected_rev = res['expected_revenue']

        # Optional oracle benchmark
        oracle_rev = None
        if run_oracle:
            oracle_res = evaluate_actuals_oracle(
                bess_params,
                res['dam_schedule'],
                day_data['dam_actual'],
                day_data['rtm_actual'],
                cost_model = cost_model,
                lambda_dev = config['lambda_dev'],
            )
            oracle_rev = oracle_res['revenue'] if oracle_res else None

        if oracle_rev is not None:
            print(f"Exp: ₹{expected_rev:,.0f} | "
                  f"Causal: ₹{realized_rev:,.0f} | "
                  f"Oracle: ₹{oracle_rev:,.0f} | "
                  f"Net: ₹{net_rev:,.0f} | "
                  f"Forecast cost: ₹{oracle_rev - realized_rev:,.0f}")
        else:
            print(f"Exp: ₹{expected_rev:,.0f} | "
                  f"Realized: ₹{realized_rev:,.0f} | "
                  f"Net: ₹{net_rev:,.0f}")

        daily_output = {
            "date":                  date,
            "status":                res['status'],
            "evaluation_mode":       "causal_sequential",
            "expected_revenue":      expected_rev,
            "realized_revenue":      realized_rev,
            "oracle_revenue":        oracle_rev,
            "net_revenue":           net_rev,
            "soc_initial":           prev_soc,
            "soc_terminal":          eval_res['soc'][-1],
            "continuation_value":    bess_params.soc_terminal_value_rs_mwh,
            "dam_schedule":          res['dam_schedule'],
            "rtm_realized_schedule": eval_res['rtm_schedule'],
            "soc_realized":          eval_res['soc'],
            "actual_dam_prices":     day_data['dam_actual'].tolist(),
            "actual_rtm_prices":     day_data['rtm_actual'].tolist(),
            "rtm_forecast_q50_used": eval_res['forecast_used'],
            "scenarios":             res['scenarios'],
        }

        prev_soc = eval_res['soc'][-1]

        with open(daily_results_dir / f"result_{date}.json", 'w') as f:
            json.dump(daily_output, f, indent=2,
                      default=lambda x: float(x) if isinstance(x, np.generic) else x)

        row = {
            "target_date":      date,
            "expected_revenue": expected_rev,
            "realized_revenue": realized_rev,
            "net_revenue":      net_rev,
        }
        if oracle_rev is not None:
            row["oracle_revenue"]   = oracle_rev
            row["forecast_cost_rs"] = oracle_rev - realized_rev
        backtest_results.append(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(backtest_results)
    results_df.to_csv(results_dir / "backtest_results.csv", index=False)

    summary = {
        "evaluation_mode":          "causal_sequential",
        "n_days":                   len(results_df),
        "blocks_per_day":           T_BLOCKS,
        "dt_hours":                 DT,
        "total_expected_revenue":   results_df['expected_revenue'].sum(),
        "total_realized_revenue":   results_df['realized_revenue'].sum(),
        "total_net_revenue":        results_df['net_revenue'].sum(),
        "avg_daily_realized":       results_df['realized_revenue'].mean(),
        "avg_daily_net":            results_df['net_revenue'].mean(),
        "std_daily_realized":       results_df['realized_revenue'].std(),
        "min_daily_realized":       results_df['realized_revenue'].min(),
    }
    if 'oracle_revenue' in results_df.columns:
        summary["total_oracle_revenue"]    = results_df['oracle_revenue'].sum()
        summary["total_forecast_cost_rs"]  = results_df['forecast_cost_rs'].sum()
        summary["avg_daily_forecast_cost"] = results_df['forecast_cost_rs'].mean()

    with open(results_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n============================================================")
    print("BACKTEST COMPLETE — CAUSAL Stage 2")
    print(f"Total Realized Revenue: ₹{summary['total_realized_revenue']:,.2f}")
    print(f"Total Net Revenue:      ₹{summary['total_net_revenue']:,.2f}")
    print(f"Avg Daily Net:          ₹{summary['avg_daily_net']:,.2f}")
    if 'total_forecast_cost_rs' in summary:
        print(f"Total Forecast Cost:    ₹{summary['total_forecast_cost_rs']:,.2f}  "
              f"(oracle − causal)")
    print(f"Summary saved to {results_dir / 'summary.json'}")
    print("============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--day',     type=str,  help='Run for a specific date (YYYY-MM-DD)')
    parser.add_argument('--limit',   type=int,  help='Limit number of days')
    parser.add_argument('--oracle',  action='store_true',
                        help='Also run oracle (look-ahead) LP for benchmark comparison')
    parser.add_argument('--verbose', action='store_true',
                        help='Print block-level dispatch trace every 16 blocks')
    args = parser.parse_args()
    run_backtest(args)
