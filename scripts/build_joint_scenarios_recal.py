import pandas as pd
import numpy as np
import json
import scipy.stats
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.scenarios.joint_copula import inverse_cdf_vectorized

# ── Resolution constant ───────────────────────────────────────────────────────
BLOCKS_PER_DAY = 96


def _normalise_predictions(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Ensure the DataFrame has target_date (str) and target_block (int 1-96).
    Handles delivery_start_ist as index or column, or target_hour fallback.
    """
    df = df.copy()

    # Promote DatetimeIndex
    idx_name = df.index.name or ""
    if "delivery_start_ist" in idx_name or (
        hasattr(df.index, "dtype") and str(df.index.dtype).startswith("datetime")
    ):
        df = df.reset_index()

    # Derive target_block
    if 'target_block' not in df.columns:
        if 'delivery_start_ist' in df.columns:
            ts = pd.to_datetime(df['delivery_start_ist'])
            df['target_block'] = ts.dt.hour * 4 + ts.dt.minute // 15 + 1
        elif 'target_hour' in df.columns:
            df['target_block'] = df['target_hour'] * 4 + 1
        else:
            raise KeyError(
                f"[{name}] Cannot derive target_block. Columns: {list(df.columns)}"
            )

    # Ensure target_date
    if 'target_date' not in df.columns:
        if 'delivery_start_ist' in df.columns:
            df['target_date'] = pd.to_datetime(df['delivery_start_ist']).dt.date.astype(str)
        else:
            raise KeyError(f"[{name}] Cannot find target_date. Columns: {list(df.columns)}")
    df['target_date'] = df['target_date'].astype(str)

    return df


def run_build_scenarios():
    print("============================================================")
    print("BUILDING JOINT DAM<->RTM SCENARIOS")
    print(f"Resolution: {BLOCKS_PER_DAY} blocks/day (15-min)")
    print("============================================================")

    results_dir = Path("results")
    pred_dir = Path("Data/Predictions")
    params_path = results_dir / "joint_copula_params.json"

    if not params_path.exists():
        print(f"Error: Joint copula parameters not found at {params_path}")
        return

    with open(params_path, 'r') as f:
        params = json.load(f)

    # Support both block-level and legacy hour-level param files
    blocks_per_day = params.get('blocks_per_day', BLOCKS_PER_DAY)
    rho_list = params.get('rho_by_block', params.get('rho_by_hour'))
    if len(rho_list) != blocks_per_day:
        print(f"WARNING: rho list length {len(rho_list)} != blocks_per_day {blocks_per_day}. "
              f"Re-run fit_joint_copula.py.")

    dam_corr = np.array(params['dam_copula_correlation'])
    print(f"DAM copula matrix: {dam_corr.shape}")

    # Load backtest quantile predictions
    print("Loading backtest quantile predictions...")

    recal_dam = pred_dir / "dam_quantiles_backtest_recalibrated.parquet"
    recal_rtm = pred_dir / "rtm_quantiles_backtest_recalibrated.parquet"
    raw_dam   = pred_dir / "dam_quantiles_backtest.parquet"
    raw_rtm   = pred_dir / "rtm_quantiles_backtest.parquet"

    dam_path = recal_dam if recal_dam.exists() else raw_dam
    rtm_path = recal_rtm if recal_rtm.exists() else raw_rtm

    print(f"  DAM: {dam_path.name}")
    print(f"  RTM: {rtm_path.name}")

    dam_preds_raw = pd.read_parquet(dam_path)
    rtm_preds_raw = pd.read_parquet(rtm_path)

    dam_preds = _normalise_predictions(dam_preds_raw, "DAM-backtest")
    rtm_preds = _normalise_predictions(rtm_preds_raw, "RTM-backtest")

    print(f"  DAM normalised: {dam_preds.shape}  RTM normalised: {rtm_preds.shape}")

    # Find common dates that have all BLOCKS_PER_DAY blocks
    dam_complete = (
        dam_preds.groupby('target_date')
        .filter(lambda x: len(x) == blocks_per_day)['target_date']
        .unique()
    )
    rtm_complete = (
        rtm_preds.groupby('target_date')
        .filter(lambda x: len(x) == blocks_per_day)['target_date']
        .unique()
    )
    common_dates = sorted(set(dam_complete).intersection(set(rtm_complete)))
    n_days = len(common_dates)

    print(f"Generating scenarios for {n_days} common backtest days "
          f"({blocks_per_day} blocks/day)...")

    if n_days == 0:
        # Debug help
        dam_rpd = dam_preds.groupby('target_date').size()
        rtm_rpd = rtm_preds.groupby('target_date').size()
        print(f"  DAM dates: {len(dam_rpd)}  rows/date sample: "
              f"{dam_rpd.value_counts().head().to_dict()}")
        print(f"  RTM dates: {len(rtm_rpd)}  rows/date sample: "
              f"{rtm_rpd.value_counts().head().to_dict()}")
        print(f"  DAM date sample: {list(dam_preds['target_date'].unique()[:5])}")
        print(f"  RTM date sample: {list(rtm_preds['target_date'].unique()[:5])}")
        print("ERROR: No common dates found. Check that backtest parquets cover "
              "the same date range and have the expected number of blocks per date.")
        return

    # Cholesky decomposition
    L = np.linalg.cholesky(dam_corr)

    n_scenarios = 100
    dam_rows = []
    rtm_rows = []

    for date_idx, date in enumerate(common_dates):
        rng = np.random.default_rng(seed=42 + date_idx)

        dam_day = dam_preds[dam_preds['target_date'] == date].sort_values('target_block')
        rtm_day = rtm_preds[rtm_preds['target_date'] == date].sort_values('target_block')

        # Latent Gaussians (n_scenarios x blocks_per_day)
        z_indep = rng.standard_normal((n_scenarios, blocks_per_day))
        z_dam = z_indep @ L.T

        # Correlated uniforms
        u_dam = scipy.stats.norm.cdf(z_dam)

        # RTM correlated uniforms via per-block rho
        eps_rtm = rng.standard_normal((n_scenarios, blocks_per_day))
        z_rtm = np.zeros_like(z_dam)
        for b in range(blocks_per_day):
            rho = rho_list[b]
            z_rtm[:, b] = rho * z_dam[:, b] + np.sqrt(max(1 - rho**2, 0)) * eps_rtm[:, b]
        u_rtm = scipy.stats.norm.cdf(z_rtm)

        # Map uniforms to prices via inverse CDF
        dam_prices = np.zeros((n_scenarios, blocks_per_day))
        rtm_prices = np.zeros((n_scenarios, blocks_per_day))

        for b in range(blocks_per_day):
            row_d = dam_day.iloc[b]
            q_d = {k: row_d[k] for k in ['q10', 'q25', 'q50', 'q75', 'q90']}
            dam_prices[:, b] = inverse_cdf_vectorized(u_dam[:, b], q_d)

            row_r = rtm_day.iloc[b]
            q_r = {k: row_r[k] for k in ['q10', 'q25', 'q50', 'q75', 'q90']}
            rtm_prices[:, b] = inverse_cdf_vectorized(u_rtm[:, b], q_r)

        # Clamp non-negative (optional: also cap at IEX ceiling)
        dam_prices = np.maximum(dam_prices, 0)
        rtm_prices = np.maximum(rtm_prices, 0)
        # Uncomment next two lines to cap at IEX exchange price ceiling:
        # dam_prices = np.minimum(dam_prices, 10000.0)
        # rtm_prices = np.minimum(rtm_prices, 10000.0)

        # Store as wide rows: one row per (date, scenario)
        # Columns: b01 .. b96  (block 1..96)
        for s in range(n_scenarios):
            dam_rows.append({
                'target_date': date,
                'scenario_id': s,
                **{f'b{b+1:02d}': float(dam_prices[s, b]) for b in range(blocks_per_day)}
            })
            rtm_rows.append({
                'target_date': date,
                'scenario_id': s,
                **{f'b{b+1:02d}': float(rtm_prices[s, b]) for b in range(blocks_per_day)}
            })

        if (date_idx + 1) % 5 == 0 or (date_idx + 1) == n_days:
            print(f"  Generated {date_idx + 1}/{n_days} days")

    # Save
    dam_df = pd.DataFrame(dam_rows)
    rtm_df = pd.DataFrame(rtm_rows)

    out_dam = pred_dir / "joint_dam_scenarios_backtest_recalibrated.parquet"
    out_rtm = pred_dir / "joint_rtm_scenarios_backtest_recalibrated.parquet"

    dam_df.to_parquet(out_dam, index=False)
    dam_df.to_csv(pred_dir / "joint_dam_scenarios_backtest.csv", index=False)
    rtm_df.to_parquet(out_rtm, index=False)
    rtm_df.to_csv(pred_dir / "joint_rtm_scenarios_backtest.csv", index=False)

    print("\nJOINT SCENARIO GENERATION COMPLETE")
    print(f"Backtest dates: {n_days} days ({common_dates[0]} to {common_dates[-1]})")
    print(f"Scenarios per day: {n_scenarios}")
    print(f"Blocks per day:    {blocks_per_day}")

    all_dam = dam_df.filter(regex='^b').values.flatten()
    all_rtm = rtm_df.filter(regex='^b').values.flatten()
    print(f"\nDAM price stats: mean={np.mean(all_dam):.0f}  "
          f"std={np.std(all_dam):.0f}  "
          f"min={np.min(all_dam):.0f}  "
          f"max={np.max(all_dam):.0f}")
    print(f"RTM price stats: mean={np.mean(all_rtm):.0f}  "
          f"std={np.std(all_rtm):.0f}  "
          f"min={np.min(all_rtm):.0f}  "
          f"max={np.max(all_rtm):.0f}")

    print(f"\nSaved -> {out_dam}")
    print(f"Saved -> {out_rtm}")
    print("============================================================")


if __name__ == "__main__":
    run_build_scenarios()
