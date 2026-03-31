import pandas as pd
import numpy as np
import json
import scipy.stats
import sys
from pathlib import Path
from sklearn.covariance import LedoitWolf

sys.path.append(str(Path(__file__).resolve().parent.parent))

# ── Resolution constant ───────────────────────────────────────────────────────
# 96 for 15-min blocks, 24 for hourly
BLOCKS_PER_DAY = 96


def _normalise_predictions(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Guarantee the DataFrame has:
      - target_date  (str  YYYY-MM-DD)
      - target_block (int  1-96)

    Handles:
      A) delivery_start_ist is the INDEX  -> reset_index, derive block from it
      B) delivery_start_ist is a COLUMN   -> derive block from it
      C) target_block already present     -> nothing to do
      D) target_hour present (hourly)     -> block = hour*4 + 1
    """
    df = df.copy()

    # Promote DatetimeIndex
    idx_name = df.index.name or ""
    if "delivery_start_ist" in idx_name or (
        hasattr(df.index, "dtype") and str(df.index.dtype).startswith("datetime")
    ):
        df = df.reset_index()
        print(f"  [{name}] Reset DatetimeIndex -> delivery_start_ist column.")

    # Derive target_block (1-96)
    if 'target_block' not in df.columns:
        if 'delivery_start_ist' in df.columns:
            ts = pd.to_datetime(df['delivery_start_ist'])
            df['target_block'] = ts.dt.hour * 4 + ts.dt.minute // 15 + 1
            print(f"  [{name}] Derived target_block from delivery_start_ist.")
        elif 'target_hour' in df.columns:
            df['target_block'] = df['target_hour'] * 4 + 1
            print(f"  [{name}] Derived target_block from target_hour.")
        else:
            raise KeyError(
                f"[{name}] Cannot derive target_block. "
                f"Columns: {list(df.columns)}"
            )

    # Ensure target_date as plain string
    if 'target_date' not in df.columns:
        if 'delivery_start_ist' in df.columns:
            df['target_date'] = pd.to_datetime(df['delivery_start_ist']).dt.date.astype(str)
        else:
            raise KeyError(f"[{name}] Cannot find target_date. Columns: {list(df.columns)}")
    df['target_date'] = df['target_date'].astype(str)

    rows_per_date = df.groupby('target_date').size()
    print(f"  [{name}] shape={df.shape}  rows/date: "
          f"min={rows_per_date.min()}, max={rows_per_date.max()}")
    return df


def _compute_pit(actual: float, quantiles_dict: dict) -> float:
    """Map actual price to (0,1) using piecewise-linear quantile interpolation."""
    q_keys = ['q10', 'q25', 'q50', 'q75', 'q90']
    q_levels = [0.10, 0.25, 0.50, 0.75, 0.90]
    q_values = [quantiles_dict.get(k, np.nan) for k in q_keys]
    if any(np.isnan(v) for v in q_values):
        return 0.5
    q_values = list(np.maximum.accumulate(q_values))
    low_ext = q_values[0] - 1.5 * (q_values[1] - q_values[0])
    high_ext = q_values[4] + 1.5 * (q_values[4] - q_values[3])
    q_levels_ext = [0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
    q_values_ext = [low_ext] + q_values + [high_ext]
    u = np.interp(actual, q_values_ext, q_levels_ext)
    return float(np.clip(u, 0.001, 0.999))


def _fit_cross_market_rho_blocks(dam_val_df: pd.DataFrame,
                                  rtm_val_df: pd.DataFrame) -> dict:
    """Per-block (1-BLOCKS_PER_DAY) cross-market Spearman correlation."""
    merged = pd.merge(
        dam_val_df, rtm_val_df,
        on=['target_date', 'target_block'],
        suffixes=('_dam', '_rtm')
    )
    print(f"  Merged validation rows: {len(merged)}  "
          f"common dates: {merged['target_date'].nunique()}")

    z_dam_list, z_rtm_list = [], []
    for _, row in merged.iterrows():
        q_dam = {k: row[f'{k}_dam'] for k in ['q10', 'q25', 'q50', 'q75', 'q90']}
        q_rtm = {k: row[f'{k}_rtm'] for k in ['q10', 'q25', 'q50', 'q75', 'q90']}
        act_dam = row.get('target_mcp_rs_mwh_dam', np.nan)
        act_rtm = row.get('target_mcp_rs_mwh_rtm', np.nan)
        if np.isnan(act_dam) or np.isnan(act_rtm):
            z_dam_list.append(0.0)
            z_rtm_list.append(0.0)
        else:
            z_dam_list.append(scipy.stats.norm.ppf(_compute_pit(act_dam, q_dam)))
            z_rtm_list.append(scipy.stats.norm.ppf(_compute_pit(act_rtm, q_rtm)))

    merged['z_dam'] = z_dam_list
    merged['z_rtm'] = z_rtm_list

    global_rho = float(np.corrcoef(merged['z_dam'], merged['z_rtm'])[0, 1])

    rho_raw = []
    n_obs = []
    for b in range(1, BLOCKS_PER_DAY + 1):
        bd = merged[merged['target_block'] == b]
        n_obs.append(len(bd))
        if len(bd) > 1:
            r = np.corrcoef(bd['z_dam'], bd['z_rtm'])[0, 1]
            rho_raw.append(float(r if np.isfinite(r) else global_rho))
        else:
            rho_raw.append(float(global_rho))

    shrink = 0.3
    rho_shrunk = [
        float(np.clip(
            (1 - shrink) * rho_raw[i] + shrink * global_rho
            if n_obs[i] >= 15 else global_rho,
            -0.99, 0.99
        ))
        for i in range(BLOCKS_PER_DAY)
    ]

    return {
        'rho_by_block': rho_shrunk,
        'rho_by_block_raw': rho_raw,
        'rho_global': global_rho,
        'n_observations_by_block': n_obs,
        'shrinkage_factor': shrink,
        'n_total_observations': len(merged),
        'n_common_dates': int(merged['target_date'].nunique()),
    }


def _estimate_dam_copula_blocks(dam_val_df: pd.DataFrame) -> np.ndarray:
    """BLOCKS_PER_DAY × BLOCKS_PER_DAY Ledoit-Wolf copula correlation matrix."""
    z_data = []
    for _, row in dam_val_df.iterrows():
        q_dict = {k: row[k] for k in ['q10', 'q25', 'q50', 'q75', 'q90']}
        act = row.get('target_mcp_rs_mwh', np.nan)
        if np.isnan(act):
            continue
        u = _compute_pit(act, q_dict)
        z = scipy.stats.norm.ppf(u)
        z_data.append({'date': row['target_date'], 'block': int(row['target_block']), 'z': z})

    z_df = pd.DataFrame(z_data)
    z_pivoted = z_df.pivot(index='date', columns='block', values='z').dropna()
    print(f"  Copula fit on {z_pivoted.shape[0]} complete days, "
          f"{z_pivoted.shape[1]} blocks.")

    lw = LedoitWolf().fit(z_pivoted.values)
    cov = lw.covariance_
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)

    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-6)
    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(corr_psd))
    return corr_psd / np.outer(d, d)


def _estimate_cross_day_rho(dam_val_df: pd.DataFrame) -> float:
    """Lag-1 autocorrelation of daily-average z-scores."""
    z_data = []
    for _, row in dam_val_df.iterrows():
        q_dict = {k: row[k] for k in ['q10', 'q25', 'q50', 'q75', 'q90']}
        act = row.get('target_mcp_rs_mwh', np.nan)
        if np.isnan(act):
            continue
        u = _compute_pit(act, q_dict)
        z_data.append({'date': row['target_date'],
                       'z': float(scipy.stats.norm.ppf(u))})

    z_df = pd.DataFrame(z_data)
    daily_z = z_df.groupby('date')['z'].mean().sort_index()
    if len(daily_z) < 3:
        return 0.3
    return float(np.clip(daily_z.autocorr(lag=1), 0.0, 0.95))


def run_fitting():
    print("============================================================")
    print("FITTING JOINT DAM<->RTM COPULA")
    print(f"Resolution: {BLOCKS_PER_DAY} blocks/day (15-min)")
    print("============================================================")

    pred_dir = Path("Data/Predictions")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    dam_val_path = pred_dir / "dam_quantiles_val.parquet"
    rtm_val_path = pred_dir / "rtm_quantiles_val.parquet"

    if not dam_val_path.exists() or not rtm_val_path.exists():
        print(f"Error: Validation predictions not found in {pred_dir}")
        return

    print("Loading validation predictions...")
    dam_val_raw = pd.read_parquet(dam_val_path)
    rtm_val_raw = pd.read_parquet(rtm_val_path)

    print(f"  DAM val: {dam_val_raw.shape}  index={dam_val_raw.index.name}  "
          f"cols={list(dam_val_raw.columns)}")
    print(f"  RTM val: {rtm_val_raw.shape}  index={rtm_val_raw.index.name}  "
          f"cols={list(rtm_val_raw.columns)}")

    dam_val = _normalise_predictions(dam_val_raw, "DAM-val")
    rtm_val = _normalise_predictions(rtm_val_raw, "RTM-val")

    # Fit rho per block
    print(f"\nFitting cross-market rho ({BLOCKS_PER_DAY} blocks)...")
    rho_result = _fit_cross_market_rho_blocks(dam_val, rtm_val)

    # DAM copula
    print(f"\nEstimating DAM {BLOCKS_PER_DAY}x{BLOCKS_PER_DAY} copula matrix...")
    dam_corr = _estimate_dam_copula_blocks(dam_val)

    # Cross-day rho
    print("\nEstimating cross-day rho...")
    cross_day_rho = _estimate_cross_day_rho(dam_val)
    print(f"Cross-day rho: {cross_day_rho:.3f}")
    if not (0.1 <= cross_day_rho <= 0.9):
        print(f"WARNING: cross_day_rho={cross_day_rho:.3f} outside [0.1, 0.9]")

    # Save
    val_dates = dam_val['target_date'].unique()
    params = {
        "blocks_per_day": BLOCKS_PER_DAY,
        # Primary keys (block-level)
        "rho_by_block": rho_result['rho_by_block'],
        "rho_by_block_raw": rho_result['rho_by_block_raw'],
        # Alias kept so downstream scripts that reference rho_by_hour still work
        "rho_by_hour": rho_result['rho_by_block'],
        "rho_global": rho_result['rho_global'],
        "n_observations_by_block": rho_result['n_observations_by_block'],
        "n_common_dates": rho_result['n_common_dates'],
        "shrinkage_factor": rho_result['shrinkage_factor'],
        "dam_copula_correlation": dam_corr.tolist(),
        "cross_day_rho": cross_day_rho,
        "validation_period": f"{min(val_dates)} to {max(val_dates)}",
    }

    with open(results_dir / "joint_copula_params.json", 'w') as f:
        json.dump(params, f, indent=2)

    # Print summary
    print("\n" + "=" * 62)
    print("JOINT DAM<->RTM COPULA PARAMETERS")
    print(f"Validation: {params['validation_period']}  "
          f"({params['n_common_dates']} dates, "
          f"{rho_result['n_total_observations']} obs)")

    print(f"\n{'Block':<7}{'Time':<8}{'rho_raw':<11}{'rho_shrunk':<13}{'n_obs'}")
    for b in range(BLOCKS_PER_DAY):
        hour = b // 4
        minute = (b % 4) * 15
        print(
            f"{b+1:<7d}"
            f"{hour:02d}:{minute:02d}  "
            f"{rho_result['rho_by_block_raw'][b]:<11.3f}"
            f"{rho_result['rho_by_block'][b]:<13.3f}"
            f"{rho_result['n_observations_by_block'][b]}"
        )

    print(f"\nGlobal rho:       {rho_result['rho_global']:.3f}")
    print(f"Mean rho/block:   {np.mean(rho_result['rho_by_block']):.3f}")
    print(f"Min  rho/block:   {np.min(rho_result['rho_by_block']):.3f}")
    print(f"Max  rho/block:   {np.max(rho_result['rho_by_block']):.3f}")

    print(f"\nDAM Copula: {dam_corr.shape}")
    off = dam_corr[~np.eye(BLOCKS_PER_DAY, dtype=bool)]
    print(f"  off-diagonal mean={np.mean(off):.3f}  "
          f"min={np.min(off):.3f}  max={np.max(off):.3f}")

    print(f"\nCross-Day rho: {cross_day_rho:.3f}")
    print(f"Saved -> results/joint_copula_params.json")
    print("=" * 62)


if __name__ == "__main__":
    run_fitting()
