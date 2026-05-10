"""
Systematic Event Backtester.

Author: Diego Ascencio
Created: 2026-05-09
Project: Systematic Event Backtester


This script tests a narrow empirical question: when pairwise stock correlations
rise during broad selloffs, do the affected stocks mean-revert enough to earn
positive risk-budgeted returns after costs?

The unit of evidence is the stitched market-stress episode, not each raw trigger
or Monte Carlo draw. Candidate APC shocks are consolidated into non-overlapping
episodes, while stock-level entry lots are managed and exited independently
inside those episodes. That structure keeps APC as the event-definition signal
without double-counting the same crisis regime.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass(frozen=True)
class ResearchConfig:
    """Research bounds and reproducibility controls set before the test is run."""

    min_history: int = 252
    apc_window_min: int = 35
    apc_window_max: int = 126
    max_apc_names: int = 125
    n_simulations: int = 500
    investment_unit: float = 1.0
    round_trip_cost: float = 0.002
    seed: int = 7


def load_prices(
    file_path: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
    price_col: str = "price",
) -> pd.DataFrame:
    """Read long or wide price data and return a clean date-by-ticker panel."""
    print(f"[data] Reading the equity price panel: {file_path}")
    df = pd.read_csv(file_path, parse_dates=[date_col], low_memory=False)
    df[date_col] = pd.to_datetime(df[date_col])

    if ticker_col in df.columns and price_col in df.columns:
        print(
            "[data] Long-format observations detected; constructing the "
            "price panel."
        )
        prices = (
            df.pivot_table(
                index=date_col,
                columns=ticker_col,
                values=price_col,
                aggfunc="last",
            )
            .sort_index()
            .apply(pd.to_numeric, errors="coerce")
        )
    else:
        print("[data] Wide-format price panel detected.")
        prices = df.set_index(date_col).sort_index().apply(pd.to_numeric, errors="coerce")

    prices = prices.dropna(axis=1, how="all").dropna(axis=0, how="all")
    print(
        f"[data] Estimation sample: {prices.shape[0]:,} dates x "
        f"{prices.shape[1]:,} securities."
    )
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns while preserving the panel's original missingness."""
    returns = np.log(prices / prices.shift(1))
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def _finite_past(arr: np.ndarray, i: int) -> np.ndarray:
    past = arr[:i]
    return past[np.isfinite(past)]


def expanding_percentile_rank(series: pd.Series, min_history: int) -> pd.Series:
    """Rank each observation against the empirical distribution available before that date."""
    arr = series.to_numpy(dtype=np.float64)
    out = np.full(len(arr), np.nan)
    for i, value in enumerate(arr):
        past = _finite_past(arr, i)
        if len(past) >= min_history and np.isfinite(value):
            out[i] = float(np.mean(past <= value))
    return pd.Series(out, index=series.index, name=f"{series.name}_pct")


def dynamic_expanding_quantile(
    series: pd.Series,
    quantile_series: pd.Series,
    min_history: int,
) -> pd.Series:
    """Evaluate an expanding quantile using only information available before the date."""
    arr = series.to_numpy(dtype=np.float64)
    q_arr = quantile_series.reindex(series.index).to_numpy(dtype=np.float64)
    out = np.full(len(arr), np.nan)
    for i, q in enumerate(q_arr):
        past = _finite_past(arr, i)
        if len(past) >= min_history and np.isfinite(q):
            out[i] = float(np.quantile(past, np.clip(q, 0.01, 0.99)))
    return pd.Series(out, index=series.index, name=f"{series.name}_adaptive_q")


def adaptive_int_from_percentile(
    percentile: pd.Series,
    low: int,
    high: int,
    inverse: bool = False,
) -> pd.Series:
    """Map a state percentile into an integer window within ex ante bounds."""
    p = percentile.fillna(0.5).clip(0.0, 1.0)
    values = high - p * (high - low) if inverse else low + p * (high - low)
    return values.round().astype(int)


def build_market_state(returns: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    """Build the point-in-time market state used by the adaptive rules."""
    market_return = returns.mean(axis=1, skipna=True)
    coverage = returns.notna().sum(axis=1)
    frac_down = (returns.lt(0).sum(axis=1) / coverage.replace(0, np.nan)).rename("frac_down")

    fast_vol = market_return.shift(1).ewm(
        span=cfg.apc_window_min,
        min_periods=max(10, cfg.apc_window_min // 2),
    ).std()
    slow_vol = market_return.shift(1).ewm(
        span=cfg.apc_window_max,
        min_periods=max(30, cfg.apc_window_max // 2),
    ).std()
    vol_ratio = (fast_vol / slow_vol.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    vol_ratio.name = "vol_ratio"
    vol_percentile = expanding_percentile_rank(vol_ratio, cfg.min_history).rename("vol_percentile")
    apc_window = adaptive_int_from_percentile(
        vol_percentile,
        cfg.apc_window_min,
        cfg.apc_window_max,
        inverse=True,
    ).rename("apc_window")

    return pd.concat(
        [
            market_return.rename("market_return"),
            coverage.rename("coverage"),
            frac_down,
            fast_vol.rename("fast_market_vol"),
            slow_vol.rename("slow_market_vol"),
            vol_ratio,
            vol_percentile,
            apc_window,
        ],
        axis=1,
    )


def compute_adaptive_apc(
    returns: pd.DataFrame,
    state: pd.DataFrame,
    cfg: ResearchConfig,
) -> pd.DataFrame:
    """
    Estimate APC with an adaptive lookback and deterministic coverage sample.

    The cross-sectional sample is deterministic so that changes in results are
    attributable to the signal and data, not to random APC subsampling.
    """
    arr = returns.to_numpy(dtype=np.float64)
    dates = returns.index
    tickers = np.array(returns.columns)
    windows = state["apc_window"].reindex(dates).fillna(cfg.apc_window_max).astype(int).to_numpy()

    apc_values = np.full(len(dates), np.nan)
    apc_names = np.zeros(len(dates), dtype=int)

    print("[apc] Estimating average pairwise correlation with adaptive lookbacks.")
    for i in range(len(dates)):
        window = int(windows[i])
        if i + 1 < window:
            continue

        window_data = arr[i - window + 1 : i + 1]
        valid_idx = np.flatnonzero(np.isfinite(window_data).all(axis=0))
        if len(valid_idx) < 4:
            continue

        if len(valid_idx) > cfg.max_apc_names:
            order = np.argsort(tickers[valid_idx].astype(str))
            spaced = np.linspace(0, len(order) - 1, cfg.max_apc_names).round().astype(int)
            valid_idx = valid_idx[order[spaced]]

        wd = window_data[:, valid_idx]
        keep = wd.std(axis=0) > 1e-12
        wd = wd[:, keep]
        if wd.shape[1] < 4:
            continue

        corr = np.corrcoef(wd.T)
        upper = np.triu_indices(corr.shape[0], k=1)
        apc_values[i] = float(np.nanmean(corr[upper]))
        apc_names[i] = int(wd.shape[1])

        if i and i % 1000 == 0:
            print(f"[apc] Processed {i:,}/{len(dates):,} dates in the APC panel.")

    apc = pd.Series(apc_values, index=dates, name="apc")
    print(f"[apc] Completed APC estimation; observed range {apc.min():.3f} to {apc.max():.3f}.")
    return pd.DataFrame({"apc": apc, "apc_n_names": apc_names}, index=dates)


def build_adaptive_thresholds(
    returns: pd.DataFrame,
    state: pd.DataFrame,
    apc_frame: pd.DataFrame,
    cfg: ResearchConfig,
) -> pd.DataFrame:
    """Estimate point-in-time event thresholds from prior empirical distributions."""
    idx = returns.index
    state = state.reindex(idx)
    apc = apc_frame["apc"].reindex(idx)
    vol_pct = state["vol_percentile"].fillna(0.5).clip(0.0, 1.0)

    apc_q = (0.55 + 0.35 * vol_pct).rename("apc_threshold_quantile")
    market_q = (0.20 - 0.15 * vol_pct).rename("market_tail_quantile")
    breadth_q = (0.55 + 0.25 * vol_pct).rename("breadth_quantile")
    coverage_q = (0.20 + 0.15 * (1.0 - vol_pct)).rename("coverage_quantile")

    apc_threshold = dynamic_expanding_quantile(apc.rename("apc"), apc_q, cfg.min_history)
    apc_threshold.name = "apc_threshold"
    market_threshold = dynamic_expanding_quantile(
        state["market_return"].rename("market_return"),
        market_q,
        cfg.min_history,
    )
    market_threshold.name = "market_return_threshold"
    breadth_threshold = dynamic_expanding_quantile(
        state["frac_down"].rename("frac_down"),
        breadth_q,
        cfg.min_history,
    )
    breadth_threshold.name = "frac_down_threshold"
    coverage_floor = dynamic_expanding_quantile(
        state["coverage"].rename("coverage"),
        coverage_q,
        cfg.min_history,
    ).round()
    coverage_floor.name = "coverage_floor"

    apc_excess = (apc - apc_threshold).rename("apc_excess")
    apc_excess_pct = expanding_percentile_rank(apc_excess.clip(lower=0), cfg.min_history)
    market_stress_pct = expanding_percentile_rank(
        (-state["market_return"]).clip(lower=0).rename("market_drop_abs"),
        cfg.min_history,
    )
    breadth_pct = expanding_percentile_rank(state["frac_down"].rename("frac_down"), cfg.min_history)
    severity = pd.concat([apc_excess_pct, market_stress_pct, breadth_pct], axis=1).mean(axis=1)
    severity = severity.where(state["market_return"] < 0).rename("event_severity")

    return pd.concat(
        [
            apc_q,
            market_q,
            breadth_q,
            coverage_q,
            apc_threshold,
            market_threshold,
            breadth_threshold,
            coverage_floor,
            apc_excess,
            apc_excess_pct.rename("apc_excess_percentile"),
            market_stress_pct.rename("market_stress_percentile"),
            breadth_pct.rename("breadth_percentile"),
            severity,
        ],
        axis=1,
    )


def adaptive_episode_bridge_days(
    date: pd.Timestamp,
    state: pd.DataFrame,
    cfg: ResearchConfig,
) -> int:
    """
    Adaptive bridge used to stitch adjacent triggers into one stress episode.

    In volatile regimes, systematic shocks often arrive in waves. The bridge
    therefore treats nearby APC triggers as one unresolved regime instead of
    repeatedly re-opening the same event.
    """
    vol_pct = state.at[date, "vol_percentile"] if date in state.index else np.nan
    apc_window = state.at[date, "apc_window"] if date in state.index else cfg.apc_window_max
    vol_pct = 0.5 if not np.isfinite(vol_pct) else float(np.clip(vol_pct, 0.0, 1.0))
    apc_window = cfg.apc_window_max if not np.isfinite(apc_window) else int(apc_window)
    bridge = int(round(apc_window * (0.15 + 0.45 * vol_pct)))
    return int(np.clip(bridge, max(2, cfg.apc_window_min // 8), cfg.apc_window_max // 2))


def adaptive_stability_days(
    date: pd.Timestamp,
    state: pd.DataFrame,
    cfg: ResearchConfig,
) -> int:
    """Return the number of normalized trading days required to close an episode."""
    vol_pct = state.at[date, "vol_percentile"] if date in state.index else np.nan
    vol_pct = 0.5 if not np.isfinite(vol_pct) else float(np.clip(vol_pct, 0.0, 1.0))
    window = state.at[date, "apc_window"] if date in state.index else cfg.apc_window_max
    window = cfg.apc_window_max if not np.isfinite(window) else int(window)
    stable_days = int(round(window * (0.02 + 0.04 * (1.0 - vol_pct))))
    return int(np.clip(stable_days, 1, max(1, cfg.apc_window_min // 5)))


def find_episode_end_index(
    start_i: int,
    idx: pd.DatetimeIndex,
    state: pd.DataFrame,
    apc_frame: pd.DataFrame,
    thresholds: pd.DataFrame,
    cfg: ResearchConfig,
) -> tuple[int, str, int]:
    """
    Find the natural end of a systematic episode.

    Episodes close after APC and market stress have jointly normalized for a
    state-dependent number of trading days. This separates the research episode
    from the holding period of any individual stock-level lot.
    """
    start_date = idx[start_i]
    stable_needed = adaptive_stability_days(start_date, state, cfg)
    stable_count = 0

    for j in range(start_i + 1, len(idx)):
        date = idx[j]
        apc_today = apc_frame.at[date, "apc"]
        apc_threshold = thresholds.at[date, "apc_threshold"]
        market_return = state.at[date, "market_return"]
        market_threshold = thresholds.at[date, "market_return_threshold"]

        normalized = (
            np.isfinite(apc_today)
            and np.isfinite(apc_threshold)
            and apc_today <= apc_threshold
            and (
                not np.isfinite(market_return)
                or not np.isfinite(market_threshold)
                or market_return >= market_threshold
            )
        )

        stable_count = stable_count + 1 if normalized else 0
        if stable_count >= stable_needed:
            return j, "apc_and_market_normalized", stable_needed

    return len(idx) - 1, "sample_end", stable_needed


def identify_events(
    returns: pd.DataFrame,
    state: pd.DataFrame,
    apc_frame: pd.DataFrame,
    thresholds: pd.DataFrame,
    cfg: ResearchConfig,
) -> pd.DataFrame:
    """
    Identify non-overlapping systematic episodes.

    Candidate trigger dates are stitched into one event when they occur before
    the prior episode has normalized, or inside the adaptive bridge window after
    normalization. The resulting observations are closer to independent market
    regimes than to raw trigger dates.
    """
    idx = returns.index
    state = state.reindex(idx)
    apc_frame = apc_frame.reindex(idx)
    thresholds = thresholds.reindex(idx)

    apc_spike = apc_frame["apc"] > thresholds["apc_threshold"]
    market_down = state["market_return"] < thresholds["market_return_threshold"]
    broad_down = state["frac_down"] > thresholds["frac_down_threshold"]
    enough_data = state["coverage"] >= thresholds["coverage_floor"]
    candidate_mask = apc_spike & market_down & broad_down & enough_data
    candidate_dates = list(idx[candidate_mask.fillna(False)])
    candidate_pos = {date: idx.get_loc(date) for date in candidate_dates}

    print(f"[events] Candidate APC trigger dates before stitching: {len(candidate_dates):,}.")

    events: list[dict] = []
    i = 0
    while i < len(candidate_dates):
        start = candidate_dates[i]
        start_i = candidate_pos[start]
        end_i, exit_reason, stable_days = find_episode_end_index(
            start_i,
            idx,
            state,
            apc_frame,
            thresholds,
            cfg,
        )
        bridge_days = adaptive_episode_bridge_days(start, state, cfg)
        stitched_dates = [start]
        i += 1

        # Merge later triggers when they are still part of the same unresolved
        # APC regime, even if they occur after the first normalization date.
        while i < len(candidate_dates):
            next_start = candidate_dates[i]
            next_i = candidate_pos[next_start]
            bridge_end_i = min(len(idx) - 1, end_i + bridge_days)
            if next_i > bridge_end_i:
                break

            next_end_i, next_reason, next_stable_days = find_episode_end_index(
                next_i,
                idx,
                state,
                apc_frame,
                thresholds,
                cfg,
            )
            stitched_dates.append(next_start)
            if next_end_i >= end_i:
                end_i = next_end_i
                exit_reason = next_reason
                stable_days = max(stable_days, next_stable_days)
            bridge_days = max(bridge_days, adaptive_episode_bridge_days(next_start, state, cfg))
            i += 1

        end = idx[end_i]
        duration = int(end_i - start_i)
        trigger_rows = thresholds.loc[stitched_dates]
        peak_apc_date = apc_frame.loc[stitched_dates, "apc"].idxmax()
        worst_market_date = state.loc[stitched_dates, "market_return"].idxmin()

        events.append(
            {
                "event_date": start,
                "end_date": end,
                "duration_days": duration,
                "exit_reason": exit_reason,
                "stitched_trigger_count": len(stitched_dates),
                "first_trigger_date": stitched_dates[0],
                "last_trigger_date": stitched_dates[-1],
                "episode_bridge_days": bridge_days,
                "stability_days_required": stable_days,
                "market_return": float(state.at[start, "market_return"]),
                "worst_trigger_date": worst_market_date,
                "worst_trigger_market_return": float(state.at[worst_market_date, "market_return"]),
                "apc": float(apc_frame.at[start, "apc"]),
                "peak_apc_date": peak_apc_date,
                "peak_trigger_apc": float(apc_frame.at[peak_apc_date, "apc"]),
                "apc_threshold": float(thresholds.at[start, "apc_threshold"]),
                "apc_window": int(state.at[start, "apc_window"]),
                "apc_n_names": int(apc_frame.at[start, "apc_n_names"]),
                "frac_down": float(state.at[start, "frac_down"]),
                "frac_down_threshold": float(thresholds.at[start, "frac_down_threshold"]),
                "coverage": int(state.at[start, "coverage"]),
                "coverage_floor": float(thresholds.at[start, "coverage_floor"]),
                "vol_percentile": float(state.at[start, "vol_percentile"]),
                "event_severity": float(trigger_rows["event_severity"].max()),
            }
        )

    event_df = pd.DataFrame(events)
    print(f"[events] Independent stitched episodes retained: {len(event_df):,}.")
    if not event_df.empty:
        print(
            "[events] Episode duration range: "
            f"{event_df.duration_days.min()} to {event_df.duration_days.max()} trading days; "
            f"median={event_df.duration_days.median():.0f}."
        )
        print(
            "[events] Trigger stitching diagnostics: "
            f"mean={event_df.stitched_trigger_count.mean():.1f}; "
            f"max={event_df.stitched_trigger_count.max()}."
        )
    return event_df


def estimate_adaptive_stock_volatility(
    returns: pd.DataFrame,
    state: pd.DataFrame,
    cfg: ResearchConfig,
) -> pd.DataFrame:
    """Blend fast and slow stock-volatility estimates by market regime."""
    shifted = returns.shift(1)
    fast = shifted.rolling(
        cfg.apc_window_min,
        min_periods=max(10, cfg.apc_window_min // 2),
    ).std()
    slow = shifted.rolling(
        cfg.apc_window_max,
        min_periods=max(30, cfg.apc_window_max // 2),
    ).std()
    w = state["vol_percentile"].reindex(returns.index).fillna(0.5).clip(0.0, 1.0)
    return fast.mul(w, axis=0).add(slow.mul(1.0 - w, axis=0)).replace([np.inf, -np.inf], np.nan)


def estimate_adaptive_betas(
    returns: pd.DataFrame,
    state: pd.DataFrame,
    cfg: ResearchConfig,
) -> pd.DataFrame:
    """Estimate point-in-time market betas with regime-dependent smoothing."""
    market_return = state["market_return"].reindex(returns.index)
    stock_shifted = returns.shift(1)
    market_shifted = market_return.shift(1)

    def beta_for_window(window: int, min_periods: int) -> pd.DataFrame:
        cov = stock_shifted.rolling(window, min_periods=min_periods).cov(market_shifted)
        var = market_shifted.rolling(window, min_periods=min_periods).var()
        return cov.div(var.replace(0, np.nan), axis=0)

    fast = beta_for_window(cfg.apc_window_min, max(10, cfg.apc_window_min // 2))
    slow = beta_for_window(cfg.apc_window_max, max(30, cfg.apc_window_max // 2))
    w = state["vol_percentile"].reindex(returns.index).fillna(0.5).clip(0.0, 1.0)
    return fast.mul(w, axis=0).add(slow.mul(1.0 - w, axis=0)).replace([np.inf, -np.inf], np.nan)


def build_drop_z_threshold(
    returns: pd.DataFrame,
    stock_vol: pd.DataFrame,
    state: pd.DataFrame,
    cfg: ResearchConfig,
) -> pd.Series:
    """Estimate the entry cutoff from prior cross-sectional downside shocks."""
    z = returns.abs().div(stock_vol.replace(0, np.nan)).where(returns < 0)
    daily_tail = z.quantile(0.75, axis=1, interpolation="linear").rename("daily_negative_z_tail")
    vol_pct = state["vol_percentile"].reindex(returns.index).fillna(0.5).clip(0.0, 1.0)
    z_q = (0.60 + 0.30 * vol_pct).rename("z_threshold_quantile")
    z_threshold = dynamic_expanding_quantile(daily_tail, z_q, cfg.min_history)
    z_threshold.name = "drop_z_threshold"
    return z_threshold


def build_adaptive_lot_return_cap(
    returns: pd.DataFrame,
    state: pd.DataFrame,
    cfg: ResearchConfig,
) -> pd.Series:
    """
    Point-in-time cap for a single lot's positive payoff.

    The source data are not filtered. Instead, the estimator limits how much a
    single security can contribute when its forward payoff lies far beyond the
    prior cross-sectional positive-return tail.
    """
    vol_pct = state["vol_percentile"].reindex(returns.index).fillna(0.5).clip(0.0, 1.0)
    positive = returns.where(returns > 0)
    q85 = positive.quantile(0.85, axis=1, interpolation="linear")
    q95 = positive.quantile(0.95, axis=1, interpolation="linear")
    q99 = positive.quantile(0.99, axis=1, interpolation="linear")
    daily_tail = q85.mul(1.0 - vol_pct).add(q99.mul(vol_pct)).clip(lower=q95)
    daily_tail.name = "daily_positive_log_return_tail"

    cap_q = (0.70 + 0.25 * vol_pct).rename("lot_return_cap_quantile")
    cap = dynamic_expanding_quantile(daily_tail, cap_q, cfg.min_history)
    cap.name = "adaptive_daily_lot_log_return_cap"
    return cap.clip(lower=0)


def build_event_windows(events: pd.DataFrame, price_index: pd.DatetimeIndex) -> list[dict]:
    """Convert event metadata into contiguous windows on the price index."""
    windows: list[dict] = []
    for _, event in events.iterrows():
        start = pd.Timestamp(event["event_date"])
        end = pd.Timestamp(event["end_date"])
        dates = price_index[(price_index >= start) & (price_index <= end)]
        if len(dates) >= 2:
            windows.append({"event": event, "dates": dates})
    return windows


def adaptive_portfolio_size(eligible_count: int, frac_down: float) -> int:
    """Choose portfolio breadth as a function of eligible names and selloff breadth."""
    if eligible_count <= 0:
        return 0
    breadth = 0.5 if not np.isfinite(frac_down) else float(np.clip(frac_down, 0.0, 1.0))
    target = int(round(np.sqrt(eligible_count) * (1.0 + breadth)))
    return int(np.clip(target, 1, eligible_count))


def build_adaptive_trade_horizon(
    events: pd.DataFrame,
    state: pd.DataFrame,
    price_index: pd.DatetimeIndex,
    cfg: ResearchConfig,
) -> pd.Series:
    """
    Build a point-in-time lot-level trade horizon.

    Stitched episodes define the independent research observation. They should
    not mechanically define how long each entry lot is held. The lot horizon
    is learned from prior completed episode durations and blended with the
    current APC window and volatility regime.
    """
    events_sorted = events.sort_values("end_date").reset_index(drop=True)
    prior_durations: list[float] = []
    event_pointer = 0
    horizons: list[int] = []

    for date in price_index:
        while event_pointer < len(events_sorted):
            end_date = pd.Timestamp(events_sorted.at[event_pointer, "end_date"])
            if end_date >= date:
                break
            prior_durations.append(float(events_sorted.at[event_pointer, "duration_days"]))
            event_pointer += 1

        vol_pct = state.at[date, "vol_percentile"] if date in state.index else np.nan
        vol_pct = 0.5 if not np.isfinite(vol_pct) else float(np.clip(vol_pct, 0.0, 1.0))

        apc_window = state.at[date, "apc_window"] if date in state.index else cfg.apc_window_max
        apc_window = cfg.apc_window_max if not np.isfinite(apc_window) else int(apc_window)

        if len(prior_durations) >= 5:
            empirical_q = 0.45 + 0.35 * vol_pct
            empirical_horizon = float(np.quantile(prior_durations, empirical_q))
        elif prior_durations:
            empirical_horizon = float(np.median(prior_durations))
        else:
            empirical_horizon = apc_window * (0.30 + 0.45 * vol_pct)

        regime_horizon = apc_window * (0.20 + 0.55 * vol_pct)
        raw_horizon = 0.50 * empirical_horizon + 0.50 * regime_horizon

        # These bounds scale with the current APC window; they are governance
        # constraints on an adaptive horizon rather than fitted constants.
        horizon_floor = max(1, int(round(apc_window * (0.05 + 0.03 * vol_pct))))
        horizon_cap = max(horizon_floor, int(round(apc_window * (0.65 + 0.55 * vol_pct))))
        horizon = int(np.clip(round(raw_horizon), horizon_floor, horizon_cap))
        horizons.append(max(1, horizon))

    return pd.Series(horizons, index=price_index, name="adaptive_trade_horizon_days")


def is_locally_normalized(
    row: int,
    price_index: pd.DatetimeIndex,
    state: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> bool:
    """Check whether APC and market stress have normalized on a given date."""
    date = price_index[row]
    if date not in state.index or date not in thresholds.index:
        return False

    apc_today = thresholds.at[date, "apc_excess"]
    market_return = state.at[date, "market_return"]
    market_threshold = thresholds.at[date, "market_return_threshold"]

    apc_ok = np.isfinite(apc_today) and apc_today <= 0
    market_ok = (
        not np.isfinite(market_return)
        or not np.isfinite(market_threshold)
        or market_return >= market_threshold
    )
    return bool(apc_ok and market_ok)


def find_lot_exit_row(
    entry_row: int,
    episode_exit_row: int,
    price_index: pd.DatetimeIndex,
    state: pd.DataFrame,
    thresholds: pd.DataFrame,
    trade_horizon_arr: np.ndarray,
    cfg: ResearchConfig,
) -> tuple[int, str]:
    """
    Find the adaptive exit row for a single entry lot.

    The lot exits on the earliest of local normalization, adaptive trade
    horizon, or the end of the stitched episode. This keeps episode identity
    separate from the realized holding period of each security.
    """
    if episode_exit_row <= entry_row:
        return episode_exit_row, "episode_end"

    horizon = trade_horizon_arr[entry_row] if entry_row < len(trade_horizon_arr) else np.nan
    if not np.isfinite(horizon) or horizon <= 0:
        date = price_index[entry_row]
        apc_window = state.at[date, "apc_window"] if date in state.index else cfg.apc_window_max
        horizon = apc_window if np.isfinite(apc_window) else cfg.apc_window_max

    horizon_end_row = min(episode_exit_row, entry_row + max(1, int(round(horizon))))
    stable_needed = adaptive_stability_days(price_index[entry_row], state, cfg)
    stable_count = 0

    for row in range(entry_row + 1, horizon_end_row + 1):
        if is_locally_normalized(row, price_index, state, thresholds):
            stable_count += 1
        else:
            stable_count = 0

        if stable_count >= stable_needed:
            return row, "local_normalization"

    if horizon_end_row < episode_exit_row:
        return horizon_end_row, "adaptive_trade_horizon"
    return horizon_end_row, "episode_end"


def estimate_episode_risk_budget(
    event: pd.Series,
    portfolio_size: int,
    event_trade_horizon: float,
    cfg: ResearchConfig,
) -> float:
    """
    Estimate an ex-ante episode risk budget for headline return reporting.

    Deployed-capital returns remain in the output, but the primary `return_pct`
    uses max(actual capital deployed, this adaptive budget) as the denominator.
    This prevents sparse deployment from mechanically inflating the headline
    event return while preserving the full underlying sample.
    """
    duration = max(1.0, float(event["duration_days"]))
    horizon = max(1.0, float(event_trade_horizon)) if np.isfinite(event_trade_horizon) else duration
    severity = float(event["event_severity"]) if np.isfinite(event["event_severity"]) else 0.5
    trigger_count = max(1.0, float(event.get("stitched_trigger_count", 1)))

    cycle_scale = max(1.0, duration / horizon)
    trigger_scale = max(1.0, np.sqrt(trigger_count))
    activity_scale = max(cycle_scale, trigger_scale)
    return float(cfg.investment_unit * portfolio_size * (0.5 + severity) * activity_scale)


def simulate_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    events: pd.DataFrame,
    state: pd.DataFrame,
    thresholds: pd.DataFrame,
    stock_vol: pd.DataFrame,
    betas: pd.DataFrame,
    drop_z_threshold: pd.Series,
    trade_horizon: pd.Series,
    lot_return_cap: pd.Series,
    cfg: ResearchConfig,
) -> pd.DataFrame:
    """Simulate stock-selection uncertainty within each stitched episode."""
    if events.empty:
        return pd.DataFrame()

    aligned_returns = returns.reindex(prices.index)
    aligned_state = state.reindex(prices.index)
    aligned_thresholds = thresholds.reindex(prices.index)
    aligned_stock_vol = stock_vol.reindex(prices.index)
    aligned_betas = betas.reindex(prices.index)
    aligned_z_threshold = drop_z_threshold.reindex(prices.index)
    aligned_trade_horizon = trade_horizon.reindex(prices.index).ffill().bfill()
    aligned_lot_return_cap = lot_return_cap.reindex(prices.index).ffill()

    prices_arr = prices.to_numpy(dtype=np.float64)
    returns_arr = aligned_returns.to_numpy(dtype=np.float64)
    stock_vol_arr = aligned_stock_vol.to_numpy(dtype=np.float64)
    beta_arr = aligned_betas.to_numpy(dtype=np.float64)
    z_threshold_arr = aligned_z_threshold.to_numpy(dtype=np.float64)
    trade_horizon_arr = aligned_trade_horizon.to_numpy(dtype=np.float64)
    lot_return_cap_arr = aligned_lot_return_cap.to_numpy(dtype=np.float64)
    severity_arr = aligned_thresholds["event_severity"].to_numpy(dtype=np.float64)
    market_arr = aligned_state["market_return"].to_numpy(dtype=np.float64)

    tickers = np.array(prices.columns)
    date_idx = {date: i for i, date in enumerate(prices.index)}
    rng = np.random.default_rng(cfg.seed)
    results: list[dict] = []

    windows = build_event_windows(events, prices.index)
    print(
        "[simulation] Sampling stock portfolios within "
        f"{len(windows):,} stitched episodes; {cfg.n_simulations:,} draws per episode."
    )

    for window in windows:
        event = window["event"]
        event_dates = window["dates"]
        start = pd.Timestamp(event["event_date"])
        end = pd.Timestamp(event["end_date"])
        start_i = date_idx[start]
        exit_i = date_idx[end]
        event_row_idx = [date_idx[d] for d in event_dates]

        betas_start = beta_arr[start_i]
        prices_start = prices_arr[start_i]
        systematic_component = betas_start * market_arr[start_i]
        eligible_idx = np.flatnonzero(
            np.isfinite(prices_start)
            & (prices_start > 0)
            & np.isfinite(betas_start)
            & (betas_start > 0)
            & (systematic_component < 0)
        )

        portfolio_size = adaptive_portfolio_size(len(eligible_idx), float(event["frac_down"]))
        if portfolio_size == 0:
            continue

        event_horizons = trade_horizon_arr[event_row_idx]
        event_horizons = event_horizons[np.isfinite(event_horizons) & (event_horizons > 0)]
        event_trade_horizon = float(np.median(event_horizons)) if len(event_horizons) else float(event["duration_days"])
        episode_risk_budget = estimate_episode_risk_budget(event, portfolio_size, event_trade_horizon, cfg)

        for sim_id in range(cfg.n_simulations):
            chosen_idx = rng.choice(eligible_idx, size=portfolio_size, replace=False)
            chosen_tickers = tickers[chosen_idx]

            invested = np.zeros(portfolio_size)
            lowest_purchase_price = np.full(portfolio_size, np.inf)
            weighted_holding_days = 0.0
            n_trade_days = 0
            total_invested = 0.0
            raw_gross_pnl = 0.0
            gross_pnl = 0.0
            transaction_cost = 0.0
            lot_return_caps = 0
            capped_pnl_reduction = 0.0
            exit_counts = {
                "local_normalization": 0,
                "adaptive_trade_horizon": 0,
                "episode_end": 0,
            }
            exit_cache: dict[int, tuple[int, str]] = {}

            for row in event_row_idx[1:-1]:
                z_min = z_threshold_arr[row]
                if not np.isfinite(z_min):
                    continue

                severity = severity_arr[row]
                if not np.isfinite(severity) or severity <= 0:
                    severity = float(event["event_severity"]) if np.isfinite(event["event_severity"]) else 0.5

                z_scores = np.zeros(portfolio_size)
                vol_scales = np.ones(portfolio_size)
                valid_mask = np.zeros(portfolio_size, dtype=bool)
                row_vols = stock_vol_arr[row, chosen_idx]
                positive_vols = row_vols[np.isfinite(row_vols) & (row_vols > 0)]
                median_vol = np.nanmedian(positive_vols) if len(positive_vols) else np.nan

                for k, col in enumerate(chosen_idx):
                    ret_today = returns_arr[row, col]
                    price_today = prices_arr[row, col]
                    sigma_today = stock_vol_arr[row, col]

                    if (
                        not np.isfinite(ret_today)
                        or ret_today >= 0
                        or not np.isfinite(price_today)
                        or price_today <= 0
                        or not np.isfinite(sigma_today)
                        or sigma_today <= 0
                    ):
                        continue

                    if price_today >= lowest_purchase_price[k]:
                        continue

                    z_i = abs(ret_today) / sigma_today
                    if z_i < z_min:
                        continue

                    z_scores[k] = z_i
                    if np.isfinite(median_vol) and median_vol > 0:
                        vol_scales[k] = np.clip(median_vol / sigma_today, 0.25, 4.0)
                    valid_mask[k] = True

                if not valid_mask.any():
                    continue

                if row not in exit_cache:
                    exit_cache[row] = find_lot_exit_row(
                        entry_row=row,
                        episode_exit_row=exit_i,
                        price_index=prices.index,
                        state=aligned_state,
                        thresholds=aligned_thresholds,
                        trade_horizon_arr=trade_horizon_arr,
                        cfg=cfg,
                    )
                lot_exit_row, lot_exit_reason = exit_cache[row]

                z_valid = z_scores[valid_mask]
                z_norm = z_valid / np.nanmean(z_valid) if np.nanmean(z_valid) > 0 else np.ones_like(z_valid)
                j = 0
                for k, col in enumerate(chosen_idx):
                    if not valid_mask[k]:
                        continue
                    price_today = prices_arr[row, col]
                    amount = cfg.investment_unit * (0.5 + severity) * z_norm[j] * vol_scales[k]
                    exit_price = prices_arr[lot_exit_row, col]
                    if not np.isfinite(exit_price) or exit_price <= 0:
                        prior_prices = prices_arr[: lot_exit_row + 1, col]
                        prior_prices = prior_prices[np.isfinite(prior_prices) & (prior_prices > 0)]
                        exit_price = prior_prices[-1] if len(prior_prices) else 0.0

                    shares = amount / price_today
                    raw_lot_return = (exit_price / price_today) - 1.0 if price_today > 0 else 0.0
                    hold_days = max(1, lot_exit_row - row)
                    daily_log_cap = lot_return_cap_arr[row] if row < len(lot_return_cap_arr) else np.nan
                    lot_return = raw_lot_return

                    if np.isfinite(daily_log_cap) and daily_log_cap > 0:
                        scaled_cap = np.expm1(daily_log_cap * np.sqrt(hold_days))
                        if np.isfinite(scaled_cap) and raw_lot_return > scaled_cap:
                            lot_return = float(scaled_cap)
                            lot_return_caps += 1
                            capped_pnl_reduction += amount * (raw_lot_return - lot_return)

                    raw_gross_pnl += amount * raw_lot_return
                    gross_pnl += amount * lot_return
                    transaction_cost += amount * cfg.round_trip_cost
                    total_invested += amount
                    invested[k] += amount
                    lowest_purchase_price[k] = price_today
                    weighted_holding_days += amount * max(0, lot_exit_row - row)
                    n_trade_days += 1
                    exit_counts[lot_exit_reason] = exit_counts.get(lot_exit_reason, 0) + 1
                    j += 1

            base_row = {
                "event_date": start,
                "end_date": end,
                "duration_days": int(event["duration_days"]),
                "exit_reason": event["exit_reason"],
                "stitched_trigger_count": int(event.get("stitched_trigger_count", 1)),
                "market_return": float(event["market_return"]),
                "worst_trigger_market_return": float(event.get("worst_trigger_market_return", event["market_return"])),
                "apc": float(event["apc"]),
                "peak_trigger_apc": float(event.get("peak_trigger_apc", event["apc"])),
                "apc_threshold": float(event["apc_threshold"]),
                "event_severity": float(event["event_severity"]),
                "eligible_count": int(len(eligible_idx)),
                "adaptive_portfolio_size": int(portfolio_size),
                "adaptive_trade_horizon_days": float(event_trade_horizon),
                "episode_risk_budget": float(episode_risk_budget),
                "sim_id": sim_id,
                "tickers": "|".join(chosen_tickers.astype(str)),
            }

            if total_invested <= 0:
                results.append(
                    {
                        **base_row,
                        "total_invested": 0.0,
                        "raw_gross_pnl": 0.0,
                        "gross_pnl": 0.0,
                        "transaction_cost": 0.0,
                        "net_pnl": 0.0,
                        "capped_pnl_reduction": 0.0,
                        "deployed_return_pct": 0.0,
                        "return_on_budget_pct": 0.0,
                        "return_pct": 0.0,
                        "lot_return_caps": 0,
                        "n_stocks_invested": 0,
                        "n_trade_days": 0,
                        "avg_hold_days": float(event["duration_days"]),
                        "local_normalization_exits": 0,
                        "adaptive_horizon_exits": 0,
                        "episode_end_exits": 0,
                    }
                )
                continue

            net_pnl = gross_pnl - transaction_cost
            avg_hold = weighted_holding_days / total_invested if total_invested > 0 else np.nan
            deployed_return_pct = net_pnl / total_invested * 100.0
            budget_denominator = max(total_invested, episode_risk_budget)
            return_on_budget_pct = net_pnl / budget_denominator * 100.0
            results.append(
                {
                    **base_row,
                    "total_invested": total_invested,
                    "raw_gross_pnl": float(raw_gross_pnl),
                    "gross_pnl": float(gross_pnl),
                    "transaction_cost": float(transaction_cost),
                    "net_pnl": float(net_pnl),
                    "capped_pnl_reduction": float(capped_pnl_reduction),
                    "deployed_return_pct": float(deployed_return_pct),
                    "return_on_budget_pct": float(return_on_budget_pct),
                    "return_pct": float(return_on_budget_pct),
                    "lot_return_caps": int(lot_return_caps),
                    "n_stocks_invested": int((invested > 0).sum()),
                    "n_trade_days": int(n_trade_days),
                    "avg_hold_days": float(avg_hold),
                    "local_normalization_exits": int(exit_counts.get("local_normalization", 0)),
                    "adaptive_horizon_exits": int(exit_counts.get("adaptive_trade_horizon", 0)),
                    "episode_end_exits": int(exit_counts.get("episode_end", 0)),
                }
            )

    result_df = pd.DataFrame(results)
    print(f"[simulation] Generated {len(result_df):,} episode-simulation observations.")
    return result_df


def compute_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Collapse simulation draws to the independent episode level."""
    if results.empty:
        return pd.DataFrame()

    summary = (
        results.groupby("event_date", as_index=False)
        .agg(
            end_date=("end_date", "first"),
            duration_days=("duration_days", "first"),
            exit_reason=("exit_reason", "first"),
            stitched_trigger_count=("stitched_trigger_count", "first"),
            market_return=("market_return", "first"),
            worst_trigger_market_return=("worst_trigger_market_return", "first"),
            apc=("apc", "first"),
            peak_trigger_apc=("peak_trigger_apc", "first"),
            apc_threshold=("apc_threshold", "first"),
            event_severity=("event_severity", "first"),
            eligible_count=("eligible_count", "first"),
            adaptive_portfolio_size=("adaptive_portfolio_size", "first"),
            adaptive_trade_horizon_days=("adaptive_trade_horizon_days", "first"),
            mean_episode_risk_budget=("episode_risk_budget", "mean"),
            mean_return_pct=("return_pct", "mean"),
            median_return_pct=("return_pct", "median"),
            trimmed_mean_return_pct=("return_pct", lambda x: stats.trim_mean(x.dropna(), 0.10) if x.notna().sum() else np.nan),
            p05_return_pct=("return_pct", lambda x: x.quantile(0.05)),
            p95_return_pct=("return_pct", lambda x: x.quantile(0.95)),
            std_return_pct=("return_pct", "std"),
            mean_deployed_return_pct=("deployed_return_pct", "mean"),
            median_deployed_return_pct=("deployed_return_pct", "median"),
            mean_return_on_budget_pct=("return_on_budget_pct", "mean"),
            win_rate=("return_pct", lambda x: (x > 0).mean()),
            deployment_rate=("total_invested", lambda x: (x > 0).mean()),
            mean_invested=("total_invested", "mean"),
            mean_raw_gross_pnl=("raw_gross_pnl", "mean"),
            mean_net_pnl=("net_pnl", "mean"),
            mean_capped_pnl_reduction=("capped_pnl_reduction", "mean"),
            mean_lot_return_caps=("lot_return_caps", "mean"),
            total_lot_return_caps=("lot_return_caps", "sum"),
            mean_transaction_cost=("transaction_cost", "mean"),
            mean_hold_days=("avg_hold_days", "mean"),
            local_normalization_exits=("local_normalization_exits", "sum"),
            adaptive_horizon_exits=("adaptive_horizon_exits", "sum"),
            episode_end_exits=("episode_end_exits", "sum"),
            n_simulations=("sim_id", "count"),
        )
        .sort_values("event_date")
    )
    summary["headline_return_pct"] = summary["trimmed_mean_return_pct"]
    summary["raw_mean_return_pct"] = summary["mean_return_pct"]
    summary["event_sharpe"] = summary["mean_return_pct"] / summary["std_return_pct"].replace(0, np.nan)
    return summary


def event_level_research_report(summary: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Compute inference on stitched episodes, not on Monte Carlo rows."""
    if summary.empty:
        return pd.DataFrame(columns=["metric", "value"])

    headline_col = "headline_return_pct" if "headline_return_pct" in summary.columns else "mean_return_pct"
    investable = summary.loc[summary["deployment_rate"] > 0, headline_col].dropna()
    raw_means = summary.loc[summary["deployment_rate"] > 0, "mean_return_pct"].dropna()
    medians = summary.loc[summary["deployment_rate"] > 0, "median_return_pct"].dropna()
    trimmed = summary.loc[summary["deployment_rate"] > 0, "trimmed_mean_return_pct"].dropna()
    deployed = summary.loc[summary["deployment_rate"] > 0, "mean_deployed_return_pct"].dropna()
    rng = np.random.default_rng(seed)

    metrics: list[tuple[str, float | int | str]] = [
        ("n_events", int(len(summary))),
        ("n_investable_events", int(len(investable))),
        ("first_event", str(pd.to_datetime(summary["event_date"].min()).date())),
        ("last_event", str(pd.to_datetime(summary["event_date"].max()).date())),
        ("mean_event_return_pct", float(investable.mean()) if len(investable) else np.nan),
        ("median_event_return_pct", float(investable.median()) if len(investable) else np.nan),
        ("mean_raw_event_return_pct", float(raw_means.mean()) if len(raw_means) else np.nan),
        ("mean_deployed_event_return_pct", float(deployed.mean()) if len(deployed) else np.nan),
        ("mean_event_median_return_pct", float(medians.mean()) if len(medians) else np.nan),
        ("mean_event_trimmed_return_pct", float(trimmed.mean()) if len(trimmed) else np.nan),
        ("mean_lot_return_caps", float(summary["mean_lot_return_caps"].mean()) if "mean_lot_return_caps" in summary.columns else np.nan),
        ("total_lot_return_caps", int(summary["total_lot_return_caps"].sum()) if "total_lot_return_caps" in summary.columns else 0),
        ("mean_capped_pnl_reduction", float(summary["mean_capped_pnl_reduction"].mean()) if "mean_capped_pnl_reduction" in summary.columns else np.nan),
        ("event_hit_rate", float((investable > 0).mean()) if len(investable) else np.nan),
        ("mean_deployment_rate", float(summary["deployment_rate"].mean())),
        ("mean_duration_days", float(summary["duration_days"].mean())),
        ("median_duration_days", float(summary["duration_days"].median())),
        ("mean_stitched_trigger_count", float(summary["stitched_trigger_count"].mean())),
        ("max_stitched_trigger_count", int(summary["stitched_trigger_count"].max())),
    ]

    if len(investable) >= 2:
        t_stat, p_two_sided = stats.ttest_1samp(investable, 0.0, nan_policy="omit")
        boot = np.array(
            [
                rng.choice(investable.to_numpy(), size=len(investable), replace=True).mean()
                for _ in range(10000)
            ]
        )
        metrics.extend(
            [
                ("event_level_t_stat", float(t_stat)),
                ("event_level_one_sided_p_value", float(p_two_sided / 2.0)),
                ("bootstrap_mean_ci_2p5", float(np.quantile(boot, 0.025))),
                ("bootstrap_mean_ci_97p5", float(np.quantile(boot, 0.975))),
                ("bootstrap_prob_mean_le_0", float(np.mean(boot <= 0.0))),
            ]
        )

    return pd.DataFrame(metrics, columns=["metric", "value"])


def print_report(summary: pd.DataFrame, report: pd.DataFrame) -> None:
    """Print a compact empirical summary for the terminal."""
    if summary.empty or report.empty:
        print(
            "[report] No empirical report is available because no investable "
            "episodes were produced."
        )
        return

    metric = dict(zip(report["metric"], report["value"]))
    print("\n" + "=" * 72)
    print("  ADAPTIVE APC EVENT-STUDY REPORT")
    print("=" * 72)
    print("Empirical sample")
    print("-" * 72)
    print(f"Stitched APC episodes              : {metric.get('n_events')}")
    print(f"Investable episodes                : {metric.get('n_investable_events')}")
    print(
        f"Sample period                      : {metric.get('first_event')} "
        f"to {metric.get('last_event')}"
    )
    print(
        "Mean / median duration             : "
        f"{float(metric.get('mean_duration_days', np.nan)):6.1f} / "
        f"{float(metric.get('median_duration_days', np.nan)):6.1f} trading days"
    )
    print(
        "Mean / max stitched triggers       : "
        f"{float(metric.get('mean_stitched_trigger_count', np.nan)):6.1f} / "
        f"{metric.get('max_stitched_trigger_count')}"
    )
    print("\nEpisode-level return estimates")
    print("-" * 72)
    print(
        "Robust mean return                 : "
        f"{float(metric.get('mean_event_return_pct', np.nan)):8.2f}%"
    )
    print(
        "Robust median return               : "
        f"{float(metric.get('median_event_return_pct', np.nan)):8.2f}%"
    )
    print(
        "Raw mean risk-budget return        : "
        f"{float(metric.get('mean_raw_event_return_pct', np.nan)):8.2f}%"
    )
    print(
        "Mean deployed-capital return       : "
        f"{float(metric.get('mean_deployed_event_return_pct', np.nan)):8.2f}%"
    )
    print(
        "Mean of episode medians            : "
        f"{float(metric.get('mean_event_median_return_pct', np.nan)):8.2f}%"
    )
    print(
        "Mean trimmed episode return        : "
        f"{float(metric.get('mean_event_trimmed_return_pct', np.nan)):8.2f}%"
    )
    print(f"Event hit rate                     : {float(metric.get('event_hit_rate', np.nan)):8.1%}")
    print(f"Mean deployment rate               : {float(metric.get('mean_deployment_rate', np.nan)):8.1%}")
    print(f"Lot payoff caps applied            : {metric.get('total_lot_return_caps')}")

    if "event_level_t_stat" in metric:
        print("\nInference on independent episodes")
        print("-" * 72)
        print(
            "One-sample t-statistic             : "
            f"{float(metric['event_level_t_stat']):8.3f}"
        )
        print(
            "One-sided p-value                  : "
            f"{float(metric['event_level_one_sided_p_value']):8.4f}"
        )
        print(
            "Bootstrap mean 95% interval        : "
            f"[{float(metric['bootstrap_mean_ci_2p5']):.2f}%, "
            f"{float(metric['bootstrap_mean_ci_97p5']):.2f}%]"
        )

    print("\nLargest robust-return episodes")
    print("-" * 72)
    ranking_col = (
        "headline_return_pct"
        if "headline_return_pct" in summary.columns
        else "mean_return_pct"
    )
    cols = [
        "event_date",
        "end_date",
        "duration_days",
        "stitched_trigger_count",
        ranking_col,
        "mean_return_pct",
        "win_rate",
    ]
    print(
        summary.nlargest(5, ranking_col)[cols].to_string(
            index=False,
            float_format="{:.2f}".format,
        )
    )

    print("\nSmallest robust-return episodes")
    print("-" * 72)
    print(
        summary.nsmallest(5, ranking_col)[cols].to_string(
            index=False,
            float_format="{:.2f}".format,
        )
    )
    print("=" * 72 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an adaptive APC event study on a historical equity price panel"
    )
    parser.add_argument("--file", required=True, help="Path to the historical price CSV")
    parser.add_argument(
        "--date-col",
        default="date",
        help="Date column used to index observations",
    )
    parser.add_argument(
        "--ticker-col",
        default="ticker",
        help="Ticker identifier for long-format data",
    )
    parser.add_argument(
        "--price-col",
        default="price",
        help="Adjusted price column for long-format data",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=252,
        help="Minimum prior observations required for adaptive empirical thresholds",
    )
    parser.add_argument(
        "--apc-window-min",
        type=int,
        default=35,
        help="Lower governance bound for the adaptive APC lookback",
    )
    parser.add_argument(
        "--apc-window-max",
        type=int,
        default=126,
        help="Upper governance bound for the adaptive APC lookback",
    )
    parser.add_argument(
        "--max-apc-names",
        type=int,
        default=125,
        help="Maximum deterministic cross-section used in APC estimation",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=500,
        help="Monte Carlo stock-selection draws per stitched episode",
    )
    parser.add_argument(
        "--investment",
        type=float,
        default=1.0,
        help="Research capital unit before adaptive severity scaling",
    )
    parser.add_argument(
        "--round-trip-cost",
        type=float,
        default=0.002,
        help="Round-trip transaction cost as a fraction of invested capital",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible portfolio sampling",
    )
    parser.add_argument(
        "--out-prefix",
        default="adaptive_apc",
        help="Prefix for generated research outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ResearchConfig(
        min_history=args.min_history,
        apc_window_min=args.apc_window_min,
        apc_window_max=args.apc_window_max,
        max_apc_names=args.max_apc_names,
        n_simulations=args.n_simulations,
        investment_unit=args.investment,
        round_trip_cost=args.round_trip_cost,
        seed=args.seed,
    )

    prices = load_prices(args.file, args.date_col, args.ticker_col, args.price_col)
    returns = compute_returns(prices)

    print("[research] Building the point-in-time market state.")
    state = build_market_state(returns, cfg)
    apc_frame = compute_adaptive_apc(returns, state, cfg)
    thresholds = build_adaptive_thresholds(returns, state, apc_frame, cfg)

    events = identify_events(returns, state, apc_frame, thresholds, cfg)
    if events.empty:
        print("[research] No APC episodes satisfied the adaptive event-study filters.")
        return

    print("[research] Estimating adaptive volatility, beta, entry, horizon, and payoff-cap rules.")
    stock_vol = estimate_adaptive_stock_volatility(returns, state, cfg)
    betas = estimate_adaptive_betas(returns, state, cfg)
    drop_z_threshold = build_drop_z_threshold(returns, stock_vol, state, cfg)
    trade_horizon = build_adaptive_trade_horizon(events, state, prices.index, cfg)
    lot_return_cap = build_adaptive_lot_return_cap(returns, state, cfg)

    results = simulate_strategy(
        prices=prices,
        returns=returns,
        events=events,
        state=state,
        thresholds=thresholds,
        stock_vol=stock_vol,
        betas=betas,
        drop_z_threshold=drop_z_threshold,
        trade_horizon=trade_horizon,
        lot_return_cap=lot_return_cap,
        cfg=cfg,
    )
    summary = compute_summary(results)
    report = event_level_research_report(summary, seed=cfg.seed)
    print_report(summary, report)

    daily = pd.concat([state, apc_frame, thresholds, drop_z_threshold, trade_horizon, lot_return_cap], axis=1)
    daily.index.name = "date"

    out_prefix = Path(args.out_prefix)
    results.to_csv(f"{out_prefix}_results.csv", index=False)
    summary.to_csv(f"{out_prefix}_summary.csv", index=False)
    events.to_csv(f"{out_prefix}_events.csv", index=False)
    daily.reset_index().to_csv(f"{out_prefix}_daily_diagnostics.csv", index=False)
    report.to_csv(f"{out_prefix}_research_report.csv", index=False)

    print("[output] Research artifacts written:")
    print(f"  {out_prefix}_results.csv")
    print(f"  {out_prefix}_summary.csv")
    print(f"  {out_prefix}_events.csv")
    print(f"  {out_prefix}_daily_diagnostics.csv")
    print(f"  {out_prefix}_research_report.csv")


if __name__ == "__main__":
    main()
