"""
Backtest a systematic-event mean-reversion.

The program identifies broad market stress episodes using rolling Average
Pairwise Correlation (APC), a volatility-adjusted market-return filter, and a
minimum data-coverage requirement. For each event, it simulates randomized
portfolios drawn from stocks with evidence of systematic downside exposure,
adds capital only on sufficiently large within-event declines, and exits when
the correlation shock dissipates or the maximum holding period is reached.

Primary outputs:
    systematic_event_backtest_results.csv
    systematic_event_backtest_summary.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Data loading and normalization

def load_prices(file_path: str, date_col: str = "date",
                ticker_col: str = "ticker", price_col: str = "price") -> pd.DataFrame:
    """Load long or wide price data and return a date-by-ticker price matrix."""
    print(f"[load] Reading price data from {file_path}...")
    df = pd.read_csv(file_path, parse_dates=[date_col], low_memory=False)
    df[date_col] = pd.to_datetime(df[date_col])

    # Normalize long and wide inputs into the same date-by-ticker structure.
    if ticker_col in df.columns and price_col in df.columns:
        print("[load] Long-format data detected; pivoting observations to a wide price matrix.")
        df = (df.pivot_table(index=date_col, columns=ticker_col,
                             values=price_col, aggfunc="last")
                .sort_index())
    else:
        print("[load] Wide-format data detected.")
        df = df.set_index(date_col).sort_index()
        df = df.apply(pd.to_numeric, errors="coerce")

    # Exclude dates and tickers with no usable observations.
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    print(f"[load] Cleaned sample: {df.shape[0]:,} dates by "
          f"{df.shape[1]:,} tickers.")
    return df


# Return construction

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily log returns from the adjusted price matrix."""
    # Log returns are additive over time and convenient for correlation estimates.
    returns = np.log(prices / prices.shift(1))
    return returns.dropna(how="all")


# Systematic event detection

def compute_rolling_apc(returns: pd.DataFrame, window: int,
                        sample_size: int) -> pd.Series:
    """
    Estimate rolling Average Pairwise Correlation using a fixed stock subsample.

    Estimating the full pairwise correlation matrix is expensive for a large
    equity universe. A reproducible random subsample provides a lower-cost APC
    estimate while preserving the central object of interest: market-wide
    co-movement.
    """
    n_stocks = returns.shape[1]
    actual_sample = min(sample_size, n_stocks)

    rng = np.random.default_rng(42)
    sampled_cols = rng.choice(returns.columns, size=actual_sample, replace=False)
    sub = returns[sampled_cols].copy()

    print(f"[apc] Estimating rolling APC with a {window}-day window and "
          f"{actual_sample:,} sampled stocks...")

    apc_values = []
    dates = []

    # Use NumPy arrays inside the rolling loop to reduce pandas overhead.
    arr = sub.to_numpy(dtype=np.float64)
    col_dates = sub.index

    for t in range(window - 1, len(arr)):
        window_data = arr[t - window + 1 : t + 1]          # shape: (window, sample)
        # Correlation is undefined for flat or nearly flat return series.
        stds = window_data.std(axis=0)
        valid = stds > 1e-10
        wd = window_data[:, valid]
        if wd.shape[1] < 4:                                 # need at least four stocks
            apc_values.append(np.nan)
        else:
            # Average unique stock-pair correlations.
            corr = np.corrcoef(wd.T)                        # shape: (k, k)
            # Use the upper triangle so that each pair enters exactly once.
            k = corr.shape[0]
            idx = np.triu_indices(k, k=1)
            apc_values.append(float(np.nanmean(corr[idx])))
        dates.append(col_dates[t])

    apc = pd.Series(apc_values, index=dates, name="APC")
    print(f"[apc] APC estimation complete. Sample range: "
          f"[{apc.min():.3f}, {apc.max():.3f}].")
    return apc


# Event identification

def identify_events(returns: pd.DataFrame, apc: pd.Series,
                    apc_threshold_quantile: float,
                    cooldown_days: int,
                    min_stocks_available: int,
                    min_history_days: int = 252,
                    mkt_drop_vol_window: int = 60,
                    mkt_drop_vol_mult: float = 1.0,
                    max_hold_days_scan: int = 756) -> pd.DataFrame:
    """
    Identify non-overlapping systematic events.

    A date qualifies when APC exceeds its expanding historical threshold, the
    equal-weighted market return is below its rolling volatility floor, and the
    cross section contains enough valid stock returns. Once a trigger is
    accepted, subsequent candidates are ignored until the estimated event window
    has closed.
    """
    # Expanding APC threshold: only information available before date t is used.
    apc_expanding_thresh = (
        apc.shift(1)
           .expanding(min_periods=min_history_days)
           .quantile(apc_threshold_quantile)
    )
    print(f"[events] Expanding APC threshold: {apc_threshold_quantile:.0%} quantile; "
          f"first valid date = {apc_expanding_thresh.first_valid_index()}.")
    print(f"[events] APC threshold range: "
          f"[{apc_expanding_thresh.min():.4f}, {apc_expanding_thresh.max():.4f}].")

    # Equal-weighted market return, used as a broad market proxy.
    mkt_ret = returns.mean(axis=1)

    # Volatility-adjusted market drop filter.
    rolling_std        = (mkt_ret.shift(1)
                                  .rolling(window=mkt_drop_vol_window, min_periods=20)
                                  .std())
    mkt_drop_threshold = -mkt_drop_vol_mult * rolling_std
    print(f"[events] Market filter: equal-weighted return below "
          f"{mkt_drop_vol_mult:.2f} rolling standard deviations "
          f"({mkt_drop_vol_window}-day window).")
    print(f"[events] Market-return threshold range: "
          f"[{mkt_drop_threshold.min():.4f}, {mkt_drop_threshold.max():.4f}].")

    # Align dates before applying event filters.
    common = (apc.index
                 .intersection(apc_expanding_thresh.dropna().index)
                 .intersection(returns.index))

    apc_a       = apc.loc[common]
    thresh_a    = apc_expanding_thresh.loc[common]
    mkt_a       = mkt_ret.loc[common]
    mkt_floor_a = mkt_drop_threshold.reindex(common).fillna(0.0)
    ret_a       = returns.loc[common]

    # Candidate event filters.
    apc_spike   = apc_a > thresh_a
    mkt_down    = mkt_a < mkt_floor_a
    enough_data = ret_a.notna().sum(axis=1) >= min_stocks_available

    candidates = common[apc_spike & mkt_down & enough_data]

    # Keep event windows from overlapping; accepted events block later candidates
    # until the estimated event end date.
    price_dates_list = list(prices_idx) if hasattr(prices_idx := returns.index, '__iter__') else list(returns.index)
    price_dates_set  = set(price_dates_list)

    events       = []
    last_end     = pd.Timestamp("1900-01-01")   # end date of the last active event

    for d in candidates:
        # Skip candidates that fall within the previous event window.
        if d <= last_end:
            continue

        # Estimate the event end date using the same APC rule as the simulator.
        future = [fd for fd in price_dates_list if fd >= d][:max_hold_days_scan + 1]
        end_date = future[-1] if future else d

        for fd in future:
            if fd > d:
                apc_fd   = apc_a.get(fd, np.nan) if fd in apc_a.index else np.nan
                thr_fd   = thresh_a.get(fd, np.nan) if fd in thresh_a.index else np.nan
                if not (np.isnan(apc_fd) or np.isnan(thr_fd)) and apc_fd < thr_fd:
                    end_date = fd
                    break

        n_down  = (ret_a.loc[d] < 0).sum()
        n_valid = ret_a.loc[d].notna().sum()
        events.append({
            "date"           : d,
            "end_date_est"   : end_date,   # used only to prevent overlapping events
            "market_return"  : float(mkt_a.loc[d]),
            "mkt_threshold"  : float(mkt_floor_a.loc[d]),
            "apc"            : float(apc_a.loc[d]),
            "apc_threshold"  : float(thresh_a.loc[d]),
            "n_stocks_down"  : int(n_down),
            "n_stocks_valid" : int(n_valid),
        })
        last_end = end_date

    event_df = pd.DataFrame(events)
    print(f"[events] Identified {len(event_df):,} non-overlapping systematic events.")
    if len(event_df):
        worst = event_df.loc[event_df.market_return.idxmin()]
        print(f"[events] Largest market decline: {worst.date.date()} | "
              f"market return={worst.market_return:.2%}, "
              f"market floor={worst.mkt_threshold:.2%}, "
              f"APC={worst.apc:.3f}, threshold={worst.apc_threshold:.3f}.")
    return event_df, apc_expanding_thresh


# Z-score threshold calibration

def calibrate_z_threshold(prices: pd.DataFrame, returns: pd.DataFrame,
                           events: pd.DataFrame, apc: pd.Series,
                           apc_expanding_thresh: pd.Series,
                           roll_std_arr: np.ndarray,
                           round_trip_cost: float = 0.002,
                           z_candidates: np.ndarray = None) -> float:
    """
    Select the z-score cutoff with the highest average forward net return.

    Candidate thresholds are evaluated using all eligible stock-days within the
    detected event windows. Transaction costs are subtracted before scoring.
    Since this is an in-sample calibration, the result should be interpreted as
    exploratory research rather than a finalized trading rule.
    """
    if z_candidates is None:
        z_candidates = np.arange(0.25, 3.25, 0.25)

    prices_arr  = prices.to_numpy(dtype=np.float64)
    returns_arr = returns.reindex(prices.index).to_numpy(dtype=np.float64)
    date_idx    = {d: i for i, d in enumerate(prices.index)}
    tickers     = prices.columns.tolist()
    ticker_idx  = {t: i for i, t in enumerate(tickers)}

    # Build event windows once for the calibration pass.
    windows = build_event_windows(prices, apc, apc_expanding_thresh,
                                  events, max_hold_days=252)

    # Use every eligible stock-day rather than adding Monte Carlo sampling noise.
    print("[calibrate] Collecting stock-day observations across detected event windows...")
    z_obs  = []   # drop z-score on the investment date
    fr_obs = []   # forward return through the event exit date

    for w in windows:
        t_start     = w["date"]
        event_dates = w["event_dates"]

        if t_start not in date_idx:
            continue

        event_row_idx = [date_idx[d] for d in event_dates if d in date_idx]
        if len(event_row_idx) < 2:
            continue
        exit_row = event_row_idx[-1]

        # Evaluate every stock on each interior event day.
        for row in event_row_idx[1:-1]:
            for col in range(len(tickers)):
                ret_today   = returns_arr[row, col]
                price_today = prices_arr[row, col]
                std_today   = roll_std_arr[row, col]
                exit_price  = prices_arr[exit_row, col]

                if (np.isnan(ret_today)   or ret_today >= 0
                        or np.isnan(price_today) or price_today <= 0
                        or np.isnan(std_today)   or std_today <= 0
                        or np.isnan(exit_price)  or exit_price <= 0):
                    continue

                z   = abs(ret_today) / std_today
                fwd = (exit_price / price_today) - 1.0
                z_obs.append(z)
                fr_obs.append(fwd)

    z_obs  = np.array(z_obs,  dtype=np.float64)
    fr_obs = np.array(fr_obs, dtype=np.float64)
    print(f"[calibrate] Eligible stock-day observations: {len(z_obs):,}.")

    if len(z_obs) == 0:
        print("[calibrate] No eligible observations found; using z_min = 1.00.")
        return 1.0

    # Score each candidate threshold.
    print("[calibrate] Evaluating candidate z-score thresholds...")
    best_z      = z_candidates[0]
    best_metric = -np.inf
    results_cal = []

    for z_c in z_candidates:
        mask = z_obs >= z_c
        n    = mask.sum()
        if n < 30:          # require at least 30 observations for a stable estimate
            results_cal.append((z_c, np.nan, n))
            continue

        net_returns = fr_obs[mask] - round_trip_cost
        metric      = net_returns.mean()
        results_cal.append((z_c, metric, n))

        if metric > best_metric:
            best_metric = metric
            best_z      = z_c

    # Print the calibration table.
    print(f"\n{'-'*52}")
    print(f"  {'z_min':>6}  {'net_return':>12}  {'n_obs':>10}")
    print(f"{'-'*52}")
    for z_c, metric, n in results_cal:
        marker = "  selected" if abs(z_c - best_z) < 1e-9 else ""
        metric_str = f"{metric:>12.4f}" if not np.isnan(metric) else f"{'(insufficient)':>12}"
        print(f"  {z_c:>6.2f}  {metric_str}  {n:>10,}{marker}")
    print(f"{'-'*52}")
    print(f"  Selected z_min = {best_z:.2f}  "
          f"(net return {best_metric:.4f} per dollar)\n")

    return float(best_z)




def build_event_windows(prices: pd.DataFrame, apc: pd.Series,
                        apc_expanding_thresh: pd.Series,
                        events: pd.DataFrame,
                        max_hold_days: int) -> list:
    """
    Construct the trading-date window associated with each event.

    The window starts on the trigger date and ends when APC falls below its
    expanding threshold, unless the maximum holding period binds first. The exit
    date is included because positions are liquidated on that date.
    """
    price_dates_set = set(prices.index)
    windows = []

    for _, ev in events.iterrows():
        t_start = ev["date"]

        # Candidate price dates from the trigger through the holding-period cap.
        future_dates = prices.index[prices.index >= t_start][:max_hold_days + 1]

        # End the event after APC falls back below its threshold.
        event_dates = []
        end_date    = future_dates[-1]   # default: hard stop

        for d in future_dates:
            event_dates.append(d)
            # The trigger day is always included; exit checks begin afterward.
            if d > t_start:
                apc_today   = apc.get(d, np.nan)
                thresh_today = apc_expanding_thresh.get(d, np.nan)
                if np.isnan(apc_today) or np.isnan(thresh_today):
                    continue
                if apc_today < thresh_today:
                    end_date = d
                    break

        windows.append({
            "date"          : t_start,
            "end_date"      : end_date,
            "event_dates"   : event_dates,   # includes both trigger and exit dates
            "duration_days" : len(event_dates) - 1,
            "market_return" : ev["market_return"],
            "apc"           : ev["apc"],
            "apc_threshold" : ev["apc_threshold"],
        })

    return windows


def simulate_strategy(prices: pd.DataFrame, returns: pd.DataFrame,
                      events: pd.DataFrame, apc: pd.Series,
                      apc_expanding_thresh: pd.Series,
                      portfolio_size: int, max_hold_days: int,
                      n_simulations: int,
                      investment_per_stock: float = 1.0,
                      z_min: float = 0.0,
                      roll_std_arr: np.ndarray = None) -> pd.DataFrame:
    """
    Run event-level Monte Carlo portfolio simulations.

    Each simulation samples eligible stocks on the trigger date, invests only on
    later qualifying negative-return days, scales capital by event severity, and
    liquidates the portfolio at the event exit. The low rule prevents the
    strategy from averaging up within an event window.
    """
    # Estimate rolling beta for each stock against the equal-weighted market.
    print("[sim] Estimating rolling 252-day OLS betas for each stock...")
    mkt_ret = returns.mean(axis=1)
    mkt_aligned = mkt_ret.reindex(prices.index)

    win = 252
    ret_shifted = returns.shift(1).reindex(prices.index)
    mkt_shifted = mkt_aligned.shift(1)

    roll_cov = ret_shifted.rolling(win, min_periods=60).cov(mkt_shifted)
    roll_var = mkt_shifted.rolling(win, min_periods=60).var()
    beta_df  = roll_cov.div(roll_var, axis=0)
    beta_arr = beta_df.to_numpy(dtype=np.float64)
    print("[sim] Rolling beta estimates complete.")

    # Rolling volatility is reused when the caller has already calculated it.
    if roll_std_arr is None:
        print("[sim] Estimating rolling 60-day stock-level return volatility...")
        roll_std_df  = (returns.shift(1)
                               .reindex(prices.index)
                               .rolling(60, min_periods=20)
                               .std())
        roll_std_arr = roll_std_df.to_numpy(dtype=np.float64)
        print("[sim] Rolling volatility estimates complete.")
    else:
        print("[sim] Using the precomputed rolling volatility array.")

    print(f"[sim] Investment filter: z_min = {z_min:.2f}; "
          f"capital is deployed only when |r_i,t|/sigma_i,t >= {z_min:.2f}.")

    # Composite Severity Score by date.
    print("[sim] Computing the Composite Severity Score (CSS)...")
    apc_aligned    = apc.reindex(prices.index)
    thresh_aligned = apc_expanding_thresh.reindex(prices.index)
    mkt_arr_full   = mkt_aligned.to_numpy(dtype=np.float64)

    # S1: APC excess, scaled by its recent volatility.
    apc_excess     = apc_aligned - thresh_aligned
    apc_excess_std = apc_excess.shift(1).rolling(60, min_periods=20).std()
    S1 = (apc_excess / apc_excess_std.replace(0, np.nan)).clip(lower=0)

    # S2: market drop size relative to recent volatility.
    mkt_std = (mkt_aligned.shift(1).rolling(60, min_periods=20).std())
    S2 = (mkt_aligned.abs() / mkt_std.replace(0, np.nan)).clip(lower=0)

    # S3: fraction of stocks down, scaled so broad selloffs score higher.
    frac_down = (returns.reindex(prices.index) < 0).mean(axis=1)
    S3 = ((frac_down - 0.5) / 0.5).clip(lower=0)

    # Average the three signals and cap extreme values.
    CSS = ((S1 + S2 + S3) / 3).clip(lower=0, upper=3)
    css_arr = CSS.to_numpy(dtype=np.float64)

    # Report a concise diagnostic for the severity measure.
    css_valid = CSS.dropna()
    print(f"[sim] CSS range: [{css_valid.min():.3f}, {css_valid.max():.3f}]; "
          f"mean={css_valid.mean():.3f}; median={css_valid.median():.3f}.")

    # Event windows used by the simulations.
    print("[sim] Constructing event windows from the APC exit rule...")
    windows   = build_event_windows(prices, apc, apc_expanding_thresh,
                                    events, max_hold_days)
    durations = [w["duration_days"] for w in windows]
    print(f"[sim] Event-window duration: min={min(durations)} days; "
          f"median={int(np.median(durations))} days; max={max(durations)} days.")

    prices_arr  = prices.to_numpy(dtype=np.float64)
    returns_arr = returns.reindex(prices.index).to_numpy(dtype=np.float64)
    tickers     = prices.columns.tolist()
    ticker_idx  = {t: i for i, t in enumerate(tickers)}
    date_idx    = {d: i for i, d in enumerate(prices.index)}

    rng     = np.random.default_rng(0)
    results = []

    for w in windows:
        t_start     = w["date"]
        event_dates = w["event_dates"]
        duration    = w["duration_days"]

        if t_start not in date_idx:
            continue
        t0_idx = date_idx[t_start]

        # Beta-based eligibility on the trigger day.
        mkt_ret_t0           = mkt_arr_full[t0_idx]
        betas_t0             = beta_arr[t0_idx]
        prices_t0            = prices_arr[t0_idx]
        systematic_component = betas_t0 * mkt_ret_t0

        eligible = [
            tickers[i]
            for i in range(len(tickers))
            if (not np.isnan(prices_t0[i])   and prices_t0[i] > 0
                and not np.isnan(betas_t0[i]) and betas_t0[i] > 0
                and systematic_component[i] < 0)
        ]

        if len(eligible) < portfolio_size:
            continue

        event_row_idx = [date_idx[d] for d in event_dates if d in date_idx]
        if len(event_row_idx) < 2:
            continue
        exit_row = event_row_idx[-1]

        for sim_id in range(n_simulations):
            chosen     = rng.choice(eligible, size=portfolio_size, replace=False)
            chosen_idx = [ticker_idx[t] for t in chosen]

            shares      = np.zeros(portfolio_size)
            invested    = np.zeros(portfolio_size)
            # Track the lowest entry price for the new-low averaging rule.
            lowest_purchase_price = np.full(portfolio_size, np.inf)

            # CSS-adjusted investment on each qualifying day.
            for row in event_row_idx[1:-1]:
                css = css_arr[row]
                if np.isnan(css) or css <= 0:
                    css = 1.0

                z_scores   = np.zeros(portfolio_size)
                valid_mask = np.zeros(portfolio_size, dtype=bool)

                for k, col in enumerate(chosen_idx):
                    ret_today   = returns_arr[row, col]
                    price_today = prices_arr[row, col]
                    std_today   = roll_std_arr[row, col]

                    if (not np.isnan(ret_today)   and ret_today < 0
                            and not np.isnan(price_today) and price_today > 0
                            and not np.isnan(std_today)   and std_today > 0):

                        # Add only below the previous lowest purchase price.
                        if price_today >= lowest_purchase_price[k]:
                            continue

                        z_i = abs(ret_today) / std_today
                        if z_i < z_min:
                            continue
                        z_scores[k]   = z_i
                        valid_mask[k] = True

                n_down = valid_mask.sum()
                if n_down == 0:
                    continue

                # CSS sets total capital; z-scores distribute capital across stocks.
                z_valid = z_scores[valid_mask]
                z_mean  = z_valid.mean()
                z_norm  = z_valid / z_mean if z_mean > 0 else np.ones(n_down)

                j = 0
                for k, col in enumerate(chosen_idx):
                    if not valid_mask[k]:
                        continue
                    price_today = prices_arr[row, col]
                    amount = investment_per_stock * css * z_norm[j]
                    shares[k]   += amount / price_today
                    invested[k] += amount
                    # Store the new price floor for this stock.
                    lowest_purchase_price[k] = price_today
                    j += 1

            # Liquidate on the event exit day.
            total_invested = invested.sum()
            if total_invested == 0:
                results.append({
                    "event_date"       : t_start,
                    "end_date"         : w["end_date"],
                    "duration_days"    : duration,
                    "market_return"    : w["market_return"],
                    "apc"              : w["apc"],
                    "sim_id"           : sim_id,
                    "tickers"          : "|".join(chosen),
                    "total_invested"   : 0.0,
                    "total_pnl"        : 0.0,
                    "return_pct"       : 0.0,
                    "n_stocks_invested": 0,
                    "avg_hold_days"    : float(duration),
                })
                continue

            total_pnl = 0.0
            for k, col in enumerate(chosen_idx):
                if shares[k] == 0:
                    continue
                exit_price = prices_arr[exit_row, col]
                if np.isnan(exit_price) or exit_price <= 0:
                    col_prices = prices_arr[:exit_row + 1, col]
                    valid_p    = col_prices[~np.isnan(col_prices)]
                    exit_price = valid_p[-1] if len(valid_p) else 0.0
                total_pnl += shares[k] * exit_price - invested[k]

            results.append({
                "event_date"       : t_start,
                "end_date"         : w["end_date"],
                "duration_days"    : duration,
                "market_return"    : w["market_return"],
                "apc"              : w["apc"],
                "sim_id"           : sim_id,
                "tickers"          : "|".join(chosen),
                "total_invested"   : float(total_invested),
                "total_pnl"        : float(total_pnl),
                "return_pct"       : float(total_pnl / total_invested * 100),
                "n_stocks_invested": int((invested > 0).sum()),
                "avg_hold_days"    : float(duration),
            })

    return pd.DataFrame(results)


# Summary statistics

def compute_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate simulation results to the event level."""
    if results.empty:
        return pd.DataFrame()

    agg = (results.groupby("event_date")
           .agg(
               market_return      = ("market_return",    "first"),
               apc                = ("apc",               "first"),
               end_date           = ("end_date",          "first"),
               duration_days      = ("duration_days",     "first"),
               mean_return_pct    = ("return_pct",        "mean"),
               std_return_pct     = ("return_pct",        "std"),
               win_rate           = ("return_pct",        lambda x: (x > 0).mean()),
               median_return      = ("return_pct",        "median"),
               mean_invested      = ("total_invested",    "mean"),
               mean_hold_days     = ("avg_hold_days",     "mean"),
               n_simulations      = ("sim_id",            "count"),
           )
           .reset_index())

    agg["sharpe_ratio"] = agg["mean_return_pct"] / agg["std_return_pct"].replace(0, np.nan)
    return agg.sort_values("event_date")


def print_report(summary: pd.DataFrame, results: pd.DataFrame) -> None:
    """Print a compact research summary of the backtest results."""
    if summary.empty or results.empty:
        print("\n[report] No simulation results are available for reporting.")
        return

    sep = "-" * 64
    print(f"\n{'='*64}")
    print("  SYSTEMATIC EVENT MEAN-REVERSION: BACKTEST REPORT")
    print(f"{'='*64}")

    print("\nEVENT SAMPLE")
    print(sep)
    print(f"  Total events detected           : {len(summary):>8}")
    print(f"  Date range                      : "
          f"{summary.event_date.min().date()} to {summary.event_date.max().date()}")
    print(f"  Avg market return on event day  : "
          f"{summary.market_return.mean():>8.2%}")
    print(f"  Avg APC on event day            : "
          f"{summary.apc.mean():>8.3f}")
    print(f"  Avg event duration              : "
          f"{summary.duration_days.mean():>8.1f} days")
    print(f"  Median event duration           : "
          f"{summary.duration_days.median():>8.1f} days")

    print("\nCAPITAL DEPLOYMENT")
    print(sep)
    print(f"  Avg total invested per simulation : "
          f"${results.total_invested.mean():>8.2f}")
    print(f"  Median total invested per simulation: "
          f"${results.total_invested.median():>8.2f}")
    print(f"  Maximum total invested in one simulation: "
          f"${results.total_invested.max():>8.2f}")

    print("\nRETURNS ACROSS ALL EVENTS AND SIMULATIONS")
    print(sep)
    all_rets = results["return_pct"]
    # Exclude simulations in which no capital was deployed.
    all_rets_nonzero = results.loc[results.total_invested > 0, "return_pct"]
    print(f"  Mean return per portfolio       : {all_rets_nonzero.mean():>8.2f}%")
    print(f"  Median return per portfolio     : {all_rets_nonzero.median():>8.2f}%")
    print(f"  Std dev of returns              : {all_rets_nonzero.std():>8.2f}%")
    print(f"  Overall win rate                : "
          f"{(all_rets_nonzero > 0).mean():>8.1%}")
    print(f"  Best portfolio return           : {all_rets_nonzero.max():>8.2f}%")
    print(f"  Worst portfolio return          : {all_rets_nonzero.min():>8.2f}%")

    # One-sample test of whether average return is above zero.
    t_stat, p_val = stats.ttest_1samp(all_rets_nonzero.dropna(), 0)
    print("\nSTATISTICAL INFERENCE")
    print(sep)
    print("  Null hypothesis: mean portfolio return = 0%")
    print(f"  t-statistic                     : {t_stat:>8.3f}")
    print(f"  p-value (one-sided)             : {p_val/2:>8.4f}")
    sig = "reject at the 5% level" if p_val/2 < 0.05 else "do not reject at the 5% level"
    print(f"  Result                          : {sig}")

    print("\nHOLDING PERIOD")
    print(sep)
    print(f"  Mean event duration (days)      : "
          f"{results.avg_hold_days.mean():>8.1f}")
    print(f"  Median event duration (days)    : "
          f"{results.avg_hold_days.median():>8.1f}")

    print("\nTOP 5 EVENTS BY MEAN RETURN")
    print(sep)
    top5 = summary.nlargest(5, "mean_return_pct")[
        ["event_date", "end_date", "duration_days",
         "market_return", "mean_return_pct", "win_rate"]]
    print(top5.to_string(index=False, float_format="{:.2f}".format))

    print("\nBOTTOM 5 EVENTS BY MEAN RETURN")
    print(sep)
    bot5 = summary.nsmallest(5, "mean_return_pct")[
        ["event_date", "end_date", "duration_days",
         "market_return", "mean_return_pct", "win_rate"]]
    print(bot5.to_string(index=False, float_format="{:.2f}".format))
    print(f"\n{'='*64}\n")


# Command-line interface

def parse_args():
    p = argparse.ArgumentParser(
        description="Systematic Event Mean-Reversion Backtester")
    p.add_argument("--file",           required=True,  help="Path to the price CSV file")
    p.add_argument("--date-col",       default="date", help="Name of date column")
    p.add_argument("--ticker-col",     default="ticker",help="Name of ticker column (long format)")
    p.add_argument("--price-col",      default="price", help="Name of price column (long format)")
    p.add_argument("--window",         type=int,   default=60,
                   help="Rolling window for APC estimation (trading days, default 60)")
    p.add_argument("--apc-threshold",  type=float, default=0.65,
                   help="APC quantile used to define a correlation spike (default 0.65)")
    p.add_argument("--cooldown",       type=int,   default=30,
                   help="Minimum spacing between accepted events (default 30 days)")
    p.add_argument("--portfolio-size", type=int,   default=10,
                   help="Number of stocks sampled in each simulated portfolio (default 10)")
    p.add_argument("--max-hold",       type=int,   default=756,
                   help="Maximum holding period in trading days (default 756, approximately 3 years)")
    p.add_argument("--apc-sample",     type=int,   default=50,
                   help="Number of stocks subsampled for the APC estimate (default 50)")
    p.add_argument("--n-simulations",  type=int,   default=200,
                   help="Number of random portfolio draws per event (default 200)")
    p.add_argument("--min-stocks",     type=int,   default=20,
                   help="Minimum stocks with valid data required on an event day (default 20)")
    p.add_argument("--min-history",    type=int,   default=252,
                   help="Minimum past APC observations before event detection begins (default 252)")
    p.add_argument("--mkt-drop-vol-window", type=int, default=60,
                   help="Rolling window for the volatility-adjusted market drop filter (default 60)")
    p.add_argument("--mkt-drop-vol-mult",   type=float, default=1.5,
                   help="Standard-deviation multiplier for the market drop filter (default 1.5)")
    p.add_argument("--investment",     type=float, default=1.0,
                   help="Base dollars invested per qualifying stock-day (default $1)")
    p.add_argument("--round-trip-cost", type=float, default=0.002,
                   help="Round-trip transaction cost as fraction of trade value "
                        "(default 0.002 = 0.2%%). Used in z_min calibration.")
    p.add_argument("--out-prefix",     default="systematic_event_backtest",
                   help="Prefix for output CSV files")
    return p.parse_args()


def main():
    args = parse_args()

    # Load prices and construct daily returns.
    prices  = load_prices(args.file, args.date_col,
                          args.ticker_col, args.price_col)
    returns = compute_returns(prices)

    # Detect systematic events.
    apc    = compute_rolling_apc(returns, window=args.window,
                                 sample_size=args.apc_sample)
    events, apc_expanding_thresh = identify_events(
                             returns, apc,
                             apc_threshold_quantile=args.apc_threshold,
                             cooldown_days=args.cooldown,
                             min_stocks_available=args.min_stocks,
                             min_history_days=args.min_history,
                             mkt_drop_vol_window=args.mkt_drop_vol_window,
                             mkt_drop_vol_mult=args.mkt_drop_vol_mult,
                             max_hold_days_scan=args.max_hold)

    if events.empty:
        print("[main] No systematic events satisfied the selection criteria. "
              "Consider lowering --apc-threshold or reviewing the market-drop filter.")
        return

    # Run simulations.
    print(f"\n[sim] Running {args.n_simulations:,} simulations for each of "
          f"{len(events):,} detected events...")

    # Rolling volatility is shared by calibration and simulation.
    roll_std_df  = (returns.shift(1)
                           .reindex(prices.index)
                           .rolling(60, min_periods=20)
                           .std())
    roll_std_arr = roll_std_df.to_numpy(dtype=np.float64)

    # Calibrate the z-score filter.
    print("\n[calibrate] Selecting the z_min threshold from in-sample event observations...")
    z_min = calibrate_z_threshold(
        prices=prices, returns=returns, events=events,
        apc=apc, apc_expanding_thresh=apc_expanding_thresh,
        roll_std_arr=roll_std_arr,
        round_trip_cost=args.round_trip_cost,
    )

    results = simulate_strategy(
        prices=prices, returns=returns, events=events,
        apc=apc, apc_expanding_thresh=apc_expanding_thresh,
        portfolio_size=args.portfolio_size,
        max_hold_days=args.max_hold,
        n_simulations=args.n_simulations,
        investment_per_stock=args.investment,
        z_min=z_min,
        roll_std_arr=roll_std_arr,
    )
    print(f"[sim] Simulation complete. Total result rows: {len(results):,}.")

    # Summarize and print the report.
    summary = compute_summary(results)
    print_report(summary, results)

    # Save CSV outputs.
    out_results = f"{args.out_prefix}_results.csv"
    out_summary = f"{args.out_prefix}_summary.csv"
    results.to_csv(out_results, index=False)
    summary.to_csv(out_summary, index=False)
    print(f"[save] Wrote output files:\n  {out_results}\n  {out_summary}")


if __name__ == "__main__":
    main()
