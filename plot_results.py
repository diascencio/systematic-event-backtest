"""
Figures for the adaptive APC stitched-episode study.

The plots are meant to read like a compact research appendix: first define the
APC stress episodes, then show robust event-level payoffs, and finally check
whether the result depends on duration, trigger clustering, or a few large wins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "#0f1419",
        "axes.facecolor": "#0f1419",
        "savefig.facecolor": "#0f1419",
        "text.color": "#e6edf3",
        "axes.labelcolor": "#c9d1d9",
        "xtick.color": "#9aa4af",
        "ytick.color": "#9aa4af",
        "grid.color": "#2a3138",
        "axes.edgecolor": "#2a3138",
    }
)

BLUE = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
YELLOW = "#d29922"
MUTED = "#8b949e"


def load_outputs(
    prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the CSV artifacts produced by the APC event-study script."""
    base = Path(prefix)
    results = pd.read_csv(
        f"{base}_results.csv",
        parse_dates=["event_date", "end_date"],
    )
    summary = pd.read_csv(
        f"{base}_summary.csv",
        parse_dates=["event_date", "end_date"],
    )
    events = pd.read_csv(f"{base}_events.csv", parse_dates=["event_date", "end_date"])
    daily = pd.read_csv(f"{base}_daily_diagnostics.csv", parse_dates=["date"])
    report_path = Path(f"{base}_research_report.csv")
    report = pd.read_csv(report_path) if report_path.exists() else pd.DataFrame()
    return results, summary, events, daily, report


def shade_events(ax: plt.Axes, events: pd.DataFrame, alpha: float = 0.10) -> None:
    """Shade stitched APC episodes on a time-series axis."""
    for _, event in events.iterrows():
        ax.axvspan(
            event["event_date"],
            event["end_date"],
            color=YELLOW,
            alpha=alpha,
            lw=0,
        )


def robust_return(summary: pd.DataFrame) -> pd.Series:
    """Select the preferred robust episode-level return estimator."""
    if "headline_return_pct" in summary.columns:
        return summary["headline_return_pct"]
    if "trimmed_mean_return_pct" in summary.columns:
        return summary["trimmed_mean_return_pct"]
    return summary["mean_return_pct"]


def event_time_apc_excess(
    events: pd.DataFrame,
    daily: pd.DataFrame,
    pre_days: int = 60,
    post_days: int = 120,
) -> pd.DataFrame:
    """Align APC stress around event starts and summarize the event-time panel."""
    d = daily.sort_values("date").reset_index(drop=True).copy()
    if "apc_excess" not in d.columns:
        d["apc_excess"] = d["apc"] - d["apc_threshold"]

    date_to_row = {date: i for i, date in enumerate(d["date"])}
    rows: list[dict[str, float | int]] = []
    for _, event in events.iterrows():
        event_date = event["event_date"]
        if event_date not in date_to_row:
            continue

        event_row = date_to_row[event_date]
        for rel_day in range(-pre_days, post_days + 1):
            row = event_row + rel_day
            if row < 0 or row >= len(d):
                continue
            rows.append(
                {
                    "event_time": rel_day,
                    "apc_excess": float(d.at[row, "apc_excess"]),
                }
            )

    panel = pd.DataFrame(rows)
    if panel.empty:
        return pd.DataFrame(columns=["event_time", "median", "q25", "q75"])

    return (
        panel.groupby("event_time")["apc_excess"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )


def winner_exclusion_profile(returns: pd.Series) -> pd.DataFrame:
    """Measure how much the mean depends on the largest positive episodes."""
    clean = returns.dropna().sort_values(ascending=False)
    checks = [("Full sample", 0), ("Ex top 1", 1), ("Ex top 2", 2), ("Ex top 5", 5)]
    rows: list[dict[str, float | int | str]] = []
    for label, n_excluded in checks:
        retained = clean.iloc[n_excluded:]
        if retained.empty:
            continue
        rows.append(
            {
                "sample": label,
                "mean_return": float(retained.mean()),
                "median_return": float(retained.median()),
                "hit_rate": float((retained > 0).mean()),
                "n_events": int(len(retained)),
            }
        )
    return pd.DataFrame(rows)


def fig_story(
    summary: pd.DataFrame,
    events: pd.DataFrame,
    daily: pd.DataFrame,
    report: pd.DataFrame,
    prefix: str,
) -> None:
    """Create the narrative figure that explains the empirical claim."""
    plot_returns = robust_return(summary)
    investable = plot_returns.loc[summary["deployment_rate"] > 0].dropna()
    event_panel = event_time_apc_excess(events, daily)
    robustness = winner_exclusion_profile(investable)
    metrics = dict(zip(report["metric"], report["value"])) if not report.empty else {}

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "APC Stress Episodes and Subsequent Robust Payoffs",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[0.95, 1.15],
        hspace=0.34,
        wspace=0.26,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    if not event_panel.empty:
        x = event_panel["event_time"]
        ax1.fill_between(
            x,
            event_panel["q25"],
            event_panel["q75"],
            color=BLUE,
            alpha=0.18,
            label="Interquartile range",
        )
        ax1.plot(x, event_panel["median"], color=BLUE, lw=2.0, label="Median")
    ax1.axvline(0, color=YELLOW, lw=1.4, ls="--", label="Event start")
    ax1.axhline(0, color=MUTED, lw=0.8)
    ax1.set_title("Event-Time APC Stress")
    ax1.set_xlabel("Trading days from APC episode start")
    ax1.set_ylabel("APC minus adaptive threshold")
    ax1.legend(fontsize=8, loc="upper right")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    n_events = metrics.get("n_investable_events", len(investable))
    mean_return = metrics.get("mean_event_return_pct", investable.mean())
    median_return = metrics.get("median_event_return_pct", investable.median())
    hit_rate = metrics.get("event_hit_rate", (investable > 0).mean())
    prob_nonpositive = metrics.get("bootstrap_prob_mean_le_0", np.nan)
    ci_low = metrics.get("bootstrap_mean_ci_2p5", np.nan)
    ci_high = metrics.get("bootstrap_mean_ci_97p5", np.nan)
    statement = (
        "The test asks whether APC stress identifies broad, temporary "
        "dislocations rather than isolated stock-level accidents."
    )
    ax2.text(
        0.0,
        0.95,
        "Research Claim",
        fontsize=14,
        fontweight="bold",
        color="#e6edf3",
    )
    ax2.text(0.0, 0.78, statement, fontsize=11, color="#c9d1d9", wrap=True)
    ax2.text(
        0.0,
        0.56,
        f"Independent investable episodes: {int(float(n_events))}",
        fontsize=11,
    )
    ax2.text(
        0.0,
        0.43,
        "Robust mean / median return: "
        f"{float(mean_return):.2f}% / {float(median_return):.2f}%",
        fontsize=11,
    )
    ax2.text(0.0, 0.30, f"Episode hit rate: {float(hit_rate):.1%}", fontsize=11)
    if np.isfinite(float(ci_low)) and np.isfinite(float(ci_high)):
        ax2.text(
            0.0,
            0.17,
            f"Bootstrap mean interval: [{float(ci_low):.2f}%, {float(ci_high):.2f}%]",
            fontsize=11,
        )
    if np.isfinite(float(prob_nonpositive)):
        ax2.text(0.0, 0.04, f"P(mean <= 0): {float(prob_nonpositive):.2%}", fontsize=11)

    ax3 = fig.add_subplot(gs[1, 0])
    waterfall = (
        summary.assign(plot_return=plot_returns)
        .sort_values("plot_return")
        .reset_index(drop=True)
    )
    colors = np.where(waterfall["plot_return"] >= 0, GREEN, RED)
    ax3.bar(np.arange(len(waterfall)), waterfall["plot_return"], color=colors, alpha=0.85)
    ax3.axhline(0, color=MUTED, lw=0.8)
    if len(investable):
        ax3.axhline(investable.mean(), color=BLUE, lw=1.2, ls="--", label="Mean")
        ax3.legend(fontsize=8, loc="upper left")
    ax3.set_title("Episode Payoff Waterfall")
    ax3.set_xlabel("Independent APC episodes sorted by robust return")
    ax3.set_ylabel("Robust risk-budget return (%)")

    label_positions = sorted(
        set([0, 1, len(waterfall) - 3, len(waterfall) - 2, len(waterfall) - 1])
    )
    for pos in label_positions:
        if pos < 0 or pos >= len(waterfall):
            continue
        row = waterfall.iloc[pos]
        year = pd.Timestamp(row["event_date"]).year
        y = row["plot_return"]
        va = "top" if y < 0 else "bottom"
        offset = -0.18 if y < 0 else 0.18
        ax3.text(
            pos,
            y + offset,
            str(year),
            ha="center",
            va=va,
            fontsize=8,
            color="#c9d1d9",
        )

    ax4 = fig.add_subplot(gs[1, 1])
    if not robustness.empty:
        bar_colors = [GREEN if value >= 0 else RED for value in robustness["mean_return"]]
        ax4.bar(
            robustness["sample"],
            robustness["mean_return"],
            color=bar_colors,
            alpha=0.85,
        )
        ax4.plot(
            robustness["sample"],
            robustness["median_return"],
            color=BLUE,
            marker="o",
            lw=1.4,
            label="Median retained return",
        )
        for x_label, _, hit, n_events in robustness[
            ["sample", "mean_return", "hit_rate", "n_events"]
        ].itertuples(index=False):
            ax4.text(
                x_label,
                0,
                f" n={n_events}\n hit={hit:.0%}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#c9d1d9",
            )
        ax4.legend(fontsize=8, loc="upper right")
    ax4.axhline(0, color=MUTED, lw=0.8)
    ax4.set_title("Dependence on the Largest Winning Episodes")
    ax4.set_ylabel("Mean robust return after exclusion (%)")
    ax4.tick_params(axis="x", rotation=15)

    output = f"{prefix}_research_story.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    print(f"[figures] Saved narrative research figure: {output}")


def fig_overview(
    summary: pd.DataFrame,
    events: pd.DataFrame,
    daily: pd.DataFrame,
    report: pd.DataFrame,
    prefix: str,
) -> None:
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "Adaptive APC Event-Study Evidence",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        height_ratios=[1.15, 1.0, 1.0],
        hspace=0.38,
        wspace=0.28,
    )

    ax1 = fig.add_subplot(gs[0, :])
    shade_events(ax1, events, alpha=0.08)
    ax1.plot(daily["date"], daily["apc"], color=BLUE, lw=1.1, label="APC")
    ax1.plot(
        daily["date"],
        daily["apc_threshold"],
        color=YELLOW,
        lw=1.0,
        label="Adaptive APC threshold",
    )
    ax1.set_title("APC Signal, Adaptive Threshold, and Stitched Stress Episodes")
    ax1.set_ylabel("Average Pairwise Correlation")
    ax1.legend(loc="upper left", ncols=2, fontsize=9)
    ax1.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2 = fig.add_subplot(gs[1, 0])
    plot_returns = robust_return(summary)
    colors = np.where(plot_returns >= 0, GREEN, RED)
    ax2.bar(summary["event_date"], plot_returns, width=60, color=colors, alpha=0.85)
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.set_title("Robust Episode Returns on Risk-Budget Capital")
    ax2.set_ylabel("Robust risk-budget return (%)")
    ax2.tick_params(axis="x", rotation=30)

    ax3 = fig.add_subplot(gs[1, 1])
    investable = plot_returns.loc[summary["deployment_rate"] > 0].dropna()
    if len(investable):
        bins = np.linspace(investable.quantile(0.02), investable.quantile(0.98), 28)
        ax3.hist(
            investable[investable >= 0],
            bins=bins,
            color=GREEN,
            alpha=0.85,
            label="Positive",
        )
        ax3.hist(
            investable[investable < 0],
            bins=bins,
            color=RED,
            alpha=0.85,
            label="Negative",
        )
        ax3.axvline(
            investable.mean(),
            color=BLUE,
            ls="--",
            lw=1.5,
            label=f"Mean {investable.mean():.2f}%",
        )
    ax3.axvline(0, color=MUTED, lw=0.8)
    ax3.set_title("Cross-Episode Distribution of Robust Returns")
    ax3.set_xlabel("Robust episode return (%)")
    ax3.set_ylabel("Episodes")
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[2, 0])
    size_col = summary.get("stitched_trigger_count", pd.Series(1, index=summary.index))
    scatter = ax4.scatter(
        summary["event_severity"],
        plot_returns,
        c=summary["duration_days"],
        cmap="viridis",
        s=35 + 15 * size_col.clip(lower=1),
        alpha=0.82,
        edgecolors="none",
    )
    ax4.axhline(0, color=MUTED, lw=0.8)
    ax4.set_title("Robust Return vs. Ex Ante Event Severity")
    ax4.set_xlabel("Event severity percentile blend")
    ax4.set_ylabel("Robust risk-budget return (%)")
    cb = fig.colorbar(scatter, ax=ax4)
    cb.set_label("Duration days")
    cb.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=MUTED)

    ax5 = fig.add_subplot(gs[2, 1])
    if not report.empty:
        values = dict(zip(report["metric"], report["value"]))
        labels = ["Hit rate", "Deployment", "P(mean <= 0)"]
        vals = [
            float(values.get("event_hit_rate", np.nan)),
            float(values.get("mean_deployment_rate", np.nan)),
            float(values.get("bootstrap_prob_mean_le_0", np.nan)),
        ]
        ax5.bar(labels, vals, color=[GREEN, BLUE, RED], alpha=0.85)
        ax5.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax5.set_ylim(0, max(1.0, np.nanmax(vals) * 1.15))
    ax5.set_title("Episode-Level Inference Summary")
    ax5.set_ylabel("Rate / Probability")

    output = f"{prefix}_research_overview.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    print(f"[figures] Saved overview diagnostic: {output}")


def fig_diagnostics(
    summary: pd.DataFrame,
    events: pd.DataFrame,
    prefix: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Adaptive APC Event-Study Diagnostics",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    s = summary.sort_values("event_date").copy()
    s["plot_return"] = robust_return(s)
    s["rolling_5_event_return"] = s["plot_return"].rolling(5, min_periods=2).mean()
    s["rolling_5_hit_rate"] = (s["plot_return"] > 0).rolling(5, min_periods=2).mean()

    ax1 = axes[0, 0]
    ax1.plot(
        s["event_date"],
        s["rolling_5_event_return"],
        color=BLUE,
        lw=1.5,
        label="5-episode robust return",
    )
    ax1.axhline(0, color=MUTED, lw=0.8)
    ax1b = ax1.twinx()
    ax1b.plot(
        s["event_date"],
        s["rolling_5_hit_rate"],
        color=GREEN,
        lw=1.2,
        label="5-episode hit rate",
    )
    ax1b.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax1.set_title("Rolling Five-Episode Robust Performance")
    ax1.set_ylabel("Robust risk-budget return (%)")
    ax1b.set_ylabel("Hit rate")
    ax1.tick_params(axis="x", rotation=30)

    ax2 = axes[0, 1]
    plot_returns = robust_return(summary)
    colors = np.where(plot_returns >= 0, GREEN, RED)
    ax2.scatter(summary["duration_days"], plot_returns, c=colors, s=70, alpha=0.8)
    ax2.axhline(0, color=MUTED, lw=0.8)
    ax2.set_title("Episode Duration and Robust Return")
    ax2.set_xlabel("Episode duration (trading days)")
    ax2.set_ylabel("Robust risk-budget return (%)")

    ax3 = axes[1, 0]
    trigger_count = summary.get(
        "stitched_trigger_count",
        pd.Series(1, index=summary.index),
    )
    ax3.scatter(
        trigger_count,
        plot_returns,
        c=summary["duration_days"],
        cmap="plasma",
        s=80,
        alpha=0.82,
    )
    ax3.axhline(0, color=MUTED, lw=0.8)
    ax3.set_title("Trigger Clustering and Robust Return")
    ax3.set_xlabel("APC triggers stitched into episode")
    ax3.set_ylabel("Robust risk-budget return (%)")

    ax4 = axes[1, 1]
    by_reason = (
        summary.assign(plot_return=plot_returns)
        .groupby("exit_reason")
        .agg(plot_return=("plot_return", "mean"), n_events=("event_date", "count"))
        .sort_values("plot_return")
    )
    reason_colors = np.where(by_reason["plot_return"] >= 0, GREEN, RED)
    ax4.barh(by_reason.index.astype(str), by_reason["plot_return"], color=reason_colors, alpha=0.85)
    for y, (_, row) in enumerate(by_reason.iterrows()):
        ax4.text(
            row["plot_return"],
            y,
            f"  n={int(row['n_events'])}",
            va="center",
            color="#c9d1d9",
            fontsize=9,
        )
    ax4.axvline(0, color=MUTED, lw=0.8)
    ax4.set_title("Return Attribution by Episode Exit Rule")
    ax4.set_xlabel("Mean robust risk-budget return (%)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output = f"{prefix}_research_diagnostics.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    print(f"[figures] Saved supplemental diagnostic: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create figures for the adaptive APC event-study outputs"
    )
    parser.add_argument("--prefix", default="adaptive_apc", help="Prefix shared by the backtest CSV artifacts")
    parser.add_argument("--show", action="store_true", help="Display figures after saving them")
    args = parser.parse_args()

    _, summary, events, daily, report = load_outputs(args.prefix)
    fig_story(summary, events, daily, report, args.prefix)
    fig_overview(summary, events, daily, report, args.prefix)
    fig_diagnostics(summary, events, args.prefix)

    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
