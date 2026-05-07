"""
Visualize the systematic-event mean-reversion backtest.

This script reads the event-level results and summary files produced by
systematic_event_strategy.py and generates two figures: a main performance
overview and a holding-period diagnostic.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "figure.facecolor": "#0d1117",
    "axes.facecolor"  : "#0d1117",
    "text.color"      : "#c9d1d9",
    "axes.labelcolor" : "#c9d1d9",
    "xtick.color"     : "#8b949e",
    "ytick.color"     : "#8b949e",
    "grid.color"      : "#21262d",
    "axes.edgecolor"  : "#21262d",
})

ACCENT   = "#58a6ff"
POSITIVE = "#3fb950"
NEGATIVE = "#f85149"
NEUTRAL  = "#d29922"


def load(results_path, summary_path):
    """Read the simulation-level and event-level CSV outputs."""
    r = pd.read_csv(results_path, parse_dates=["event_date"])
    s = pd.read_csv(summary_path, parse_dates=["event_date"])
    return r, s


def fig_overview(results, summary):
    """Create the main performance overview figure."""
    fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    fig.suptitle("Systematic Event Mean-Reversion Strategy: Backtest Results",
                 fontsize=15, color="#e6edf3", y=0.98, fontweight="bold")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    # Distribution of simulated portfolio returns across all event windows.
    ax1 = fig.add_subplot(gs[0, 0])
    rets = results["return_pct"].dropna()
    bins = np.linspace(rets.quantile(0.01), rets.quantile(0.99), 60)
    ax1.hist(rets[rets >= 0], bins=bins, color=POSITIVE, alpha=0.8, label="Profitable")
    ax1.hist(rets[rets <  0], bins=bins, color=NEGATIVE, alpha=0.8, label="Loss")
    ax1.axvline(rets.mean(), color=ACCENT, lw=1.5, ls="--",
                label=f"Mean {rets.mean():.2f}%")
    ax1.set_xlabel("Portfolio Return (%)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Return Distribution Across Events and Simulations")
    ax1.legend(fontsize=8)

    # Average simulated return for each detected systematic event.
    ax2 = fig.add_subplot(gs[0, 1])
    colors = [POSITIVE if v > 0 else NEGATIVE for v in summary["mean_return_pct"]]
    ax2.bar(summary["event_date"], summary["mean_return_pct"],
            width=40, color=colors, alpha=0.85)
    ax2.axhline(0, color="#8b949e", lw=0.8)
    ax2.set_xlabel("Event Date")
    ax2.set_ylabel("Mean Return (%)")
    ax2.set_title("Mean Return per Systematic Event")
    ax2.tick_params(axis="x", rotation=30)

    # Fraction of simulations with positive returns for each event.
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(summary["event_date"], summary["win_rate"] * 100,
                c=[POSITIVE if w >= 50 else NEGATIVE for w in summary["win_rate"]*100],
                s=40, alpha=0.85, zorder=3)
    ax3.axhline(50, color=NEUTRAL, lw=1, ls="--", label="50% baseline")
    ax3.yaxis.set_major_formatter(PercentFormatter())
    ax3.set_xlabel("Event Date")
    ax3.set_ylabel("Win Rate (%)")
    ax3.set_title("Event-Level Win Rate Across Simulations")
    ax3.legend(fontsize=8)
    ax3.tick_params(axis="x", rotation=30)

    # Relationship between event severity and subsequent simulated returns.
    ax4 = fig.add_subplot(gs[1, 1])
    sc = ax4.scatter(summary["market_return"] * 100, summary["mean_return_pct"],
                     c=summary["apc"], cmap="plasma", s=50, alpha=0.8)
    ax4.axhline(0, color="#8b949e", lw=0.8)
    ax4.axvline(0, color="#8b949e", lw=0.8)
    cb = fig.colorbar(sc, ax=ax4)
    cb.set_label("APC on event day", color="#c9d1d9")
    cb.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color="#8b949e")
    ax4.set_xlabel("Market Return on Event Day (%)")
    ax4.set_ylabel("Mean Portfolio Return (%)")
    ax4.set_title("Strategy Return vs. Event Severity (Color = APC)")

    plt.savefig("backtest_overview.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("[plot] Wrote main performance figure: backtest_overview.png")
    plt.show()


def fig_holding(results):
    """Create the holding-period diagnostic figure."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="#0d1117")
    fig.suptitle("Holding Period Analysis", fontsize=13, color="#e6edf3",
                 fontweight="bold")

    hold = results["avg_hold_days"].dropna()

    ax = axes[0]
    ax.hist(hold, bins=50, color=ACCENT, alpha=0.85)
    ax.axvline(hold.median(), color=NEUTRAL, lw=1.5, ls="--",
               label=f"Median {hold.median():.0f}d")
    ax.set_xlabel("Avg Hold (days)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Average Holding Periods")
    ax.legend()

    ax2 = axes[1]
    ax2.scatter(results["avg_hold_days"], results["return_pct"],
                alpha=0.2, s=6, color=ACCENT)
    ax2.axhline(0, color="#8b949e", lw=0.8)
    ax2.set_xlabel("Avg Hold (days)")
    ax2.set_ylabel("Portfolio Return (%)")
    ax2.set_title("Return vs. Holding Period")

    plt.tight_layout()
    plt.savefig("backtest_holding.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("[plot] Wrote holding-period figure: backtest_holding.png")
    plt.show()


def main():
    p = argparse.ArgumentParser(
        description="Create diagnostic figures from the backtest CSV outputs.")
    p.add_argument("--results", default="systematic_event_strategy_results.csv")
    p.add_argument("--summary", default="systematic_event_strategy_summary.csv")
    args = p.parse_args()

    results, summary = load(args.results, args.summary)
    fig_overview(results, summary)
    fig_holding(results)


if __name__ == "__main__":
    main()
