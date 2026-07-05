"""Live paper-trading and RL backtest pages for the Streamlit dashboard.

Live data is published by the scheduled GitHub Actions tick to the
`live-data` branch; this module fetches it from raw.githubusercontent.com
(5-minute cache) and falls back to local files for development.
"""

from __future__ import annotations

import io
import json
import os
import pathlib

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent

LIVE_DATA_URL = os.environ.get(
    "LIVE_DATA_URL",
    "https://raw.githubusercontent.com/shoaib-kk/electricity-price-solar-analysis/live-data",
)

# Fixed entity -> color assignment (validated categorical palette).
# Color follows the strategy everywhere; seeds other than the primary are
# de-emphasized tints of the same entity hue.
C_RL = "#2a78d6"
C_RL_OTHER = "#9ec5f4"
C_THRESHOLD = "#1baf7a"
C_PERFECT = "#eda100"
C_PRICE = "#52514e"
C_CHARGE = "#008300"
C_DISCHARGE = "#e34948"
GRID = dict(alpha=0.25, linewidth=0.8)

DISCLAIMER = (
    "**Simulated paper trading only.** No real money, market participation, or "
    "battery hardware is involved. Prices are real AEMO dispatch data; every "
    "trade is hypothetical."
)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_remote(path: str) -> bytes | None:
    try:
        r = requests.get(f"{LIVE_DATA_URL}/{path}", timeout=10)
        if r.status_code == 200:
            return r.content
    except requests.RequestException:
        pass
    return None


def load_live_file(path: str) -> bytes | None:
    remote = _fetch_remote(path)
    if remote is not None:
        return remote
    local = ROOT_DIR / path
    if local.exists():
        return local.read_bytes()
    return None


def load_live_ledger() -> pd.DataFrame | None:
    raw = load_live_file("artifacts/live/ledger.csv")
    if raw is None:
        return None
    df = pd.read_csv(io.BytesIO(raw), parse_dates=["interval_start"])
    return df.sort_values(["strategy", "interval_start"])


def load_meta() -> dict | None:
    raw = load_live_file("models/deployed/meta.json")
    return json.loads(raw) if raw else None


def _fmt_age(latest: pd.Timestamp) -> str:
    # Ledger stamps are NEM market time (UTC+10, no DST).
    now = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=10)
    mins = max(0.0, (now - latest).total_seconds() / 60.0)
    return f"{mins/60:.1f} h ago" if mins >= 90 else f"{mins:.0f} min ago"


def _style_time_axis(ax):
    ax.grid(True, **GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))


def strategy_color(name: str, primary: str) -> tuple[str, float, str]:
    """(color, linewidth, label) for a strategy line."""
    if name == primary:
        return C_RL, 2.4, f"RL agent ({primary})"
    if name.startswith("rl_seed"):
        return C_RL_OTHER, 1.0, None  # grouped under one legend entry
    if name == "threshold":
        return C_THRESHOLD, 2.0, "Threshold baseline"
    if name == "perfect_foresight":
        return C_PERFECT, 2.0, "Perfect foresight (bound)"
    return "#898781", 1.2, name


def plot_live_price(df_primary: pd.DataFrame, hours: int = 48):
    cutoff = df_primary["interval_start"].max() - pd.Timedelta(hours=hours)
    w = df_primary[df_primary["interval_start"] >= cutoff]
    fig, ax = plt.subplots(figsize=(12, 4))
    if {"h1_q05", "h1_q95"}.issubset(w.columns) and w["h1_q05"].notna().any():
        ax.fill_between(w["interval_start"], w["h1_q05"], w["h1_q95"],
                        color=C_RL, alpha=0.15, label="Forecast 90% band")
    ax.plot(w["interval_start"], w["price_mwh"], color=C_PRICE, lw=1.6,
            label="Price (settled)")
    charges = w[w["action"].str.startswith("charge")]
    discharges = w[w["action"].str.startswith("discharge")]
    ax.scatter(charges["interval_start"], charges["price_mwh"], marker="^",
               s=48, color=C_CHARGE, zorder=3, label="Agent charged")
    ax.scatter(discharges["interval_start"], discharges["price_mwh"], marker="v",
               s=48, color=C_DISCHARGE, zorder=3, label="Agent discharged")
    ax.set_ylabel("Price ($/MWh)")
    ax.set_title(f"VIC1 30-min price, last {hours}h — RL agent decisions")
    _style_time_axis(ax)
    ax.legend(loc="upper left", frameon=False, ncols=2, fontsize=9)
    fig.tight_layout()
    return fig


def plot_live_pnl(ledger: pd.DataFrame, primary: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    seen_other = False
    for name, chunk in ledger.groupby("strategy"):
        color, lw, label = strategy_color(name, primary)
        if label is None:
            label = None if seen_other else "RL agent (other seeds)"
            seen_other = True
        ax.plot(chunk["interval_start"], chunk["cumulative_profit"],
                color=color, lw=lw, label=label)
    ax.axhline(0, color="#898781", lw=0.8)
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title("Live paper-trading P&L — RL agent vs threshold baseline")
    _style_time_axis(ax)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_live_soc(df_primary: pd.DataFrame, hours: int = 48):
    cutoff = df_primary["interval_start"].max() - pd.Timedelta(hours=hours)
    w = df_primary[df_primary["interval_start"] >= cutoff]
    fig, ax = plt.subplots(figsize=(12, 2.2))
    ax.fill_between(w["interval_start"], 0, w["soc"] * 100, color=C_RL, alpha=0.25)
    ax.plot(w["interval_start"], w["soc"] * 100, color=C_RL, lw=1.8)
    ax.set_ylabel("SoC (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Battery state of charge (RL agent)")
    _style_time_axis(ax)
    fig.tight_layout()
    return fig


def render_live_page():
    st.title("Live Paper Trading")
    st.warning(DISCLAIMER)

    ledger = load_live_ledger()
    meta = load_meta()
    if ledger is None or ledger.empty:
        st.info(
            "No live ledger available yet. The scheduled tick publishes to the "
            "`live-data` branch every 30 minutes once deployed."
        )
        return

    primary = f"rl_seed{meta['primary_seed']}" if meta else "rl_seed0"
    if primary not in set(ledger["strategy"]):
        primary = sorted(ledger["strategy"].unique())[0]
    df_primary = ledger[ledger["strategy"] == primary]
    last = df_primary.iloc[-1]
    thr = ledger[ledger["strategy"] == "threshold"]
    thr_pnl = float(thr["cumulative_profit"].iloc[-1]) if len(thr) else float("nan")

    k = st.columns(6)
    k[0].metric("Latest interval", str(last["interval_start"])[:16],
                _fmt_age(last["interval_start"]))
    k[1].metric("Settled price", f"${last['price_mwh']:,.0f}/MWh")
    k[2].metric("Agent action", str(last["action"]).replace("_", " "))
    k[3].metric("Agent SoC", f"{last['soc'] * 100:.0f}%")
    k[4].metric("Agent live P&L", f"${last['cumulative_profit']:,.2f}")
    k[5].metric("Baseline live P&L",
                "N/A" if pd.isna(thr_pnl) else f"${thr_pnl:,.2f}",
                delta=None if pd.isna(thr_pnl)
                else f"{last['cumulative_profit'] - thr_pnl:+,.2f} agent edge")

    st.pyplot(plot_live_price(df_primary))
    st.pyplot(plot_live_pnl(ledger, primary))
    st.pyplot(plot_live_soc(df_primary))

    if meta:
        with st.expander("Deployment details"):
            st.markdown(
                f"- **Trained at (UTC):** {meta.get('trained_at_utc')}\n"
                f"- **Training data ends:** {meta.get('data_end')}\n"
                f"- **Seeds deployed:** {meta.get('seeds')} "
                f"(primary = median-validation seed {meta.get('primary_seed')})\n"
                f"- **Threshold baseline:** charge < "
                f"${meta['thresholds']['low_mwh']:.0f}/MWh, discharge > "
                f"${meta['thresholds']['high_mwh']:.0f}/MWh (train-window 30/70 pct)\n"
                f"- Battery: 100 kWh / 50 kW, 90% one-way efficiency, 1% fees, "
                f"$0.03/kWh degradation cost"
            )
    with st.expander("Recent decisions (all strategies)"):
        recent = ledger.sort_values("interval_start").groupby("strategy").tail(12)
        st.dataframe(
            recent.sort_values(["interval_start", "strategy"], ascending=False),
            width="stretch", hide_index=True,
        )


def load_backtest() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    d = ROOT_DIR / "artifacts" / "backtest"
    summary = cum = None
    if (d / "wf_fold_summary.csv").exists():
        summary = pd.read_csv(d / "wf_fold_summary.csv",
                              parse_dates=["test_start", "test_end"])
    if (d / "wf_cumulative_pnl.csv").exists():
        cum = pd.read_csv(d / "wf_cumulative_pnl.csv", index_col=0, parse_dates=True)
    return summary, cum


def plot_backtest_cum(cum: pd.DataFrame):
    rl_cols = [c for c in cum.columns if c.startswith("rl_seed")]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for c in rl_cols:
        ax.plot(cum.index, cum[c], color=C_RL_OTHER, lw=0.9,
                label="RL seeds" if c == rl_cols[0] else None)
    if rl_cols:
        ax.plot(cum.index, cum[rl_cols].mean(axis=1), color=C_RL, lw=2.4,
                label="RL agent (seed mean)")
    if "threshold" in cum:
        ax.plot(cum.index, cum["threshold"], color=C_THRESHOLD, lw=2.0,
                label="Threshold baseline")
    if "perfect_foresight" in cum:
        ax.plot(cum.index, cum["perfect_foresight"], color=C_PERFECT, lw=2.0,
                label="Perfect foresight (bound)")
    ax.axhline(0, color="#898781", lw=0.8)
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title("Walk-forward test windows, stitched — cumulative P&L")
    _style_time_axis(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def plot_fold_profits(summary: pd.DataFrame):
    rl = summary[summary["strategy"].str.startswith("rl_seed")]
    rl_stats = rl.groupby("fold_id")["total_profit"].agg(["mean", "std"])
    thr = summary[summary["strategy"] == "threshold"].set_index("fold_id")["total_profit"]
    pf = summary[summary["strategy"] == "perfect_foresight"].set_index("fold_id")["total_profit"]
    folds = rl_stats.index.to_numpy()
    width = 0.27
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(folds - width, rl_stats["mean"], width, yerr=rl_stats["std"].fillna(0),
           color=C_RL, capsize=3, label="RL agent (mean ± std across seeds)")
    ax.bar(folds, thr.reindex(folds), width, color=C_THRESHOLD,
           label="Threshold baseline")
    ax.bar(folds + width, pf.reindex(folds), width, color=C_PERFECT,
           label="Perfect foresight (bound)")
    ax.axhline(0, color="#898781", lw=0.8)
    ax.set_xlabel("Walk-forward fold")
    ax.set_ylabel("Test-window profit ($)")
    ax.set_title("Profit per fold")
    ax.grid(True, axis="y", **GRID)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks(folds)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def render_backtest_page():
    st.title("RL Walk-Forward Backtest")
    st.caption(
        "Each fold trains on 1 year + 30 days calibration, then trades the next "
        "90 days out-of-sample. RL results are reported across all training "
        "seeds — no cherry-picking."
    )
    summary, cum = load_backtest()
    if summary is None or summary.empty:
        st.info("No backtest artifacts found. Run `python scripts/run_backtest.py`.")
        return

    rl = summary[summary["strategy"].str.startswith("rl_seed")]
    per_seed = rl.groupby("strategy")["total_profit"].sum()
    thr_total = summary.loc[summary["strategy"] == "threshold", "total_profit"].sum()
    pf_total = summary.loc[
        summary["strategy"] == "perfect_foresight", "total_profit"
    ].sum()

    k = st.columns(4)
    k[0].metric("RL total P&L (mean ± std over seeds)",
                f"${per_seed.mean():,.0f} ± {per_seed.std():,.0f}")
    k[1].metric("Threshold baseline", f"${thr_total:,.0f}")
    k[2].metric("Perfect foresight bound", f"${pf_total:,.0f}")
    k[3].metric("RL capture of bound",
                f"{100 * per_seed.mean() / pf_total:.0f}%" if pf_total else "N/A")

    if cum is not None and not cum.empty:
        st.pyplot(plot_backtest_cum(cum))
    st.pyplot(plot_fold_profits(summary))

    st.subheader("Metrics by strategy (summed / averaged over folds)")
    agg = (
        summary.assign(
            group=summary["strategy"].where(
                ~summary["strategy"].str.startswith("rl_seed"), "rl (per seed)"
            )
        )
        .groupby(["group", "strategy"])
        .agg(
            total_profit=("total_profit", "sum"),
            sharpe_mean=("sharpe_daily_ann", "mean"),
            max_drawdown=("max_drawdown", "max"),
            cycles=("equivalent_cycles", "sum"),
        )
        .round(2)
    )
    st.dataframe(agg, width="stretch")
    with st.expander("Per-fold detail"):
        st.dataframe(summary.round(2), width="stretch", hide_index=True)
