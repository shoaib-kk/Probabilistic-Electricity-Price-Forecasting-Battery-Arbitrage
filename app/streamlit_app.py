from __future__ import annotations

import pathlib
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

METRICS_FILE = ARTIFACTS_DIR / "final_metrics.csv"
CUM_PROFIT_FILE = ARTIFACTS_DIR / "cumulative_profit.csv"
FORECAST_FILE = ARTIFACTS_DIR / "conformal_forecast.csv"
BACKTEST_FILE = ARTIFACTS_DIR / "backtest_summary.csv"
ACTIONS_FILE = ARTIFACTS_DIR / "actions.csv"

st.set_page_config(page_title="Battery Arbitrage Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def load_csv(path: pathlib.Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def parse_timestamp_column(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    if column in df.columns:
        work = df.copy()
        work[column] = pd.to_datetime(work[column], errors="coerce")
        work = work.dropna(subset=[column])
        return work
    return df


def ensure_strategy_column(df: pd.DataFrame, default_name: str = "strategy") -> pd.DataFrame:
    work = df.copy()
    if "strategy" not in work.columns:
        work["strategy"] = default_name
    work["strategy"] = work["strategy"].astype(str)
    return work


def pick_profit_column(df: pd.DataFrame) -> str | None:
    for col in ["total_profit_with_terminal", "total_profit", "profit"]:
        if col in df.columns:
            return col
    return None


def pick_drawdown_column(df: pd.DataFrame) -> str | None:
    for col in ["max_drawdown", "drawdown", "max_equity_drawdown"]:
        if col in df.columns:
            return col
    return None


def pick_cycles_column(df: pd.DataFrame) -> str | None:
    for col in ["equivalent_cycles", "cycles"]:
        if col in df.columns:
            return col
    return None


def pick_profit_per_cycle_column(df: pd.DataFrame) -> str | None:
    for col in ["profit_per_cycle"]:
        if col in df.columns:
            return col
    return None


def derive_cumulative_from_actions(actions_df: pd.DataFrame) -> pd.DataFrame | None:
    required = {"timestamp", "strategy"}
    if not required.issubset(actions_df.columns):
        return None

    work = parse_timestamp_column(actions_df)
    work = ensure_strategy_column(work)
    work = work.sort_values(["strategy", "timestamp"])

    if "cumulative_profit" in work.columns:
        out = work[["timestamp", "strategy", "cumulative_profit"]].copy()
        return out.dropna(subset=["cumulative_profit"])

    if "profit" in work.columns:
        work["cumulative_profit"] = work.groupby("strategy")["profit"].cumsum()
        return work[["timestamp", "strategy", "cumulative_profit"]]

    return None


def prepare_cumulative_df(cum_df: pd.DataFrame | None, actions_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if cum_df is not None:
        work = parse_timestamp_column(cum_df)
        work = ensure_strategy_column(work)
        if {"timestamp", "strategy", "cumulative_profit"}.issubset(work.columns):
            return work.sort_values(["strategy", "timestamp"])

    if actions_df is not None:
        return derive_cumulative_from_actions(actions_df)

    return None


def prepare_actions_df(actions_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if actions_df is None:
        return None
    required = {"timestamp", "price_kwh", "action"}
    if not required.issubset(actions_df.columns):
        return None

    work = parse_timestamp_column(actions_df)
    work = ensure_strategy_column(work)
    work = work.sort_values(["strategy", "timestamp"]).reset_index(drop=True)

    if "cumulative_profit" not in work.columns and "profit" in work.columns:
        work["cumulative_profit"] = work.groupby("strategy")["profit"].cumsum()

    if "cumulative_profit" in work.columns:
        work["running_max"] = work.groupby("strategy")["cumulative_profit"].cummax()
        work["drawdown"] = work["running_max"] - work["cumulative_profit"]

    return work


def resolve_date_bounds(*dfs: pd.DataFrame | None) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    mins: list[pd.Timestamp] = []
    maxs: list[pd.Timestamp] = []
    for df in dfs:
        if df is None or "timestamp" not in df.columns or df.empty:
            continue
        mins.append(df["timestamp"].min())
        maxs.append(df["timestamp"].max())
    if not mins:
        return None
    return min(mins), max(maxs)


def apply_date_filter(df: pd.DataFrame | None, start: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame | None:
    if df is None or "timestamp" not in df.columns:
        return df
    work = df
    if start is not None:
        work = work[work["timestamp"] >= start]
    if end is not None:
        work = work[work["timestamp"] < end + pd.Timedelta(days=1)]
    return work


def style_best_legend_item(ax, best_strategy: str | None):
    if best_strategy is None:
        return
    legend = ax.get_legend()
    if legend is None:
        return
    for text in legend.get_texts():
        if text.get_text() == best_strategy:
            text.set_fontweight("bold")
            break


def plot_cumulative_profit_lines(df: pd.DataFrame, strategies: Iterable[str], title: str, best_strategy: str | None = None):
    fig, ax = plt.subplots(figsize=(12, 4))
    for strategy in strategies:
        chunk = df[df["strategy"] == strategy]
        if chunk.empty:
            continue
        lw = 2.8 if strategy == best_strategy else 1.6
        ax.plot(chunk["timestamp"], chunk["cumulative_profit"], label=strategy, linewidth=lw)
    ax.set_title(title)
    ax.set_ylabel("Cumulative Profit ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    style_best_legend_item(ax, best_strategy)
    if best_strategy is not None:
        ax.annotate(
            f"Best strategy: {best_strategy}",
            xy=(0.01, 0.98),
            xycoords="axes fraction",
            va="top",
            ha="left",
            fontsize=10,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "#999999"},
        )
    fig.autofmt_xdate()
    return fig


def plot_drawdown_lines(df: pd.DataFrame, strategies: Iterable[str], title: str, best_strategy: str | None = None):
    fig, ax = plt.subplots(figsize=(12, 4))
    for strategy in strategies:
        chunk = df[df["strategy"] == strategy]
        if chunk.empty or "drawdown" not in chunk.columns:
            continue
        lw = 2.8 if strategy == best_strategy else 1.6
        ax.plot(chunk["timestamp"], chunk["drawdown"], label=strategy, linewidth=lw)
    ax.set_title(title)
    ax.set_ylabel("Drawdown ($)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    style_best_legend_item(ax, best_strategy)
    fig.autofmt_xdate()
    return fig


def plot_strategy_bars(values: pd.Series, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(12, 4))
    values.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    return fig


def pick_best_strategy(summary_df: pd.DataFrame | None, cumulative_df: pd.DataFrame | None) -> tuple[str | None, dict[str, float | str]]:
    if summary_df is not None and not summary_df.empty:
        work = ensure_strategy_column(summary_df, default_name="summary")
        profit_col = pick_profit_column(work)
        drawdown_col = pick_drawdown_column(work)
        cycles_col = pick_cycles_column(work)
        ppc_col = pick_profit_per_cycle_column(work)
        if profit_col is not None:
            best_idx = work[profit_col].astype(float).idxmax()
            row = work.loc[best_idx]
            profit_value = float(row[profit_col])
            cycles_value = float(row[cycles_col]) if cycles_col and pd.notna(row.get(cycles_col)) else np.nan
            ppc_value = float(row[ppc_col]) if ppc_col and pd.notna(row.get(ppc_col)) else np.nan
            if pd.isna(ppc_value) and pd.notna(cycles_value) and cycles_value > 0:
                ppc_value = profit_value / cycles_value
            return str(row["strategy"]), {
                "total_profit": profit_value,
                "max_drawdown": float(row[drawdown_col]) if drawdown_col and pd.notna(row.get(drawdown_col)) else np.nan,
                "equivalent_cycles": cycles_value,
                "profit_per_cycle": ppc_value,
            }

    if cumulative_df is not None and not cumulative_df.empty:
        terminal = cumulative_df.sort_values("timestamp").groupby("strategy")["cumulative_profit"].last()
        if not terminal.empty:
            best_name = str(terminal.idxmax())
            return best_name, {
                "total_profit": float(terminal.max()),
                "max_drawdown": np.nan,
                "equivalent_cycles": np.nan,
                "profit_per_cycle": np.nan,
            }

    return None, {"total_profit": np.nan, "max_drawdown": np.nan, "equivalent_cycles": np.nan, "profit_per_cycle": np.nan}


def get_profit_series(summary_df: pd.DataFrame | None, cumulative_df: pd.DataFrame | None) -> pd.Series | None:
    if summary_df is not None and not summary_df.empty:
        work = ensure_strategy_column(summary_df)
        profit_col = pick_profit_column(work)
        if profit_col is not None:
            profit_series = pd.to_numeric(work[profit_col], errors="coerce")
            return pd.Series(profit_series.values, index=work["strategy"]).dropna()
    if cumulative_df is not None and not cumulative_df.empty:
        terminal = cumulative_df.sort_values("timestamp").groupby("strategy")["cumulative_profit"].last()
        return terminal.dropna()
    return None


def get_summary_metric_series(summary_df: pd.DataFrame | None, metric_col: str) -> pd.Series | None:
    if summary_df is None or summary_df.empty:
        return None
    if metric_col not in summary_df.columns:
        return None
    work = ensure_strategy_column(summary_df)
    metric = pd.to_numeric(work[metric_col], errors="coerce")
    return pd.Series(metric.values, index=work["strategy"]).dropna()


def plot_price_with_trades(df: pd.DataFrame, strategy: str):
    chunk = df[df["strategy"] == strategy].copy()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(chunk["timestamp"], chunk["price_kwh"], color="black", linewidth=1.2, label="Price")
    charges = chunk[chunk["action"] == "charge"]
    discharges = chunk[chunk["action"] == "discharge"]
    ax.scatter(charges["timestamp"], charges["price_kwh"], color="green", marker="^", s=24, label="Charge")
    ax.scatter(discharges["timestamp"], discharges["price_kwh"], color="red", marker="v", s=24, label="Discharge")
    ax.set_title(f"Price + Trade Markers ({strategy})")
    ax.set_ylabel("Price ($/kWh)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    return fig


def render_explanation_panel():
    st.subheader("Model and Strategy Summary")
    st.markdown(
        """
- **Baseline policy**: Charges below a low historical price threshold and discharges above a high threshold.
- **Conformal policy**: Uses forecast median and uncertainty interval to decide if expected edge is large enough to trade.
- **Forecast interval**: The lower/upper conformal band is an uncertainty range around the forecast, calibrated to hit target coverage over time.
        """
    )


def render_overview(summary_df: pd.DataFrame | None, cumulative_df: pd.DataFrame | None):
    st.title("Battery Arbitrage Dashboard")
    st.caption("Cloud-first offline dashboard: reads precomputed artifacts only.")
    st.info(
        "This dashboard evaluates battery energy arbitrage strategies using probabilistic electricity price forecasts.\n\n"
        "Pipeline: Data -> Forecast Model -> Conformal Prediction -> Trading Policy -> Backtest"
    )

    best_name, stats = pick_best_strategy(summary_df, cumulative_df)
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Best Strategy", best_name or "N/A")
    k2.metric("Total Profit", "N/A" if pd.isna(stats["total_profit"]) else f"${stats['total_profit']:,.2f}")
    k3.metric("Max Drawdown", "N/A" if pd.isna(stats["max_drawdown"]) else f"${stats['max_drawdown']:,.2f}")
    k4.metric("Equivalent Cycles", "N/A" if pd.isna(stats["equivalent_cycles"]) else f"{stats['equivalent_cycles']:,.1f}")
    k5.metric("Profit / Cycle", "N/A" if pd.isna(stats["profit_per_cycle"]) else f"${stats['profit_per_cycle']:,.2f}")

    profit_series = get_profit_series(summary_df, cumulative_df)
    if profit_series is not None and not profit_series.empty:
        st.subheader("Profit by Strategy")
        st.caption("Final profit includes terminal battery value when available in backtest summary.")
        st.pyplot(plot_strategy_bars(profit_series.sort_values(ascending=False), "Final Profit by Strategy (with terminal)", "Profit ($)"))

    st.subheader("Cumulative Cash Profit")
    st.caption("This chart excludes terminal battery value and tracks realized cash PnL over time.")
    if cumulative_df is None or cumulative_df.empty:
        st.info("No cumulative profit artifact found. Add `cumulative_profit.csv` or include `profit/cumulative_profit` in `actions.csv`.")
    else:
        strategies = sorted(cumulative_df["strategy"].unique())
        st.pyplot(plot_cumulative_profit_lines(cumulative_df, strategies, "Cumulative Profit by Strategy", best_strategy=best_name))

    render_explanation_panel()

    with st.expander("Optional data sample"):
        if summary_df is not None:
            st.write("Backtest summary")
            st.dataframe(summary_df.head(20), width="stretch")
        if cumulative_df is not None:
            st.write("Cumulative profit")
            st.dataframe(cumulative_df.head(20), width="stretch")


def render_strategy_comparison(actions_df: pd.DataFrame | None, summary_df: pd.DataFrame | None):
    st.title("Strategy Comparison")

    if actions_df is None or actions_df.empty:
        st.info("`actions.csv` not found or missing required columns (`timestamp`, `price_kwh`, `action`).")
        return

    available = sorted(actions_df["strategy"].unique())
    preferred = [s for s in available if "baseline" in s.lower() or "conformal" in s.lower()]
    default_selection = preferred if len(preferred) >= 2 else available[: min(2, len(available))]

    selected = st.multiselect("Strategies", options=available, default=default_selection)
    if not selected:
        st.warning("Select at least one strategy.")
        return

    bounds = resolve_date_bounds(actions_df)
    start = end = None
    if bounds is not None:
        min_ts, max_ts = bounds
        date_range = st.date_input("Date filter", [min_ts.date(), max_ts.date()])
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

    filtered_actions = apply_date_filter(actions_df, start, end)
    filtered_actions = filtered_actions[filtered_actions["strategy"].isin(selected)]
    best_selected = None
    if "cumulative_profit" in filtered_actions.columns and not filtered_actions.empty:
        terminal = filtered_actions.sort_values("timestamp").groupby("strategy")["cumulative_profit"].last()
        if not terminal.empty:
            best_selected = str(terminal.idxmax())

    if filtered_actions.empty:
        st.warning("No rows after filtering.")
        return

    final_profit_series = get_summary_metric_series(summary_df, "total_profit_with_terminal")
    if final_profit_series is not None and not final_profit_series.empty:
        show_profit = final_profit_series.reindex(selected).dropna()
        if not show_profit.empty:
            st.subheader("Final Profit Comparison")
            st.caption("Uses `total_profit_with_terminal` from backtest summary.")
            st.pyplot(plot_strategy_bars(show_profit.sort_values(ascending=False), "Final Profit by Strategy (with terminal)", "Profit ($)"))

    if "cumulative_profit" not in filtered_actions.columns:
        st.info("Cumulative and drawdown charts need `profit` or `cumulative_profit` columns in `actions.csv`.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_cumulative_profit_lines(filtered_actions, selected, "Cumulative Profit (Filtered)", best_strategy=best_selected))
        with c2:
            st.pyplot(plot_drawdown_lines(filtered_actions, selected, "Drawdown (Filtered)", best_strategy=best_selected))

    st.subheader("Trade Activity")
    trade_mask = filtered_actions["action"].isin(["charge", "discharge"])
    trade_counts = filtered_actions[trade_mask].groupby("strategy")["action"].count().reindex(selected).fillna(0)

    bars_col1, bars_col2 = st.columns(2)
    with bars_col1:
        st.pyplot(plot_strategy_bars(trade_counts, "Trade Count", "Trades"))

    with bars_col2:
        if summary_df is not None and "strategy" in summary_df.columns and pick_cycles_column(summary_df):
            cycle_col = pick_cycles_column(summary_df)
            cycle_values = ensure_strategy_column(summary_df).set_index("strategy")[cycle_col].reindex(selected).astype(float)
            st.pyplot(plot_strategy_bars(cycle_values.fillna(0), "Equivalent Cycles (Full Backtest)", "Cycles"))
        else:
            st.info("Equivalent cycles bar needs `backtest_summary.csv` with an `equivalent_cycles` column.")

    st.subheader("Price and Trade Behavior")
    focus_default = best_selected if best_selected in selected else selected[0]
    focus_strategy = st.selectbox("Strategy for price/trade markers", options=selected, index=selected.index(focus_default))
    st.pyplot(plot_price_with_trades(filtered_actions, focus_strategy))

    with st.expander("Optional action sample"):
        st.dataframe(filtered_actions.head(30), width="stretch")


def render_forecast_viewer(forecast_df: pd.DataFrame | None):
    st.title("Forecast Viewer")

    if forecast_df is None or forecast_df.empty:
        st.info("`conformal_forecast.csv` not found.")
        return

    required = {"timestamp", "y_true", "median", "lower_conformal", "upper_conformal"}
    if not required.issubset(forecast_df.columns):
        st.error(f"Forecast artifact missing columns: {sorted(required - set(forecast_df.columns))}")
        return

    work = parse_timestamp_column(forecast_df)
    horizons = sorted(work["horizon"].dropna().unique()) if "horizon" in work.columns else []
    if horizons:
        horizon = st.selectbox("Horizon", horizons)
        work = work[work["horizon"] == horizon]

    if work.empty:
        st.warning("No forecast rows after horizon filter.")
        return

    min_ts, max_ts = work["timestamp"].min(), work["timestamp"].max()
    date_range = st.date_input("Date filter", [min_ts.date(), max_ts.date()])
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        work = apply_date_filter(work, pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))

    if work is None or work.empty:
        st.warning("No forecast rows after date filter.")
        return

    coverage = float(((work["y_true"] >= work["lower_conformal"]) & (work["y_true"] <= work["upper_conformal"])).mean())
    target_coverage = 0.90
    if "target_coverage" in work.columns:
        target_coverage = float(pd.to_numeric(work["target_coverage"], errors="coerce").dropna().iloc[0]) if not pd.to_numeric(work["target_coverage"], errors="coerce").dropna().empty else 0.90
    st.metric("Interval Coverage", f"{coverage * 100:.2f}%", delta=f"target {target_coverage * 100:.1f}%")

    fig, ax = plt.subplots(figsize=(12, 4))
    t = work["timestamp"]
    ax.plot(t, work["y_true"], label="Actual", color="black", linewidth=1)
    ax.plot(t, work["median"], label="Median", color="#1f77b4", linewidth=1)
    ax.fill_between(t, work["lower_conformal"], work["upper_conformal"], color="#1f77b4", alpha=0.2, label="Conformal interval")
    ax.set_title("Forecast vs Actual")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    st.pyplot(fig)

    with st.expander("Optional forecast sample"):
        st.dataframe(work.head(30), width="stretch")


def main():
    from app import live_dashboard

    page = st.sidebar.selectbox(
        "Page",
        [
            "Live Paper Trading",
            "RL Backtest",
            "Overview (legacy)",
            "Strategy Comparison (legacy)",
            "Forecast Viewer (legacy)",
        ],
    )

    st.sidebar.warning(
        "Simulated paper trading — no real money or hardware. "
        "Real AEMO prices, hypothetical trades."
    )
    st.sidebar.markdown("### Deployment Notes")
    st.sidebar.caption(
        "Live pages read the `live-data` branch published by the scheduled "
        "GitHub Actions tick (30-min cadence). Legacy pages show the original "
        "conformal-policy backtest artifacts."
    )

    if page == "Live Paper Trading":
        live_dashboard.render_live_page()
    elif page == "RL Backtest":
        live_dashboard.render_backtest_page()
    elif page == "Overview (legacy)":
        summary_df = load_csv(BACKTEST_FILE)
        actions_df = prepare_actions_df(load_csv(ACTIONS_FILE))
        cumulative_df = prepare_cumulative_df(load_csv(CUM_PROFIT_FILE), actions_df)
        render_overview(summary_df, cumulative_df)
    elif page == "Strategy Comparison (legacy)":
        summary_df = load_csv(BACKTEST_FILE)
        actions_df = prepare_actions_df(load_csv(ACTIONS_FILE))
        render_strategy_comparison(actions_df, summary_df)
    else:
        render_forecast_viewer(load_csv(FORECAST_FILE))


if __name__ == "__main__":
    main()
