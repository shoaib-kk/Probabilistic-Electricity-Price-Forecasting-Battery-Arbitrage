"""Perfect-foresight upper bound via linear programming.

Given the full realized price path (which no causal strategy can see),
choose charge/discharge energies maximizing total profit net of fees and
degradation, subject to SoC dynamics and power limits. Continuous power
is allowed, so this is a strict upper bound on the Discrete(5) policies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from src import config
from src.rl.env import EnvData


def solve_perfect_foresight(
    data: EnvData,
    battery_cfg: config.BatteryConfig = config.BatteryConfig(),
    costs: config.CostConfig = config.CostConfig(),
    dt_hours: float = config.DT_HOURS,
) -> pd.DataFrame:
    """Returns a per-interval ledger DataFrame like env.rollout_policy."""
    p_kwh = np.asarray(data.prices_mwh, dtype=float) / 1000.0
    n = len(p_kwh)
    cap = battery_cfg.capacity_kwh
    e_max = battery_cfg.max_power_kw * dt_hours  # kWh per interval at full power
    eta_c = battery_cfg.charge_efficiency
    eta_d = battery_cfg.discharge_efficiency
    soc0 = battery_cfg.initial_soc * cap

    # Variables x = [c_0..c_{n-1}, d_0..d_{n-1}, s_0..s_{n-1}]
    # c/d are grid-side kWh bought/sold; s is stored energy after interval t.
    fee = costs.fee_rate * np.abs(p_kwh)
    deg = costs.degradation_cost_per_kwh
    cost_c = p_kwh + fee + deg  # minimize => buying costs money
    cost_d = -p_kwh + fee + deg  # selling earns money
    c_obj = np.concatenate([cost_c, cost_d, np.zeros(n)])

    # SoC dynamics: s_t - s_{t-1} - eta_c*c_t + d_t/eta_d = 0  (s_{-1} = soc0)
    rows, cols, vals, rhs = [], [], [], []
    for t in range(n):
        rows += [t, t, t]
        cols += [t, n + t, 2 * n + t]
        vals += [-eta_c, 1.0 / eta_d, 1.0]
        if t > 0:
            rows.append(t)
            cols.append(2 * n + t - 1)
            vals.append(-1.0)
            rhs.append(0.0)
        else:
            rhs.append(soc0)
    A_eq = sparse.coo_matrix((vals, (rows, cols)), shape=(n, 3 * n))

    bounds = [(0.0, e_max)] * n + [(0.0, e_max)] * n + [(0.0, cap)] * n
    res = linprog(
        c_obj, A_eq=A_eq.tocsc(), b_eq=np.array(rhs), bounds=bounds, method="highs"
    )
    if not res.success:
        raise RuntimeError(f"Perfect-foresight LP failed: {res.message}")

    c_kwh, d_kwh, s_kwh = res.x[:n], res.x[n : 2 * n], res.x[2 * n :]
    fee_paid = costs.fee_rate * (c_kwh + d_kwh) * np.abs(p_kwh)
    degradation = deg * (c_kwh + d_kwh)
    profit = d_kwh * p_kwh - c_kwh * p_kwh - fee_paid - degradation

    df = pd.DataFrame(
        {
            "timestamp": data.timestamps,
            "price_mwh": data.prices_mwh,
            "action": np.where(
                c_kwh > 1e-6, "charge", np.where(d_kwh > 1e-6, "discharge", "hold")
            ),
            "power_kw": (c_kwh + d_kwh) / dt_hours,
            "energy_bought_kwh": c_kwh,
            "energy_sold_kwh": d_kwh,
            "soc": s_kwh / cap,
            "fee": fee_paid,
            "degradation": degradation,
            "profit": profit,
            "cumulative_profit": np.cumsum(profit),
        }
    ).set_index("timestamp")
    return df
