"""Persistable probabilistic price forecaster.

LightGBM quantile models per (horizon, quantile) with split-conformal
widening of the outer interval, trained on the compact feature set from
src.rl.features. Predictions feed the RL agent's observation vector both
in backtests and in the live loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.rl import features as F

logger = logging.getLogger(__name__)


def _make_model(quantile: float) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=quantile,
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1,
    )


@dataclass
class PriceForecaster:
    horizons: list[int] = field(default_factory=lambda: list(config.FORECAST_HORIZONS))
    quantiles: list[float] = field(default_factory=lambda: list(config.QUANTILES))
    target_coverage: float = config.TARGET_COVERAGE
    feature_cols: list[str] = field(default_factory=list)
    models: dict = field(default_factory=dict)  # {h: {q: LGBMRegressor}}
    conformal_q: dict = field(default_factory=dict)  # {h: float}

    def fit(
        self,
        feature_frame: pd.DataFrame,
        rrp: pd.Series,
        train_index: pd.Index,
        cal_index: pd.Index,
    ) -> "PriceForecaster":
        """Train on train_index rows, calibrate conformal offsets on cal_index."""
        self.feature_cols = list(feature_frame.columns)
        q_low, q_high = min(self.quantiles), max(self.quantiles)

        for h in self.horizons:
            target = F.target_for_horizon(rrp, h)
            tr = self._usable(feature_frame, target, train_index)
            ca = self._usable(feature_frame, target, cal_index)
            if len(tr) < 500 or len(ca) < 50:
                raise ValueError(
                    f"Not enough rows to fit horizon {h}: train={len(tr)}, cal={len(ca)}"
                )
            X_tr, y_tr = feature_frame.loc[tr], target.loc[tr]
            X_ca, y_ca = feature_frame.loc[ca], target.loc[ca]

            self.models[h] = {}
            for q in self.quantiles:
                model = _make_model(q)
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_ca, y_ca)],
                    eval_metric="quantile",
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                self.models[h][q] = model

            lo = self.models[h][q_low].predict(X_ca)
            hi = self.models[h][q_high].predict(X_ca)
            scores = np.maximum(lo - y_ca.to_numpy(), y_ca.to_numpy() - hi)
            scores = np.maximum(scores, 0.0)
            self.conformal_q[h] = float(
                np.quantile(scores, self.target_coverage)
            )
            logger.info(
                "h=%d trained (train=%d cal=%d) conformal_q=%.2f",
                h, len(tr), len(ca), self.conformal_q[h],
            )
        return self

    @staticmethod
    def _usable(
        feature_frame: pd.DataFrame, target: pd.Series, index: pd.Index
    ) -> pd.Index:
        rows = feature_frame.loc[feature_frame.index.intersection(index)]
        mask = ~rows.isna().any(axis=1) & target.loc[rows.index].notna()
        return rows.index[mask]

    def predict(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        """Quantile forecasts per row; outer quantiles conformally widened.

        Rows with NaN features get NaN predictions (dropped downstream).
        """
        q_low, q_high = min(self.quantiles), max(self.quantiles)
        X = feature_frame[self.feature_cols]
        ok = ~X.isna().any(axis=1)
        out = pd.DataFrame(
            index=feature_frame.index,
            columns=F.forecast_columns(self.horizons, self.quantiles),
            dtype=float,
        )
        if ok.sum() == 0:
            return out
        for h in self.horizons:
            adj = self.conformal_q.get(h, 0.0)
            for q in self.quantiles:
                pred = self.models[h][q].predict(X.loc[ok])
                if q == q_low:
                    pred = pred - adj
                elif q == q_high:
                    pred = pred + adj
                out.loc[ok, f"h{h}_q{int(q * 100):02d}"] = pred
        # Repair quantile crossing so downstream code can rely on ordering.
        for h in self.horizons:
            cols = [f"h{h}_q{int(q * 100):02d}" for q in sorted(self.quantiles)]
            vals = out[cols].to_numpy(dtype=float)
            out[cols] = np.sort(vals, axis=1)
        return out

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "PriceForecaster":
        return joblib.load(path)
