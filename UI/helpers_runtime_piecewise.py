from __future__ import annotations

from typing import Optional, Sequence, Type

import numpy as np
from sklearn.base import clone
from survivors.external import BaseSAAdapter


class PiecewiseClassifWrapSA(BaseSAAdapter):
    def __init__(self, model, times: int = 8, start_at_zero: bool = True):
        self.model = model
        self.model_cls: Type = model.__class__
        self.times = int(times)
        self.start_at_zero = start_at_zero
        self.__name__ = f"PiecewiseClassifWrapSA({self.model_cls.__name__}, times={self.times})"
        self._bounds: Optional[tuple[float, float]] = None
        self.models_: list = []
        self.bin_edges_: Optional[np.ndarray] = None

    def __call__(self, **kwargs):
        self.model = self.model_cls(**kwargs)
        return self

    def _build_piecewise_bin_edges(self, t) -> np.ndarray:
        t = np.asarray(t, float)
        t = t[np.isfinite(t)]
        if t.size == 0:
            return np.array([0.0, 1.0], float)

        left = 0.0 if self.start_at_zero else float(np.min(t))
        right = float(np.max(t))
        n_edges = max(self.times, 2)
        if right <= left:
            return np.array([left, left + 1.0], float)

        edges = np.quantile(t, np.linspace(0.0, 1.0, n_edges)).astype(float)
        edges[0] = left
        edges[-1] = right
        edges = np.unique(edges)
        if edges.size < 2:
            return np.array([left, max(right, left + 1.0)], float)
        return edges

    def _interval_training_mask(self, t, e, a, b):
        at_risk = t > a
        yj = ((e == 1) & (t > a) & (t <= b)).astype(int)
        return at_risk, yj

    def fit(self, X, y, time_col: str = "time", event_col: str = "cens"):
        t, e = self.timeWrap(y, time_col, event_col)
        left = float(np.min(t))
        right = float(np.max(t))
        if not np.isfinite(left) or not np.isfinite(right) or left >= right:
            left = 0.0
            right = 1.0
        self._bounds = (left, right)

        self.bin_edges_ = self._build_piecewise_bin_edges(t)
        self.models_ = []

        for j in range(1, len(self.bin_edges_)):
            a = self.bin_edges_[j - 1]
            b = self.bin_edges_[j]
            mask, yj = self._interval_training_mask(t, e, a, b)

            if hasattr(X, "iloc"):
                Xj = X.iloc[mask]
            else:
                Xj = X[mask]
            yj = yj[mask]

            if len(yj) == 0:
                self.models_.append(None)
                continue
            if np.all(yj == 0):
                self.models_.append(0.0)
                continue
            if np.all(yj == 1):
                self.models_.append(1.0)
                continue

            model = clone(self.model)
            model.fit(Xj, yj)
            self.models_.append(model)

        return self

    def _predict_interval_proba(self, model, X):
        n = len(X) if hasattr(X, "__len__") else 1
        if model is None:
            return np.zeros(n, float)
        if isinstance(model, (float, int)):
            return np.full(n, float(model), float)
        return self._get_proba(model, X)

    def predict_survival_function(self, X, times: Optional[Sequence[float]] = None):
        assert self._bounds is not None, "Call fit() first"
        assert self.bin_edges_ is not None, "Call fit() first"

        base_t = self.bin_edges_[1:]
        hazards = []
        for model in self.models_:
            p = self._predict_interval_proba(model, X)
            hazards.append(np.clip(np.asarray(p, float), 0.0, 1.0))

        if not hazards:
            n = len(X) if hasattr(X, "__len__") else 1
            t = np.asarray(times, float) if times is not None else np.array([], float)
            return np.ones((n, t.size), float), t

        step_survival = np.cumprod(1.0 - np.column_stack(hazards), axis=1)
        if times is None:
            return step_survival, base_t

        t = np.asarray(times, float)
        survival = np.ones((step_survival.shape[0], t.size), float)
        for k, time_value in enumerate(t):
            if time_value < base_t[0]:
                survival[:, k] = 1.0
                continue
            idx = np.searchsorted(base_t, time_value, side="left")
            idx = min(idx, step_survival.shape[1] - 1)
            survival[:, k] = step_survival[:, idx]
        return survival, t

    def predict_hazard_function(self, X, times=None):
        survival, t = self.predict_survival_function(X, times)
        hazard = -np.log(np.clip(survival, 1e-12, 1.0))
        return hazard, t

    def predict_expected_time(self, X, times=None):
        survival, t = self.predict_survival_function(X, times)
        if t.size == 0:
            n = len(X) if hasattr(X, "__len__") else 1
            return np.zeros(n, float)
        t_full = np.r_[0.0, t]
        survival_full = np.column_stack([np.ones(survival.shape[0]), survival])
        integrate = getattr(np, "trapezoid", np.trapz)
        return integrate(survival_full, t_full, axis=1)

    def predict_time(self, X):
        survival, t = self.predict_survival_function(X)
        med = np.full(survival.shape[0], np.nan, float)
        for idx, row in enumerate(survival):
            hit = np.argmax(row <= 0.5)
            if row.size and row[hit] <= 0.5:
                med[idx] = t[hit]
        return med

    def predict_proba(self, X):
        survival, _ = self.predict_survival_function(X)
        if survival.shape[1] == 0:
            p = np.zeros(survival.shape[0], float)
        else:
            p = 1.0 - survival[:, -1]
        return np.column_stack([1.0 - p, p])


class PiecewiseCensorAwareClassifWrapSA(PiecewiseClassifWrapSA):
    def __init__(self, model, times: int = 8, start_at_zero: bool = True):
        super().__init__(model, times=times, start_at_zero=start_at_zero)
        self.__name__ = f"PiecewiseCensorAwareClassifWrapSA({self.model_cls.__name__}, times={self.times})"

    def _interval_training_mask(self, t, e, a, b):
        event_in_interval = (e == 1) & (t > a) & (t <= b)
        known_survived_interval = t > b
        mask = event_in_interval | known_survived_interval
        yj = event_in_interval.astype(int)
        return mask, yj


PIECEWISE_RUNTIME_CLASSES = {
    "PiecewiseClassifWrapSA": PiecewiseClassifWrapSA,
    "PiecewiseCensorAwareClassifWrapSA": PiecewiseCensorAwareClassifWrapSA,
}
