"""Learned forecasters wrapped in the benchmark's Forecaster interface."""
import numpy as np
from windows import features, CONTEXT


class Learned:
    """Any regressor with fit/predict on (features -> 36 changes)."""

    def __init__(self, name, regressor, drivers=("vpd",)):
        self.name, self.regressor, self.drivers = name, regressor, drivers

    def fit(self, X, Y):
        self.regressor.fit(X, Y)
        return self

    def warm_up(self, context_length, horizon):
        pass

    def predict(self, history, horizon, exog_past=None, exog_future=None):
        history = np.asarray(history, dtype=float)
        x = features(history, [exog_past[d] for d in self.drivers])[None, :]
        return history[-1] + self.regressor.predict(x)[0][:horizon]
