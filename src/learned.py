"""Learned forecasters wrapped in the benchmark's Forecaster interface."""
import numpy as np
from windows import features


class Learned:
    """A regressor mapping features -> 36 changes from the current value."""

    def __init__(self, name, regressor, drivers=("vpd",), layout="full"):
        self.name, self.regressor, self.drivers, self.layout = name, regressor, drivers, layout

    def fit(self, X, Y):
        self.regressor.fit(X, Y)
        return self

    def warm_up(self, context_length, horizon):
        pass

    def predict(self, history, horizon, exog_past=None, exog_future=None):
        history = np.asarray(history, dtype=float)
        x = features(history, [exog_past[d] for d in self.drivers], self.layout)[None, :].astype(np.float32)
        return history[-1] + np.asarray(self.regressor.predict(x))[0][:horizon]
