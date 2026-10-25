"""Training windows and the input features shared by every learned model.

A window is 24 h of history (288 steps) and the next 3 h (36 steps).
The model sees the soil series relative to its current value, so sensor level
jumps do not matter, and predicts the change from the current value.
"""
import numpy as np
import pandas as pd

CONTEXT, HORIZON = 288, 36
SENSORS = ["fc_s0", "fc_s1", "fc_s2"]


def features(soil, drivers):
    """soil: (288,) array, drivers: list of (288,) arrays -> 1-D feature vector."""
    soil = np.asarray(soil, dtype=float)
    parts = [soil - soil[-1]] + [np.asarray(d, dtype=float) for d in drivers]
    return np.concatenate(parts)


def target(soil_history_last, future):
    return np.asarray(future, dtype=float) - soil_history_last


def load(path):
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


def make_windows(frame, drivers=("vpd",), stride=1, sensors=SENSORS):
    """All complete windows from every sensor of one greenhouse."""
    X, Y = [], []
    drv = [frame[d].to_numpy(float) for d in drivers]
    for s in sensors:
        soil = frame[s].to_numpy(float)
        for end in range(CONTEXT, len(soil) - HORIZON + 1, stride):
            h = soil[end - CONTEXT:end]
            f = soil[end:end + HORIZON]
            past = [d[end - CONTEXT:end] for d in drv]
            if not (np.isfinite(h).all() and np.isfinite(f).all() and all(np.isfinite(p).all() for p in past)):
                continue
            X.append(features(h, past))
            Y.append(target(h[-1], f))
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
