"""Training windows and the input features shared by every learned model.

A window is 24 h of history (288 steps) and the next 3 h (36 steps).
The model sees the soil series relative to its current value, so sensor level
jumps do not matter, and predicts the change from the current value.

Two input layouts:
  full        every 5-min step: 288 values per channel
  compressed  last 2 h at 5 min (24 values) + the 22 h before as hourly means (22 values)
              = 46 values per channel
"""
import numpy as np
import pandas as pd

CONTEXT, HORIZON = 288, 36
SENSORS = ["fc_s0", "fc_s1", "fc_s2"]
RECENT = 24          # steps kept at full resolution in the compressed layout


def _compress(x):
    x = np.asarray(x, dtype=float)
    old = x[:-RECENT].reshape(-1, 12).mean(axis=1)
    return np.concatenate([old, x[-RECENT:]])


def features(soil, drivers, layout="full"):
    """soil: (288,) array, drivers: list of (288,) arrays -> 1-D feature vector."""
    soil = np.asarray(soil, dtype=float)
    parts = [soil - soil[-1]] + [np.asarray(d, dtype=float) for d in drivers]
    if layout == "compressed":
        parts = [_compress(p) for p in parts]
    return np.concatenate(parts)


def load(path):
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


def make_windows(frame, drivers=("vpd",), stride=1, sensors=SENSORS, layout="full"):
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
            X.append(features(h, past, layout))
            Y.append(f - h[-1])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def time_split(frame, fraction=0.8):
    """First part for training, last part for validation (no shuffling: windows overlap)."""
    cut = frame.index[int(len(frame) * fraction)]
    return frame[frame.index < cut], frame[frame.index >= cut]
