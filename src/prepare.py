"""Turn the AGC exports into clean 5-minute training series.

One table per greenhouse in data/processed/<name>.csv with:
  vpd, par            air drivers (VPD in kPa, Tetens, same formula as the node)
  fc_s0, fc_s1, fc_s2 each substrate sensor as % of its own 95th percentile
  fc_mean             mean of the three sensors, scaled the same way (= the benchmark series)

Scaling follows sagriot-dss/evaluate.py (value / p95 * 100, clipped 0-100), so a
model trained on one sensor sees the same kind of numbers it is evaluated on.

Usage: python src/prepare.py data/raw data/processed
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PERM = "compartment/substrate/relative_permittivity"
MIN_PERMITTIVITY = 5.0      # below this the probe is out of the substrate (same cut as the reference EC code)


def vpd_kpa(temp, rh):
    es = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
    return np.round(es - es * rh / 100, 3)


def to_fc(series):
    return (series / series.quantile(0.95) * 100).clip(0, 100)


def prepare(path):
    raw = pd.read_csv(path, low_memory=False)
    index = pd.to_datetime(raw["time"], utc=True).dt.tz_convert("Europe/Amsterdam")
    num = lambda c: pd.Series(pd.to_numeric(raw[c], errors="coerce").to_numpy(), index=index)

    out = pd.DataFrame(index=index)
    out["vpd"] = vpd_kpa(num("compartment/air_temperature"), num("compartment/relative_humidity"))
    out["par"] = num("compartment/par")
    sensors = []
    for i, col in enumerate([PERM, PERM + ".1", PERM + ".2"]):
        s = num(col)
        s[s < MIN_PERMITTIVITY] = np.nan
        sensors.append(s)
        out[f"fc_s{i}"] = s
    out["fc_mean"] = pd.concat(sensors, axis=1).mean(axis=1)

    out = out.sort_index().resample("5min").mean().interpolate(limit=3, limit_area="inside")
    for c in ["fc_s0", "fc_s1", "fc_s2", "fc_mean"]:
        out[c] = to_fc(out[c])
    return out


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["mugrow", "agrifusion"]:
        frame = prepare(src / f"{name}.csv")
        frame.to_csv(dst / f"{name}.csv")
        print(f"{name}: {len(frame)} rows, {frame.index[0]} -> {frame.index[-1]}")
        print(frame.describe().loc[["count", "min", "50%", "max"]].round(2).to_string())
