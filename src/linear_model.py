"""The selected TinyML model: a linear (ridge) forecaster of soil-moisture change.

Input : last 24 h at 5 min -> 288 soil values relative to the current value + 288 VPD values (576 numbers)
Output: change of soil moisture for the next 36 steps (3 h), added to the current value
Size  : 576 x 36 weights + 36 biases = 20,772 parameters (one dense layer)

For each split it trains on one greenhouse (all three sensors), tests on the other
with the internship evaluation code, and saves the weights to models/.

Usage: python src/linear_model.py ../sagriot-dss
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, sys.argv[1]); sys.path.insert(0, str(Path(__file__).parent))
from sagriot.evaluate import load_frame, evaluate, summarise, level_for
from sagriot.forecasters import DrivenDrying
from windows import load, make_windows, CONTEXT, HORIZON
from learned import Learned

ALPHA = 100.0                 # ridge penalty
DRIVERS = ("vpd",)
STRIDE, SEASON = 12, 288      # benchmark: one window per hour
SPLITS = [("mugrow", "agrifusion"), ("agrifusion", "mugrow")]


def train(train_name):
    X, Y = make_windows(load(f"data/processed/{train_name}.csv"), DRIVERS, stride=3)
    model = Ridge(alpha=ALPHA).fit(X, Y)
    return model, len(X)


def save(model, train_name, n_windows):
    Path("models").mkdir(exist_ok=True)
    stem = f"models/linear_{train_name}"
    np.savez(f"{stem}.npz", weights=model.coef_.T.astype(np.float32), bias=model.intercept_.astype(np.float32))
    meta = {"model": "ridge", "alpha": ALPHA, "trained_on": train_name, "training_windows": n_windows,
            "inputs": "288 soil values minus current value, then 288 VPD values (kPa)",
            "outputs": "36 changes from the current soil value (5-min steps, 3 h)",
            "context": CONTEXT, "horizon": HORIZON, "parameters": int(model.coef_.size + model.intercept_.size)}
    Path(f"{stem}.json").write_text(json.dumps(meta, indent=2))


rows = []
for train_name, test_name in SPLITS:
    model, n = train(train_name)
    save(model, train_name, n)
    path = f"data/raw/{test_name}.csv"
    frame = load_frame(path, ["air_temp", "air_humidity", "par", "soil_fc"], resample="5min")
    level, direction, _ = level_for(path, "soil_fc", frame["soil_fc"].to_numpy(float))
    result = evaluate(frame, "soil_fc", [DrivenDrying(("vpd",)), Learned("linear", model, DRIVERS)],
                      CONTEXT, HORIZON, STRIDE, SEASON, threshold=level, direction=direction,
                      drivers=("vpd", "par"))
    table = summarise(result, 5).reset_index().rename(columns={"index": "model"})
    table.insert(0, "test", test_name); table.insert(0, "train", train_name)
    rows.append(table)
    print(f"\ntrain {train_name} -> test {test_name} ({n} training windows)")
    print(table[["model", "mae", "events", "recall", "precision", "false_alarm", "cross_err_min"]].to_string(index=False))

pd.concat(rows).to_csv("results/linear_model.csv", index=False)
