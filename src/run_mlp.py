"""Step 4: MLP variants vs. ridge and physics, cross-greenhouse, 3 seeds each.

Usage: python src/run_mlp.py ../sagriot-dss
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, sys.argv[1]); sys.path.insert(0, str(Path(__file__).parent))
from sagriot.evaluate import load_frame, evaluate, summarise, level_for
from sagriot.forecasters import DrivenDrying
from windows import load, make_windows, time_split
from learned import Learned
from mlp import MLPRegressor

CONTEXT, HORIZON, STRIDE, SEASON = 288, 36, 12, 288
SPLITS = [("mugrow", "agrifusion"), ("agrifusion", "mugrow")]
SEEDS = [0, 1, 2]
VARIANTS = [   # name, layout, hidden layers, extra options
    ("mlp_full_32", "full", (32,), {}),
    ("mlp_comp_64", "compressed", (64,), {}),
    ("mlp_comp_64_32", "compressed", (64, 32), {}),
    ("mlp_comp_32_reg", "compressed", (32,), {"l2": 1e-3, "dropout": 0.2}),
    ("mlp_comp_32_skip", "compressed", (32,), {"l2": 1e-3, "dropout": 0.2, "skip": True}),
]

rows = []
for train_name, test_name in SPLITS:
    print(f"\n=== train {train_name} -> test {test_name}", flush=True)
    frame_train = load(f"data/processed/{train_name}.csv")
    fit_part, val_part = time_split(frame_train)
    data = {}
    for layout in ["full", "compressed"]:
        Xa, Ya = make_windows(frame_train, stride=3, layout=layout)
        Xf, Yf = make_windows(fit_part, stride=3, layout=layout)
        Xv, Yv = make_windows(val_part, stride=3, layout=layout)
        data[layout] = (Xa, Ya, Xf, Yf, Xv, Yv)

    models = [DrivenDrying(("vpd",))]
    for layout in ["full", "compressed"]:
        Xa, Ya = data[layout][:2]
        models.append(Learned(f"ridge_{layout}", Ridge(alpha=100.0), layout=layout).fit(Xa, Ya))
    info = {}
    for name, layout, hidden, opts in VARIANTS:
        _, _, Xf, Yf, Xv, Yv = data[layout]
        for seed in SEEDS:
            t = time.time()
            reg = MLPRegressor(hidden, seed, **opts).fit(Xf, Yf, Xv, Yv)
            m = Learned(f"{name}_s{seed}", reg, layout=layout)
            models.append(m)
            info[m.name] = {"params": reg.params(), "epochs": reg.epochs_run, "val_mae": reg.val_mae}
            print(f"  {m.name}: {reg.params()} params, {reg.epochs_run} epochs, val MAE {reg.val_mae:.3f}, {time.time()-t:.0f}s", flush=True)

    path = f"data/raw/{test_name}.csv"
    frame = load_frame(path, ["air_temp", "air_humidity", "par", "soil_fc"], resample="5min")
    level, direction, _ = level_for(path, "soil_fc", frame["soil_fc"].to_numpy(float))
    result = evaluate(frame, "soil_fc", models, CONTEXT, HORIZON, STRIDE, SEASON,
                      threshold=level, direction=direction, drivers=("vpd", "par"))
    table = summarise(result, 5).reset_index().rename(columns={"index": "model"})
    table["params"] = table["model"].map(lambda n: info.get(n, {}).get("params"))
    table.insert(0, "test", test_name); table.insert(0, "train", train_name)
    rows.append(table)
    print(table[["model", "params", "mae", "recall", "precision", "false_alarm", "cross_err_min"]].to_string(index=False), flush=True)

Path("results").mkdir(exist_ok=True)
out = pd.concat(rows)
out.to_csv("results/step4_mlp.csv", index=False)
out["family"] = out["model"].str.replace(r"_s\d$", "", regex=True)
summary = out.groupby(["test", "family"])[["mae", "recall", "precision", "false_alarm", "cross_err_min"]].agg(["mean", "std"]).round(3)
summary.to_csv("results/step4_summary.csv")
print(summary.to_string())
