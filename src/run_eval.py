"""Cross-greenhouse evaluation: train on one greenhouse, test on the other.

Test uses the benchmark's own loader, event level and evaluate() from sagriot-dss,
so every number is comparable with the internship report.

Usage: python src/run_eval.py ../sagriot-dss
"""
import sys
from pathlib import Path
import pandas as pd
from sklearn.linear_model import Ridge

sys.path.insert(0, sys.argv[1]); sys.path.insert(0, str(Path(__file__).parent))
from sagriot.evaluate import load_frame, evaluate, summarise, level_for
from sagriot.forecasters import Persistence, SeasonalNaive, DrivenDrying
from windows import load, make_windows
from learned import Learned

CONTEXT, HORIZON, STRIDE, SEASON = 288, 36, 12, 288
SPLITS = [("mugrow", "agrifusion"), ("agrifusion", "mugrow")]


def learned_models(train):
    models = []
    for drivers in [("vpd",), ("vpd", "par")]:
        X, Y = make_windows(train, drivers, stride=3)
        tag = "+".join(drivers)
        models.append(Learned(f"ridge({tag})", Ridge(alpha=100.0), drivers).fit(X, Y))
        print(f"  ridge({tag}): {len(X)} training windows, {X.shape[1]} inputs")
    return models


rows = []
for train_name, test_name in SPLITS:
    print(f"\n=== train {train_name} -> test {test_name}")
    train = load(f"data/processed/{train_name}.csv")
    models = [Persistence(), SeasonalNaive(SEASON), DrivenDrying(("vpd",))] + learned_models(train)

    path = f"data/raw/{test_name}.csv"
    frame = load_frame(path, ["air_temp", "air_humidity", "par", "soil_fc"], resample="5min")
    level, direction, _ = level_for(path, "soil_fc", frame["soil_fc"].to_numpy(float))
    result = evaluate(frame, "soil_fc", models, CONTEXT, HORIZON, STRIDE, SEASON,
                      threshold=level, direction=direction, drivers=("vpd", "par"))
    table = summarise(result, 5)
    print(table[["mae", "events", "recall", "precision", "false_alarm", "cross_err_min"]].to_string())
    table = table.reset_index().rename(columns={"index": "model"})
    table.insert(0, "test", test_name); table.insert(0, "train", train_name)
    rows.append(table)

Path("results").mkdir(exist_ok=True)
pd.concat(rows).to_csv("results/step3_baselines.csv", index=False)
