"""Reproduce the internship benchmark for the physics model on the two AGC greenhouses.

Uses the evaluation code from zehragzl/sagriot-dss unchanged, so our numbers are
directly comparable with the internship report.

Usage:
  git clone https://github.com/zehragzl/sagriot-dss ../sagriot-dss
  python src/reproduce_baseline.py ../sagriot-dss data/raw   # data/raw/{mugrow,agrifusion}.csv
"""
import sys
from pathlib import Path

dss, data = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(dss))
from sagriot.evaluate import load_frame, evaluate, summarise, level_for
from sagriot.forecasters import Persistence, SeasonalNaive, DrivenDrying

CONTEXT, HORIZON, STRIDE, SEASON = 288, 36, 12, 288   # 24 h in, 3 h out, every hour

rows = []
for name in ["mugrow", "agrifusion"]:
    path = str(data / f"{name}.csv")
    frame = load_frame(path, ["air_temp", "air_humidity", "par", "soil_fc"], resample="5min")
    level, direction, source = level_for(path, "soil_fc", frame["soil_fc"].to_numpy(float))
    result = evaluate(frame, "soil_fc", [Persistence(), SeasonalNaive(SEASON), DrivenDrying(("vpd",))],
                      CONTEXT, HORIZON, STRIDE, SEASON, threshold=level, direction=direction,
                      drivers=("vpd", "par"))
    table = summarise(result, 5)
    print(f"\n{name}: event level {level:.2f} ({source})\n{table.to_string()}")
    table = table.reset_index(); table.insert(0, "dataset", name); rows.append(table)

import pandas as pd
Path("results").mkdir(exist_ok=True)
pd.concat(rows).to_csv("results/baseline_reproduced.csv", index=False)
