# SAgrIoT – TinyML soil-moisture forecaster

Part of the GTÜ CSE495/496 graduation project P51 (SAgrIoT, edge vs. cloud irrigation decisions).
This repo holds work package A: a small forecasting model that runs on the ESP32 (int8) and in the cloud (float).

## Task
- **Input:** recent soil moisture and VPD (window length to be decided from the data).
- **Output:** soil moisture 30, 60, 120 and 180 minutes ahead.
- **Decision:** a simple rule on the forecast: warn if soil moisture will fall below the threshold (70% FC) within 1 hour. The same rule runs on the physics model's forecast for comparison.

## Data
- 4th Autonomous Greenhouse Challenge – Dwarf Tomato Timeseries (4TU.ResearchData, doi:10.4121/fa102772-32db-4b30-bace-12f2016722ce). Raw files go in `data/raw/` and are not committed.
- Internship testbed record and school greenhouse record (later).

## Success criteria (first presentation)
| # | Target |
|---|---|
| Early warning | Catch ≥ 40% of dry-soil events in two greenhouses not used in training (physics model: 26% and 16%) |
| Model on the ESP32 | ≤ 50 KB model, ≤ 100 KB RAM, ≤ 100 ms per decision; same decision as the cloud model in ≥ 90% of cases |

## Selected model
`src/linear_model.py` trains the linear (ridge) forecaster on one greenhouse, tests it on the other
with the internship evaluation code, and saves the weights to `models/linear_<train>.npz`.

| Train → test | Events caught (recall) | Warnings that were real (precision) | Physics model recall |
|---|---|---|---|
| MuGrow → AgriFusion | 54.6% | 86.8% | 14.8% |
| AgriFusion → MuGrow | 45.2% | 88.7% | 24.0% |

Run: `python src/prepare.py data/raw data/processed`, then `python src/linear_model.py ../sagriot-dss`.

## Layout
```
data/raw/        original downloads (git-ignored)
data/processed/  cleaned 5-minute tables (git-ignored)
notebooks/       exploration
src/             data preparation, windows, models, evaluation
models/          trained and converted models
results/         tables and figures
```
