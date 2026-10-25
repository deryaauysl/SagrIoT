# Data notes: Autonomous Greenhouse Challenge (AGC)

First look at the two test greenhouses, exported from Numbers (25 Oct 2026).

## Files
| Greenhouse | Rows | Period (UTC) | Step |
|---|---|---|---|
| AgriFusion | 22,621 | 2024-09-02 → 2024-11-20 | 5 min, no gaps in timestamps |
| MuGrow | 20,029 | 2024-09-02 → 2024-11-11 | 5 min, no gaps in timestamps |

`mugrow_2` is an exact copy of `mugrow`.

## Columns we need
| Need | Column | Note |
|---|---|---|
| Soil moisture | `compartment/substrate/relative_permittivity` (+ `.1`, `.2`) | **Permittivity, not VWC.** Three sensors per greenhouse. Converted with the Topp equation for now. |
| Air temperature | `compartment/air_temperature` | °C |
| Relative humidity | `compartment/relative_humidity` | % |
| VPD | computed from temperature and RH (Tetens, same as the node) | `humidity_deficit` is in g/m³, not kPa: not used |
| Irrigation | `compartment/water_supply/water_flow_duration` | Looks like a daily counter of irrigation pulses (steps up, resets at night) |

## What the data looks like
- Clear daily sawtooth: soil dries overnight, irrigation in the morning refills it.
- The three sensors sit at different levels (different slabs / positions).
- Level jumps around 18 Sep and 3 Oct (sensor moved?), a gap around 14–16 Oct, and a drop to ~0 at the very last sample (sensor removed). These need cleaning.
- About 4% of substrate values are missing.

## How the internship benchmark uses this data (`sagriot-dss/sagriot/evaluate.py`)
This is what our model must match to be compared with the 26% / 16% figures.

- **Soil series:** the **mean of the three permittivity sensors**, no Topp conversion.
- **"% FC":** that mean divided by its own 95th percentile, × 100, clipped to 0–100.
- **Event level:** the **15th percentile** of that series (MuGrow 78.9, AgriFusion 82.0), not 70% FC.
- **Windows:** 24 h in, 3 h out, one window every hour. An event = a window that starts above the level and goes below it within 3 h. Events are counted per window (MuGrow 104, AgriFusion 108).
- **Physics model's future VPD:** the VPD 24 h earlier (seasonal).

## Reproduced baseline (`src/reproduce_baseline.py`, `results/baseline_reproduced.csv`)
| Greenhouse | Model | Events | Recall | Precision | Timing error |
|---|---|---|---|---|---|
| MuGrow | driven drying (physics) | 104 | 0.240 | 0.781 | 27.6 min |
| MuGrow | seasonal naive | 104 | 0.596 | 0.408 | 61.0 min |
| AgriFusion | driven drying (physics) | 108 | 0.148 | 0.762 | 41.3 min |
| AgriFusion | seasonal naive | 108 | 0.787 | 0.669 | 55.2 min |

Internship report: 0.26 and 0.157 with 100 and 108 events. The small gap on MuGrow is most likely the export (Numbers → CSV); AgriFusion matches.

## Decisions (25 Oct 2026)
- The AGC training greenhouses are not available, so **MuGrow and AgriFusion are used for training**. To keep the "not seen in training" claim, the plan is to train on one and test on the other, both ways.
- **Each sensor is a separate series** for training. Evaluation uses the same mean series as the benchmark, so the comparison stays fair.

## Open points
- Permittivity → VWC: Topp is made for mineral soil, not rockwool. Because the decision uses % of field capacity, the exact conversion matters less, but it should be stated as a limitation.
- One series per sensor (3 per greenhouse) or the mean of the three?

## Step 4 result: MLP vs. linear (25 Oct 2026, `results/step4_summary.csv`)
Five MLP variants, three seeds each, against ridge (linear) and the physics model.
No MLP beat the linear model on both greenhouses. Larger MLPs catch a few more
events on AgriFusion but give more false alarms, and on MuGrow they are worse
than the physics model in MAE: they fit the training greenhouse too closely.
Regularized MLPs end up close to the linear model. **Decision: the linear model
(one Dense layer) is the model taken to the ESP32.**
