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

## Open points
- Training greenhouses (ideas, tomatonuts, trigger, reference) are not here yet.
- Permittivity → VWC: Topp is made for mineral soil, not rockwool. Because the decision uses % of field capacity, the exact conversion matters less, but it should be stated as a limitation.
- One series per sensor (3 per greenhouse) or the mean of the three?
