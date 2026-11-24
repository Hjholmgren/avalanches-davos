# Avalanche day predictor: Davos

**Question:** how likely are natural avalanches around Davos on a given day of the snow season (November-May), and how does a simple model compare with the official SLF avalanche forecasters?

**Answer:** a logistic regression on daily weather (plus whether avalanches happened yesterday) beats the official danger level in **98% of 200 random train/test splits** (average AUC **0.74 vs 0.69**).

## How to run

```bash
uv sync                                   # install the packages
# download the SLF data (see below) into data/
# open avalanche_model.ipynb in VS Code, select the .venv kernel, Run All
```

The weather is downloaded automatically the first time (Open-Meteo, free, no API key).

## Data

| Source | What | Period |
|---|---|---|
| [SLF: Snow avalanche data Davos](https://www.envidat.ch/dataset/snow-avalanche-data-davos) (doi:10.16904/envidat.134) | One row per winter day: number of avalanches by size, type and trigger, and the official danger level (1-5). Download `avalanche_activity_data.csv` into `data/`. | 1999-2019, 3,533 days |
| [Open-Meteo archive](https://open-meteo.com/en/docs/historical-weather-api) | Daily weather at 2,500 m: snowfall, snow depth, temperature, wind gusts, sun | same days |

**Target:** an *avalanche day* = at least one **natural** avalanche of **size 2+** (big enough to bury a person). Snow season November-May (3,530 days, 25% avalanche days). November and May have few avalanches, so each month gets its own baseline.

## Method

1. **Features** (only today and earlier days, no future data):

| Avalanche type | Cause | Features |
|---|---|---|
| New snow slab (dry) | New snow overloads the snowpack | snow 72h / 7d, snow depth |
| Wind slab (dry) | Wind packs loose snow into slabs | gust x new snow, gusts |
| Weak-layer slab (dry) | Cold air over thin snow grows weak crystals; dangerous once new snow falls on top | temperature gradient, snow on a weak layer (> 10 C/m) |
| Release after storms (dry) | Fresh snow that suddenly warms | new snow x warming |
| Wet snow | Warmth, sudden warming, sunshine | temp max / min, temp change, warm sunny day, snow depth |
| All | Season; avalanche cycles | one column per month, avalanche yesterday |

2. **Two logistic regressions:** dry avalanches (driven by snow) and wet avalanches (driven by warmth), combined as P = 1 - (1 - P_dry)(1 - P_wet).
3. **Split by whole winters, at random:** train on 15 winters, test on 6. Whole winters, because neighbouring days are almost identical.
4. **Benchmark:** the official SLF danger level, scored on the same test days.

## Results

One random split (test winters 1999, 2004, 2005, 2008, 2009, 2012):

| | AUC | Hit rate | False alarms |
|---|---|---|---|
| Forecasters (danger level) | 0.66 | 63% | 66% |
| **Our model** | **0.72** | **66%** | **63%** |

Example, avalanche winter 1999 (a test winter): on 23 Feb 1999 (official danger level 5) the model gives **95% -> high**.

Repeated over 200 random splits: model **0.737**, forecasters **0.687**, model better in **98%** of splits.

*AUC: how often an avalanche day gets a higher score than a quiet day (0.5 = coin flip, 1 = perfect). Hit rate: share of avalanche days with a warning. False alarms: share of warnings with no avalanche.*

![ROC curves](outputs/roc_comparison.png)

**What drives the model:** dry avalanches follow new snow (last 3-7 days), helped by wind and weak layers; wet avalanches follow temperature. Every weight points in the physically expected direction except *snow on a weak layer*, which is about zero (-0.03): its effect is already carried by the temperature gradient and 7-day snow. Avalanches yesterday raise the risk in both.

![Feature importance](outputs/feature_importance.png)

**One test winter, day by day:** peaks in the predicted risk line up with avalanche cycles and with high official danger levels.

![Example winter](outputs/example_winter.png)

![Calibration](outputs/calibration.png)

## Limitations

- **Modelled weather:** Open-Meteo is reanalysis for one ~10 km grid cell, not station measurements. Wind is the weakest part (correlation only 0.43 with measured wind at Weissfluhjoch).
- **Estimated snowpack gradient:** (0 C - mean air temperature) / snow depth, not measured. Checked against measured snow temperatures at Weissfluhjoch (2000-2017): correlation 0.85.
- **One yes/no per day for the whole region:** no slope direction, so the effects of wind loading and sun on south slopes cannot be separated. (A model per slope direction was tested and did not help.)
- **Recording changed over time:** about twice as many avalanches were recorded per day after 2015.
- **Avalanche yesterday** needs yesterday's observations, and part of its effect may come from how release dates are estimated.
- **Not a replacement for the official [SLF avalanche bulletin](https://www.slf.ch/en/avalanche-bulletin-and-snow-situation/).**

## Next steps

- A leaner model with fewer, non-overlapping features
- Correct the predicted probabilities (calibration)
- Human-triggered avalanches, i.e. danger to people
- An LLM agent that explains the daily risk in plain language
