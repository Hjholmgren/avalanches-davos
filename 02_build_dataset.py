"""Step 2: join avalanche activity with weather, create features and targets.

Run with:  uv run 02_build_dataset.py
Output:    data/dataset.csv (one row per winter day)
"""

import os

import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths like "data/..." work from any folder

from features import compute_features

# ---------------------------------------------------------------------------
# 1. Load the data
# ---------------------------------------------------------------------------
weather = pd.read_csv("data/weather_daily.csv")
activity = pd.read_csv("data/avalanche_activity_data.csv", sep=";", parse_dates=["date"])

# ---------------------------------------------------------------------------
# 2. Features from the weather
# ---------------------------------------------------------------------------
# Computed on the full weather series BEFORE joining, so rolling windows (e.g. snow
# over 7 days) use the real previous days even when the SLF file skips a day.
features = compute_features(weather)

# ---------------------------------------------------------------------------
# 3. Targets from the avalanche activity (one row per day)
# ---------------------------------------------------------------------------
days = pd.DataFrame()
days["date"] = activity["date"]
days["winter"] = activity["year"]                  # winter 2012 = 2011-2012
days["danger_level"] = activity["max.danger.corr"]  # human benchmark, NOT a feature

# Main target: at least one natural avalanche of size 2+.
days["avalanche_day"] = (activity["NATURAL.SIZE_234"] >= 1).astype(int)

# Separate targets for dry and wet snow avalanches (natural, size 2+), because
# they have different causes: dry = new snow and wind, wet = warming and sun.
dry_count = activity["dry.NATURAL.SIZE2"] + activity["dry.NATURAL.SIZE3"] + activity["dry.NATURAL.SIZE4"]
wet_count = activity["wet.NATURAL.SIZE2"] + activity["wet.NATURAL.SIZE3"] + activity["wet.NATURAL.SIZE4"]
days["dry_day"] = (dry_count >= 1).astype(int)
days["wet_day"] = (wet_count >= 1).astype(int)

# ---------------------------------------------------------------------------
# 4. Join: keep only days that are in the SLF file (inner join)
# ---------------------------------------------------------------------------
dataset = days.merge(features, on="date", how="inner")
print(f"Days in SLF file: {len(days)},  days after join: {len(dataset)}")
print(f"Missing values after join: {dataset.isna().sum().sum()}")

# ---------------------------------------------------------------------------
# 5. Summary and save
# ---------------------------------------------------------------------------
print(f"Avalanche days: {dataset['avalanche_day'].mean():.1%}  "
      f"(dry: {dataset['dry_day'].mean():.1%}, wet: {dataset['wet_day'].mean():.1%})")
print("\nShare of avalanche days per winter:")
print(dataset.groupby("winter")["avalanche_day"].mean().round(2).to_string())

dataset.to_csv("data/dataset.csv", index=False)
print(f"\nSaved data/dataset.csv ({len(dataset)} rows, {len(dataset.columns)} columns)")
