"""Turn raw daily weather (Open-Meteo) into model features.

Used by both 02_build_dataset.py (history) and 04_agent.py (forecast / replay),
so training and live prediction compute features in exactly the same way.
Every feature uses only the current day and the days before it (no future data).
"""

import numpy as np
import pandas as pd

# The features the models can use.
FEATURE_COLUMNS = [
    "snow_24h",         # new snow today (cm)
    "snow_72h",         # new snow over the last 3 days (cm)
    "snow_7d",          # new snow over the last 7 days (cm)
    "days_since_snow",  # days since the last snowfall of >= 10 cm (capped at 30)
    "snow_depth",       # total snow on the ground (cm)
    "rain",             # rain today (mm) - rain on snow weakens it
    "wind_max",         # max wind speed today (km/h)
    "wind_gust",        # max wind gust today (km/h)
    "wind_snow",        # wind_gust x snow_72h: wind can only move snow that is lying loose
    "wind_from_north",  # +1 = wind from the north, -1 = from the south
    "wind_from_east",   # +1 = wind from the east,  -1 = from the west
    "temp_max",         # max temperature today (C)
    "temp_min",         # min temperature today (C)
    "temp_change",      # temp_max today minus yesterday (warming signal)
    "temp_gradient",    # C per metre through the snowpack (14-day average): weak-layer formation
    "warm_3d",          # sum of above-zero temp_max over the last 3 days (snowpack warming up)
    "sun",              # solar radiation today (MJ/m2)
    "sun_warm",         # sun x above-zero temp_max: sunshine only melts snow when it is warm
    "season_month",     # season effect: Oct = 1, Nov = 2, ..., Jan = 4, ..., Jun = 9
]

# Month as yes/no columns ("is it December?"). A logistic regression can only draw a
# straight line through season_month, but avalanche risk PEAKS in mid-winter. With one
# column per month, each month gets its own baseline risk.
MONTH_NAMES = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun"]
MONTH_COLUMNS = ["month_" + name for name in MONTH_NAMES]

BIG_SNOWFALL_CM = 10
MAX_DAYS_SINCE_SNOW = 30


def count_days_since_big_snowfall(snowfall):
    """For each day, how many days ago was the last snowfall of >= 10 cm.
    0 = it snowed >= 10 cm today. Written as a plain loop so it is easy to follow."""
    result = []
    days = MAX_DAYS_SINCE_SNOW  # start as "long ago"
    for snow_today in snowfall:
        if snow_today >= BIG_SNOWFALL_CM:
            days = 0
        else:
            days = min(days + 1, MAX_DAYS_SINCE_SNOW)
        result.append(days)
    return result


def compute_features(weather):
    """weather: one row per day, sorted by date, with the Open-Meteo column names.
    The rows must be consecutive days, so rolling windows look at the real previous days."""
    weather = weather.sort_values("date").reset_index(drop=True)
    features = pd.DataFrame()
    features["date"] = pd.to_datetime(weather["date"])

    # --- Snow ---
    features["snow_24h"] = weather["snowfall_sum"]
    # rolling(3).sum() = today + the 2 days before
    features["snow_72h"] = weather["snowfall_sum"].rolling(3, min_periods=1).sum()
    features["snow_7d"] = weather["snowfall_sum"].rolling(7, min_periods=1).sum()
    features["days_since_snow"] = count_days_since_big_snowfall(weather["snowfall_sum"])
    features["snow_depth"] = weather["snow_depth_cm"]
    features["rain"] = weather["rain_sum"]

    # --- Wind ---
    features["wind_max"] = weather["wind_speed_10m_max"]
    features["wind_gust"] = weather["wind_gusts_10m_max"]
    features["wind_snow"] = features["wind_gust"] * features["snow_72h"]
    # Direction is an angle (0 = from north, 90 = from east). As a raw number 359 and 1
    # would look far apart although they are almost the same direction, so we split it
    # into a north part (cosine) and an east part (sine).
    angle = np.radians(weather["wind_direction_10m_dominant"])
    features["wind_from_north"] = np.cos(angle)
    features["wind_from_east"] = np.sin(angle)

    # --- Temperature and sun ---
    features["temp_max"] = weather["temperature_2m_max"]
    features["temp_min"] = weather["temperature_2m_min"]
    features["temp_change"] = weather["temperature_2m_max"].diff().fillna(0)  # today - yesterday

    # Temperature gradient through the snowpack. The ground under the snow stays near 0 C,
    # the surface is near the air temperature. Cold air over a THIN snowpack = a steep
    # gradient, which grows weak, sugary crystals (facets / depth hoar) that stay weak for
    # weeks. Rule of thumb: above ~10 C per metre, weak layers grow. We average the cold
    # nights over 14 days because weak layers form slowly.
    night_cold = (-weather["temperature_2m_min"]).clip(lower=0)      # degrees below 0 C
    night_cold_14d = night_cold.rolling(14, min_periods=1).mean()
    snow_depth_m = (weather["snow_depth_cm"] / 100).clip(lower=0.1)  # avoid dividing by ~0
    features["temp_gradient"] = night_cold_14d / snow_depth_m

    above_zero = weather["temperature_2m_max"].clip(lower=0)  # negative temps -> 0
    features["warm_3d"] = above_zero.rolling(3, min_periods=1).sum()
    features["sun"] = weather["shortwave_radiation_sum"]
    features["sun_warm"] = features["sun"] * above_zero

    # A plain month number puts December (12) far from January (1). Counting months
    # from the start of the winter keeps the season in order: Oct = 1, ..., Jun = 9.
    calendar_month = features["date"].dt.month
    features["season_month"] = (calendar_month - 10) % 12 + 1
    for number, column in enumerate(MONTH_COLUMNS, start=1):
        features[column] = (features["season_month"] == number).astype(int)

    numbers = features.columns.drop("date")
    features[numbers] = features[numbers].round(2)
    return features
