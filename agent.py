"""Avalanche agent for Davos.

Claude answers questions about avalanche risk. It does not know the numbers itself:
it calls tools that read the trained model (data/model.joblib) and the saved days
(data/all_days.csv), both made by avalanche_model.ipynb, plus live weather and the
official SLF bulletin.

Run with:  uv run agent.py "Is it dangerous in Davos tomorrow?"
Needs:     an Anthropic login on this machine, or ANTHROPIC_API_KEY in a .env file next to this script
"""

import json
import os
import sys
from datetime import date

import anthropic
import joblib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths like "data/..." work from any folder
load_dotenv()  # optional: reads ANTHROPIC_API_KEY from .env (never commit that file)

MODEL = "claude-sonnet-5-5"
SAVED = joblib.load("data/model.joblib")  # {"models": {"dry": ..., "wet": ...}, "cutoffs": ...}
ALL_DAYS = pd.read_csv("data/all_days.csv", parse_dates=["date"])
AVALANCHES = pd.read_csv("data/snow_avalanche_data.csv", sep=";", parse_dates=["date_release"])
MONTHS = [11, 12, 1, 2, 3, 4, 5]
MONTH_COLUMNS = ["month_Nov", "month_Dec", "month_Jan", "month_Feb", "month_Mar", "month_Apr", "month_May"]


# ---------------------------------------------------------------------------
# 1. Features: an exact copy of compute_features() from the notebook.
#    check_features() makes sure the copy gives the same numbers as the notebook.
# ---------------------------------------------------------------------------
def compute_features(weather):
    weather = weather.sort_values("date").reset_index(drop=True)
    snow, t_max = weather["snowfall_sum"], weather["temperature_2m_max"]
    f = pd.DataFrame({"date": pd.to_datetime(weather["date"])})

    f["snow_72h"] = snow.rolling(3, min_periods=1).sum()
    f["snow_7d"] = snow.rolling(7, min_periods=1).sum()
    f["snow_depth"] = weather["snow_depth_cm"]

    f["wind_gust"] = weather["wind_gusts_10m_max"]
    f["wind_snow"] = f["wind_gust"] * f["snow_72h"]

    air_cold = (-weather["temperature_2m_mean"]).clip(lower=0)
    snow_depth_m = (weather["snow_depth_cm"] / 100).clip(lower=0.1)
    f["temp_gradient"] = air_cold.rolling(14, min_periods=1).mean() / snow_depth_m
    weak_layer = f["temp_gradient"].shift(7).rolling(21, min_periods=1).max()
    f["snow_on_weak_layer"] = f["snow_7d"] * (weak_layer > 10)

    f["temp_max"] = t_max
    f["temp_min"] = weather["temperature_2m_min"]
    f["temp_change"] = t_max.diff().fillna(0)
    f["warming_after_snow"] = f["snow_72h"] * f["temp_change"].clip(lower=0)
    f["warm_sunny_day"] = ((t_max > 0) & (weather["shortwave_radiation_sum"] > 9)).astype(int)

    for month, column in zip(MONTHS, MONTH_COLUMNS):
        f[column] = (f["date"].dt.month == month).astype(int)
    return f


def check_features():
    """Recompute all past days and compare with the notebook's saved features."""
    mine = compute_features(pd.read_csv("data/weather_daily.csv"))
    both = ALL_DAYS.merge(mine, on="date", suffixes=("_notebook", "_agent"))
    for column in mine.columns.drop("date"):
        if not np.allclose(both[column + "_notebook"], both[column + "_agent"]):
            raise ValueError(f"Feature '{column}' differs from the notebook")
    print("Feature check OK: agent and notebook compute the same features.\n")


# ---------------------------------------------------------------------------
# 2. Weather for one day: past days from the saved data, other days from the
#    Open-Meteo forecast. The features look back up to 4 weeks (weak layer),
#    so we also download the last 60 days.
# ---------------------------------------------------------------------------
def fetch_forecast():
    params = {"latitude": 46.80, "longitude": 9.84, "elevation": 2500, "timezone": "Europe/Zurich",
              "past_days": 60, "forecast_days": 3, "hourly": "snow_depth",
              "daily": "snowfall_sum,temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
                       "wind_gusts_10m_max,shortwave_radiation_sum"}
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=60)
    response.raise_for_status()
    data = response.json()
    weather = pd.DataFrame(data["daily"]).rename(columns={"time": "date"})
    hourly = pd.DataFrame(data["hourly"])
    hourly["date"] = hourly["time"].str[:10]
    weather["snow_depth_cm"] = weather["date"].map(hourly.groupby("date")["snow_depth"].mean() * 100)
    return compute_features(weather)


def features_for_day(day):
    """One row of features. 'past' = a day from the SLF data (1998-2019), else 'forecast'."""
    past = ALL_DAYS[ALL_DAYS["date"] == day]
    if len(past) == 1:
        return past.iloc[0], "past"
    forecast = fetch_forecast()
    row = forecast[forecast["date"] == day]
    if len(row) == 0:
        raise ValueError(f"No weather for {day.date()}. Use a Nov-May day 1998-2019, "
                         f"or a day between {forecast['date'].min().date()} and {forecast['date'].max().date()}.")
    return row.iloc[0], "forecast"


# ---------------------------------------------------------------------------
# 3. Tools: plain Python functions that Claude can call
# ---------------------------------------------------------------------------
def get_weather(day):
    """The weather behind the features for one day."""
    row, source = features_for_day(pd.Timestamp(day))
    return {"date": day, "source": source,
            "new_snow_3_days_cm": round(row["snow_72h"], 1),
            "new_snow_7_days_cm": round(row["snow_7d"], 1),
            "snow_depth_cm": round(row["snow_depth"]),
            "max_wind_gust_kmh": round(row["wind_gust"]),
            "max_temperature_c": round(row["temp_max"], 1),
            "min_temperature_c": round(row["temp_min"], 1),
            "temperature_gradient_c_per_m": round(row["temp_gradient"], 1),
            "warm_sunny_day": bool(row["warm_sunny_day"])}


LABELS = {"snow_72h": "new snow (3 days)", "snow_7d": "new snow (7 days)", "snow_depth": "snow depth",
          "wind_snow": "wind slab (wind x new snow)", "wind_gust": "strong wind",
          "temp_gradient": "weak layer forming (cold air, thin snow)",
          "snow_on_weak_layer": "new snow on a weak layer", "warming_after_snow": "warming after snowfall",
          "temp_max": "temperature", "temp_min": "night temperature", "temp_change": "warming since yesterday",
          "warm_sunny_day": "warm sunny day", "avalanche_yesterday": "avalanches the day before (cycle)"}


def risk_level(risk):
    """Same cut-offs as the notebook."""
    cutoffs = SAVED["cutoffs"]
    if risk >= cutoffs["high"]:
        return "high"
    if risk >= cutoffs["considerable"]:
        return "considerable"
    if risk >= cutoffs["moderate"]:
        return "moderate"
    return "low"


def risk_for_row(row):
    """Run the dry and wet models on one row, like predict_risk() in the notebook,
    and list the main causes: the largest weight x z in the stronger of the two models."""
    p = {}
    causes = {}
    for kind, part in SAVED["models"].items():
        x = pd.DataFrame([row[part["features"]]])
        z = part["scaler"].transform(x)[0]
        p[kind] = part["model"].predict_proba(part["scaler"].transform(x))[0, 1]
        contributions = pd.Series(part["model"].coef_[0] * z, index=part["features"]).drop(MONTH_COLUMNS)
        causes[kind] = [LABELS[name] for name in contributions.sort_values(ascending=False).index[:3]
                        if contributions[name] > 0]
    risk = 1 - (1 - p["dry"]) * (1 - p["wet"])
    main_type = "dry" if p["dry"] >= p["wet"] else "wet"
    return {"risk_percent": round(100 * risk), "level": risk_level(risk),
            "dry_avalanche_percent": round(100 * p["dry"]), "wet_avalanche_percent": round(100 * p["wet"]),
            "main_causes": [f"{main_type}: {cause}" for cause in causes[main_type]]}


def predict_risk(day, avalanches_observed_yesterday=None):
    """Model risk for one day. For past days we know if avalanches happened the day before.
    For forecast days we don't: if the user didn't say, we give both scenarios."""
    row, source = features_for_day(pd.Timestamp(day))
    result = {"date": day, "source": source}
    if pd.Timestamp(day).month not in MONTHS:
        result["warning"] = "Outside the model's season (November-May): less reliable."
    if source == "past":
        result.update(risk_for_row(row))
        result["what_happened"] = {"avalanche_day": bool(row["avalanche_day"]),
                                   "official_danger_level": int(row["danger_level"])}
    elif avalanches_observed_yesterday is None:
        result["if_no_avalanches_yesterday"] = risk_for_row(with_yesterday(row, 0))
        result["if_avalanches_yesterday"] = risk_for_row(with_yesterday(row, 1))
    else:
        result.update(risk_for_row(with_yesterday(row, int(avalanches_observed_yesterday))))
    return result


def with_yesterday(row, value):
    """A copy of the row with 'avalanche_yesterday' set (not known for forecast days)."""
    row = row.copy()
    row["avalanche_yesterday"] = value
    return row


def natural_avalanches_on(dates):
    return AVALANCHES[AVALANCHES["date_release"].isin(dates) & (AVALANCHES["trigger_type"] == "NATURAL")
                      & (AVALANCHES["aval_size_class"] >= 2)]


def find_similar_days(day):
    """The 10 past days whose weather was most similar (distance on standardised features)."""
    row, _ = features_for_day(pd.Timestamp(day))
    weather_features = [f for f in LABELS if f != "avalanche_yesterday"]
    candidates = ALL_DAYS[(ALL_DAYS["date"] - pd.Timestamp(day)).abs().dt.days > 10]  # not the same storm
    mean, std = ALL_DAYS[weather_features].mean(), ALL_DAYS[weather_features].std()
    z_day = (row[weather_features].astype(float) - mean) / std
    z_all = (candidates[weather_features] - mean) / std
    distance = np.sqrt(((z_all - z_day) ** 2).sum(axis=1))
    similar = candidates.loc[distance.nsmallest(10).index]
    largest = natural_avalanches_on(similar["date"]).groupby("date_release")["aval_size_class"].max()
    days = [{"date": str(d.date()), "avalanche_day": bool(a), "official_danger_level": int(level),
             "largest_avalanche_size": int(largest.get(d, 0))}
            for d, a, level in zip(similar["date"], similar["avalanche_day"], similar["danger_level"])]
    return {"similar_days": days, "summary": f"{int(similar['avalanche_day'].sum())} of 10 had natural avalanches (size 2+)"}


def avalanche_rose(dates):
    """Where natural avalanches (size 2+) released on these days: slope direction and elevation."""
    found = natural_avalanches_on(pd.to_datetime(dates)).dropna(subset=["aspect_degrees"])
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    direction = [directions[int(((deg + 22.5) % 360) // 45)] for deg in found["aspect_degrees"]]
    band = pd.cut(found["max_elevation_m"], [0, 2000, 2500, 5000], labels=["below 2000 m", "2000-2500 m", "above 2500 m"])
    return {"avalanches": len(found),
            "by_slope_direction": {d: direction.count(d) for d in directions},
            "by_elevation": band.value_counts().to_dict()}


def get_official_bulletin():
    """Today's official SLF avalanche bulletin for Davos (region CH-5123). Source: SLF, www.slf.ch"""
    response = requests.get("https://aws.slf.ch/api/bulletin/caaml/en/json", timeout=90)
    response.raise_for_status()
    for bulletin in response.json()["bulletins"]:
        if any(region["regionID"] == "CH-5123" for region in bulletin["regions"]):
            return {"valid": bulletin["validTime"],
                    "danger_level": [rating["mainValue"] for rating in bulletin["dangerRatings"]],
                    "problems": [{"type": p["problemType"], "elevation": p.get("elevation"), "slopes": p.get("aspects")}
                                 for p in bulletin.get("avalancheProblems", [])],
                    "source": "SLF avalanche bulletin, www.slf.ch"}
    return {"error": "No bulletin for Davos right now (outside the winter season?)"}


# Tool descriptions for Claude: name, what it does, and its inputs
DATE = {"type": "string", "description": "Date as YYYY-MM-DD"}
TOOLS = [
    {"name": "get_weather", "description": "Weather for one day in Davos (2500 m): new snow, snow depth, wind, temperature. "
     "Past days Nov-May 1998-2019, or the last 60 days and next 3 days (forecast).",
     "input_schema": {"type": "object", "properties": {"day": DATE}, "required": ["day"]}},
    {"name": "predict_risk", "description": "The ML model's avalanche risk for one day: risk %, level, dry vs wet, main causes. "
     "For past days it also says what really happened. For other days, pass avalanches_observed_yesterday if known; "
     "otherwise both scenarios are returned.",
     "input_schema": {"type": "object", "properties": {"day": DATE, "avalanches_observed_yesterday": {"type": "boolean"}},
                      "required": ["day"]}},
    {"name": "find_similar_days", "description": "The 10 past days (1999-2019) with the most similar weather, and whether avalanches happened.",
     "input_schema": {"type": "object", "properties": {"day": DATE}, "required": ["day"]}},
    {"name": "avalanche_rose", "description": "On which slope directions and elevations natural avalanches released on the given past days.",
     "input_schema": {"type": "object", "properties": {"dates": {"type": "array", "items": DATE}}, "required": ["dates"]}},
    {"name": "get_official_bulletin", "description": "Today's official SLF avalanche bulletin for Davos (only the current day).",
     "input_schema": {"type": "object", "properties": {}}},
]
RUN_TOOL = {"get_weather": get_weather, "predict_risk": predict_risk, "find_similar_days": find_similar_days,
            "avalanche_rose": avalanche_rose, "get_official_bulletin": get_official_bulletin}


# ---------------------------------------------------------------------------
# 4. The agent loop: ask Claude -> run the tools it asks for -> send back results -> repeat
# ---------------------------------------------------------------------------
SYSTEM = f"""You explain avalanche risk around Davos (Switzerland) to skiers. Today is {date.today()}.
Use only numbers from the tools, never invent numbers. Answer in 3-4 sentences: the risk level and %,
dry or wet avalanches and the main causes, what similar past days looked like and which slopes.
If you don't know whether avalanches were observed yesterday, give both scenarios.
For today, compare with the official SLF bulletin. Always end with: follow the official SLF avalanche bulletin."""


def ask(question):
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]
    for step in range(10):  # at most 10 rounds
        response = client.beta.messages.create(
            model=MODEL, max_tokens=16000, system=SYSTEM, tools=TOOLS, messages=messages,
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",  # retry on another model if refused
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":  # Claude is done: print its answer
            print("\n" + "".join(block.text for block in response.content if block.type == "text"))
            return

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"-> {block.name}({json.dumps(block.input)})")
            try:
                result = json.dumps(RUN_TOOL[block.name](**block.input), default=str)
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
            except Exception as error:  # tell Claude what went wrong instead of crashing
                result = f"Error: {error}"
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result, "is_error": True})
            print(f"   {result[:300]}")
        messages.append({"role": "user", "content": tool_results})
    print("Stopped after 10 rounds.")


if __name__ == "__main__":
    logged_in = os.path.isdir(os.path.expanduser(os.environ.get("ANTHROPIC_CONFIG_DIR", "~/.config/anthropic")))
    if not os.environ.get("ANTHROPIC_API_KEY") and not logged_in:
        sys.exit("Not logged in: log in to Anthropic on this machine, or put ANTHROPIC_API_KEY=sk-ant-... in .env.")
    check_features()
    ask(" ".join(sys.argv[1:]) or "Is it dangerous in Davos tomorrow?")
