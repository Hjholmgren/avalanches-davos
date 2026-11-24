"""Export the saved model and data as JSON files for the Lovable frontend.

One-way data contract: this script only READS the files the notebook saved
(data/ and outputs/, plus the printed outputs inside the notebook) and WRITES
JSON to frontend_export/data/. It does not import agent.py or the notebook code.
Delete this folder and nothing else changes.

Run with:  uv run frontend_export/export_for_frontend.py
"""

import json
import math
import os
import re

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "data")

SAVED = joblib.load(os.path.join(REPO, "data/model.joblib"))  # {"models": {"dry", "wet"}, "cutoffs"}
ALL_DAYS = pd.read_csv(os.path.join(REPO, "data/all_days.csv"), parse_dates=["date"])
WEATHER = pd.read_csv(os.path.join(REPO, "data/weather_daily.csv"), parse_dates=["date"])
AVALANCHES = pd.read_csv(os.path.join(REPO, "data/snow_avalanche_data.csv"), sep=";")
RESULTS = pd.read_csv(os.path.join(REPO, "outputs/results.csv"), index_col=0)
NOTEBOOK_TEXT = "".join(
    "".join(output.get("text", ""))
    for cell in json.load(open(os.path.join(REPO, "avalanche_model.ipynb")))["cells"]
    for output in cell.get("outputs", [])
)

# How each feature is computed, in words (copied from compute_features() in the notebook).
# "today" includes the day itself; rolling windows use whatever days exist (min_periods=1).
FEATURE_DEFINITIONS = {
    "snow_72h": "sum of snowfall_sum (cm) over today and the 2 days before",
    "snow_7d": "sum of snowfall_sum (cm) over today and the 6 days before",
    "snow_depth": "snow_depth_cm: daily mean of hourly snow_depth (m) x 100",
    "wind_gust": "wind_gusts_10m_max (km/h)",
    "wind_snow": "wind_gust x snow_72h",
    "temp_gradient": "mean over today and the 13 days before of max(0, -temperature_2m_mean), "
                     "divided by max(0.1, snow_depth_cm / 100)",
    "snow_on_weak_layer": "snow_7d if the max temp_gradient from 27 to 7 days ago "
                          "(shift 7, then rolling 21) is > 10, else 0",
    "temp_max": "temperature_2m_max (C)",
    "temp_min": "temperature_2m_min (C)",
    "temp_change": "temperature_2m_max today - yesterday (0 for the first day)",
    "warming_after_snow": "snow_72h x max(0, temp_change)",
    "warm_sunny_day": "1 if temperature_2m_max > 0 and shortwave_radiation_sum > 9 (MJ/m2), else 0",
    "avalanche_yesterday": "1 if natural avalanches (size 2+) happened the day before, else 0. "
                           "Not known for forecast days: compute both 0 and 1",
    "month_*": "1 if the day is in that month, else 0 (month_Nov ... month_May)",
}
RAW_WEATHER_COLUMNS = ["snowfall_sum", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                       "wind_gusts_10m_max", "shortwave_radiation_sum", "snow_depth_cm"]


def plain_probability(part, row):
    """The logistic regression written out, exactly as the JavaScript will do it."""
    z = part["model"].intercept_[0]
    for name, mean, std, coef in zip(part["features"], part["scaler"].mean_, part["scaler"].scale_,
                                     part["model"].coef_[0]):
        z += coef * (row[name] - mean) / std
    return 1 / (1 + math.exp(-z))


def sklearn_probability(part, rows):
    return part["model"].predict_proba(part["scaler"].transform(rows[part["features"]]))[:, 1]


def risk_level(risk):
    cutoffs = SAVED["cutoffs"]
    for level in ["high", "considerable", "moderate"]:
        if risk >= cutoffs[level]:
            return level
    return "low"


def combined_risk(rows):
    p = {kind: sklearn_probability(part, rows) for kind, part in SAVED["models"].items()}
    return p, 1 - (1 - p["dry"]) * (1 - p["wet"])


def export_model_params():
    models = {}
    for kind, part in SAVED["models"].items():
        models[kind] = {"features": list(part["features"]),
                        "mean": part["scaler"].mean_.tolist(),
                        "std": part["scaler"].scale_.tolist(),
                        "coef": part["model"].coef_[0].tolist(),
                        "intercept": float(part["model"].intercept_[0])}

    # 5 test cases: past days spread from the lowest to the highest risk
    p, risk = combined_risk(ALL_DAYS)
    order = np.argsort(risk)
    picks = [order[int(q * (len(order) - 1))] for q in [0.05, 0.3, 0.6, 0.9, 0.999]]
    all_features = sorted(set(models["dry"]["features"]) | set(models["wet"]["features"]))
    test_cases = []
    for i in picks:
        row = ALL_DAYS.iloc[i]
        plain = {kind: plain_probability(part, row) for kind, part in SAVED["models"].items()}
        assert abs(plain["dry"] - p["dry"][i]) < 1e-9 and abs(plain["wet"] - p["wet"][i]) < 1e-9, \
            "the written-out formula differs from sklearn"
        test_cases.append({"date": str(row["date"].date()),
                           "inputs": {name: float(row[name]) for name in all_features},
                           "p_dry": round(float(p["dry"][i]), 6), "p_wet": round(float(p["wet"][i]), 6),
                           "risk": round(float(risk[i]), 6), "level": risk_level(risk[i])})

    # Feature test: 60 days of raw weather -> the features the notebook saved for the last day.
    # Picked a day where the weak-layer rule is active, so that branch gets tested too.
    day = ALL_DAYS[(ALL_DAYS["snow_on_weak_layer"] > 0) & (ALL_DAYS["date"].dt.month == 1)]["date"].iloc[0]
    window = WEATHER[(WEATHER["date"] > day - pd.Timedelta(days=60)) & (WEATHER["date"] <= day)]
    assert len(window) == 60, "need 60 consecutive days of weather"
    saved_row = ALL_DAYS[ALL_DAYS["date"] == day].iloc[0]
    weather_features = [f for f in all_features if f != "avalanche_yesterday"]
    feature_test = {"date": str(day.date()),
                    "raw_daily_weather": [{"date": str(r["date"].date()),
                                           **{c: float(r[c]) for c in RAW_WEATHER_COLUMNS}}
                                          for _, r in window.iterrows()],
                    "expected_features": {name: float(saved_row[name]) for name in weather_features}}

    return {"location": {"latitude": 46.80, "longitude": 9.84, "elevation": 2500, "timezone": "Europe/Zurich",
                         "past_days": 60},
            "season_months": [11, 12, 1, 2, 3, 4, 5],
            "combine": "risk = 1 - (1 - p_dry) * (1 - p_wet)",
            "formula": "p = 1 / (1 + exp(-(intercept + sum(coef_i * (x_i - mean_i) / std_i))))",
            "cutoffs": {level: float(value) for level, value in SAVED["cutoffs"].items()},
            "models": models,
            "feature_definitions": FEATURE_DEFINITIONS,
            "test_cases": test_cases,
            "feature_test": feature_test}


def export_aspect_counts():
    """Natural avalanches of size 2+ (the model's target), per 45-degree slope direction."""
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    found = AVALANCHES[(AVALANCHES["trigger_type"] == "NATURAL") & (AVALANCHES["aval_size_class"] >= 2)]
    found = found.dropna(subset=["aspect_degrees"])
    sector = ((found["aspect_degrees"] + 22.5) % 360 // 45).astype(int)
    counts = {kind: [int(((found["snow_type"] == kind) & (sector == i)).sum()) for i in range(8)]
              for kind in ["dry", "wet"]}
    return {"filter": "natural avalanches, size 2+, SLF Davos 1998-2019", "directions": directions, **counts}


def export_winter_history():
    """The most active TEST winter (not used for training), like the notebook's example plot."""
    test_winters = [int(w) for w in re.search(r"Test winters: \[([^\]]+)\]", NOTEBOOK_TEXT).group(1).split(",")]
    test = ALL_DAYS[ALL_DAYS["winter"].isin(test_winters)]
    winter = int(test.groupby("winter")["avalanche_day"].mean().idxmax())
    days = ALL_DAYS[ALL_DAYS["winter"] == winter]
    _, risk = combined_risk(days)
    return {"winter": f"{winter - 1}-{winter}", "note": "a test winter: the model never saw it in training",
            "days": [{"date": str(d.date()), "risk": round(float(r), 4), "avalanche_day": int(a),
                      "danger_level": int(level)}
                     for d, r, a, level in zip(days["date"], risk, days["avalanche_day"], days["danger_level"])]}


def export_evaluation():
    repeated = re.search(r"Average AUC: model ([\d.]+), forecasters ([\d.]+)\s+Model better in (\d+)%", NOTEBOOK_TEXT)
    split = {name.lower().replace("our ", ""): {"auc": row["AUC"], "hit_rate": row["hit rate"],
                                                "false_alarms": row["false alarms"]}
             for name, row in RESULTS.iterrows()}
    return {"test_split": split,
            "repeated_200": {"model_auc": float(repeated.group(1)), "forecasters_auc": float(repeated.group(2)),
                             "model_better_share": int(repeated.group(3)) / 100}}


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = {"model_params.json": export_model_params(), "aspect_counts.json": export_aspect_counts(),
             "winter_history.json": export_winter_history(), "evaluation.json": export_evaluation()}
    for name, content in files.items():
        with open(os.path.join(OUT, name), "w") as f:
            json.dump(content, f, indent=1)
        print(f"wrote frontend_export/data/{name} ({os.path.getsize(os.path.join(OUT, name)) // 1024} KB)")
