"""Step 3: train the model, test it on unseen winters, compare with the forecasters.

Run with:  uv run 03_train_model.py
Outputs:   outputs/*.png, outputs/results.csv, data/model.joblib, data/train_days.csv
"""

import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths like "data/..." work from any folder

from features import MONTH_COLUMNS

# Each model only gets the features that make physical sense for its avalanche type.
DRY_FEATURES = [
    "snow_24h", "snow_72h", "snow_7d", "days_since_snow", "snow_depth",
    "wind_gust", "wind_snow", "wind_from_north", "wind_from_east",
    "temp_max", "temp_gradient",
] + MONTH_COLUMNS
WET_FEATURES = [
    "temp_max", "temp_min", "temp_change", "warm_3d", "sun", "sun_warm",
    "rain", "snow_depth", "snow_24h",
] + MONTH_COLUMNS

BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 11})


def train_logistic(rows, features, target):
    """Standardise the features (mean 0, std 1) and fit a logistic regression,
    which finds the weights that minimise the log loss."""
    scaler = StandardScaler()
    x = scaler.fit_transform(rows[features])
    model = LogisticRegression(max_iter=1000)
    model.fit(x, rows[target])
    return scaler, model


def predict(scaler, model, rows, features):
    """Predicted probability of an avalanche for each row."""
    x = scaler.transform(rows[features])
    return model.predict_proba(x)[:, 1]


def risk_level(probability, cutoffs):
    """Turn a probability (0-1) into a word."""
    if probability >= cutoffs["high"]:
        return "high"
    if probability >= cutoffs["considerable"]:
        return "considerable"
    if probability >= cutoffs["moderate"]:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# 1. Split by time: learn from the past, test on later winters the model never saw.
#    (A random split would put neighbouring, almost identical days in both sets.)
# ---------------------------------------------------------------------------
data = pd.read_csv("data/dataset.csv", parse_dates=["date"])
train = data[data["winter"] <= 2014].copy()
test = data[data["winter"] >= 2015].copy()
print(f"Train: {len(train)} days (1999-2014), avalanche days {train['avalanche_day'].mean():.0%}")
print(f"Test:  {len(test)} days (2015-2019), avalanche days {test['avalanche_day'].mean():.0%}")

# ---------------------------------------------------------------------------
# 2. Train one model for dry and one for wet avalanches, then combine them.
# ---------------------------------------------------------------------------
dry_scaler, dry_model = train_logistic(train, DRY_FEATURES, "dry_day")
wet_scaler, wet_model = train_logistic(train, WET_FEATURES, "wet_day")

for rows in [train, test]:
    rows["p_dry"] = predict(dry_scaler, dry_model, rows, DRY_FEATURES)
    rows["p_wet"] = predict(wet_scaler, wet_model, rows, WET_FEATURES)
    # Chance of at least one of the two types (assuming they are independent)
    rows["p_model"] = 1 - (1 - rows["p_dry"]) * (1 - rows["p_wet"])

# ---------------------------------------------------------------------------
# 3. Risk levels. Fixed cut-offs (15/35/60%) left the top levels almost empty, so we
#    let the model use each level as often as the forecasters used theirs (training years):
#    low ~ danger level 1, moderate ~ 2, considerable ~ 3, high ~ 4-5.
# ---------------------------------------------------------------------------
cutoffs = {
    "moderate": train["p_model"].quantile((train["danger_level"] <= 1).mean()),
    "considerable": train["p_model"].quantile((train["danger_level"] <= 2).mean()),
    "high": train["p_model"].quantile((train["danger_level"] <= 3).mean()),
}
print("\nRisk cut-offs:", {name: f"{value:.0%}" for name, value in cutoffs.items()})

# ---------------------------------------------------------------------------
# 4. Compare with the forecasters on the TEST winters.
#    A "warning" = considerable or higher (forecasters: danger level 3 or higher).
# ---------------------------------------------------------------------------
avalanche = test["avalanche_day"] == 1
model_warns = test["p_model"] >= cutoffs["considerable"]
human_warns = test["danger_level"] >= 3

results = pd.DataFrame({
    "method": ["Forecasters (danger level)", "Our model (dry + wet)"],
    # AUC: how often an avalanche day gets a higher score than a quiet day (0.5 = coin flip)
    "AUC": [roc_auc_score(avalanche, test["danger_level"]), roc_auc_score(avalanche, test["p_model"])],
    # Hit rate: share of avalanche days with a warning
    "hit rate": [human_warns[avalanche].mean(), model_warns[avalanche].mean()],
    # False alarms: share of warnings where nothing happened
    "false alarms": [(~avalanche[human_warns]).mean(), (~avalanche[model_warns]).mean()],
}).round(2)
print("\n=== Test winters 2015-2019 ===")
print(results.to_string(index=False))
results.to_csv("outputs/results.csv", index=False)

print("\nWarned on dry avalanche days: model "
      f"{model_warns[test['dry_day'] == 1].mean():.0%}, forecasters {human_warns[test['dry_day'] == 1].mean():.0%}")
print("Warned on wet avalanche days: model "
      f"{model_warns[test['wet_day'] == 1].mean():.0%}, forecasters {human_warns[test['wet_day'] == 1].mean():.0%}")

# Do the levels mean something? Actual avalanche rate per level on the test winters.
test["level"] = [risk_level(p, cutoffs) for p in test["p_model"]]
level_check = test.groupby("level")["avalanche_day"].agg(["size", "mean"])
level_check = level_check.reindex(["low", "moderate", "considerable", "high"])
level_check.columns = ["days", "avalanche rate"]
print("\nRisk levels on the test winters:")
print(level_check.round(2).to_string())

# ---------------------------------------------------------------------------
# 5. Plots for the slides
# ---------------------------------------------------------------------------
# 5a. ROC: avalanche days caught vs quiet days wrongly flagged, for every possible cut-off
fig, ax = plt.subplots(figsize=(6, 5.5))
for name, score, color in [("Forecasters", test["danger_level"], GREEN), ("Our model", test["p_model"], BLUE)]:
    false_rate, hit_rate, _ = roc_curve(avalanche, score)
    ax.plot(false_rate, hit_rate, color=color, linewidth=2,
            label=f"{name} (AUC {roc_auc_score(avalanche, score):.2f})")
ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1, label="Coin flip (AUC 0.50)")
ax.set_xlabel("Quiet days wrongly flagged")
ax.set_ylabel("Avalanche days caught")
ax.set_title("Test winters 2015-2019")
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig("outputs/roc_comparison.png", dpi=150)
plt.close(fig)

# 5b. Coefficients: features are standardised, so a longer bar = a bigger effect
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, model, features, title, color in [
    (axes[0], dry_model, DRY_FEATURES, "Dry avalanche model", BLUE),
    (axes[1], wet_model, WET_FEATURES, "Wet avalanche model", ORANGE),
]:
    weights = pd.Series(model.coef_[0], index=features).drop(MONTH_COLUMNS).sort_values()
    ax.barh(weights.index, weights.values, color=color, height=0.6)
    ax.axvline(0, color="#52514e", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Effect of +1 std on log-odds\n(right = more avalanches)")
fig.tight_layout()
fig.savefig("outputs/feature_importance.png", dpi=150)
plt.close(fig)

# 5c. Calibration: when the model says 30%, do avalanches happen 30% of the time?
actual, predicted = calibration_curve(avalanche, test["p_model"], n_bins=8, strategy="quantile")
fig, ax = plt.subplots(figsize=(6, 5.5))
ax.plot(predicted, actual, marker="o", markersize=8, linewidth=2, color=BLUE, label="Our model")
ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1, label="Perfect")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Actual share of avalanche days")
ax.set_title("Calibration, test winters 2015-2019")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("outputs/calibration.png", dpi=150)
plt.close(fig)

# 5d. One test winter day by day: model risk (top) and official danger level (bottom)
winter = test[test["winter"] == 2018]
avalanche_days = winter[winter["avalanche_day"] == 1]
fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
top.plot(winter["date"], winter["p_model"] * 100, color=BLUE, linewidth=2, label="Our model: risk")
top.scatter(avalanche_days["date"], [100] * len(avalanche_days), color=RED, s=30, marker="v",
            label="Observed avalanche day")
for value in cutoffs.values():
    top.axhline(value * 100, color="#cccccc", linewidth=1, zorder=0)  # risk-level lines
top.set_ylabel("Risk (%)")
top.set_ylim(0, 105)
top.set_title("Winter 2017-2018: predicted risk vs. what happened")
top.legend(loc="upper left", bbox_to_anchor=(0, 0.9), frameon=False)
bottom.step(winter["date"], winter["danger_level"], where="mid", color=GREEN, linewidth=2)
bottom.set_ylabel("Official\ndanger level")
bottom.set_yticks([1, 2, 3, 4, 5])
fig.tight_layout()
fig.savefig("outputs/example_winter.png", dpi=150)
plt.close(fig)
print("\nSaved 4 plots to outputs/")

# ---------------------------------------------------------------------------
# 6. Save what the agent needs
# ---------------------------------------------------------------------------
joblib.dump({
    "dry_scaler": dry_scaler, "dry_model": dry_model, "dry_features": DRY_FEATURES,
    "wet_scaler": wet_scaler, "wet_model": wet_model, "wet_features": WET_FEATURES,
    "cutoffs": cutoffs,
}, "data/model.joblib")
train.to_csv("data/train_days.csv", index=False)  # for "find similar past days"
print("Saved data/model.joblib and data/train_days.csv")
