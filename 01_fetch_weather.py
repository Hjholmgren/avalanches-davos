"""Step 1: download historical daily weather for Davos from Open-Meteo.

Run with:  uv run 01_fetch_weather.py
"""

import os

import pandas as pd
import requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # paths like "data/..." work from any folder

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Davos. Avalanches start high up, so we ask Open-Meteo for weather at 2500 m
# (it adjusts temperature to that elevation).
LATITUDE = 46.80
LONGITUDE = 9.84
ELEVATION = 2500

DAILY_VARIABLES = [
    "snowfall_sum",
    "precipitation_sum",
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "rain_sum",
    "wind_gusts_10m_max",
]

# Snow depth (total snow on the ground) only exists as an hourly variable,
# so we download it hourly and average it per day ourselves.
HOURLY_VARIABLES = ["snow_depth"]

WEATHER_FILE = "data/weather_daily.csv"


def fetch_one_winter(winter):
    """Download weather for one winter. Winter 2012 = 1 Oct 2011 to 30 Jun 2012
    (same convention as the SLF 'year' column)."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "elevation": ELEVATION,
        "start_date": f"{winter - 1}-10-01",
        "end_date": f"{winter}-06-30",
        "timezone": "Europe/Zurich",
        "daily": ",".join(DAILY_VARIABLES),
        "hourly": ",".join(HOURLY_VARIABLES),
    }
    response = requests.get(ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()  # stop with a clear error if the request failed
    data = response.json()

    # Daily values: a dict like {"time": [...], "snowfall_sum": [...], ...}
    table = pd.DataFrame(data["daily"])
    table = table.rename(columns={"time": "date"})

    # Hourly snow depth (metres) -> daily mean in cm
    hourly = pd.DataFrame(data["hourly"])
    hourly["date"] = hourly["time"].str[:10]  # "2012-02-01T13:00" -> "2012-02-01"
    daily_snow_depth = hourly.groupby("date")["snow_depth"].mean() * 100
    table["snow_depth_cm"] = table["date"].map(daily_snow_depth)
    return table


# ---------------------------------------------------------------------------
# Download the weather (one request per winter keeps requests small)
# ---------------------------------------------------------------------------
if os.path.exists(WEATHER_FILE):
    print(f"{WEATHER_FILE} already exists, skipping download (delete it to re-download).")
    weather = pd.read_csv(WEATHER_FILE)
else:
    winter_tables = []
    for winter in range(1999, 2020):
        print(f"Downloading winter {winter - 1}-{winter} ...")
        winter_tables.append(fetch_one_winter(winter))
    weather = pd.concat(winter_tables, ignore_index=True)
    weather.to_csv(WEATHER_FILE, index=False)
    print(f"Saved {WEATHER_FILE}")

print("\n=== Weather data ===")
print(f"Rows: {len(weather)}")
print(f"Date range: {weather['date'].min()} to {weather['date'].max()}")
print(f"Duplicate dates: {weather['date'].duplicated().sum()}")
print("Missing values per column:")
print(weather.isna().sum().to_string())
