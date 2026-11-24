// Weather -> model features, a line-by-line copy of compute_features() in avalanche_model.ipynb.
// pandas rolling(n, min_periods=1) = use the last n days, or fewer at the start of the data.

export type DailyWeather = {
  date: string; // "YYYY-MM-DD"
  snowfall_sum: number; // cm
  temperature_2m_max: number; // C
  temperature_2m_min: number; // C
  temperature_2m_mean: number; // C
  wind_gusts_10m_max: number; // km/h
  shortwave_radiation_sum: number; // MJ/m2
  snow_depth_cm: number; // cm (daily mean of hourly snow_depth in m, x 100)
};

export type Features = Record<string, number> & { date: string };

const MONTH_COLUMNS: Record<number, string> = {
  11: "month_Nov", 12: "month_Dec", 1: "month_Jan", 2: "month_Feb", 3: "month_Mar", 4: "month_Apr", 5: "month_May",
};

function window(values: number[], i: number, size: number): number[] {
  return values.slice(Math.max(0, i - size + 1), i + 1);
}
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const mean = (xs: number[]) => sum(xs) / xs.length;

export function computeFeatures(weatherIn: DailyWeather[]): Features[] {
  const weather = [...weatherIn].sort((a, b) => a.date.localeCompare(b.date));
  const snow = weather.map((w) => w.snowfall_sum);
  const airCold = weather.map((w) => Math.max(0, -w.temperature_2m_mean));

  // temp_gradient: 14-day mean of cold air / snow depth in m (at least 0.1 m)
  const tempGradient = weather.map((w, i) => mean(window(airCold, i, 14)) / Math.max(0.1, w.snow_depth_cm / 100));

  return weather.map((w, i) => {
    const snow72h = sum(window(snow, i, 3));
    const snow7d = sum(window(snow, i, 7));
    // weak layer: max temp_gradient from 27 to 7 days ago (shift 7, rolling 21)
    const weakLayer = i >= 7 ? Math.max(...tempGradient.slice(Math.max(0, i - 27), i - 7 + 1)) : NaN;
    const tempChange = i === 0 ? 0 : w.temperature_2m_max - weather[i - 1]!.temperature_2m_max;
    const month = Number(w.date.slice(5, 7));

    const f: Features = {
      date: w.date,
      snow_72h: snow72h,
      snow_7d: snow7d,
      snow_depth: w.snow_depth_cm,
      wind_gust: w.wind_gusts_10m_max,
      wind_snow: w.wind_gusts_10m_max * snow72h,
      temp_gradient: tempGradient[i]!,
      snow_on_weak_layer: weakLayer > 10 ? snow7d : 0, // NaN > 10 is false, like pandas
      temp_max: w.temperature_2m_max,
      temp_min: w.temperature_2m_min,
      temp_change: tempChange,
      warming_after_snow: snow72h * Math.max(0, tempChange),
      warm_sunny_day: w.temperature_2m_max > 0 && w.shortwave_radiation_sum > 9 ? 1 : 0,
    } as unknown as Features; // date is a string, all other values are numbers
    for (const [m, column] of Object.entries(MONTH_COLUMNS)) f[column] = month === Number(m) ? 1 : 0;
    return f;
  });
}

// Open-Meteo forecast -> one DailyWeather per day (same request as fetch_forecast() in agent.py)
export async function fetchDailyWeather(): Promise<DailyWeather[]> {
  const params = new URLSearchParams({
    latitude: "46.80", longitude: "9.84", elevation: "2500", timezone: "Europe/Zurich",
    past_days: "60", forecast_days: "3", hourly: "snow_depth",
    daily: "snowfall_sum,temperature_2m_max,temperature_2m_min,temperature_2m_mean,wind_gusts_10m_max,shortwave_radiation_sum",
  });
  const response = await fetch(`https://api.open-meteo.com/v1/forecast?${params}`);
  if (!response.ok) throw new Error(`Open-Meteo error ${response.status}`);
  const data = await response.json();

  // snow depth exists only hourly (in m): daily mean, skipping missing hours, x 100 -> cm
  const depthByDay: Record<string, number[]> = {};
  data.hourly.time.forEach((t: string, i: number) => {
    const value = data.hourly.snow_depth[i];
    if (value !== null) (depthByDay[t.slice(0, 10)] ??= []).push(value);
  });

  return data.daily.time.map((date: string, i: number) => {
    const row: DailyWeather = {
      date,
      snowfall_sum: data.daily.snowfall_sum[i],
      temperature_2m_max: data.daily.temperature_2m_max[i],
      temperature_2m_min: data.daily.temperature_2m_min[i],
      temperature_2m_mean: data.daily.temperature_2m_mean[i],
      wind_gusts_10m_max: data.daily.wind_gusts_10m_max[i],
      shortwave_radiation_sum: data.daily.shortwave_radiation_sum[i],
      snow_depth_cm: depthByDay[date] ? mean(depthByDay[date]) * 100 : NaN,
    };
    for (const [key, value] of Object.entries(row)) {
      if (value === null || Number.isNaN(value)) throw new Error(`Open-Meteo has no ${key} for ${date}`);
    }
    return row;
  });
}
