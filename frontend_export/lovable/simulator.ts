// "What if?" simulator: a few understandable inputs -> the full feature row the model needs.
// Derived features are rebuilt exactly like compute_features() in the notebook:
//   wind_snow = wind_gust x snow_72h, warming_after_snow = snow_72h x max(0, temp_change),
//   snow_on_weak_layer = snow_7d if a weak layer exists, warm_sunny_day = temp_max > 0 and sunny.
import type { Part } from "./model";

export type SimInputs = {
  snow_72h: number; // cm new snow, last 3 days
  snow_7d: number; // cm new snow, last 7 days (at least snow_72h)
  snow_depth: number; // cm
  wind_gust: number; // km/h
  temp_max: number; // C
  temp_min: number; // C
  temp_change: number; // C, today's max minus yesterday's max
  temp_gradient: number; // C/m, cold air over thin snow (> 10 = weak crystals grow)
  weak_layer: boolean; // was there a weak layer 1-4 weeks ago (temp_gradient > 10)?
  sunny: boolean; // shortwave radiation > 9 MJ/m2
  avalanche_yesterday: boolean;
  month: number; // 1-12
};

const MONTH_COLUMNS: Record<number, string> = {
  11: "month_Nov", 12: "month_Dec", 1: "month_Jan", 2: "month_Feb", 3: "month_Mar", 4: "month_Apr", 5: "month_May",
};

export function featuresFromInputs(s: SimInputs): Record<string, number> {
  const snow7d = Math.max(s.snow_7d, s.snow_72h);
  const f: Record<string, number> = {
    snow_72h: s.snow_72h,
    snow_7d: snow7d,
    snow_depth: s.snow_depth,
    wind_gust: s.wind_gust,
    wind_snow: s.wind_gust * s.snow_72h,
    temp_gradient: s.temp_gradient,
    snow_on_weak_layer: s.weak_layer ? snow7d : 0,
    temp_max: s.temp_max,
    temp_min: s.temp_min,
    temp_change: s.temp_change,
    warming_after_snow: s.snow_72h * Math.max(0, s.temp_change),
    warm_sunny_day: s.temp_max > 0 && s.sunny ? 1 : 0,
    avalanche_yesterday: s.avalanche_yesterday ? 1 : 0,
  };
  for (const [m, column] of Object.entries(MONTH_COLUMNS)) f[column] = s.month === Number(m) ? 1 : 0;
  return f;
}

// The reverse: start the sliders from a real feature row (tomorrow, or a test case).
export function inputsFromFeatures(f: Record<string, number>, month: number): SimInputs {
  return {
    snow_72h: f["snow_72h"] ?? 0,
    snow_7d: f["snow_7d"] ?? 0,
    snow_depth: f["snow_depth"] ?? 0,
    wind_gust: f["wind_gust"] ?? 0,
    temp_max: f["temp_max"] ?? 0,
    temp_min: f["temp_min"] ?? 0,
    temp_change: f["temp_change"] ?? 0,
    temp_gradient: f["temp_gradient"] ?? 0,
    weak_layer: (f["snow_on_weak_layer"] ?? 0) > 0 || ((f["snow_7d"] ?? 0) === 0 && (f["temp_gradient"] ?? 0) > 10),
    sunny: (f["warm_sunny_day"] ?? 0) === 1,
    avalanche_yesterday: (f["avalanche_yesterday"] ?? 0) === 1,
    month,
  };
}

// "Why this risk?": each feature's push on the log-odds, coef x standardised value (months left out),
// the same as the agent's main_causes. Positive = raises the risk.
export function contributions(part: Part, x: Record<string, number>): { feature: string; value: number }[] {
  return part.features
    .map((name, i) => ({ feature: name, value: (part.coef[i]! * (x[name]! - part.mean[i]!)) / part.std[i]! }))
    .filter((c) => !c.feature.startsWith("month_"))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
}

export const LABELS: Record<string, string> = {
  snow_72h: "New snow (3 days)", snow_7d: "New snow (7 days)", snow_depth: "Snow depth",
  wind_snow: "Wind slab (wind x new snow)", wind_gust: "Strong wind",
  temp_gradient: "Weak layer forming (cold, thin snow)", snow_on_weak_layer: "New snow on a weak layer",
  warming_after_snow: "Warming after snowfall", temp_max: "Daytime temperature", temp_min: "Night temperature",
  temp_change: "Warming since yesterday", warm_sunny_day: "Warm sunny day",
  avalanche_yesterday: "Avalanches the day before",
};
