// The two logistic regressions from model_params.json, written out (no sklearn in the browser).
// p = 1 / (1 + exp(-(intercept + sum(coef_i * (x_i - mean_i) / std_i))))
// risk = 1 - (1 - p_dry) * (1 - p_wet): the chance of at least one dry OR wet avalanche.

export type Part = { features: string[]; mean: number[]; std: number[]; coef: number[]; intercept: number };
export type ModelParams = {
  models: { dry: Part; wet: Part };
  cutoffs: { moderate: number; considerable: number; high: number };
  season_months: number[];
};
export type Level = "low" | "moderate" | "considerable" | "high";

export function probability(part: Part, x: Record<string, number>): number {
  let z = part.intercept;
  part.features.forEach((name, i) => {
    z += (part.coef[i]! * (x[name]! - part.mean[i]!)) / part.std[i]!;
  });
  return 1 / (1 + Math.exp(-z));
}

export function riskLevel(risk: number, cutoffs: ModelParams["cutoffs"]): Level {
  if (risk >= cutoffs.high) return "high";
  if (risk >= cutoffs.considerable) return "considerable";
  if (risk >= cutoffs.moderate) return "moderate";
  return "low";
}

export function predict(params: ModelParams, x: Record<string, number>) {
  const pDry = probability(params.models.dry, x);
  const pWet = probability(params.models.wet, x);
  const risk = 1 - (1 - pDry) * (1 - pWet);
  return { pDry, pWet, risk, level: riskLevel(risk, params.cutoffs) };
}
