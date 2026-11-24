// Offline check: do features.ts and model.ts give the same numbers as Python?
// Run with:  node frontend_export/lovable/check.ts
import { readFileSync } from "node:fs";
import { computeFeatures } from "./features.ts";
import { predict } from "./model.ts";
import { featuresFromInputs, inputsFromFeatures } from "./simulator.ts";

const params = JSON.parse(readFileSync(new URL("../data/model_params.json", import.meta.url), "utf8"));
let failures = 0;

for (const t of params.test_cases) {
  const r = predict(params, t.inputs);
  const ok = Math.abs(r.pDry - t.p_dry) < 5e-4 && Math.abs(r.pWet - t.p_wet) < 5e-4
    && Math.abs(r.risk - t.risk) < 5e-4 && r.level === t.level;
  if (!ok) failures++;
  console.log(`${ok ? "OK  " : "FAIL"} ${t.date}: JS ${r.risk.toFixed(3)} ${r.level} | Python ${t.risk.toFixed(3)} ${t.level}`);
}

const last = computeFeatures(params.feature_test.raw_daily_weather).at(-1)!;
for (const [name, expected] of Object.entries(params.feature_test.expected_features) as [string, number][]) {
  const ok = Math.abs(last[name] - expected) < 1e-6;
  if (!ok) { failures++; console.log(`FAIL feature ${name}: JS ${last[name]} | Python ${expected}`); }
}
console.log(`Features for ${params.feature_test.date}: ${failures === 0 ? "all match" : "see failures above"}`);

// Simulator round trip: real feature row -> slider inputs -> feature row must give the same risk
for (const t of params.test_cases) {
  const month = Number(t.date.slice(5, 7));
  const r = predict(params, featuresFromInputs(inputsFromFeatures(t.inputs, month)));
  const ok = Math.abs(r.risk - t.risk) < 5e-4;
  if (!ok) failures++;
  console.log(`${ok ? "OK  " : "FAIL"} simulator ${t.date}: ${r.risk.toFixed(3)} vs Python ${t.risk.toFixed(3)}`);
}
process.exit(failures ? 1 : 0);
