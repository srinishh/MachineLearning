// Evaluation helpers. `idx` selects which rows (train or test) to score.

export const accuracy = (predict, Y, idx) => idx.filter((i) => predict(i) === Y[i]).length / idx.length;

export const mse = (predict, Y, idx) => idx.reduce((s, i) => s + (predict(i) - Y[i]) ** 2, 0) / idx.length;

// R² = 1 − SSE / SST, measured around the mean of the scored rows.
export function r2(predict, Y, idx) {
  const mean = idx.reduce((s, i) => s + Y[i], 0) / idx.length;
  const sst = idx.reduce((s, i) => s + (Y[i] - mean) ** 2, 0);
  const sse = idx.reduce((s, i) => s + (predict(i) - Y[i]) ** 2, 0);
  return 1 - sse / sst;
}

export const pct = (v) => `${(100 * v).toFixed(1)}%`;
