// Small CART implementations shared by Random Forest and Gradient Boosting.
// X: array of feature vectors, idx: rows to fit on.

// Classification tree (Gini). Leaves store P(class 1).
// `features(rand)` picks which feature indices each split may consider.
export function fitClassTree(X, Y, idx, { maxDepth = 3, minLeaf = 2, features = null, rand = Math.random } = {}) {
  const nf = X[0].length;
  function grow(rows, depth) {
    const pos = rows.reduce((s, i) => s + Y[i], 0);
    const p = rows.length ? pos / rows.length : 0.5;
    const node = { p, n: rows.length };
    if (depth >= maxDepth || rows.length < 2 * minLeaf || p === 0 || p === 1) return node;
    const gini = (a, n) => (n ? 1 - (a / n) ** 2 - (1 - a / n) ** 2 : 0);
    let best = { score: gini(pos, rows.length) - 1e-9 };
    const feats = features ? features(rand) : [...Array(nf).keys()];
    for (const f of feats) {
      const sorted = [...rows].sort((a, b) => X[a][f] - X[b][f]);
      let left = 0;
      for (let s = 0; s < sorted.length - 1; s++) {
        left += Y[sorted[s]];
        const nl = s + 1;
        const nr = sorted.length - nl;
        if (nl < minLeaf || nr < minLeaf || X[sorted[s]][f] === X[sorted[s + 1]][f]) continue;
        const score = (nl * gini(left, nl) + nr * gini(pos - left, nr)) / sorted.length;
        if (score < best.score) best = { score, f, thr: (X[sorted[s]][f] + X[sorted[s + 1]][f]) / 2 };
      }
    }
    if (best.f === undefined) return node;
    node.f = best.f;
    node.thr = best.thr;
    node.left = grow(rows.filter((i) => X[i][best.f] <= best.thr), depth + 1);
    node.right = grow(rows.filter((i) => X[i][best.f] > best.thr), depth + 1);
    return node;
  }
  return grow(idx, 0);
}

// Regression tree (squared error). Leaves store the mean target.
export function fitRegTree(X, R, idx, { maxDepth = 2, minLeaf = 3 } = {}) {
  const nf = X[0].length;
  function grow(rows, depth) {
    const sum = rows.reduce((s, i) => s + R[i], 0);
    const node = { value: rows.length ? sum / rows.length : 0, n: rows.length };
    if (depth >= maxDepth || rows.length < 2 * minLeaf) return node;
    const sq = rows.reduce((s, i) => s + R[i] * R[i], 0);
    let best = { score: sq - (sum * sum) / rows.length - 1e-9 };
    for (let f = 0; f < nf; f++) {
      const sorted = [...rows].sort((a, b) => X[a][f] - X[b][f]);
      let ls = 0;
      let lq = 0;
      for (let s = 0; s < sorted.length - 1; s++) {
        const r = R[sorted[s]];
        ls += r;
        lq += r * r;
        const nl = s + 1;
        const nr = sorted.length - nl;
        if (nl < minLeaf || nr < minLeaf || X[sorted[s]][f] === X[sorted[s + 1]][f]) continue;
        const rs = sum - ls;
        const score = (lq - (ls * ls) / nl) + (sq - lq - (rs * rs) / nr);
        if (score < best.score) best = { score, f, thr: (X[sorted[s]][f] + X[sorted[s + 1]][f]) / 2 };
      }
    }
    if (best.f === undefined) return node;
    node.f = best.f;
    node.thr = best.thr;
    node.left = grow(rows.filter((i) => X[i][best.f] <= best.thr), depth + 1);
    node.right = grow(rows.filter((i) => X[i][best.f] > best.thr), depth + 1);
    return node;
  }
  return grow(idx, 0);
}

export function predictTree(node, x) {
  let n = node;
  while (n.f !== undefined) n = x[n.f] <= n.thr ? n.left : n.right;
  return n.value ?? n.p;
}

// Human-readable path of yes/no answers a sample takes through a tree.
export function treePath(node, x, describe) {
  const steps = [];
  let n = node;
  while (n.f !== undefined) {
    const yes = x[n.f] <= n.thr;
    steps.push(describe(n.f, n.thr, yes));
    n = yes ? n.left : n.right;
  }
  return { steps, leaf: n };
}
