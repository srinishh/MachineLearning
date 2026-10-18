import * as THREE from 'three';
import { rng, gauss, grid2D, sphereCloud, setInstance, setInstanceColor, commitInstances, PALETTE, damp, clamp } from '../helpers.js';

const SX = 6; // display scale for x in [-1, 1]
const SY = 2.6; // display scale for y
const SAMPLES = 220;
const MAX_DEG = 14;
const truth = (x) => Math.sin(Math.PI * x);

// Legendre polynomials: same function space as x^k, far better conditioned.
function basis(x, d) {
  const out = [1];
  if (d >= 1) out.push(x);
  for (let n = 1; n < d; n++) out.push(((2 * n + 1) * x * out[n] - n * out[n - 1]) / (n + 1));
  return out;
}

function solve(A, b) {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    [M[col], M[piv]] = [M[piv], M[col]];
    const d = M[col][col] || 1e-12;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const k = M[r][col] / d;
      for (let c = col; c <= n; c++) M[r][c] -= k * M[col][c];
    }
  }
  return M.map((row, i) => row[n] / (row[i] || 1e-12));
}

function fit(xs, ys, d, lambda) {
  const m = d + 1;
  const A = Array.from({ length: m }, () => new Array(m).fill(0));
  const b = new Array(m).fill(0);
  xs.forEach((x, i) => {
    const phi = basis(x, d);
    for (let r = 0; r < m; r++) {
      b[r] += phi[r] * ys[i];
      for (let c = 0; c < m; c++) A[r][c] += phi[r] * phi[c];
    }
  });
  for (let r = 1; r < m; r++) A[r][r] += lambda * xs.length + 1e-12;
  return solve(A, b);
}

const predict = (coef, x) => basis(x, coef.length - 1).reduce((s, v, i) => s + v * coef[i], 0);
const mse = (coef, xs, ys) => xs.reduce((s, x, i) => s + (predict(coef, x) - ys[i]) ** 2, 0) / xs.length;

export default {
  id: 'overfitting',
  title: 'Overfitting & Regularization',
  category: 'Foundations',
  camera: { position: [0, -3, 18], target: [0, 0, 0] },
  formula: 'ŷ = Σₖ wₖ·xᵏ  (k = 0…degree)<br>Loss = MSE + λ·Σ wₖ²   (ridge)',
  hint: 'Cyan = training data the model sees · Pink = unseen test data',
  info: `
    <h3>The core tension</h3>
    <p>A model should capture the <strong>pattern</strong> in the data and ignore the
    <strong>noise</strong>. Here we fit polynomials of increasing degree to noisy samples of a sine
    wave (the dashed line).</p>
    <ul>
      <li><strong>Underfitting (high bias)</strong>: the model is too simple, so a straight line can't bend to follow the wave. Both training and test error are high.</li>
      <li><strong>Overfitting (high variance)</strong>: the model is so flexible that it threads through every training point, noise included. Training error is near zero, but the curve swings wildly and test error grows.</li>
    </ul>
    <h3>The U-shaped curve</h3>
    <p>The chart shows error against model complexity. Training error always falls as degree goes up, while test
    error falls and then rises again. The best model sits at the bottom of the test curve.</p>
    <h3>Regularization</h3>
    <p><strong>Ridge regularization</strong> adds a penalty <code>λ·Σw²</code> that discourages large
    weights. With a high degree, raise λ and watch the wild curve calm down. Regularization lets you keep a
    flexible model without overfitting.</p>
  `,
  create({ scene, ui }) {
    scene.add(grid2D(14, 14));
    const rand = rng(21);
    let trainX;
    let trainY;
    let testX;
    let testY;
    let degree = 3;
    let lambda = 0;

    const train = sphereCloud(15, 0.16);
    const test = sphereCloud(40, 0.1);
    scene.add(train, test);

    // Dashed ground-truth curve.
    const truthPts = [];
    for (let i = 0; i <= 120; i++) {
      const x = -1 + (2 * i) / 120;
      truthPts.push(new THREE.Vector3(x * SX, truth(x) * SY, -0.05));
    }
    const truthLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(truthPts),
      new THREE.LineDashedMaterial({ color: 0x94a3b8, dashSize: 0.25, gapSize: 0.18 }),
    );
    truthLine.computeLineDistances();
    scene.add(truthLine);

    const curveMat = new THREE.MeshStandardMaterial({ color: PALETTE[2], emissive: PALETTE[2], emissiveIntensity: 0.6 });
    let tube = null;
    const shown = new Float32Array(SAMPLES);
    const target = new Float32Array(SAMPLES);

    const chart = ui.html('<canvas width="640" height="300" aria-label="Error versus degree chart"></canvas>');
    const canvas = chart.querySelector('canvas');
    const trainStat = ui.stat('Train MSE');
    const testStat = ui.stat('Test MSE');
    const verdictStat = ui.stat('Diagnosis', { wide: true });
    let curves = { train: [], test: [] };

    function newData() {
      trainX = [];
      trainY = [];
      testX = [];
      testY = [];
      for (let i = 0; i < 15; i++) {
        const x = -1 + (2 * (i + 0.5 + (rand() - 0.5) * 0.6)) / 15;
        trainX.push(x);
        trainY.push(truth(x) + gauss(rand) * 0.25);
      }
      for (let i = 0; i < 40; i++) {
        const x = rand() * 2 - 1;
        testX.push(x);
        testY.push(truth(x) + gauss(rand) * 0.25);
      }
      trainX.forEach((x, i) => { setInstance(train, i, x * SX, trainY[i] * SY, 0); setInstanceColor(train, i, PALETTE[0]); });
      testX.forEach((x, i) => { setInstance(test, i, x * SX, testY[i] * SY, 0); setInstanceColor(test, i, PALETTE[1]); });
      commitInstances(train);
      commitInstances(test);
      refit();
    }

    function refit() {
      const coef = fit(trainX, trainY, degree, lambda);
      for (let i = 0; i < SAMPLES; i++) {
        const x = -1 + (2 * i) / (SAMPLES - 1);
        target[i] = clamp(predict(coef, x) * SY, -7, 7);
      }
      const tr = mse(coef, trainX, trainY);
      const te = mse(coef, testX, testY);
      trainStat(tr.toFixed(4));
      testStat(te.toFixed(4));
      if (tr > 0.12) verdictStat('Underfitting: too simple (high bias)');
      else if (te > 1.8 * tr && te > 0.11) verdictStat('Overfitting: memorising noise (high variance)');
      else verdictStat('Good fit: captures the pattern ✓');
      curves = { train: [], test: [] };
      for (let d = 0; d <= MAX_DEG; d++) {
        const c = fit(trainX, trainY, d, lambda);
        curves.train.push(mse(c, trainX, trainY));
        curves.test.push(mse(c, testX, testY));
      }
      drawChart();
    }

    function drawChart() {
      const g = canvas.getContext('2d');
      const W = canvas.width;
      const H = canvas.height;
      const pad = { l: 56, r: 16, t: 26, b: 40 };
      g.clearRect(0, 0, W, H);
      const lo = -3.5;
      const hi = 1.5;
      const X = (d) => pad.l + (d / MAX_DEG) * (W - pad.l - pad.r);
      const Y = (v) => pad.t + (1 - (clamp(Math.log10(Math.max(v, 1e-6)), lo, hi) - lo) / (hi - lo)) * (H - pad.t - pad.b);
      g.font = '20px Inter, sans-serif';
      g.fillStyle = '#94a3b8';
      g.strokeStyle = 'rgba(148,163,184,0.18)';
      g.lineWidth = 1;
      for (let e = -3; e <= 1; e++) {
        g.beginPath();
        g.moveTo(pad.l, Y(10 ** e));
        g.lineTo(W - pad.r, Y(10 ** e));
        g.stroke();
        g.fillText(`1e${e}`, 6, Y(10 ** e) + 6);
      }
      g.fillText('polynomial degree →', W / 2 - 90, H - 8);
      g.fillText('error (log)', pad.l, 18);
      g.strokeStyle = 'rgba(255,209,102,0.8)';
      g.setLineDash([6, 6]);
      g.beginPath();
      g.moveTo(X(degree), pad.t);
      g.lineTo(X(degree), H - pad.b);
      g.stroke();
      g.setLineDash([]);
      for (const [key, color] of [['train', '#4cc9f0'], ['test', '#f72585']]) {
        g.strokeStyle = color;
        g.fillStyle = color;
        g.lineWidth = 3;
        g.beginPath();
        curves[key].forEach((v, d) => (d ? g.lineTo(X(d), Y(v)) : g.moveTo(X(d), Y(v))));
        g.stroke();
        g.beginPath();
        g.arc(X(degree), Y(curves[key][degree]), 6, 0, Math.PI * 2);
        g.fill();
      }
      g.fillStyle = '#4cc9f0';
      g.fillText('train', W - 130, 18);
      g.fillStyle = '#f72585';
      g.fillText('test', W - 66, 18);
    }

    function rebuildTube() {
      if (tube) {
        scene.remove(tube);
        tube.geometry.dispose();
      }
      const pts = [];
      for (let i = 0; i < SAMPLES; i++) pts.push(new THREE.Vector3((-1 + (2 * i) / (SAMPLES - 1)) * SX, shown[i], 0));
      tube = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), SAMPLES * 2, 0.06, 8), curveMat);
      scene.add(tube);
    }

    ui.slider({ label: 'Polynomial degree', min: 0, max: MAX_DEG, step: 1, value: degree, onInput: (v) => { degree = v; refit(); } });
    ui.slider({
      label: 'Regularization λ', min: -7, max: 0, step: 0.25, value: -7,
      format: (v) => (v <= -7 ? '0' : `1e${v}`),
      onInput: (v) => { lambda = v <= -7 ? 0 : 10 ** v; refit(); },
    });
    ui.toggle({ label: 'Show true function', value: true, onChange: (v) => { truthLine.visible = v; } });
    ui.toggle({ label: 'Show test data', value: true, onChange: (v) => { test.visible = v; } });
    ui.button('New samples', newData, { primary: true });

    newData();
    shown.set(target);
    rebuildTube();

    return {
      update(dt) {
        const k = damp(6, dt);
        let moved = 0;
        for (let i = 0; i < SAMPLES; i++) {
          const d = target[i] - shown[i];
          shown[i] += d * k;
          moved = Math.max(moved, Math.abs(d));
        }
        if (moved > 1e-3) rebuildTube();
      },
      dispose() { curveMat.dispose(); },
    };
  },
};
