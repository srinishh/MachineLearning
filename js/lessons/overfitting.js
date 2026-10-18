import * as THREE from 'three';
import { rng, grid2D, sphereCloud, setInstance, setInstanceColor, commitInstances, textSprite, clearGroup, PALETTE, damp, clamp } from '../helpers.js';
import { DAILY } from '../data/covid.js';

const SX = 6; // display scale for x in [-1, 1]
const SY = 5; // display units per normalised case unit
const Y0 = -3; // display y of zero cases
const SAMPLES = 220;
const MAX_DEG = 14;
const N_TRAIN = 15;
const N_TEST = 37;
const MONTHS = ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb'];

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
  camera: { position: [0, -2, 18], target: [0, 0, 0] },
  formula: 'cases(t) ≈ Σₖ wₖ·tᵏ  (k = 0…degree)<br>Loss = MSE + λ·Σ wₖ²   (ridge)',
  hint: 'Cyan = weeks the model trains on · Pink = held-out weeks · Dashed = full 7-day average curve',
  info: `
    <h3>The core tension</h3>
    <p>A model should capture the real <strong>pattern</strong> in the data, not just the particular examples it was shown.
    Here we fit polynomials of increasing degree to a country's COVID case curve over the first pandemic year.</p>
    <h3>On the COVID data</h3>
    <p>Each country reported its cases once a week; each dot is one weekly report (shown as cases per million per day). The model
    only sees <strong>15 randomly chosen weeks</strong> (cyan). The other weeks (pink) form a <strong>test set</strong> it never
    trains on, and the dashed line is the full curve.</p>
    <ul>
      <li><strong>Underfitting (high bias)</strong>: a low-degree curve can't follow the waves, so training and test error are both high.</li>
      <li><strong>Overfitting (high variance)</strong>: a high-degree curve passes exactly through all 15 training weeks but swings wildly in the gaps between them. Training error drops to zero while test error explodes.</li>
    </ul>
    <h3>The U-shaped curve</h3>
    <p>The chart shows error against model complexity. Training error always falls as degree goes up, while test error falls and
    then rises again. The best model sits at the bottom of the test curve.</p>
    <h3>Regularization</h3>
    <p><strong>Ridge regularization</strong> adds a penalty <code>λ·Σw²</code> that discourages large weights. At a high degree,
    raise λ and watch the wild curve calm down.</p>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>Early in the pandemic, curve-fitting forecasts made headlines. In May 2020 a widely criticised “cubic fit” of US deaths,
      published by White House economic advisers, implied deaths would fall to near zero within weeks. A flexible curve that matches
      the recent data can say almost anything outside it. Serious forecasters avoided this with validation on held-out data,
      regularization and models grounded in epidemiology.</p>
    </div>
  `,
  create({ scene, ui }) {
    scene.add(grid2D(14, 14));
    const rand = rng(21);
    let country = DAILY.countries[0];
    let trainX;
    let trainY;
    let testX;
    let testY;
    let degree = 4;
    let lambda = 0;
    let scale = 1;

    const train = sphereCloud(N_TRAIN, 0.16);
    const test = sphereCloud(N_TEST, 0.1);
    scene.add(train, test);
    const decor = new THREE.Group();
    scene.add(decor);
    let truthLine = null;
    let showTruth = true;

    const curveMat = new THREE.MeshStandardMaterial({ color: PALETTE[2], emissive: PALETTE[2], emissiveIntensity: 0.6 });
    let tube = null;
    const shown = new Float32Array(SAMPLES).fill(Y0);
    const target = new Float32Array(SAMPLES);

    const chart = ui.html('<canvas width="640" height="300" aria-label="Error versus degree chart"></canvas>');
    const canvas = chart.querySelector('canvas');
    const trainStat = ui.stat('Train MSE');
    const testStat = ui.stat('Test MSE');
    const verdictStat = ui.stat('Diagnosis', { wide: true });
    let curves = { train: [], test: [] };

    const dayToX = (d) => -1 + (2 * d) / (DAILY.days - 1);

    function newSample() {
      scale = Math.max(...country.smooth);
      const reports = [...country.reports];
      for (let i = reports.length - 1; i > 0; i--) {
        const j = Math.floor(rand() * (i + 1));
        [reports[i], reports[j]] = [reports[j], reports[i]];
      }
      const tr = reports.slice(0, N_TRAIN);
      const te = reports.slice(N_TRAIN, N_TRAIN + N_TEST);
      trainX = tr.map(([d]) => dayToX(d));
      trainY = tr.map(([, v]) => v / scale);
      testX = te.map(([d]) => dayToX(d));
      testY = te.map(([, v]) => v / scale);
      const yd = (v) => Math.min(Y0 + v * SY, 7);
      trainX.forEach((x, i) => { setInstance(train, i, x * SX, yd(trainY[i]), 0.05); setInstanceColor(train, i, PALETTE[0]); });
      testX.forEach((x, i) => { setInstance(test, i, x * SX, yd(testY[i]), 0.05); setInstanceColor(test, i, PALETTE[1]); });
      commitInstances(train);
      commitInstances(test);
      buildDecor();
      refit();
    }

    function buildDecor() {
      clearGroup(decor);
      const pts = country.smooth.map((v, d) => new THREE.Vector3(dayToX(d) * SX, Y0 + (v / scale) * SY, -0.05));
      truthLine = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineDashedMaterial({ color: 0x94a3b8, dashSize: 0.25, gapSize: 0.18 }));
      truthLine.computeLineDistances();
      truthLine.visible = showTruth;
      decor.add(truthLine);
      MONTHS.forEach((m, i) => {
        const s = textSprite(m, { height: 0.3, color: '#94a3b8' });
        s.position.set(dayToX(i * 30.4 + 15) * SX, Y0 - 0.45, 0);
        decor.add(s);
      });
      for (const f of [0.5, 1]) {
        const s = textSprite(Math.round(scale * f).toLocaleString('en-US'), { height: 0.3, color: '#94a3b8' });
        s.position.set(-SX - 0.9, Y0 + f * SY, 0);
        decor.add(s);
      }
      const title = textSprite(`${country.name}: new cases per million per day, Mar 2020 – Feb 2021`, { height: 0.38, color: '#e2e8f0' });
      title.position.set(0, 7.7, 0);
      decor.add(title);
    }

    function refit() {
      const coef = fit(trainX, trainY, degree, lambda);
      for (let i = 0; i < SAMPLES; i++) {
        const x = -1 + (2 * i) / (SAMPLES - 1);
        target[i] = clamp(Y0 + predict(coef, x) * SY, -7, 7);
      }
      curves = { train: [], test: [] };
      for (let d = 0; d <= MAX_DEG; d++) {
        const c = fit(trainX, trainY, d, lambda);
        curves.train.push(mse(c, trainX, trainY));
        curves.test.push(mse(c, testX, testY));
      }
      const tr = curves.train[degree];
      const te = curves.test[degree];
      const best = curves.test.indexOf(Math.min(...curves.test));
      trainStat(tr.toFixed(4));
      testStat(te.toFixed(4));
      if (te <= curves.test[best] * 1.25) verdictStat(`Good fit ✓ (best test error is at degree ${best})`);
      else if (degree < best) verdictStat(`Underfitting: too simple (best is degree ${best})`);
      else verdictStat(`Overfitting: fits training weeks, fails between them (best is degree ${best})`);
      drawChart();
    }

    function drawChart() {
      const g = canvas.getContext('2d');
      const W = canvas.width;
      const H = canvas.height;
      const pad = { l: 56, r: 16, t: 26, b: 40 };
      g.clearRect(0, 0, W, H);
      const all = [...curves.train, ...curves.test].map((v) => Math.log10(Math.max(v, 1e-6)));
      const lo = Math.floor(Math.min(...all));
      const hi = Math.min(Math.ceil(Math.max(...all)), lo + 6);
      const X = (d) => pad.l + (d / MAX_DEG) * (W - pad.l - pad.r);
      const Y = (v) => pad.t + (1 - (clamp(Math.log10(Math.max(v, 1e-6)), lo, hi) - lo) / (hi - lo)) * (H - pad.t - pad.b);
      g.font = '20px Inter, sans-serif';
      g.fillStyle = '#94a3b8';
      g.strokeStyle = 'rgba(148,163,184,0.18)';
      g.lineWidth = 1;
      for (let e = lo; e <= hi; e++) {
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

    ui.select({
      label: 'Country',
      options: DAILY.countries.map((c) => ({ value: c.iso, label: c.name })),
      value: country.iso,
      onChange: (iso) => { country = DAILY.countries.find((c) => c.iso === iso); newSample(); },
    });
    ui.slider({ label: 'Polynomial degree', min: 0, max: MAX_DEG, step: 1, value: degree, onInput: (v) => { degree = v; refit(); } });
    ui.slider({
      label: 'Regularization λ', min: -7, max: 0, step: 0.25, value: -7,
      format: (v) => (v <= -7 ? '0' : `1e${v}`),
      onInput: (v) => { lambda = v <= -7 ? 0 : 10 ** v; refit(); },
    });
    ui.toggle({ label: 'Show full curve', value: true, onChange: (v) => { showTruth = v; if (truthLine) truthLine.visible = v; } });
    ui.toggle({ label: 'Show test weeks', value: true, onChange: (v) => { test.visible = v; } });
    ui.button('Resample weeks', newSample, { primary: true });

    newSample();
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
