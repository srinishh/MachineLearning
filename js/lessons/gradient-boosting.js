import * as THREE from 'three';
import { grid2D, textSprite, heat, PALETTE } from '../helpers.js';
import { COUNTRIES, FIELDS, TRAIN_IDX, TEST_IDX, IS_TEST, scaler } from '../data/covid.js';
import { splitPoints, SPLIT_LEGEND } from '../ml/split-points.js';
import { predictor } from '../ml/predictor.js';
import { fitRegTree, predictTree } from '../ml/trees.js';
import { r2, pct } from '../ml/metrics.js';

const S = 1.8; // display units per standard deviation (floor)
const HZ = 1.1; // display height per standard deviation of the target
const HALF = 6;
const SEG = 48;
const STEP_EVERY = 0.3; // seconds per boosting round

export default {
  id: 'gradient-boosting',
  title: 'Gradient Boosting (GBM / XGBoost)',
  category: 'Supervised Learning',
  camera: { position: [12, -16, 12], target: [0, 0, 0.3] },
  formula: 'F₀ = mean(y)<br>repeat: rᵢ = yᵢ − F(xᵢ)  (residuals)  →  fit small tree hₘ to r  →  F ← F + ν·hₘ',
  goal: 'A <strong>number</strong>: a country\'s expected COVID deaths per million, built as a <strong>sum of many small trees</strong>. Each tree corrects the mistakes left by the ones before it.',
  usage: 'Start from the base value, pass the new country through every tree, and add up their small corrections (each scaled by the learning rate). The total is the prediction. “Use the trained model” shows the base value, the first few corrections and the final answer.',
  hint: 'Surface = model prediction · Points sit at their true value · Lines are the remaining errors (residuals)',
  info: `
    <h3>What it is</h3>
    <p>Gradient boosting builds a strong model from many <strong>weak</strong> ones, small trees with only a few splits. Unlike a random
    forest, where trees are independent and vote, boosting trains trees <strong>one after another</strong>. Each new tree is fitted to the
    <strong>residuals</strong>, the errors the current model still makes, so it focuses on what is still wrong.</p>
    <h3>Why “gradient”?</h3>
    <p>For squared error, the residuals are exactly the negative <strong>gradient</strong> of the loss. Each tree is therefore a step of
    gradient descent, taken in the space of functions rather than parameters. The <strong>learning rate ν</strong> shrinks every step, so many
    small careful corrections beat a few big ones.</p>
    <h3>On the COVID data</h3>
    <p>The model predicts <strong>deaths per million</strong> (log scale) from <strong>median age</strong> and <strong>cases per
    million</strong>. Start: a flat surface at the average. Every round a depth-2 tree adds a few rectangular “steps” and the surface bends
    toward the points while the error lines shrink. The chart shows training error falling steadily, but test error levels off. With a high
    learning rate and many rounds it can even rise again, which is overfitting. Compare the test R² with linear regression's.</p>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>Gradient-boosted trees (<strong>XGBoost, LightGBM, CatBoost</strong>) dominate machine-learning competitions on tabular data and
      run behind many production systems: credit risk, insurance pricing, search ranking, and hospital models that predicted COVID
      patients' risk of deterioration from routine blood tests.</p>
    </div>
  `,
  create({ scene, ui, onClick }) {
    scene.add(grid2D(HALF * 2, 12));
    const age = scaler('median_age');
    const cases = scaler('cases_pm');
    const target = scaler('deaths_pm', { log: true });
    const X = COUNTRIES.map((_, i) => [age.z[i], cases.z[i]]);
    const Y = target.z;
    const N = X.length;
    let nu = 0.2;
    let rounds = 80;
    let depth = 2;
    let trees = [];
    let F0 = 0;
    let F = [];
    let timer = 0;
    const history = [];

    const pts = splitPoints(N, 0.12);
    scene.add(pts.group);
    X.forEach(([a, c], i) => pts.set(i, a * S, c * S, Y[i] * HZ, 1, IS_TEST[i] ? PALETTE[1] : PALETTE[0]));
    pts.commit();

    const surfGeo = new THREE.PlaneGeometry(HALF * 2, HALF * 2, SEG, SEG);
    const surfColors = new Float32Array(surfGeo.attributes.position.count * 3);
    surfGeo.setAttribute('color', new THREE.BufferAttribute(surfColors, 3));
    const surface = new THREE.Mesh(surfGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false }));
    const wire = new THREE.Mesh(surfGeo, new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.06 }));
    scene.add(surface, wire);

    const resPos = new Float32Array(N * 6);
    const resGeo = new THREE.BufferGeometry();
    resGeo.setAttribute('position', new THREE.BufferAttribute(resPos, 3));
    const residuals = new THREE.LineSegments(resGeo, new THREE.LineBasicMaterial({ color: PALETTE[2], transparent: true, opacity: 0.7 }));
    residuals.frustumCulled = false;
    scene.add(residuals);

    const marker = new THREE.Mesh(new THREE.OctahedronGeometry(0.3), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.8 }));
    scene.add(marker);

    const tick = (text, x, y, z = 0, color = '#94a3b8', height = 0.3) => {
      const s = textSprite(text, { height, color });
      s.position.set(x, y, z);
      scene.add(s);
    };
    for (const zv of [-2, 0, 2]) {
      tick(age.format(age.fromZ(zv)), zv * S, -HALF - 0.45);
      tick(cases.format(cases.fromZ(zv)), -HALF - 1, zv * S);
    }
    tick('Median age →', 3.5, -HALF - 1.1, 0, '#e2e8f0', 0.36);
    tick('Cases/M (log) ↑', -HALF - 1.2, HALF + 0.4, 0, '#e2e8f0', 0.36);
    for (const d of [10, 100, 1000]) tick(`${d.toLocaleString('en-US')} deaths/M`, HALF + 0.6, HALF, target.toZ(d) * HZ, '#ffd166', 0.3);

    const pipe = ui.pipeline([
      ['Prepare data', 'X = median age, log cases/M · Y = deaths per million (log)'],
      ['Split', `${TRAIN_IDX.length} train / ${TEST_IDX.length} test countries`],
      ['Train & optimise', 'Add depth-2 trees one at a time, each fitted to the current residuals (gradient descent in function space)'],
      ['Evaluate', ''],
    ]);
    const chartBox = ui.html('<canvas width="640" height="240" aria-label="Error per boosting round"></canvas>');
    const canvas = chartBox.querySelector('canvas');
    const roundStat = ui.stat('Boosting round');
    const r2Stat = ui.stat('R² train / test');
    const pickStat = ui.stat('Selected country', { wide: true });
    pickStat('Click a point');

    const predictX = (x) => F0 + trees.reduce((s, t) => s + nu * predictTree(t, x), 0);
    const rmse = (idx) => Math.sqrt(idx.reduce((s, i) => s + (F[i] - Y[i]) ** 2, 0) / idx.length);

    function reset() {
      F0 = TRAIN_IDX.reduce((s, i) => s + Y[i], 0) / TRAIN_IDX.length;
      F = X.map(() => F0);
      trees = [];
      history.length = 0;
      history.push([rmse(TRAIN_IDX), rmse(TEST_IDX)]);
      timer = 0;
      drawSurface();
      report();
    }

    function boostRound() {
      const R = Y.map((y, i) => y - F[i]);
      const tree = fitRegTree(X, R, TRAIN_IDX, { maxDepth: depth, minLeaf: 3 });
      trees.push(tree);
      F = F.map((v, i) => v + nu * predictTree(tree, X[i]));
      history.push([rmse(TRAIN_IDX), rmse(TEST_IDX)]);
      drawSurface();
      report();
    }

    const col = new THREE.Color();
    function drawSurface() {
      const p = surfGeo.attributes.position;
      for (let v = 0; v < p.count; v++) {
        const z = predictX([p.getX(v) / S, p.getY(v) / S]);
        p.setZ(v, z * HZ);
        heat(0.15 + 0.85 * ((z + 2.5) / 4.5), col);
        surfColors.set([col.r, col.g, col.b], v * 3);
      }
      p.needsUpdate = true;
      surfGeo.attributes.color.needsUpdate = true;
      for (let i = 0; i < N; i++) {
        resPos.set([X[i][0] * S, X[i][1] * S, Y[i] * HZ, X[i][0] * S, X[i][1] * S, F[i] * HZ], i * 6);
      }
      resGeo.attributes.position.needsUpdate = true;
    }

    function report() {
      const trR2 = r2((i) => F[i], Y, TRAIN_IDX);
      const teR2 = r2((i) => F[i], Y, TEST_IDX);
      roundStat(`${trees.length} / ${rounds}`);
      r2Stat(`${pct(Math.max(0, trR2))} / ${pct(Math.max(0, teR2))}`);
      pipe(3, `Test R² ${pct(Math.max(0, teR2))} on log deaths (train ${pct(Math.max(0, trR2))}) after ${trees.length} trees`);
      drawChart();
    }

    function drawChart() {
      const g = canvas.getContext('2d');
      const W = canvas.width;
      const H = canvas.height;
      g.clearRect(0, 0, W, H);
      g.font = '20px Inter, sans-serif';
      g.fillStyle = '#94a3b8';
      g.fillText('RMSE per boosting round', 10, 24);
      const max = Math.max(...history.flat());
      const min = Math.min(...history.flat());
      const xOf = (k) => 10 + (k / Math.max(rounds, 1)) * (W - 20);
      const yOf = (v) => 40 + (1 - (v - min) / (max - min || 1)) * (H - 60);
      for (const [j, color] of [[0, '#4cc9f0'], [1, '#f72585']]) {
        g.strokeStyle = color;
        g.lineWidth = 3;
        g.beginPath();
        history.forEach((h, k) => (k ? g.lineTo(xOf(k), yOf(h[j])) : g.moveTo(xOf(k), yOf(h[j]))));
        g.stroke();
      }
      g.fillStyle = '#4cc9f0';
      g.fillText('train', W - 130, 24);
      g.fillStyle = '#f72585';
      g.fillText('test', W - 66, 24);
    }

    ui.section('Train');
    ui.slider({ label: 'Learning rate ν', min: 0.02, max: 1, step: 0.02, value: nu, format: (v) => v.toFixed(2), onInput: (v) => { nu = v; reset(); } });
    ui.slider({ label: 'Number of trees', min: 5, max: 200, step: 5, value: rounds, onInput: (v) => { rounds = v; reset(); } });
    ui.slider({ label: 'Tree depth', min: 1, max: 4, step: 1, value: depth, onInput: (v) => { depth = v; reset(); } });
    ui.button('Restart boosting', reset, { primary: true });
    ui.button('Finish instantly', () => { while (trees.length < rounds) boostRound(); });
    ui.html(`${SPLIT_LEGEND}<div class="legend"><span style="--c:#4cc9f0">train country</span><span style="--c:#f72585">test country</span></div>`);
    const use = predictor(ui, ['median_age', 'cases_pm']);

    onClick((ray) => {
      const i = pts.pick(ray);
      if (i < 0) return;
      const c = COUNTRIES[i];
      pickStat(`${c.name}: predicted ${FIELDS.deaths_pm.fmt(Math.max(0, target.fromZ(F[i])))} vs actual ${FIELDS.deaths_pm.fmt(c.deaths_pm)} deaths/M`);
    });

    reset();

    return {
      update(dt) {
        timer += dt;
        if (trees.length < rounds && timer >= STEP_EVERY) {
          timer = 0;
          boostRound();
        }
        const q = use.values();
        const z = predictX(q);
        marker.position.set(q[0] * S, q[1] * S, z * HZ + 0.4);
        marker.rotation.z += dt * 2;
        const firsts = trees.slice(0, 3).map((t) => nu * predictTree(t, q));
        const breakdown = `base ${F0.toFixed(2)}${firsts.map((v) => ` ${v >= 0 ? '+' : '−'} ${Math.abs(v).toFixed(2)}`).join('')}${trees.length > 3 ? ` + ${trees.length - 3} more` : ''} = ${z.toFixed(2)}σ`;
        use.show(`${FIELDS.deaths_pm.fmt(Math.max(0, target.fromZ(z)))} deaths per million (${breakdown})${use.actual((c) => `${FIELDS.deaths_pm.fmt(c.deaths_pm)} deaths/M`)}`);
      },
    };
  },
};
