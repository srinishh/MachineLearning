import * as THREE from 'three';
import { grid2D, textSprite, clearGroup, damp, PALETTE } from '../helpers.js';
import { COUNTRIES, FIELDS, HIGH_MORTALITY, MEDIAN_DEATHS, IS_TEST, TRAIN_IDX, TEST_IDX, scaler } from '../data/covid.js';
import { splitPoints, SPLIT_LEGEND } from '../ml/split-points.js';
import { pct } from '../ml/metrics.js';

const N = COUNTRIES.length;
const S = 1.45; // display units per standard deviation
const OFF = 5.2; // horizontal offset of the train / test panels
const GRAY = 0x64748b;
const GREEN = 0x22c55e;
const RED = 0xef4444;
const sigmoid = (z) => 1 / (1 + Math.exp(-z));

const STEPS = [
  {
    title: '1 · Data preparation',
    text: `Gather the data and attach the correct answer to each example. Every country starts as grey inputs <strong>X</strong>
      (median age and cases per million). Then each one gets its target label <strong>Y</strong>: <span style="color:#f72585">high</span>
      or <span style="color:#4cc9f0">low</span> COVID mortality. The labels must be accurate, because the model can only be as good as its answers.`,
  },
  {
    title: '2 · Splitting',
    text: `Divide the dataset. About 80% become the <strong>training set</strong> (spheres, left), which teaches the model. The other 20% become
      the <strong>test set</strong> (cubes, right). The test set is locked away during training so it can measure real-world performance later.`,
  },
  {
    title: '3 · Training & optimisation',
    text: `Feed the training data to the algorithm, here logistic regression. It starts with random parameters and
      <strong>gradient descent</strong> adjusts them step by step to reduce the error (the loss curve below). The yellow decision boundary
      swings into place as the parameters improve.`,
  },
  {
    title: '4 · Evaluation',
    text: `Freeze the model and run it on the unseen test set. Green cubes were predicted correctly and red ones were wrong. If test accuracy is
      close to training accuracy, the model <strong>generalises</strong>. A big gap would mean it has <strong>overfit</strong> the training data.`,
  },
];

export default {
  id: 'supervised-process',
  title: 'How Supervised Learning Works',
  category: 'Supervised Learning',
  camera: { position: [0, -3, 22], target: [0, 0, 0] },
  formula: 'Data (X, Y) → split → train: minimise Loss(model(X_train), Y_train) → evaluate on X_test',
  goal: 'A model that maps inputs <strong>X</strong> (facts about a country) to the right output <strong>Y</strong> (high or low COVID mortality), including for countries it has never seen.',
  usage: 'Once evaluated, the model is <em>deployed</em>: give it the features of a new case and it returns a prediction. Retrain it when new data arrives or when its accuracy starts to slip.',
  hint: 'Use Next / Previous to step through the four stages',
  narrateOnLoad: false,
  info: `
    <h3>The supervised learning recipe</h3>
    <p>Every supervised model in this section, from linear regression to gradient boosting, follows the same four steps. Only the
    model in step 3 changes.</p>
    <ol>
      <li><strong>Data preparation</strong>: gather data and assign accurate target labels (Y) to the input features (X).</li>
      <li><strong>Splitting</strong>: divide the dataset into a training set (to teach the model) and a test set (to measure its real-world performance).</li>
      <li><strong>Training & optimisation</strong>: feed the training data to the algorithm, which adjusts its internal parameters with an optimiser such as gradient descent to minimise the error.</li>
      <li><strong>Evaluation</strong>: test the final model on the unseen test data to check that it generalises without overfitting.</li>
    </ol>
    <h3>On the COVID data</h3>
    <p>All supervised lessons share <strong>one fixed split</strong>: ${TRAIN_IDX.length} training and ${TEST_IDX.length} test countries.
    Training points are drawn as spheres and test points as cubes, so you can compare test scores across models. With only
    ${TEST_IDX.length} test countries, each one is worth about ${(100 / TEST_IDX.length).toFixed(0)} percentage points of accuracy, so
    small differences between models are not meaningful. Real projects use larger test sets or <strong>cross-validation</strong>.</p>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>A hospital building a model to predict which patients will deteriorate follows exactly these steps. It labels past patients
      (did they need ICU care?), holds out recent patients as a test set, trains the model, and only deploys it if test performance
      is good. Regulators increasingly require evidence from <strong>external validation</strong>, a test on data from other hospitals,
      because a model that only works on its own training data is dangerous.</p>
    </div>
  `,
  create({ scene, ui, onClick, narrate }) {
    const age = scaler('median_age');
    const cases = scaler('cases_pm');
    const X = COUNTRIES.map((_, i) => [age.z[i], cases.z[i]]);
    const Y = HIGH_MORTALITY;
    let step = 0;
    let w = [-1.2, 0.8];
    let b = 0.6;
    let epoch = 0;
    let acc = 0;
    const losses = [];

    const pts = splitPoints(N, 0.12);
    scene.add(pts.group);
    const cur = X.map(([x, y]) => [x * S, y * S, 0]);
    const curColor = COUNTRIES.map(() => new THREE.Color(GRAY));

    const panels = new THREE.Group();
    scene.add(panels);
    const single = grid2D(9, 9);
    scene.add(single);
    const trainGrid = grid2D(8, 8);
    trainGrid.position.x = -OFF;
    const testGrid = grid2D(8, 8);
    testGrid.position.x = OFF;
    panels.add(trainGrid, testGrid);
    const label = (text, x, y, color, h = 0.42) => {
      const s = textSprite(text, { height: h, color });
      s.position.set(x, y, 0);
      return s;
    };
    panels.add(label(`Training set · ${TRAIN_IDX.length} countries`, -OFF, 4.6, '#4cc9f0'));
    panels.add(label(`Test set · ${TEST_IDX.length} countries`, OFF, 4.6, '#f72585'));
    const axisNote = label('x: median age · y: cases per million (log)', 0, -5.6, '#94a3b8', 0.32);
    scene.add(axisNote);

    const boundaries = new THREE.Group();
    scene.add(boundaries);
    const lineMat = new THREE.LineBasicMaterial({ color: PALETTE[2] });
    const evalLabels = new THREE.Group();
    scene.add(evalLabels);

    // ---------- Panel ----------
    const stepBox = ui.html('');
    ui.button('← Previous', () => go(step - 1));
    ui.button('Next step →', () => go(step + 1), { primary: true });
    const pipe = ui.pipeline(STEPS.map((s) => [s.title.slice(4), '']));
    const pipeItems = document.querySelectorAll('.pipeline li');
    const chartBox = ui.html('<canvas width="640" height="220" aria-label="Training loss chart"></canvas>');
    const canvas = chartBox.querySelector('canvas');
    ui.html(SPLIT_LEGEND);
    const stats = {
      trainAcc: ui.stat('Train accuracy'),
      testAcc: ui.stat('Test accuracy'),
      verdict: ui.stat('Verdict', { wide: true }),
      pick: ui.stat('Selected country', { wide: true }),
    };
    stats.pick('Click a point');

    const predict = (i) => (sigmoid(w[0] * X[i][0] + w[1] * X[i][1] + b) >= 0.5 ? 1 : 0);
    const accuracy = (idx) => idx.filter((i) => predict(i) === Y[i]).length / idx.length;
    const logLoss = () => -TRAIN_IDX.reduce((s, i) => {
      const p = sigmoid(w[0] * X[i][0] + w[1] * X[i][1] + b);
      return s + (Y[i] ? Math.log(p + 1e-9) : Math.log(1 - p + 1e-9));
    }, 0) / TRAIN_IDX.length;

    function trainStep() {
      let g0 = 0;
      let g1 = 0;
      let gb = 0;
      for (const i of TRAIN_IDX) {
        const e = sigmoid(w[0] * X[i][0] + w[1] * X[i][1] + b) - Y[i];
        g0 += e * X[i][0];
        g1 += e * X[i][1];
        gb += e;
      }
      const n = TRAIN_IDX.length;
      w = [w[0] - (0.4 * g0) / n, w[1] - (0.4 * g1) / n];
      b -= (0.4 * gb) / n;
      epoch++;
      losses.push(logLoss());
    }

    function drawBoundary() {
      clearGroup(boundaries);
      const norm = Math.hypot(w[0], w[1]);
      if (norm < 1e-6) return;
      const p0 = [(-b * w[0]) / (norm * norm), (-b * w[1]) / (norm * norm)];
      const d = [-w[1] / norm, w[0] / norm];
      const offsets = step >= 3 ? [-OFF, OFF] : step === 2 ? [-OFF] : [];
      for (const ox of offsets) {
        // Clip the line to the panel square |x|,|y| <= 4.
        const pts3 = [];
        for (let t = -6; t <= 6; t += 0.05) {
          const x = (p0[0] + d[0] * t) * S;
          const y = (p0[1] + d[1] * t) * S;
          if (Math.abs(x) <= 4 && Math.abs(y) <= 4) pts3.push(new THREE.Vector3(x + ox, y, 0.05));
        }
        if (pts3.length > 1) boundaries.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts3), lineMat));
      }
    }

    function drawChart() {
      const g = canvas.getContext('2d');
      const W = canvas.width;
      const H = canvas.height;
      g.clearRect(0, 0, W, H);
      g.font = '20px Inter, sans-serif';
      g.fillStyle = '#94a3b8';
      g.fillText('training loss (cross-entropy) per epoch', 10, 24);
      if (losses.length < 2) return;
      const max = Math.max(...losses);
      const min = Math.min(...losses);
      g.strokeStyle = '#ffd166';
      g.lineWidth = 3;
      g.beginPath();
      losses.forEach((l, k) => {
        const x = 10 + (k / Math.max(losses.length - 1, 1)) * (W - 20);
        const y = 40 + (1 - (l - min) / (max - min || 1)) * (H - 60);
        if (k) g.lineTo(x, y);
        else g.moveTo(x, y);
      });
      g.stroke();
      g.fillText(`epoch ${epoch} · loss ${losses[losses.length - 1].toFixed(3)}`, W - 300, H - 8);
    }

    function targetFor(i) {
      const [x, y] = X[i];
      const ox = step === 0 ? 0 : IS_TEST[i] ? OFF : -OFF;
      let color = step === 0 && timerInStep < 1.6 ? GRAY : Y[i] ? PALETTE[1] : PALETTE[0];
      if (step === 3 && IS_TEST[i]) color = predict(i) === Y[i] ? GREEN : RED;
      return { pos: [x * S + ox, y * S, 0], color };
    }

    function script() {
      const st = STEPS[step];
      const d = document.createElement('div');
      d.innerHTML = st.text;
      return `${st.title.slice(4)}. ${d.textContent.replace(/\s+/g, ' ')}`;
    }

    let timerInStep = 0;
    function go(s) {
      step = Math.max(0, Math.min(STEPS.length - 1, s));
      timerInStep = 0;
      stepBox.innerHTML = `<div class="step-card"><div class="step-count">Step ${step + 1} of ${STEPS.length}</div><div class="step-title">${STEPS[step].title}</div><p>${STEPS[step].text}</p></div>`;
      pipeItems.forEach((li, k) => li.classList.toggle('active', k === step));
      single.visible = step === 0;
      panels.visible = step > 0;
      if (step === 2 && epoch === 0) losses.length = 0;
      clearGroup(evalLabels);
      if (step === 3) {
        const tr = accuracy(TRAIN_IDX);
        const te = accuracy(TEST_IDX);
        evalLabels.add(label(`train ${pct(tr)}`, -OFF, -4.6, '#4cc9f0'));
        evalLabels.add(label(`test ${pct(te)}`, OFF, -4.6, '#22c55e'));
      }
      drawBoundary();
      narrate(script());
    }

    onClick((ray) => {
      const i = pts.pick(ray);
      if (i < 0) return;
      const c = COUNTRIES[i];
      const p = sigmoid(w[0] * X[i][0] + w[1] * X[i][1] + b);
      stats.pick(`${c.name} (${IS_TEST[i] ? 'test' : 'train'}): age ${FIELDS.median_age.fmt(c.median_age)}, ${FIELDS.cases_pm.fmt(c.cases_pm)} cases/M → P(high) ${(p * 100).toFixed(0)}%, actual ${Y[i] ? 'high' : 'low'} (${FIELDS.deaths_pm.fmt(c.deaths_pm)} deaths/M)`);
    });

    pipe(0, `X = median age + cases/M; Y = high/low mortality (cut-off ${Math.round(MEDIAN_DEATHS)} deaths/M)`);
    pipe(1, `${TRAIN_IDX.length} train / ${TEST_IDX.length} test countries (fixed, random)`);
    pipe(2, 'Logistic regression, gradient descent on cross-entropy');
    pipe(3, 'Accuracy on the unseen test countries');
    go(0);

    return {
      script,
      update(dt) {
        timerInStep += dt;
        if (step === 2) {
          acc += dt;
          while (acc >= 1 / 12 && epoch < 400) {
            acc -= 1 / 12;
            trainStep();
            drawBoundary();
          }
          drawChart();
        }
        if (step === 3 && losses.length === 0) {
          for (let k = 0; k < 300; k++) trainStep();
          drawChart();
          drawBoundary();
          go(3);
        }
        const k = damp(4, dt);
        for (let i = 0; i < N; i++) {
          const t = targetFor(i);
          for (let j = 0; j < 3; j++) cur[i][j] += (t.pos[j] - cur[i][j]) * k;
          curColor[i].lerp(new THREE.Color(t.color), k);
          pts.set(i, cur[i][0], cur[i][1], cur[i][2], 1, curColor[i]);
        }
        pts.commit();
        const tr = accuracy(TRAIN_IDX);
        const te = accuracy(TEST_IDX);
        stats.trainAcc(epoch ? pct(tr) : '–');
        stats.testAcc(step === 3 ? pct(te) : 'locked until step 4');
        stats.verdict(step < 3 ? '–' : Math.abs(tr - te) < 0.12 ? 'Generalises well: test is close to train ✓' : 'Large train/test gap: possible overfitting');
        pipe(3, step === 3 ? `Test accuracy ${pct(te)} vs train ${pct(tr)}` : 'Accuracy on the unseen test countries');
      },
    };
  },
};
