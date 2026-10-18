import * as THREE from 'three';
import { floorGrid, unitCylinder, placeCylinder, textSprite, PALETTE } from '../helpers.js';
import { COUNTRIES, FIELDS, HIGH_MORTALITY, MEDIAN_DEATHS, TRAIN_IDX, TEST_IDX, scaler } from '../data/covid.js';
import { splitPoints, SPLIT_LEGEND } from '../ml/split-points.js';
import { predictor } from '../ml/predictor.js';
import { pct } from '../ml/metrics.js';

const H = 4; // display height of probability 1
const S = 2.2; // display units per standard deviation
const N = COUNTRIES.length;
const sigmoid = (z) => 1 / (1 + Math.exp(-z));

export default {
  id: 'logistic-regression',
  title: 'Logistic Regression',
  category: 'Supervised Learning',
  camera: { position: [10, 8, 12], target: [0, 1.6, 0] },
  formula: 'P(high mortality) = σ(w₁·age + w₂·log cases + b)<br>σ(z) = 1 / (1 + e⁻ᶻ)<br>Loss = −Σ [y·log p + (1−y)·log(1−p)]',
  goal: 'A <strong>probability between 0 and 1</strong> that a country is in the high-mortality half, which becomes a yes/no label by cutting at 0.5. The trained model is three numbers: one weight per feature plus a bias.',
  usage: 'For a new country, compute <em>σ(w₁·age + w₂·log cases + b)</em> to get its risk. Below 0.5 means “low”, above means “high”. Move the sliders in “Use the trained model” or load a test country and watch the white marker ride the probability sheet.',
  hint: 'Sheet height = predicted probability of high mortality · Spheres = train, cubes = test · Click a point to inspect it',
  info: `
    <h3>What it is</h3>
    <p>Logistic regression is the basic <strong>classification</strong> model. It takes a weighted sum of the inputs, as linear
    regression does, then passes it through the <strong>sigmoid</strong> <code>σ</code>, an S-curve that turns any number into a
    probability between 0 and 1.</p>
    <h3>On the COVID data</h3>
    <p>The question is: <em>is this country in the high-mortality half</em> (deaths per million above the median)? The model uses
    two features, the country's <strong>median age</strong> and its <strong>cases per million</strong> (log scale).</p>
    <ul>
      <li>Floor axes: median age (left–right) and log cases per million (front–back), both standardised.</li>
      <li><span style="color:#4cc9f0">Cyan</span> low-mortality countries sit on the floor; <span style="color:#f72585">pink</span> high-mortality countries float at height 1.</li>
      <li>The sheet is the model's probability everywhere. The yellow line is the <strong>decision boundary</strong> (p = 0.5).</li>
      <li>Misclassified countries are drawn <strong>larger</strong>. Click them to see who the model gets wrong.</li>
    </ul>
    <p>Both learned weights end up positive: older populations and more infections both push the probability up. The model is
    trained only on the spheres. On the unseen test cubes it gets roughly 4 out of 5 countries right.</p>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>Hospitals used logistic-regression risk scores to triage COVID patients. The UK's <strong>ISARIC 4C Mortality Score</strong>
      turns a patient's age, sex, comorbidities and blood results into an in-hospital mortality risk. It is a logistic model simplified into
      points a clinician can add up at the bedside. Banks use the same technique for credit scoring.</p>
    </div>
  `,
  create({ scene, ui, onClick }) {
    scene.add(floorGrid(12, 12));
    const age = scaler('median_age');
    const cases = scaler('cases_pm');
    const X = COUNTRIES.map((_, i) => [age.z[i], cases.z[i]]);
    const Y = HIGH_MORTALITY;
    let w1 = 0;
    let w2 = 0;
    let b = 0;
    let lr = 0.3;
    let epoch = 0;
    let acc = 0;

    const pts = splitPoints(N, 0.13);
    scene.add(pts.group);
    const marker = new THREE.Mesh(new THREE.OctahedronGeometry(0.32), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.8 }));
    scene.add(marker);

    const sheetGeo = new THREE.PlaneGeometry(12, 12, 60, 60);
    sheetGeo.rotateX(-Math.PI / 2);
    const sheetColors = new Float32Array(sheetGeo.attributes.position.count * 3);
    sheetGeo.setAttribute('color', new THREE.BufferAttribute(sheetColors, 3));
    const sheet = new THREE.Mesh(sheetGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.5, side: THREE.DoubleSide, depthWrite: false }));
    const sheetWire = new THREE.Mesh(sheetGeo, new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.06 }));
    scene.add(sheet, sheetWire);

    const boundary = unitCylinder(PALETTE[2], { emissive: 0.8 });
    scene.add(boundary);

    const tick = (text, x, y, z, color = '#94a3b8', height = 0.3) => {
      const s = textSprite(text, { height, color });
      s.position.set(x, y, z);
      scene.add(s);
    };
    for (const zv of [-2, 0, 2]) {
      tick(age.format(age.fromZ(zv)), zv * S, 0.05, 6.6);
      tick(`${cases.format(cases.fromZ(zv))}`, 6.9, 0.05, zv * S);
    }
    tick('Median age →', 0, 0.05, 7.4, '#e2e8f0', 0.36);
    tick('Cases per million (log) →', 7.4, 0.6, -3.5, '#e2e8f0', 0.36);
    tick('p = 1 (high mortality)', -6, H + 0.4, -6, '#f72585', 0.34);

    const pipe = ui.pipeline([
      ['Prepare data', `X = median age, log cases/M · Y = high (≥ ${Math.round(MEDIAN_DEATHS)} deaths/M) or low`],
      ['Split', `${TRAIN_IDX.length} train / ${TEST_IDX.length} test countries`],
      ['Train & optimise', 'Gradient descent on cross-entropy loss'],
      ['Evaluate', ''],
    ]);
    const stats = {
      epoch: ui.stat('Epoch'),
      loss: ui.stat('Cross-entropy (train)'),
      acc: ui.stat('Accuracy train / test'),
      w: ui.stat('w_age, w_cases, b'),
      pick: ui.stat('Selected country', { wide: true }),
    };
    stats.pick(`Click a dot · high mortality = ≥ ${Math.round(MEDIAN_DEATHS)} deaths/M`);

    function restart() {
      w1 = w2 = b = 0;
      epoch = 0;
    }

    function step() {
      let g1 = 0;
      let g2 = 0;
      let gb = 0;
      for (const i of TRAIN_IDX) {
        const e = sigmoid(w1 * X[i][0] + w2 * X[i][1] + b) - Y[i];
        g1 += e * X[i][0];
        g2 += e * X[i][1];
        gb += e;
      }
      const n = TRAIN_IDX.length;
      w1 -= (lr * g1) / n;
      w2 -= (lr * g2) / n;
      b -= (lr * gb) / n;
      epoch++;
    }

    ui.slider({ label: 'Learning rate η', min: 0.01, max: 2, step: 0.01, value: lr, format: (v) => v.toFixed(2), onInput: (v) => { lr = v; } });
    ui.button('Restart training', restart, { primary: true });
    ui.html(SPLIT_LEGEND);
    const use = predictor(ui, ['median_age', 'cases_pm']);

    onClick((ray) => {
      const i = pts.pick(ray);
      if (i < 0) return;
      const c = COUNTRIES[i];
      const p = sigmoid(w1 * X[i][0] + w2 * X[i][1] + b);
      stats.pick(`${c.name}: age ${FIELDS.median_age.fmt(c.median_age)}, ${FIELDS.cases_pm.fmt(c.cases_pm)} cases/M → P(high) = ${(p * 100).toFixed(0)}% · actual ${FIELDS.deaths_pm.fmt(c.deaths_pm)} deaths/M (${Y[i] ? 'high' : 'low'})`);
    });

    const col = new THREE.Color();
    const c0 = new THREE.Color(PALETTE[0]);
    const c1 = new THREE.Color(PALETTE[1]);
    const a = new THREE.Vector3();
    const c = new THREE.Vector3();

    return {
      update(dt) {
        acc += dt;
        while (acc >= 1 / 30) {
          acc -= 1 / 30;
          step();
        }

        const pos = sheetGeo.attributes.position;
        for (let i = 0; i < pos.count; i++) {
          const p = sigmoid(w1 * (pos.getX(i) / S) + w2 * (pos.getZ(i) / S) + b);
          pos.setY(i, p * H);
          col.copy(c0).lerp(c1, p);
          sheetColors.set([col.r, col.g, col.b], i * 3);
        }
        pos.needsUpdate = true;
        sheetGeo.attributes.color.needsUpdate = true;

        let loss = 0;
        const correct = { train: 0, test: 0 };
        const isTest = new Set(TEST_IDX);
        for (let i = 0; i < N; i++) {
          const p = sigmoid(w1 * X[i][0] + w2 * X[i][1] + b);
          if (!isTest.has(i)) loss -= Y[i] ? Math.log(p + 1e-9) : Math.log(1 - p + 1e-9);
          const ok = (p > 0.5 ? 1 : 0) === Y[i];
          if (ok) correct[isTest.has(i) ? 'test' : 'train']++;
          pts.set(i, X[i][0] * S, Y[i] * H, X[i][1] * S, ok ? 1 : 1.7, Y[i] ? PALETTE[1] : PALETTE[0]);
        }
        pts.commit();

        const [qa, qc] = use.values();
        const qp = sigmoid(w1 * qa + w2 * qc + b);
        marker.position.set(qa * S, qp * H + 0.35, qc * S);
        marker.rotation.y += dt * 2;
        use.show(`P(high mortality) = ${(qp * 100).toFixed(0)}% → ${qp >= 0.5 ? 'HIGH' : 'LOW'}${use.actual((c) => `${FIELDS.deaths_pm.fmt(c.deaths_pm)} deaths/M (${c.deaths_pm >= MEDIAN_DEATHS ? 'high' : 'low'})`)}`);

        const norm = Math.hypot(w1, w2);
        boundary.visible = norm > 1e-3;
        if (boundary.visible) {
          const px = ((-b * w1) / (norm * norm)) * S;
          const pz = ((-b * w2) / (norm * norm)) * S;
          const dx = -w2 / norm;
          const dz = w1 / norm;
          a.set(px - dx * 8, H / 2, pz - dz * 8);
          c.set(px + dx * 8, H / 2, pz + dz * 8);
          placeCylinder(boundary, a, c, 0.05);
        }

        stats.epoch(epoch);
        const trAcc = correct.train / TRAIN_IDX.length;
        const teAcc = correct.test / TEST_IDX.length;
        stats.loss((loss / TRAIN_IDX.length).toFixed(4));
        stats.acc(`${pct(trAcc)} / ${pct(teAcc)}`);
        pipe(3, `Test accuracy ${pct(teAcc)} on ${TEST_IDX.length} unseen countries (train ${pct(trAcc)})`);
        stats.w(`${w1.toFixed(2)}, ${w2.toFixed(2)}, ${b.toFixed(2)}`);
      },
    };
  },
};
