import * as THREE from 'three';
import {
  floorGrid, sphereCloud, setInstance, setInstanceColor, commitInstances,
  unitCylinder, placeCylinder, textSprite, PALETTE,
} from '../helpers.js';
import { COUNTRIES, FIELDS, HIGH_MORTALITY, MEDIAN_DEATHS, scaler } from '../data/covid.js';

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
  hint: 'Height of the sheet = predicted probability that a country has high COVID mortality · Click a dot to inspect it',
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
    <p>Both learned weights end up positive: older populations and more infections both push the probability up. The model gets
    roughly 85–87% of countries right.</p>
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

    const pts = sphereCloud(N, 0.13);
    scene.add(pts);

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

    const stats = {
      epoch: ui.stat('Epoch'),
      loss: ui.stat('Cross-entropy'),
      acc: ui.stat('Accuracy'),
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
      for (let i = 0; i < N; i++) {
        const e = sigmoid(w1 * X[i][0] + w2 * X[i][1] + b) - Y[i];
        g1 += e * X[i][0];
        g2 += e * X[i][1];
        gb += e;
      }
      w1 -= (lr * g1) / N;
      w2 -= (lr * g2) / N;
      b -= (lr * gb) / N;
      epoch++;
    }

    ui.slider({ label: 'Learning rate η', min: 0.01, max: 2, step: 0.01, value: lr, format: (v) => v.toFixed(2), onInput: (v) => { lr = v; } });
    ui.button('Restart training', restart, { primary: true });

    onClick((ray) => {
      const hit = ray.intersectObject(pts)[0];
      if (hit?.instanceId === undefined) return;
      const i = hit.instanceId;
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
        let correct = 0;
        for (let i = 0; i < N; i++) {
          const p = sigmoid(w1 * X[i][0] + w2 * X[i][1] + b);
          loss -= Y[i] ? Math.log(p + 1e-9) : Math.log(1 - p + 1e-9);
          const ok = (p > 0.5 ? 1 : 0) === Y[i];
          if (ok) correct++;
          setInstance(pts, i, X[i][0] * S, Y[i] * H, X[i][1] * S, ok ? 1 : 1.7);
          setInstanceColor(pts, i, Y[i] ? PALETTE[1] : PALETTE[0]);
        }
        commitInstances(pts);

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
        stats.loss((loss / N).toFixed(4));
        stats.acc(`${((100 * correct) / N).toFixed(1)}%`);
        stats.w(`${w1.toFixed(2)}, ${w2.toFixed(2)}, ${b.toFixed(2)}`);
      },
    };
  },
};
