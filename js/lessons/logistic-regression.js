import * as THREE from 'three';
import {
  rng, gauss, floorGrid, sphereCloud, setInstance, setInstanceColor, commitInstances,
  unitCylinder, placeCylinder, PALETTE,
} from '../helpers.js';

const H = 4; // display height of probability 1
const sigmoid = (z) => 1 / (1 + Math.exp(-z));

export default {
  id: 'logistic-regression',
  title: 'Logistic Regression',
  category: 'Supervised Learning',
  camera: { position: [10, 8, 12], target: [0, 1.6, 0] },
  formula: 'p(y=1 | x) = σ(w₁x₁ + w₂x₂ + b)<br>σ(z) = 1 / (1 + e⁻ᶻ)<br>Loss = −Σ [y·log p + (1−y)·log(1−p)]',
  hint: 'Height of the sheet = predicted probability of the pink class',
  info: `
    <h3>What it is</h3>
    <p>Logistic regression is the basic <strong>classification</strong> model. It takes a weighted sum of
    the inputs, as linear regression does, then passes it through the <strong>sigmoid</strong>
    <code>σ</code>, an S-shaped curve that squashes any number into a probability between 0 and 1.</p>
    <h3>Reading the animation</h3>
    <ul>
      <li>Cyan points (class 0) sit on the floor and pink points (class 1) float at height 1.</li>
      <li>The coloured sheet is the model's probability <code>p</code> at every point on the floor.</li>
      <li>The yellow line is the <strong>decision boundary</strong>, where <code>p = 0.5</code>. Points on one side are predicted pink, on the other cyan.</li>
      <li>Points that are currently misclassified are drawn <strong>larger</strong>.</li>
    </ul>
    <h3>How it learns</h3>
    <p>Training minimises <strong>cross-entropy loss</strong> with gradient descent. This loss heavily
    penalises confident wrong answers. As the weights grow, the sheet goes from flat to a steep cliff.</p>
    <h3>Try this</h3>
    <p>Increase the class overlap. The boundary can't separate the classes perfectly, so the sheet stays
    gentle and the model stays uncertain where they mix.</p>
  `,
  create({ scene, ui }) {
    scene.add(floorGrid(12, 12));
    const rand = rng(17);
    const N = 100;
    let X = [];
    let Y = [];
    let w1 = 0;
    let w2 = 0;
    let b = 0;
    let lr = 0.3;
    let spread = 1.2;
    let epoch = 0;
    let acc = 0;

    const pts = sphereCloud(N, 0.14);
    scene.add(pts);

    const sheetGeo = new THREE.PlaneGeometry(12, 12, 60, 60);
    sheetGeo.rotateX(-Math.PI / 2);
    const sheetColors = new Float32Array(sheetGeo.attributes.position.count * 3);
    sheetGeo.setAttribute('color', new THREE.BufferAttribute(sheetColors, 3));
    const sheet = new THREE.Mesh(sheetGeo, new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false }));
    const sheetWire = new THREE.Mesh(sheetGeo, new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.06 }));
    scene.add(sheet, sheetWire);

    const boundary = unitCylinder(PALETTE[2], { emissive: 0.8 });
    scene.add(boundary);

    const stats = {
      epoch: ui.stat('Epoch'),
      loss: ui.stat('Cross-entropy'),
      acc: ui.stat('Accuracy'),
      w: ui.stat('w₁, w₂, b'),
    };

    function newData() {
      X = [];
      Y = [];
      for (let i = 0; i < N; i++) {
        const y = i % 2;
        const cx = y ? 1.6 : -1.6;
        const cz = y ? 1.2 : -1.2;
        X.push([cx + gauss(rand) * spread, cz + gauss(rand) * spread]);
        Y.push(y);
      }
      restart();
    }

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
    ui.slider({ label: 'Class overlap', min: 0.4, max: 2.5, step: 0.1, value: spread, format: (v) => v.toFixed(1), onInput: (v) => { spread = v; newData(); } });
    ui.button('Restart training', restart, { primary: true });
    ui.button('New data', newData);

    newData();

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
          const p = sigmoid(w1 * pos.getX(i) + w2 * pos.getZ(i) + b);
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
          setInstance(pts, i, X[i][0], Y[i] * H, X[i][1], ok ? 1 : 1.7);
          setInstanceColor(pts, i, Y[i] ? PALETTE[1] : PALETTE[0]);
        }
        commitInstances(pts);

        const norm = Math.hypot(w1, w2);
        boundary.visible = norm > 1e-3;
        if (boundary.visible) {
          const px = (-b * w1) / (norm * norm);
          const pz = (-b * w2) / (norm * norm);
          const dx = -w2 / norm;
          const dz = w1 / norm;
          a.set(px - dx * 8, H / 2, pz - dz * 8);
          c.set(px + dx * 8, H / 2, pz + dz * 8);
          placeCylinder(boundary, a, c, 0.05);
        }

        stats.epoch(epoch);
        stats.loss((loss / N).toFixed(4));
        stats.acc(`${((100 * correct) / N).toFixed(0)}%`);
        stats.w(`${w1.toFixed(2)}, ${w2.toFixed(2)}, ${b.toFixed(2)}`);
      },
    };
  },
};
