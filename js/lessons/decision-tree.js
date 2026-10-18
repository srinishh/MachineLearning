import * as THREE from 'three';
import { grid2D, textSprite, clearGroup, PALETTE, smoothstep } from '../helpers.js';
import { COUNTRIES, FIELDS, HIGH_MORTALITY, MEDIAN_DEATHS, TRAIN_IDX, TEST_IDX, scaler } from '../data/covid.js';
import { splitPoints, SPLIT_LEGEND } from '../ml/split-points.js';
import { predictor } from '../ml/predictor.js';
import { pct } from '../ml/metrics.js';

const S = 1.6; // display units per standard deviation
const AGE = scaler('median_age');
const CASES = scaler('cases_pm');
const AXES = [AGE, CASES];
const AXIS_NAMES = ['median age', 'cases/M'];
const X = COUNTRIES.map((_, i) => [AGE.z[i] * S, CASES.z[i] * S]);
const HALF = Math.max(5, Math.ceil(Math.max(...X.flat().map(Math.abs))) + 0.5);
const WALL_H = 1.1;
const REVEAL = 0.9; // seconds per split

const gini = (c0, c1) => {
  const n = c0 + c1;
  return n ? 1 - (c0 / n) ** 2 - (c1 / n) ** 2 : 0;
};

// CART with Gini impurity, fitted on `rows` only. Nodes keep their region bounds for drawing.
function buildTree(Y, rows, maxDepth) {
  function grow(idx, depth, bounds) {
    const counts = [0, 0];
    idx.forEach((i) => counts[Y[i]]++);
    const node = { idx, depth, bounds, counts, majority: counts[1] > counts[0] ? 1 : 0 };
    const g = gini(counts[0], counts[1]);
    if (depth >= maxDepth || idx.length < 4 || g === 0) return node;

    let best = { score: g - 1e-9 };
    for (const axis of [0, 1]) {
      const sorted = [...idx].sort((a, b) => X[a][axis] - X[b][axis]);
      const left = [0, 0];
      const right = [...counts];
      for (let s = 0; s < sorted.length - 1; s++) {
        left[Y[sorted[s]]]++;
        right[Y[sorted[s]]]--;
        const v0 = X[sorted[s]][axis];
        const v1 = X[sorted[s + 1]][axis];
        if (v0 === v1) continue;
        const nl = s + 1;
        const nr = sorted.length - nl;
        const score = (nl * gini(left[0], left[1]) + nr * gini(right[0], right[1])) / sorted.length;
        if (score < best.score) best = { score, axis, thr: (v0 + v1) / 2 };
      }
    }
    if (best.axis === undefined) return node;

    node.axis = best.axis;
    node.thr = best.thr;
    const lb = { ...bounds };
    const rb = { ...bounds };
    if (best.axis === 0) { lb.x1 = best.thr; rb.x0 = best.thr; } else { lb.y1 = best.thr; rb.y0 = best.thr; }
    node.left = grow(idx.filter((i) => X[i][best.axis] <= best.thr), depth + 1, lb);
    node.right = grow(idx.filter((i) => X[i][best.axis] > best.thr), depth + 1, rb);
    return node;
  }
  return grow(rows, 0, { x0: -HALF, x1: HALF, y0: -HALF, y1: HALF });
}

const ruleText = (n) => `${AXIS_NAMES[n.axis]} ≤ ${AXES[n.axis].format(AXES[n.axis].fromZ(n.thr / S))}`;

export default {
  id: 'decision-tree',
  title: 'Decision Trees',
  category: 'Supervised Learning',
  camera: { position: [0, -13, 15.5], target: [0, 0, 0] },
  formula: 'Gini(node) = 1 − Σ pₖ²<br>best split = argmax [ Gini(parent) − weighted Gini(children) ]',
  goal: 'A <strong>class label</strong> (high or low mortality) produced by a flowchart of yes/no questions. The trained model <em>is</em> that flowchart: each wall in the scene is one question, and each coloured box is an answer.',
  usage: 'Start at the top question and follow the yes/no branches with the new country\'s values until you reach a box. That box\'s majority class is the prediction. The “Use the trained model” section shows the exact path of questions for any country you enter.',
  hint: 'Each wall is one yes/no question · Tiles show the predicted label · Spheres = train, cubes = test',
  info: `
    <h3>What it is</h3>
    <p>A decision tree classifies by asking a sequence of <strong>yes/no questions</strong> about the features, such as
    “is the median age ≤ 31?”, then “are cases ≤ 50,000 per million?”. Every question splits the space with a straight wall along one
    axis, and each resulting box predicts its majority class.</p>
    <h3>On the COVID data</h3>
    <p>The tree predicts whether a country had <span style="color:#f72585">high</span> or <span style="color:#4cc9f0">low</span>
    COVID mortality from its <strong>median age</strong> (left–right) and <strong>cases per million</strong> (bottom–top, log). These are
    the same features used by logistic regression, k-NN, Naive Bayes and Random Forest, so you can compare test scores. Each label
    states its rule in real units.</p>
    <h3>How it picks questions</h3>
    <p>At each step the tree tries every possible split of the <strong>training</strong> countries and keeps the one that makes the two
    sides <strong>purest</strong>, measured by <strong>Gini impurity</strong> (0 when a region holds only one class). It then repeats inside
    each side. This is a <strong>greedy, recursive</strong> algorithm.</p>
    <h3>Try this</h3>
    <p>Raise the max depth. Training accuracy keeps climbing, but test accuracy stops improving or drops, because the tree builds tiny
    boxes around single unusual countries. That's overfitting. <strong>Random forests</strong> and <strong>gradient boosting</strong>
    combine many trees to fix this.</p>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>Decision trees are popular in medicine because a person can follow them. Clinical triage rules such as <strong>CURB-65</strong>,
      used to judge pneumonia severity (including COVID pneumonia), are essentially small trees of yes/no questions about confusion,
      urea, breathing rate, blood pressure and age ≥ 65.</p>
    </div>
  `,
  create({ scene, ui, onClick }) {
    scene.add(grid2D(HALF * 2, 10));
    const Y = HIGH_MORTALITY;
    const N = COUNTRIES.length;
    let maxDepth = 3;
    let root = null;
    let order = [];
    let revealed = 0;
    let timer = 0;
    let hold = 0;

    const pts = splitPoints(N, 0.12);
    scene.add(pts.group);
    const walls = new THREE.Group();
    const tiles = new THREE.Group();
    scene.add(walls, tiles);
    const marker = new THREE.Mesh(new THREE.OctahedronGeometry(0.28), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.8 }));
    scene.add(marker);

    const pipe = ui.pipeline([
      ['Prepare data', `X = median age, log cases/M · Y = high (≥ ${Math.round(MEDIAN_DEATHS)} deaths/M) or low`],
      ['Split', `${TRAIN_IDX.length} train / ${TEST_IDX.length} test countries`],
      ['Train & optimise', 'Greedy splits that minimise Gini impurity (no gradient descent)'],
      ['Evaluate', ''],
    ]);
    const depthStat = ui.stat('Splits shown');
    const leafStat = ui.stat('Regions');
    const accStat = ui.stat('Accuracy train / test', { wide: true });

    X.forEach(([x, y], i) => pts.set(i, x, y, 0.12, 1, PALETTE[Y[i] ? 1 : 0]));
    pts.commit();

    const tick = (text, x, y, color = '#94a3b8', height = 0.3) => {
      const s = textSprite(text, { height, color });
      s.position.set(x, y, 0);
      scene.add(s);
    };
    for (const zv of [-2, 0, 2]) {
      tick(AGE.format(AGE.fromZ(zv)), zv * S, -HALF - 0.4);
      tick(CASES.format(CASES.fromZ(zv)), -HALF - 1, zv * S);
    }
    tick('Median age →', 3, -HALF - 1, '#e2e8f0', 0.36);
    tick('Cases per million (log) ↑', -3, HALF + 0.5, '#e2e8f0', 0.36);
    const pickStat = ui.stat('Selected country', { wide: true });
    pickStat('Click a point');
    onClick((ray) => {
      const i = pts.pick(ray);
      if (i < 0) return;
      const c = COUNTRIES[i];
      pickStat(`${c.name}: age ${FIELDS.median_age.fmt(c.median_age)}, ${FIELDS.cases_pm.fmt(c.cases_pm)} cases/M → ${FIELDS.deaths_pm.fmt(c.deaths_pm)} deaths/M (${Y[i] ? 'high' : 'low'})`);
    });

    // Prediction using only the splits revealed so far.
    function predictRevealed(x) {
      const revealedSet = new Set(order.slice(0, revealed));
      let n = root;
      while (revealedSet.has(n)) n = x[n.axis] <= n.thr ? n.left : n.right;
      return n.majority;
    }

    function rebuild() {
      root = buildTree(Y, TRAIN_IDX, maxDepth);
      order = [];
      const queue = [root];
      while (queue.length) {
        const n = queue.shift();
        if (n.axis === undefined) continue;
        order.push(n);
        queue.push(n.left, n.right);
      }
      clearGroup(walls);
      for (const n of order) {
        const { x0, x1, y0, y1 } = n.bounds;
        const color = PALETTE[2 + (n.depth % 4)];
        const len = n.axis === 0 ? y1 - y0 : x1 - x0;
        const geo = n.axis === 0 ? new THREE.BoxGeometry(0.05, len, WALL_H) : new THREE.BoxGeometry(len, 0.05, WALL_H);
        const wall = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.35, transparent: true, opacity: 0.75 }));
        wall.position.set(n.axis === 0 ? n.thr : (x0 + x1) / 2, n.axis === 0 ? (y0 + y1) / 2 : n.thr, WALL_H / 2);
        wall.scale.setScalar(0.0001);
        const label = textSprite(ruleText(n), { height: 0.36 - n.depth * 0.04, background: 'rgba(2,6,23,0.75)', color: '#' + color.toString(16).padStart(6, '0') });
        label.position.set(wall.position.x, wall.position.y, WALL_H + 0.35);
        label.visible = false;
        walls.add(wall, label);
        n.wall = wall;
        n.label = label;
      }
      revealed = 0;
      timer = 0;
      hold = 0;
      paintTiles();
    }

    function frontier() {
      const out = [];
      const revealedSet = new Set(order.slice(0, revealed));
      (function walk(n) {
        if (revealedSet.has(n)) { walk(n.left); walk(n.right); } else out.push(n);
      })(root);
      return out;
    }

    function paintTiles() {
      clearGroup(tiles);
      const leaves = frontier();
      for (const n of leaves) {
        const { x0, x1, y0, y1 } = n.bounds;
        const total = n.counts[0] + n.counts[1];
        const purity = total ? Math.max(...n.counts) / total : 0.5;
        const tile = new THREE.Mesh(
          new THREE.PlaneGeometry(x1 - x0 - 0.04, y1 - y0 - 0.04),
          new THREE.MeshBasicMaterial({ color: PALETTE[n.majority ? 1 : 0], transparent: true, opacity: 0.08 + 0.32 * (purity - 0.5) * 2, depthWrite: false }),
        );
        tile.position.set((x0 + x1) / 2, (y0 + y1) / 2, 0.005);
        tiles.add(tile);
      }
      const accOn = (idx) => idx.filter((i) => predictRevealed(X[i]) === Y[i]).length / idx.length;
      const tr = accOn(TRAIN_IDX);
      const te = accOn(TEST_IDX);
      depthStat(`${revealed} / ${order.length}`);
      leafStat(leaves.length);
      accStat(`${pct(tr)} / ${pct(te)}`);
      pipe(3, `Test accuracy ${pct(te)} (train ${pct(tr)}) with ${revealed} split${revealed === 1 ? '' : 's'}`);
    }

    ui.section('Train');
    ui.slider({ label: 'Max depth', min: 1, max: 7, step: 1, value: maxDepth, onInput: (v) => { maxDepth = v; rebuild(); } });
    ui.button('Replay', rebuild, { primary: true });
    ui.html(SPLIT_LEGEND);
    const use = predictor(ui, ['median_age', 'cases_pm']);
    const pathBox = ui.html('');

    rebuild();

    return {
      update(dt) {
        if (revealed < order.length) {
          timer += dt;
          const n = order[revealed];
          const k = smoothstep(timer / REVEAL);
          n.wall.scale.set(n.axis === 0 ? 1 : k, n.axis === 0 ? k : 1, 1);
          if (timer >= REVEAL) {
            n.wall.scale.set(1, 1, 1);
            n.label.visible = true;
            revealed++;
            timer = 0;
            paintTiles();
          }
        } else {
          hold += dt;
          if (hold > 6) rebuild();
        }

        // Follow the full trained tree for the "use the model" inputs.
        const [qa, qc] = use.values();
        const q = [qa * S, qc * S];
        marker.position.set(q[0], q[1], 0.6);
        marker.rotation.z += dt * 2;
        const steps = [];
        let n = root;
        while (n.axis !== undefined) {
          const yes = q[n.axis] <= n.thr;
          steps.push(`${ruleText(n)}? <strong>${yes ? 'yes' : 'no'}</strong>`);
          n = yes ? n.left : n.right;
        }
        const label = n.majority ? 'HIGH' : 'LOW';
        const html = `<div class="field-head"><span>Path through the tree</span></div><ol class="tree-path">${steps.map((s) => `<li>${s}</li>`).join('')}<li>→ predict <strong>${label}</strong> (${n.counts[1]} of ${n.counts[0] + n.counts[1]} training countries in this box were high)</li></ol>`;
        if (pathBox.innerHTML !== html) pathBox.innerHTML = html;
        use.show(`${label} mortality${use.actual((c) => `${FIELDS.deaths_pm.fmt(c.deaths_pm)} deaths/M (${c.deaths_pm >= MEDIAN_DEATHS ? 'high' : 'low'})`)}`);
      },
    };
  },
};
