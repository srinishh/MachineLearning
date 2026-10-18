import * as THREE from 'three';
import {
  grid2D, sphereCloud, setInstance, setInstanceColor, commitInstances,
  textSprite, clearGroup, PALETTE, smoothstep,
} from '../helpers.js';
import { COUNTRIES, FIELDS, HIGH_MORTALITY, MEDIAN_DEATHS, scaler } from '../data/covid.js';

const HALF = 5;
const S = 1.9; // display units per standard deviation
const AGE = scaler('median_age');
const VAX = scaler('vax');
const AXES = [AGE, VAX];
const WALL_H = 1.1;
const REVEAL = 0.9; // seconds per split

const gini = (c0, c1) => {
  const n = c0 + c1;
  return n ? 1 - (c0 / n) ** 2 - (c1 / n) ** 2 : 0;
};

function buildTree(X, Y, maxDepth) {
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
    node.gain = g - best.score;
    const lb = { ...bounds };
    const rb = { ...bounds };
    if (best.axis === 0) { lb.x1 = best.thr; rb.x0 = best.thr; } else { lb.y1 = best.thr; rb.y0 = best.thr; }
    node.left = grow(idx.filter((i) => X[i][best.axis] <= best.thr), depth + 1, lb);
    node.right = grow(idx.filter((i) => X[i][best.axis] > best.thr), depth + 1, rb);
    return node;
  }
  return grow(X.map((_, i) => i), 0, { x0: -HALF, x1: HALF, y0: -HALF, y1: HALF });
}

export default {
  id: 'decision-tree',
  title: 'Decision Trees',
  category: 'Supervised Learning',
  camera: { position: [0, -10, 13], target: [0, 0, 0] },
  formula: 'Gini(node) = 1 − Σ pₖ²<br>best split = argmax [ Gini(parent) − weighted Gini(children) ]',
  hint: 'Each wall is one yes/no question about a country · Tiles show the predicted label · Click a dot to name it',
  info: `
    <h3>What it is</h3>
    <p>A decision tree classifies by asking a sequence of <strong>yes/no questions</strong> about the features, such as
    “is the median age ≤ 31?”, then “is vaccination ≤ 55%?”. Every question splits the space with a straight wall along one axis,
    and each resulting box predicts its majority class.</p>
    <h3>On the COVID data</h3>
    <p>The tree predicts whether a country had <span style="color:#f72585">high</span> or <span style="color:#4cc9f0">low</span>
    COVID mortality (above or below the median deaths per million) from its <strong>median age</strong> (left–right) and
    <strong>% fully vaccinated</strong> (bottom–top). The first and most important split is on age, matching the regression
    lessons. Later splits refine the picture, and each label states its rule in real units.</p>
    <h3>How it picks questions</h3>
    <p>At each step the tree tries every possible split and keeps the one that makes the two sides
    <strong>purest</strong>, measured by <strong>Gini impurity</strong> (0 when a region holds only one class).
    It then repeats inside each side. This is a <strong>greedy, recursive</strong> algorithm.</p>
    <h3>Reading the animation</h3>
    <ul>
      <li>Walls appear in breadth-first order. Taller label = earlier, more important split.</li>
      <li>Tile colour = predicted class; brighter tiles are purer.</li>
    </ul>
    <h3>Try this</h3>
    <p>Raise the max depth. Training accuracy keeps climbing, but at depth 6–7 the tree builds tiny boxes
    around single unusual countries. That's overfitting. <strong>Random forests</strong> average many such trees
    to smooth this out.</p>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>Decision trees are popular in medicine because a person can follow them. Clinical triage rules such as
      <strong>CURB-65</strong>, used to judge pneumonia severity (including COVID pneumonia), are essentially small trees of yes/no
      questions about confusion, urea, breathing rate, blood pressure and age ≥ 65. Tree ensembles such as gradient-boosted
      trees are among the most accurate models for tabular data like this dataset.</p>
    </div>
  `,
  create({ scene, ui, onClick }) {
    scene.add(grid2D(HALF * 2, 10));
    const X = COUNTRIES.map((_, i) => [AGE.z[i] * S, VAX.z[i] * S]);
    const Y = HIGH_MORTALITY;
    let maxDepth = 4;
    let root = null;
    let order = [];
    let revealed = 0;
    let timer = 0;
    let hold = 0;

    const N = COUNTRIES.length;
    const pts = sphereCloud(N, 0.12);
    scene.add(pts);
    const walls = new THREE.Group();
    const tiles = new THREE.Group();
    scene.add(walls, tiles);

    const depthStat = ui.stat('Splits shown');
    const leafStat = ui.stat('Regions');
    const accStat = ui.stat('Train accuracy', { wide: true });

    X.forEach(([x, y], i) => {
      setInstance(pts, i, x, y, 0.12);
      setInstanceColor(pts, i, PALETTE[Y[i] ? 1 : 0]);
    });
    commitInstances(pts);

    const tick = (text, x, y, color = '#94a3b8', height = 0.3) => {
      const s = textSprite(text, { height, color });
      s.position.set(x, y, 0);
      scene.add(s);
    };
    for (const zv of [-2, 0, 2]) {
      tick(AGE.format(AGE.fromZ(zv)), zv * S, -HALF - 0.4);
      tick(VAX.format(VAX.fromZ(zv)), -HALF - 0.8, zv * S);
    }
    tick('Median age →', 3, -HALF - 1, '#e2e8f0', 0.36);
    tick('% fully vaccinated ↑', -3, HALF + 0.5, '#e2e8f0', 0.36);
    const pickStat = ui.stat('Selected country', { wide: true });
    pickStat(`Click a dot · high mortality = ≥ ${Math.round(MEDIAN_DEATHS)} deaths/M`);
    onClick((ray) => {
      const hit = ray.intersectObject(pts)[0];
      if (hit?.instanceId === undefined) return;
      const c = COUNTRIES[hit.instanceId];
      pickStat(`${c.name}: age ${FIELDS.median_age.fmt(c.median_age)}, ${FIELDS.vax.fmt(c.vax)} vaccinated → ${FIELDS.deaths_pm.fmt(c.deaths_pm)} deaths/M (${Y[hit.instanceId] ? 'high' : 'low'})`);
    });

    function rebuild() {
      root = buildTree(X, Y, maxDepth);
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
        const label = textSprite(`${n.axis === 0 ? 'age' : 'vaccinated'} ≤ ${AXES[n.axis].format(AXES[n.axis].fromZ(n.thr / S))}`, { height: 0.36 - n.depth * 0.04, background: 'rgba(2,6,23,0.75)', color: '#' + color.toString(16).padStart(6, '0') });
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
      let correct = 0;
      for (const n of leaves) {
        const { x0, x1, y0, y1 } = n.bounds;
        const total = n.counts[0] + n.counts[1];
        const purity = total ? Math.max(...n.counts) / total : 0.5;
        correct += Math.max(...n.counts);
        const tile = new THREE.Mesh(
          new THREE.PlaneGeometry(x1 - x0 - 0.04, y1 - y0 - 0.04),
          new THREE.MeshBasicMaterial({ color: PALETTE[n.majority ? 1 : 0], transparent: true, opacity: 0.08 + 0.32 * (purity - 0.5) * 2, depthWrite: false }),
        );
        tile.position.set((x0 + x1) / 2, (y0 + y1) / 2, 0.005);
        tiles.add(tile);
      }
      depthStat(`${revealed} / ${order.length}`);
      leafStat(leaves.length);
      accStat(`${((100 * correct) / N).toFixed(1)}%`);
    }

    ui.slider({ label: 'Max depth', min: 1, max: 7, step: 1, value: maxDepth, onInput: (v) => { maxDepth = v; rebuild(); } });
    ui.button('Replay', rebuild, { primary: true });

    rebuild();

    return {
      update(dt) {
        if (revealed < order.length) {
          timer += dt;
          const n = order[revealed];
          const k = smoothstep(timer / REVEAL);
          n.wall.scale.set(n.axis === 0 ? 1 : k, n.axis === 0 ? k : 1, 1);
          n.wall.scale.z = 1;
          if (timer >= REVEAL) {
            n.wall.scale.set(1, 1, 1);
            n.label.visible = true;
            revealed++;
            timer = 0;
            paintTiles();
          }
        } else {
          hold += dt;
          if (hold > 5) rebuild();
        }
      },
    };
  },
};
