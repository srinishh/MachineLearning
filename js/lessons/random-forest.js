import * as THREE from 'three';
import { grid2D, textSprite, rng, damp, PALETTE } from '../helpers.js';
import { COUNTRIES, FIELDS, HIGH_MORTALITY, MEDIAN_DEATHS, TRAIN_IDX, TEST_IDX, scaler } from '../data/covid.js';
import { splitPoints, SPLIT_LEGEND } from '../ml/split-points.js';
import { predictor } from '../ml/predictor.js';
import { fitClassTree, predictTree } from '../ml/trees.js';
import { pct } from '../ml/metrics.js';

const S = 1.8; // display units per standard deviation
const HALF = 6;
const RES = 80;
const LAYER_RES = 48;
const MAX_LAYERS = 8;
const GROW_EVERY = 0.45; // seconds per new tree
const LOW = new THREE.Color(PALETTE[0]);
const HIGH = new THREE.Color(PALETTE[1]);

export default {
  id: 'random-forest',
  title: 'Random Forest',
  category: 'Supervised Learning',
  camera: { position: [0, -17, 13], target: [0, 0, 1.8] },
  formula: 'Tree t is trained on a bootstrap sample (random rows) with random features per split<br>P(high | x) = ¹⁄ₜ Σₜ treeₜ(x)   →   majority vote',
  goal: 'A <strong>class label plus a confidence</strong>: the share of trees voting “high mortality”. Averaging many deliberately different trees gives a smoother, more reliable answer than any single tree.',
  usage: 'Send the new country down <em>every</em> tree, count the votes, and report the majority (and the vote share as confidence). Use “Use the trained model” to see how many of the trees vote high for any country.',
  hint: 'Floating layers = individual trees · Floor = the forest\'s averaged vote · Spheres = train, cubes = test',
  info: `
    <h3>What it is</h3>
    <p>A random forest is a <strong>team of decision trees</strong>. One deep tree overfits, memorising quirks of its training data.
    A forest trains many trees, each on a slightly different view of the data, and lets them <strong>vote</strong>. Their individual
    mistakes are different, so they cancel out. This idea is called <strong>bagging</strong> (bootstrap aggregating).</p>
    <h3>Two kinds of randomness</h3>
    <ul>
      <li><strong>Bootstrap rows</strong>: each tree trains on a random sample of the training countries drawn <em>with replacement</em>, so some countries appear twice and others not at all.</li>
      <li><strong>Random features</strong>: at each split a tree may only consider a random subset of the features. This makes the trees disagree more, which helps the vote.</li>
    </ul>
    <h3>On the COVID data</h3>
    <p>Each tree predicts high vs. low mortality from <strong>median age</strong> and <strong>cases per million</strong>, the same task
    as the single decision tree. The floating layers are the last few trees: each is blocky and a bit odd. The floor is the forest's
    average: smoother, with soft purple zones where the trees disagree. Compare the forest's test accuracy with a single tree's.</p>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>Random forests are a workhorse for tabular data. Hospitals have used them to predict COVID patients' risk of ICU admission
      from vital signs and lab results, banks use them for fraud detection, and ecologists use them to map species habitats. They
      need little tuning, handle mixed feature types, and report <strong>feature importance</strong>.</p>
    </div>
  `,
  create({ scene, ui, onClick }) {
    const grid = grid2D(HALF * 2, 12);
    scene.add(grid);
    const age = scaler('median_age');
    const cases = scaler('cases_pm');
    const X = COUNTRIES.map((_, i) => [age.z[i], cases.z[i]]);
    const Y = HIGH_MORTALITY;
    const N = X.length;
    let rand = rng(7);
    let nTrees = 30;
    let maxDepth = 5;
    let forest = [];
    let timer = 0;

    const pts = splitPoints(N, 0.12);
    scene.add(pts.group);
    X.forEach(([a, c], i) => pts.set(i, a * S, c * S, 0.12, 1, Y[i] ? PALETTE[1] : PALETTE[0]));
    pts.commit();

    // Floor: the forest's averaged probability.
    const baseData = new Uint8Array(RES * RES * 4);
    const baseTex = new THREE.DataTexture(baseData, RES, RES);
    baseTex.colorSpace = THREE.SRGBColorSpace;
    baseTex.magFilter = THREE.LinearFilter;
    const base = new THREE.Mesh(new THREE.PlaneGeometry(HALF * 2, HALF * 2), new THREE.MeshBasicMaterial({ map: baseTex, transparent: true, opacity: 0.55, depthWrite: false }));
    base.position.z = -0.02;
    scene.add(base);

    const layers = [];
    const layerGroup = new THREE.Group();
    scene.add(layerGroup);

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
    tick('individual trees ↑', HALF + 1.6, HALF, 2.5, '#ffd166', 0.36);

    const pipe = ui.pipeline([
      ['Prepare data', `X = median age, log cases/M · Y = high (≥ ${Math.round(MEDIAN_DEATHS)} deaths/M) or low`],
      ['Split', `${TRAIN_IDX.length} train / ${TEST_IDX.length} test countries`],
      ['Train & optimise', 'Grow each tree greedily (Gini) on a bootstrap sample with random features'],
      ['Evaluate', ''],
    ]);
    const treeStat = ui.stat('Trees in forest');
    const singleStat = ui.stat('Single tree test acc.');
    const accStat = ui.stat('Forest accuracy train / test', { wide: true });
    const pickStat = ui.stat('Selected country', { wide: true });
    pickStat('Click a point');

    const voteAt = (x) => (forest.length ? forest.reduce((s, t) => s + predictTree(t, x), 0) / forest.length : 0.5);
    const accOn = (f, idx) => idx.filter((i) => (f(X[i]) >= 0.5 ? 1 : 0) === Y[i]).length / idx.length;

    const toZ = (i, res) => (-HALF + ((i + 0.5) / res) * HALF * 2) / S;
    function paint(data, res, f, alpha = 255) {
      const col = new THREE.Color();
      for (let j = 0; j < res; j++) {
        for (let i = 0; i < res; i++) {
          col.copy(LOW).lerp(HIGH, f([toZ(i, res), toZ(j, res)]));
          const o = (j * res + i) * 4;
          data[o] = col.r * 255;
          data[o + 1] = col.g * 255;
          data[o + 2] = col.b * 255;
          data[o + 3] = alpha;
        }
      }
    }

    function addLayer(tree) {
      const data = new Uint8Array(LAYER_RES * LAYER_RES * 4);
      paint(data, LAYER_RES, (x) => predictTree(tree, x));
      const tex = new THREE.DataTexture(data, LAYER_RES, LAYER_RES);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.needsUpdate = true;
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(HALF * 2 * 0.55, HALF * 2 * 0.55), new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }));
      mesh.position.set(0, 0, 1.2 + MAX_LAYERS * 0.55);
      layerGroup.add(mesh);
      layers.push(mesh);
      if (layers.length > MAX_LAYERS) {
        const old = layers.shift();
        layerGroup.remove(old);
        old.geometry.dispose();
        old.material.map.dispose();
        old.material.dispose();
      }
    }

    function refresh() {
      paint(baseData, RES, voteAt, 255);
      baseTex.needsUpdate = true;
      const tr = accOn(voteAt, TRAIN_IDX);
      const te = accOn(voteAt, TEST_IDX);
      const single = forest[0] ? accOn((x) => predictTree(forest[0], x), TEST_IDX) : 0;
      treeStat(`${forest.length} / ${nTrees}`);
      singleStat(forest[0] ? pct(single) : '–');
      accStat(`${pct(tr)} / ${pct(te)}`);
      pipe(3, `Forest test accuracy ${pct(te)} vs ${pct(single)} for one tree (train ${pct(tr)})`);
    }

    function growTree() {
      const n = TRAIN_IDX.length;
      const boot = TRAIN_IDX.map(() => TRAIN_IDX[Math.floor(rand() * n)]);
      const tree = fitClassTree(X, Y, boot, {
        maxDepth,
        minLeaf: 1,
        rand,
        features: (r) => [Math.floor(r() * 2)], // one random feature per split
      });
      forest.push(tree);
      addLayer(tree);
      refresh();
    }

    function regrow() {
      rand = rng(Math.floor(Math.random() * 1e9));
      forest = [];
      while (layers.length) {
        const old = layers.pop();
        layerGroup.remove(old);
        old.geometry.dispose();
        old.material.map.dispose();
        old.material.dispose();
      }
      timer = 0;
      refresh();
    }

    ui.section('Train');
    ui.slider({
      label: 'Number of trees', min: 1, max: 80, step: 1, value: nTrees,
      onInput: (v) => {
        nTrees = v;
        if (forest.length > v) {
          forest = forest.slice(0, v);
          refresh();
        }
      },
    });
    ui.slider({ label: 'Max depth per tree', min: 1, max: 8, step: 1, value: maxDepth, onInput: (v) => { maxDepth = v; regrow(); } });
    ui.button('Regrow forest', regrow, { primary: true });
    ui.html(SPLIT_LEGEND);
    const use = predictor(ui, ['median_age', 'cases_pm']);

    onClick((ray) => {
      const i = pts.pick(ray);
      if (i < 0) return;
      const c = COUNTRIES[i];
      const votes = forest.filter((t) => predictTree(t, X[i]) >= 0.5).length;
      pickStat(`${c.name}: ${votes} of ${forest.length} trees vote high · actual ${FIELDS.deaths_pm.fmt(c.deaths_pm)} deaths/M (${Y[i] ? 'high' : 'low'})`);
    });

    refresh();

    return {
      update(dt) {
        timer += dt;
        if (forest.length < nTrees && timer >= GROW_EVERY) {
          timer = 0;
          growTree();
        }
        const k = damp(5, dt);
        layers.forEach((m, idx) => {
          const targetZ = 2 + (layers.length - 1 - idx) * 0.6;
          m.position.z += (targetZ - m.position.z) * k;
          m.material.opacity += (0.32 - m.material.opacity) * k;
        });

        const q = use.values();
        const votes = forest.filter((t) => predictTree(t, q) >= 0.5).length;
        const share = forest.length ? votes / forest.length : 0;
        marker.position.set(q[0] * S, q[1] * S, 0.5);
        marker.rotation.z += dt * 2;
        use.show(`${share >= 0.5 ? 'HIGH' : 'LOW'} mortality: ${votes} of ${forest.length} trees vote high (${(share * 100).toFixed(0)}%)${use.actual((c) => `${FIELDS.deaths_pm.fmt(c.deaths_pm)} deaths/M (${c.deaths_pm >= MEDIAN_DEATHS ? 'high' : 'low'})`)}`);
      },
    };
  },
};
