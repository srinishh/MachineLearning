import * as THREE from 'three';
import { grid2D, textSprite, PALETTE, damp } from '../helpers.js';
import { COUNTRIES, FIELDS, HIGH_MORTALITY, MEDIAN_DEATHS, TRAIN_IDX, TEST_IDX, scaler } from '../data/covid.js';
import { splitPoints, SPLIT_LEGEND } from '../ml/split-points.js';
import { predictor } from '../ml/predictor.js';
import { pct } from '../ml/metrics.js';

const RES = 110;
const HALF = 6;
const S = 2.1; // display units per standard deviation
const NAMES = ['low', 'high'];
const CLASS_COLORS = [PALETTE[0], PALETTE[1]];

export default {
  id: 'knn',
  title: 'k-Nearest Neighbors',
  category: 'Supervised Learning',
  camera: { position: [0, -5, 17], target: [0, 0, 0] },
  formula: 'ŷ(x) = majority vote of the k training countries closest to x<br>distance = √(Δage² + Δlog cases²)   (standardised)',
  goal: 'A <strong>class label</strong> (high or low mortality) for a new country, decided by a vote of the k most similar countries it has seen. There are no learned weights: the training data <em>is</em> the model.',
  usage: 'Measure the new country\'s distance to every stored training country, take the k closest, and return the majority label. The vote share doubles as a confidence (e.g. 4 of 5 neighbours = 80%). Set values in “Use the trained model” or click the plane to place a country.',
  hint: 'Click anywhere to place a hypothetical country · Spheres = training memory, cubes = test countries',
  info: `
    <h3>What it is</h3>
    <p>k-NN is the simplest classifier there is. There's no training step. To label something new, find the <strong>k</strong> most
    <strong>similar</strong> known examples and take a <strong>majority vote</strong>.</p>
    <h3>On the COVID data</h3>
    <p>Countries are placed by <strong>median age</strong> (x) and <strong>cases per million</strong> (y, log), both standardised so
    neither dominates the distance. The ${TRAIN_IDX.length} training countries form the model's memory. Each test country (cube) is
    classified by its neighbours among them, which gives the test accuracy.</p>
    <ul>
      <li>The white-ringed sphere is a <strong>query</strong>: a hypothetical country. Lines connect it to its k nearest training countries.</li>
      <li>The tinted background is the <strong>decision map</strong>: the label k-NN would predict at every location.</li>
    </ul>
    <h3>Choosing k</h3>
    <ul>
      <li><strong>k = 1</strong>: islands form around single unusual countries, so training accuracy is perfect but the model overfits.</li>
      <li><strong>Large k</strong>: smooth regions, but small groups get outvoted, which is underfitting.</li>
    </ul>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>“Find the most similar cases” is everywhere. Health agencies compared their region with <strong>similar regions further
      along the epidemic</strong> to anticipate what was coming. Hospitals use patient-similarity search, and recommender systems
      suggest products bought by similar customers. k-NN also <strong>fills in missing values</strong>: a country with no
      hospital-bed data can borrow the average of its nearest neighbours.</p>
    </div>
  `,
  create({ scene, ui, onClick }) {
    scene.add(grid2D(HALF * 2, 12));
    const age = scaler('median_age');
    const cases = scaler('cases_pm');
    const X = COUNTRIES.map((_, i) => [age.z[i] * S, cases.z[i] * S]);
    const Y = HIGH_MORTALITY;
    const N = X.length;
    let k = 5;
    let auto = true;
    const query = [0, 0];
    const shownQuery = new THREE.Vector3();

    const pts = splitPoints(N, 0.13);
    scene.add(pts.group);

    const map = new Uint8Array(RES * RES * 4);
    const tex = new THREE.DataTexture(map, RES, RES);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.magFilter = THREE.LinearFilter;
    tex.minFilter = THREE.LinearFilter;
    const bg = new THREE.Mesh(new THREE.PlaneGeometry(HALF * 2, HALF * 2), new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.38, depthWrite: false }));
    bg.position.z = -0.02;
    scene.add(bg);

    const tick = (text, x, y, color = '#94a3b8', height = 0.3) => {
      const s = textSprite(text, { height, color });
      s.position.set(x, y, 0);
      scene.add(s);
    };
    for (const zv of [-2, 0, 2]) {
      tick(age.format(age.fromZ(zv)), zv * S, -HALF - 0.45);
      tick(cases.format(cases.fromZ(zv)), -HALF - 1, zv * S);
    }
    tick('Median age →', 3.5, -HALF - 1.1, '#e2e8f0', 0.36);
    tick('Cases per million (log) ↑', -3.4, HALF + 0.5, '#e2e8f0', 0.36);

    const qMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.5 });
    const qMesh = new THREE.Mesh(new THREE.SphereGeometry(0.26, 24, 18), qMat);
    const qRing = new THREE.Mesh(new THREE.TorusGeometry(0.36, 0.04, 8, 40), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    scene.add(qMesh, qRing);

    const linePos = new Float32Array(15 * 6);
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.BufferAttribute(linePos, 3));
    const lines = new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7 }));
    lines.frustumCulled = false;
    scene.add(lines);

    const circlePts = [];
    for (let i = 0; i <= 96; i++) circlePts.push(new THREE.Vector3(Math.cos((i / 96) * Math.PI * 2), Math.sin((i / 96) * Math.PI * 2), 0));
    const circle = new THREE.Line(new THREE.BufferGeometry().setFromPoints(circlePts), new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.04, gapSize: 0.04, transparent: true, opacity: 0.6 }));
    circle.computeLineDistances();
    scene.add(circle);

    const pipe = ui.pipeline([
      ['Prepare data', `X = median age, log cases/M (standardised) · Y = high (≥ ${Math.round(MEDIAN_DEATHS)} deaths/M) or low`],
      ['Split', `${TRAIN_IDX.length} train / ${TEST_IDX.length} test countries`],
      ['Train & optimise', 'Nothing to optimise: just store the training countries (“lazy learning”)'],
      ['Evaluate', ''],
    ]);
    const accStat = ui.stat('Accuracy train / test', { wide: true });
    const queryStat = ui.stat('Query country', { wide: true });
    const voteStat = ui.stat('Votes (low / high)');
    const predStat = ui.stat('Prediction');
    const nbStat = ui.stat('Nearest neighbours', { wide: true });

    // Neighbours among the training countries only (the model's memory).
    function neighbours(qx, qy, kk) {
      const d = TRAIN_IDX.map((i) => [(X[i][0] - qx) ** 2 + (X[i][1] - qy) ** 2, i]);
      d.sort((a, b) => a[0] - b[0]);
      return d.slice(0, kk);
    }

    function vote(nb) {
      const votes = [0, 0];
      // Tiny rank bonus breaks ties toward the nearest neighbours.
      nb.forEach(([, i], r) => { votes[Y[i]] += 1 + (nb.length - r) * 1e-6; });
      return { cls: votes[1] > votes[0] ? 1 : 0, votes: votes.map(Math.floor) };
    }

    const colors = CLASS_COLORS.map((c) => new THREE.Color(c));
    function refit() {
      for (let j = 0; j < RES; j++) {
        for (let i = 0; i < RES; i++) {
          const x = -HALF + ((i + 0.5) / RES) * HALF * 2;
          const y = -HALF + ((j + 0.5) / RES) * HALF * 2;
          const c = colors[vote(neighbours(x, y, k)).cls];
          const o = (j * RES + i) * 4;
          map[o] = c.r * 255;
          map[o + 1] = c.g * 255;
          map[o + 2] = c.b * 255;
          map[o + 3] = 255;
        }
      }
      tex.needsUpdate = true;
      const accOn = (idx) => idx.filter((i) => vote(neighbours(X[i][0], X[i][1], k)).cls === Y[i]).length / idx.length;
      const tr = accOn(TRAIN_IDX);
      const te = accOn(TEST_IDX);
      accStat(`${pct(tr)} / ${pct(te)}`);
      pipe(3, `Test accuracy ${pct(te)} (train ${pct(tr)}) with k = ${k}`);
    }

    X.forEach((p, i) => pts.set(i, p[0], p[1], 0, 1, CLASS_COLORS[Y[i]]));
    pts.commit();

    ui.section('Train');
    ui.slider({ label: 'k (neighbours)', min: 1, max: 15, step: 1, value: k, onInput: (v) => { k = v; refit(); } });
    const autoToggle = ui.toggle({ label: 'Move query automatically', value: true, onChange: (v) => { auto = v; } });
    ui.toggle({ label: 'Show decision map', value: true, onChange: (v) => { bg.visible = v; } });
    ui.html(SPLIT_LEGEND);
    const use = predictor(ui, ['median_age', 'cases_pm'], {
      onChange: () => {
        const [a, c] = use.values();
        query[0] = a * S;
        query[1] = c * S;
        auto = false;
        autoToggle.set(false);
      },
    });

    onClick((ray) => {
      const hit = ray.intersectObject(bg)[0];
      if (!hit) return;
      query[0] = hit.point.x;
      query[1] = hit.point.y;
      auto = false;
      autoToggle.set(false);
    });

    refit();
    let t = 0;
    const qColor = new THREE.Color();
    const goal = new THREE.Vector3();

    return {
      update(dt) {
        t += dt;
        if (auto) {
          query[0] = 4.6 * Math.sin(t * 0.31);
          query[1] = 3.8 * Math.sin(t * 0.47 + 1);
        }
        shownQuery.lerp(goal.set(query[0], query[1], 0.05), damp(8, dt) || 1);
        const qx = shownQuery.x;
        const qy = shownQuery.y;
        qMesh.position.copy(shownQuery);
        qRing.position.copy(shownQuery);

        const nb = neighbours(qx, qy, k);
        const { cls, votes } = vote(nb);
        const near = new Set(nb.map(([, i]) => i));
        X.forEach((p, i) => pts.set(i, p[0], p[1], 0, near.has(i) ? 1.5 : 1));
        pts.commit();

        linePos.fill(0);
        nb.forEach(([, i], r) => linePos.set([qx, qy, 0.02, X[i][0], X[i][1], 0.02], r * 6));
        lineGeo.setDrawRange(0, nb.length * 2);
        lineGeo.attributes.position.needsUpdate = true;

        const radius = Math.sqrt(nb[nb.length - 1][0]);
        circle.position.set(qx, qy, 0.01);
        circle.scale.setScalar(Math.max(radius, 0.01));

        qColor.set(CLASS_COLORS[cls]);
        qMat.color.copy(qColor);
        qMat.emissive.copy(qColor);
        queryStat(`median age ${FIELDS.median_age.fmt(age.fromZ(qx / S))} · ${FIELDS.cases_pm.fmt(cases.fromZ(qy / S))} cases/M`);
        predStat(NAMES[cls]);
        voteStat(votes.join(' / '));
        nbStat(nb.slice(0, 5).map(([, i]) => COUNTRIES[i].name).join(', ') + (nb.length > 5 ? '…' : ''));
        const conf = Math.max(...votes) / k;
        use.show(`${NAMES[cls].toUpperCase()} mortality (${votes[cls]} of ${k} neighbours, ${(conf * 100).toFixed(0)}%)${use.actual((c) => `${FIELDS.deaths_pm.fmt(c.deaths_pm)} deaths/M (${c.deaths_pm >= MEDIAN_DEATHS ? 'high' : 'low'})`)}`);
      },
    };
  },
};
