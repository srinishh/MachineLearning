import * as THREE from 'three';
import { grid2D, textSprite, damp, PALETTE } from '../helpers.js';
import { COUNTRIES, FIELDS, HIGH_MORTALITY, MEDIAN_DEATHS, TRAIN_IDX, TEST_IDX, scaler } from '../data/covid.js';
import { splitPoints, SPLIT_LEGEND } from '../ml/split-points.js';
import { predictor } from '../ml/predictor.js';
import { pct } from '../ml/metrics.js';

const S = 1.8; // display units per standard deviation
const HALF = 6;
const HZ = 14; // display height per unit of density × prior
const RES = 80;
const SEG = 70;
const LOW = new THREE.Color(PALETTE[0]);
const HIGH = new THREE.Color(PALETTE[1]);
const gaussian = (x, mu, sd) => Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));

export default {
  id: 'naive-bayes',
  title: 'Naive Bayes',
  category: 'Supervised Learning',
  camera: { position: [0, -16, 12], target: [0, 0, 1.2] },
  formula: 'P(class | x) ∝ P(class) · Πⱼ P(xⱼ | class)<br>Gaussian:  P(xⱼ | class) = N(xⱼ; μⱼ,class, σⱼ,class)',
  goal: 'A <strong>probability for each class</strong>, computed with Bayes\' rule from simple per-class statistics: a mean and spread for each feature, plus how common each class is. The trained model is just those few numbers.',
  usage: 'For a new country, multiply the class prior by how typical each of its feature values is for that class, do the same for the other class, and normalise. The bigger share wins. “Use the trained model” shows this arithmetic step by step.',
  hint: 'Hills = how typical each location is for low (cyan) and high (pink) countries · Floor colour = P(high | x)',
  info: `
    <h3>What it is</h3>
    <p>Naive Bayes is a <strong>probabilistic</strong> classifier built on <strong>Bayes' rule</strong>. For each class it learns what
    typical examples look like, then asks: <em>which class makes this new example most likely?</em></p>
    <h3>Why “naive”?</h3>
    <p>It assumes the features are <strong>independent</strong> within each class, so it can multiply their probabilities. That is rarely
    exactly true (older countries also tend to report more cases), but the shortcut makes the model tiny, fast and surprisingly
    accurate. The curves on the back walls are the one-feature distributions, and each hill is simply their product.</p>
    <h3>On the COVID data</h3>
    <p>For <span style="color:#4cc9f0">low-</span> and <span style="color:#f72585">high-mortality</span> training countries it fits a bell curve
    to <strong>median age</strong> and to <strong>cases per million</strong> (log). The two hills show <em>prior × likelihood</em> for each
    class over the plane. Where the pink hill is taller, the model predicts high mortality. The floor colour is the resulting
    probability, and the boundary is curved even though each ingredient is a simple bell curve.</p>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>Naive Bayes powered the first generation of <strong>email spam filters</strong> by multiplying the spam-probabilities of individual
      words. It is still used for fast text classification, such as sorting COVID news, public-health tweets or patient messages by topic,
      and for quick medical screening scores where the inputs are roughly independent.</p>
    </div>
  `,
  create({ scene, ui, onClick }) {
    scene.add(grid2D(HALF * 2, 12));
    const age = scaler('median_age');
    const cases = scaler('cases_pm');
    const X = COUNTRIES.map((_, i) => [age.z[i], cases.z[i]]);
    const Y = HIGH_MORTALITY;
    const N = X.length;

    // ---------- Fit (closed form: counts, means, standard deviations) ----------
    const model = [0, 1].map((c) => {
      const rows = TRAIN_IDX.filter((i) => Y[i] === c);
      const mu = [0, 1].map((f) => rows.reduce((s, i) => s + X[i][f], 0) / rows.length);
      const sd = [0, 1].map((f) => Math.sqrt(rows.reduce((s, i) => s + (X[i][f] - mu[f]) ** 2, 0) / rows.length));
      return { prior: rows.length / TRAIN_IDX.length, mu, sd, n: rows.length };
    });
    const joint = (c, x) => model[c].prior * gaussian(x[0], model[c].mu[0], model[c].sd[0]) * gaussian(x[1], model[c].mu[1], model[c].sd[1]);
    const posterior = (x) => {
      const a = joint(0, x);
      const b = joint(1, x);
      return a + b > 0 ? b / (a + b) : 0.5;
    };
    const accOn = (idx) => idx.filter((i) => (posterior(X[i]) >= 0.5 ? 1 : 0) === Y[i]).length / idx.length;

    const pts = splitPoints(N, 0.12);
    scene.add(pts.group);
    X.forEach(([a, c], i) => pts.set(i, a * S, c * S, 0.1, 1, Y[i] ? PALETTE[1] : PALETTE[0]));
    pts.commit();

    // Floor: posterior P(high | x).
    const data = new Uint8Array(RES * RES * 4);
    const col = new THREE.Color();
    for (let j = 0; j < RES; j++) {
      for (let i = 0; i < RES; i++) {
        const x = [(-HALF + ((i + 0.5) / RES) * HALF * 2) / S, (-HALF + ((j + 0.5) / RES) * HALF * 2) / S];
        col.copy(LOW).lerp(HIGH, posterior(x));
        const o = (j * RES + i) * 4;
        data.set([col.r * 255, col.g * 255, col.b * 255, 255], o);
      }
    }
    const tex = new THREE.DataTexture(data, RES, RES);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.magFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(HALF * 2, HALF * 2), new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.45, depthWrite: false }));
    floor.position.z = -0.02;
    scene.add(floor);

    // Hills: prior × likelihood for each class.
    const hills = [0, 1].map((c) => {
      const geo = new THREE.PlaneGeometry(HALF * 2, HALF * 2, SEG, SEG);
      const p = geo.attributes.position;
      for (let v = 0; v < p.count; v++) p.setZ(v, joint(c, [p.getX(v) / S, p.getY(v) / S]) * HZ);
      geo.computeVertexNormals();
      const color = c ? PALETTE[1] : PALETTE[0];
      const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.25, transparent: true, opacity: 0.45, side: THREE.DoubleSide, depthWrite: false }));
      const wire = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color, wireframe: true, transparent: true, opacity: 0.12 }));
      const g = new THREE.Group();
      g.add(mesh, wire);
      g.scale.z = 0.001;
      scene.add(g);
      return g;
    });

    // 1-D class-conditional curves on the back walls: the "naive" ingredients.
    const wallCurves = new THREE.Group();
    scene.add(wallCurves);
    for (const c of [0, 1]) {
      const color = c ? PALETTE[1] : PALETTE[0];
      const ageCurve = [];
      const caseCurve = [];
      for (let t = -HALF; t <= HALF; t += 0.1) {
        ageCurve.push(new THREE.Vector3(t, HALF, gaussian(t / S, model[c].mu[0], model[c].sd[0]) * 3));
        caseCurve.push(new THREE.Vector3(-HALF, t, gaussian(t / S, model[c].mu[1], model[c].sd[1]) * 3));
      }
      wallCurves.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(ageCurve), new THREE.LineBasicMaterial({ color })));
      wallCurves.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(caseCurve), new THREE.LineBasicMaterial({ color })));
    }

    const marker = new THREE.Mesh(new THREE.OctahedronGeometry(0.28), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.8 }));
    const stemGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 0, 1)]);
    const stem = new THREE.Line(stemGeo, new THREE.LineBasicMaterial({ color: 0xffffff }));
    scene.add(marker, stem);

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
    tick('P(age | class)', 0, HALF + 0.3, 1.9, '#94a3b8', 0.32);
    tick('P(cases | class)', -HALF - 0.3, 0, 1.9, '#94a3b8', 0.32);

    const pipe = ui.pipeline([
      ['Prepare data', `X = median age, log cases/M · Y = high (≥ ${Math.round(MEDIAN_DEATHS)} deaths/M) or low`],
      ['Split', `${TRAIN_IDX.length} train / ${TEST_IDX.length} test countries`],
      ['Train & optimise', 'No iterations: count classes, compute each feature\'s mean & spread per class'],
      ['Evaluate', ''],
    ]);
    const tr = accOn(TRAIN_IDX);
    const te = accOn(TEST_IDX);
    pipe(3, `Test accuracy ${pct(te)} (train ${pct(tr)})`);
    ui.stat('Accuracy train / test', { wide: true })(`${pct(tr)} / ${pct(te)}`);
    ui.stat('Learned parameters', { wide: true })(model.map((m, c) => `${c ? 'High' : 'Low'}: prior ${pct(m.prior)}, mean age ${age.format(age.fromZ(m.mu[0]))}, mean cases ${cases.format(cases.fromZ(m.mu[1]))}/M`).join(' · '));
    const pickStat = ui.stat('Selected country', { wide: true });
    pickStat('Click a point');

    let showHills = true;
    ui.section('Display');
    ui.toggle({ label: 'Show class hills (prior × likelihood)', value: true, onChange: (v) => { showHills = v; } });
    ui.toggle({ label: 'Show per-feature curves on the walls', value: true, onChange: (v) => { wallCurves.visible = v; } });
    ui.html(SPLIT_LEGEND);
    const use = predictor(ui, ['median_age', 'cases_pm']);
    const mathBox = ui.html('');

    onClick((ray) => {
      const i = pts.pick(ray);
      if (i < 0) return;
      const c = COUNTRIES[i];
      pickStat(`${c.name}: P(high) = ${(posterior(X[i]) * 100).toFixed(0)}% · actual ${FIELDS.deaths_pm.fmt(c.deaths_pm)} deaths/M (${Y[i] ? 'high' : 'low'})`);
    });

    return {
      update(dt) {
        const k = damp(2.5, dt);
        hills.forEach((h) => {
          h.scale.z += ((showHills ? 1 : 0.001) - h.scale.z) * k;
          h.visible = h.scale.z > 0.01;
        });

        const q = use.values();
        const p = posterior(q);
        const top = Math.max(joint(0, q), joint(1, q)) * HZ;
        marker.position.set(q[0] * S, q[1] * S, top + 0.35);
        marker.rotation.z += dt * 2;
        stem.position.set(q[0] * S, q[1] * S, 0);
        stem.scale.z = top + 0.35;
        const parts = [0, 1].map((c) => {
          const m = model[c];
          const la = gaussian(q[0], m.mu[0], m.sd[0]);
          const lc = gaussian(q[1], m.mu[1], m.sd[1]);
          return `<li><strong style="color:${c ? '#f72585' : '#4cc9f0'}">${c ? 'High' : 'Low'}</strong>: prior ${m.prior.toFixed(2)} × P(age) ${la.toFixed(3)} × P(cases) ${lc.toFixed(3)} = ${(m.prior * la * lc).toFixed(4)}</li>`;
        });
        const html = `<div class="field-head"><span>Bayes' rule, step by step</span></div><ul class="profiles">${parts.join('')}<li>Normalise → P(high) = ${(p * 100).toFixed(0)}%</li></ul>`;
        if (mathBox.innerHTML !== html) mathBox.innerHTML = html;
        use.show(`P(high mortality) = ${(p * 100).toFixed(0)}% → ${p >= 0.5 ? 'HIGH' : 'LOW'}${use.actual((c) => `${FIELDS.deaths_pm.fmt(c.deaths_pm)} deaths/M (${c.deaths_pm >= MEDIAN_DEATHS ? 'high' : 'low'})`)}`);
      },
    };
  },
};
