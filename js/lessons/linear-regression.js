import * as THREE from 'three';
import { grid2D, unitCylinder, placeCylinder, textSprite, clearGroup, rng, PALETTE } from '../helpers.js';
import { COUNTRIES, FIELDS, TRAIN_IDX, TEST_IDX, scaler } from '../data/covid.js';
import { splitPoints, SPLIT_LEGEND } from '../ml/split-points.js';
import { r2, pct } from '../ml/metrics.js';

const N = COUNTRIES.length;
const SX = 2.2; // display units per standard deviation of x
const SY = 1.5; // display units per standard deviation of y
const PREDICTORS = ['median_age', 'aged_65', 'life_exp', 'vax'];
const PER = {
  median_age: 'year of median age',
  aged_65: 'point of population aged 65+',
  life_exp: 'year of life expectancy',
  vax: 'point of vaccination coverage',
};

export default {
  id: 'linear-regression',
  title: 'Linear Regression',
  category: 'Supervised Learning',
  camera: { position: [0, -4, 17], target: [0, 0, 0] },
  formula: 'deaths/M = w · feature + b<br>Loss = ¹⁄ₙ Σ (ŷᵢ − yᵢ)²<br>w ← w − η·∂Loss/∂w',
  goal: 'A <strong>number</strong>: the expected COVID deaths per million for a country, given one feature such as its median age. The trained model is just a straight line, two numbers <code>w</code> and <code>b</code>.',
  usage: 'Plug a new value into the line: <em>deaths/M = w × age + b</em>. Use the “Use the trained model” sliders to set a median age and read off the prediction. The slope also tells you how much the outcome changes per unit of the feature.',
  hint: 'Spheres = training countries · Cubes = test countries · Pink lines are errors · Click a point to name it',
  info: `
    <h3>What it is</h3>
    <p>Linear regression predicts a <strong>number</strong> by drawing the best straight line through data. The model has
    two parameters: the slope <code>w</code> and the intercept <code>b</code>.</p>
    <h3>On the COVID data</h3>
    <p>Each point is a country. The model predicts <strong>COVID deaths per million</strong> from one feature, by default the
    country's <strong>median age</strong>. The slope reads like a sentence: <em>“each extra year of median age goes with about
    N more deaths per million.”</em> <strong>R²</strong> is the share of the variation between countries that the line explains,
    reported separately for training and test countries.</p>
    <h3>How it learns</h3>
    <p>Each pink line is a <strong>residual</strong>, the gap between a training country's real death rate and the prediction.
    Training squares and averages them to get the <strong>mean squared error</strong>, then uses <strong>gradient descent</strong> to
    nudge <code>w</code> and <code>b</code> downhill. Only training countries (spheres) are used, and the test countries (cubes) are
    scored at the end.</p>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>Public-health planners use regression to estimate the expected burden of a disease from population structure, such as how many
      hospital beds or ICU staff a region with an older population might need. Insurers and actuaries use the same idea to set expected mortality.</p>
    </div>
    <h3>Try this</h3>
    <ul>
      <li>Switch the feature to <em>% vaccinated</em>. R² collapses, because vaccination alone explains little of the cross-country difference (it was tangled up with wealth and reporting).</li>
      <li>Push the learning rate above about <code>1.0</code> and watch training <strong>diverge</strong>.</li>
    </ul>
  `,
  create({ scene, ui, onClick }) {
    scene.add(grid2D(14, 14));
    const rand = rng(11);
    const ys = scaler('deaths_pm', { log: false });
    let xs;

    const pts = splitPoints(N, 0.13);
    scene.add(pts.group);
    const line = unitCylinder(PALETTE[2], { emissive: 0.6 });
    scene.add(line);
    const ticks = new THREE.Group();
    scene.add(ticks);
    const marker = new THREE.Mesh(new THREE.OctahedronGeometry(0.3), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.8 }));
    scene.add(marker);

    const resPos = new Float32Array(N * 6);
    const resGeo = new THREE.BufferGeometry();
    resGeo.setAttribute('position', new THREE.BufferAttribute(resPos, 3));
    const residuals = new THREE.LineSegments(resGeo, new THREE.LineBasicMaterial({ color: PALETTE[1], transparent: true, opacity: 0.55 }));
    residuals.frustumCulled = false;
    scene.add(residuals);

    let w = 0;
    let b = 0;
    let epoch = 0;
    let lr = 0.05;
    let acc = 0;
    let status = 'Training…';
    let feature = 'median_age';
    let query = 35;

    const pipe = ui.pipeline([
      ['Prepare data', ''],
      ['Split', `${TRAIN_IDX.length} train / ${TEST_IDX.length} test countries`],
      ['Train & optimise', 'Gradient descent on mean squared error'],
      ['Evaluate', ''],
    ]);
    const statSlope = ui.stat('Slope (real units)', { wide: true });
    const statR2 = ui.stat('R² train / test', { wide: true });
    const statEpoch = ui.stat('Epoch');
    const statStatus = ui.stat('Status');
    const statPick = ui.stat('Selected country', { wide: true });
    statPick('Click a point');

    const loss = () => TRAIN_IDX.reduce((s, i) => s + (w * xs.z[i] + b - ys.z[i]) ** 2, 0) / TRAIN_IDX.length;
    const predictZ = (i) => w * xs.z[i] + b;

    function restart() {
      w = (rand() * 2 - 1) * 1.5;
      b = (rand() * 2 - 1) * 1.5;
      epoch = 0;
      status = 'Training…';
    }

    function setFeature(key) {
      feature = key;
      xs = scaler(key, { log: false });
      COUNTRIES.forEach((c, i) => pts.set(i, xs.z[i] * SX, ys.z[i] * SY, 0, 1, PALETTE[0]));
      pts.commit();
      clearGroup(ticks);
      for (const z of [-2, -1, 0, 1, 2]) {
        const t = textSprite(xs.format(xs.fromZ(z)), { height: 0.3, color: '#94a3b8' });
        t.position.set(z * SX, -6.6, 0);
        ticks.add(t);
      }
      for (const d of [0, 2000, 4000, 6000]) {
        const t = textSprite(d.toLocaleString('en-US'), { height: 0.3, color: '#94a3b8' });
        t.position.set(-7.6, ys.toZ(d) * SY, 0);
        ticks.add(t);
      }
      const xl = textSprite(`${FIELDS[key].label} →`, { height: 0.36, color: '#e2e8f0' });
      xl.position.set(4.6, -7.3, 0);
      const yl = textSprite('Deaths per million ↑', { height: 0.36, color: '#e2e8f0' });
      yl.position.set(-5, 7.4, 0);
      ticks.add(xl, yl);
      const vals = COUNTRIES.map((c) => c[key]);
      const lo = Math.floor(Math.min(...vals));
      const hi = Math.ceil(Math.max(...vals));
      query = Math.round((lo + hi) / 2);
      querySlider.setRange(lo, hi, 0.5);
      querySlider.set(query);
      pipe(0, `X = ${FIELDS[key].label.toLowerCase()}; Y = deaths per million`);
      restart();
    }

    function step() {
      let gw = 0;
      let gb = 0;
      for (const i of TRAIN_IDX) {
        const e = w * xs.z[i] + b - ys.z[i];
        gw += e * xs.z[i];
        gb += e;
      }
      const n = TRAIN_IDX.length;
      w -= (lr * 2 * gw) / n;
      b -= (lr * 2 * gb) / n;
      epoch++;
      if (!Number.isFinite(w) || Math.abs(w) > 200) {
        restart();
        status = 'Diverged! η too high';
      } else if (Math.abs(gw) / n < 1e-5 && Math.abs(gb) / n < 1e-5) {
        status = 'Converged ✓';
      }
    }

    ui.section('Train');
    ui.select({
      label: 'Feature (x-axis)',
      options: PREDICTORS.map((k) => ({ value: k, label: FIELDS[k].label })),
      value: feature,
      onChange: setFeature,
    });
    ui.slider({
      label: 'Learning rate η', min: 0.005, max: 1.2, step: 0.005, value: lr,
      format: (v) => v.toFixed(3), onInput: (v) => { lr = v; },
    });
    ui.button('Restart training', restart, { primary: true });
    ui.html(SPLIT_LEGEND);

    ui.section('Use the trained model');
    const querySlider = ui.slider({
      label: 'Feature value for a new country', min: 0, max: 100, step: 0.5, value: query,
      format: (v) => FIELDS[feature].fmt(v),
      onInput: (v) => { query = v; },
    });
    const statPred = ui.stat('Model prediction', { wide: true });

    onClick((ray) => {
      const i = pts.pick(ray);
      if (i < 0) return;
      const c = COUNTRIES[i];
      statPick(`${c.name}: ${FIELDS[feature].fmt(c[feature])} → actual ${FIELDS.deaths_pm.fmt(c.deaths_pm)}, predicted ${FIELDS.deaths_pm.fmt(Math.max(0, ys.fromZ(predictZ(i))))} deaths/M`);
    });

    setFeature(feature);
    const a = new THREE.Vector3();
    const c = new THREE.Vector3();

    return {
      update(dt) {
        acc += dt;
        while (acc >= 1 / 20) {
          acc -= 1 / 20;
          step();
        }
        // Line drawn in display space: y_disp = SY * (w * x_disp / SX + b)
        const yAt = (xd) => SY * (w * (xd / SX) + b);
        a.set(-7, yAt(-7), 0);
        c.set(7, yAt(7), 0);
        placeCylinder(line, a, c, 0.05);
        for (let i = 0; i < N; i++) {
          const train = !TEST_IDX.includes(i);
          const x = xs.z[i] * SX;
          resPos.set(train ? [x, ys.z[i] * SY, 0, x, yAt(x), 0] : [0, 0, 0, 0, 0, 0], i * 6);
        }
        resGeo.attributes.position.needsUpdate = true;

        const qv = query;
        const qz = xs.toZ(qv);
        marker.position.set(qz * SX, yAt(qz * SX), 0.2);
        marker.rotation.y += dt * 2;
        statPred(`${FIELDS[feature].fmt(qv)} → ${FIELDS.deaths_pm.fmt(Math.max(0, ys.fromZ(w * qz + b)))} deaths per million`);

        const slopeReal = (w * ys.sd) / xs.sd;
        const trR2 = r2(predictZ, ys.z, TRAIN_IDX);
        const teR2 = r2(predictZ, ys.z, TEST_IDX);
        statSlope(`${slopeReal >= 0 ? '+' : ''}${slopeReal.toFixed(1)} deaths/M per ${PER[feature]}`);
        statR2(`${pct(Math.max(0, trR2))} / ${pct(Math.max(0, teR2))}`);
        statEpoch(epoch);
        statStatus(status);
        pipe(3, `Test R² ${pct(Math.max(0, teR2))} (train ${pct(Math.max(0, trR2))}), training loss ${loss().toFixed(3)}`);
      },
    };
  },
};
