import * as THREE from 'three';
import {
  grid2D, sphereCloud, setInstance, setInstanceColor, commitInstances,
  unitCylinder, placeCylinder, textSprite, clearGroup, rng, PALETTE,
} from '../helpers.js';
import { COUNTRIES, FIELDS, scaler } from '../data/covid.js';

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
  category: 'Foundations',
  camera: { position: [0, -4, 17], target: [0, 0, 0] },
  formula: 'deaths/M = w · feature + b<br>Loss = ¹⁄ₙ Σ (ŷᵢ − yᵢ)²<br>w ← w − η·∂Loss/∂w',
  hint: 'Each dot is a country · Pink lines are the errors the line is trying to shrink · Click a dot to name it',
  info: `
    <h3>What it is</h3>
    <p>Linear regression predicts a <strong>number</strong> by drawing the best straight line through data. The model has
    two parameters: the slope <code>w</code> and the intercept <code>b</code>.</p>
    <h3>On the COVID data</h3>
    <p>Each dot is a country. The model predicts <strong>COVID deaths per million</strong> from one feature, by default the
    country's <strong>median age</strong>. The slope reads like a sentence: <em>“each extra year of median age goes with about
    N more deaths per million.”</em> <strong>R²</strong> tells you what share of the variation between countries the line explains.</p>
    <h3>How it learns</h3>
    <p>Each pink line is a <strong>residual</strong>, the gap between a country's real death rate and the line's prediction. Training
    squares and averages these to get the <strong>mean squared error</strong>, then uses <strong>gradient descent</strong> to nudge
    <code>w</code> and <code>b</code> downhill a little each step. (Training runs on standardised values; the stats are converted back
    to real units.)</p>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>Public-health planners use regression to estimate the expected burden of a disease from population structure, such as
      how many hospital beds or ICU staff a region with an older population might need. Insurers and actuaries use the same
      idea to set expected mortality.</p>
    </div>
    <h3>Try this</h3>
    <ul>
      <li>Switch the feature to <em>% vaccinated</em>. R² collapses, because vaccination alone explains little of the cross-country difference (it was also tangled up with wealth and reporting).</li>
      <li>Push the learning rate above about <code>1.0</code> and watch training <strong>diverge</strong>.</li>
      <li>Click the dots far above the line (Peru, Bulgaria) to see the countries the model under-predicts most.</li>
    </ul>
  `,
  create({ scene, ui, onClick }) {
    scene.add(grid2D(14, 14));
    const rand = rng(11);
    const ys = scaler('deaths_pm', { log: false });
    let xs;

    const pts = sphereCloud(N, 0.13);
    scene.add(pts);
    const line = unitCylinder(PALETTE[2], { emissive: 0.6 });
    scene.add(line);
    const ticks = new THREE.Group();
    scene.add(ticks);

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

    const statSlope = ui.stat('Slope (real units)', { wide: true });
    const statLoss = ui.stat('Loss (MSE, std. units)');
    const statR2 = ui.stat('R² (variance explained)');
    const statEpoch = ui.stat('Epoch');
    const statStatus = ui.stat('Status');
    const statPick = ui.stat('Selected country', { wide: true });
    statPick('Click a dot');

    const loss = () => xs.z.reduce((s, x, i) => s + (w * x + b - ys.z[i]) ** 2, 0) / N;

    function restart() {
      w = (rand() * 2 - 1) * 1.5;
      b = (rand() * 2 - 1) * 1.5;
      epoch = 0;
      status = 'Training…';
    }

    function setFeature(key) {
      feature = key;
      xs = scaler(key, { log: false });
      COUNTRIES.forEach((c, i) => {
        setInstance(pts, i, xs.z[i] * SX, ys.z[i] * SY, 0);
        setInstanceColor(pts, i, PALETTE[0]);
      });
      commitInstances(pts);
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
      restart();
    }

    function step() {
      let gw = 0;
      let gb = 0;
      for (let i = 0; i < N; i++) {
        const e = w * xs.z[i] + b - ys.z[i];
        gw += e * xs.z[i];
        gb += e;
      }
      w -= (lr * 2 * gw) / N;
      b -= (lr * 2 * gb) / N;
      epoch++;
      if (!Number.isFinite(w) || Math.abs(w) > 200) {
        restart();
        status = 'Diverged! η too high';
      } else if (Math.abs(gw) / N < 1e-5 && Math.abs(gb) / N < 1e-5) {
        status = 'Converged ✓';
      }
    }

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

    onClick((ray) => {
      const hit = ray.intersectObject(pts)[0];
      if (hit?.instanceId === undefined) return;
      const i = hit.instanceId;
      const c = COUNTRIES[i];
      const pred = ys.fromZ(w * xs.z[i] + b);
      statPick(`${c.name}: ${FIELDS[feature].fmt(c[feature])} → actual ${FIELDS.deaths_pm.fmt(c.deaths_pm)}, predicted ${FIELDS.deaths_pm.fmt(Math.max(0, pred))} deaths/M`);
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
          resPos.set([xs.z[i] * SX, ys.z[i] * SY, 0, xs.z[i] * SX, yAt(xs.z[i] * SX), 0], i * 6);
        }
        resGeo.attributes.position.needsUpdate = true;
        const L = loss();
        const slopeReal = (w * ys.sd) / xs.sd;
        statSlope(`${slopeReal >= 0 ? '+' : ''}${slopeReal.toFixed(1)} deaths/M per ${PER[feature]}`);
        statLoss(L.toFixed(4));
        statR2(`${Math.max(0, (1 - L) * 100).toFixed(1)}%`);
        statEpoch(epoch);
        statStatus(status);
      },
    };
  },
};
