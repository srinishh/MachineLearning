import * as THREE from 'three';
import { rng, gauss, textSprite, clearGroup, placeCylinder, PALETTE, clamp } from '../helpers.js';
import { COUNTRIES, FIELDS, scaler } from '../data/covid.js';

const SIZES = [3, 5, 5, 2];
const DUR = 0.7;
const PHASES = [
  { type: 'fwd', l: 0 }, { type: 'fwd', l: 1 }, { type: 'fwd', l: 2 },
  { type: 'bwd', l: 2 }, { type: 'bwd', l: 1 }, { type: 'bwd', l: 0 },
  { type: 'update' },
];
const NAMES = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];
const IN_FEATS = ['median_age', 'gdp', 'vax'];
const IN_LABELS = ['Median age', 'GDP (log)', 'Vaccinated'];
const OUT_FEATS = ['cases_pm', 'deaths_pm'];
const OUT_LABELS = ['Cases/M', 'Deaths/M'];
const IN_SCALE = 2.5; // inputs: z / 2.5, clamped to [-1, 1]
const OUT_SCALE = 3; // targets: z / 3, clamped to [-0.9, 0.9] (inside tanh's range)
const POS = new THREE.Color(PALETTE[0]);
const NEG = new THREE.Color(PALETTE[1]);
const GRAD = new THREE.Color(PALETTE[5]);
const IDLE = new THREE.Color(0x334155);

export default {
  id: 'neural-network',
  title: 'Neural Networks & Backprop',
  category: 'Deep Learning',
  camera: { position: [-0.2, 0.3, 23.5], target: [-0.2, 0, 0] },
  formula: 'Forward:  aˡ = tanh(Wˡ·aˡ⁻¹ + bˡ)<br>Backward: δᴸ = (a − y)⊙(1 − a²);  δˡ = (Wˡ⁺¹)ᵀδˡ⁺¹ ⊙ (1 − (aˡ)²)<br>Update:  Wˡ ← Wˡ − η·δˡ(aˡ⁻¹)ᵀ',
  goal: '<strong>Two numbers at once</strong>: a country\'s predicted cases and deaths per million, produced by layers of weighted sums and tanh activations whose weights were learned by backpropagation.',
  usage: 'Feed the new country\'s standardised age, GDP and vaccination into the input layer and run one forward pass; the output neurons are the predictions (convert back from standardised log units). Press “Train 500 steps” first so the weights are trained.',
  hint: 'Blue/pink pulses = signals moving forward · Orange pulses = error flowing backward · One country per training step',
  info: `
    <h3>What it is</h3>
    <p>A neural network is layers of simple units (<strong>neurons</strong>). Each one computes a weighted sum of its inputs, adds a
    bias, and applies a non-linear <strong>activation function</strong> (here <code>tanh</code>). Stacking layers lets the network
    represent complex relationships that a single line can't.</p>
    <h3>On the COVID data</h3>
    <p>This small network takes three facts about a country (<strong>median age</strong>, <strong>GDP per capita</strong> and
    <strong>% vaccinated</strong>) and predicts two outcomes at once: <strong>cases per million</strong> and <strong>deaths per
    million</strong> (both log scale and standardised). Each cycle shows one country flowing through the network, then
    <strong>backpropagation</strong> nudges every weight to reduce that country's error. This is <strong>stochastic gradient
    descent</strong>: one example at a time, many times over.</p>
    <h3>Reading the animation</h3>
    <ul>
      <li>Cyan edges are positive weights, pink are negative, and thicker means stronger. Neuron brightness shows activation.</li>
      <li>Orange pulses carry the error backwards. The chain rule tells each weight how much it contributed.</li>
      <li>Watch <strong>dataset loss</strong> fall over many steps. It is the error averaged over all countries, not just the current one.</li>
    </ul>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>Hospitals trained neural networks on vital signs and lab results to predict which COVID patients would need ICU care or
      oxygen in the next 24–48 hours, giving staff time to prepare beds and ventilators. Similar networks forecast regional hospital
      demand from case counts, mobility and vaccination data.</p>
    </div>
  `,
  create({ scene, ui }) {
    const rand = rng(81);
    const nodePos = SIZES.map((n, l) => Array.from({ length: n }, (_, i) => new THREE.Vector3(-6 + l * 4, ((n - 1) / 2 - i) * 1.5, 0)));

    const inScalers = IN_FEATS.map((f) => scaler(f));
    const outScalers = OUT_FEATS.map((f) => scaler(f, { log: true }));
    const INPUTS = COUNTRIES.map((_, i) => inScalers.map((s) => clamp(s.z[i] / IN_SCALE, -1, 1)));
    const TARGETS = COUNTRIES.map((_, i) => outScalers.map((s) => clamp(s.z[i] / OUT_SCALE, -0.9, 0.9)));
    const toReal = (v, j) => outScalers[j].fromZ(v * OUT_SCALE);

    let W = [];
    let B = [];
    let sample = 0;
    let input = INPUTS[0];
    let target = TARGETS[0];
    let acts = [];
    let deltas = [];
    let lr = 0.15;
    let cycle = 0;
    let pi = 0;
    let timer = 0;
    let startLoss = 0;
    const shown = SIZES.map(() => null);
    const flash = SIZES.map(() => 0);

    const nodeGeo = new THREE.SphereGeometry(0.34, 28, 20);
    const nodes = nodePos.map((layer) => layer.map((p) => {
      const m = new THREE.Mesh(nodeGeo, new THREE.MeshStandardMaterial({ color: IDLE, emissive: IDLE, emissiveIntensity: 0.2 }));
      m.position.copy(p);
      scene.add(m);
      return m;
    }));

    const edgeGeo = new THREE.CylinderGeometry(1, 1, 1, 8);
    const edges = [];
    for (let l = 0; l < SIZES.length - 1; l++) {
      edges.push(nodePos[l + 1].map((b) => nodePos[l].map((a) => {
        const m = new THREE.Mesh(edgeGeo, new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false }));
        placeCylinder(m, a, b, 0.02);
        scene.add(m);
        return m;
      })));
    }

    const MAXE = 25;
    const pulses = new THREE.InstancedMesh(new THREE.SphereGeometry(0.1, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffffff }), MAXE);
    pulses.frustumCulled = false;
    scene.add(pulses);

    const label = (text, pos, opts) => {
      const s = textSprite(text, { height: 0.36, color: '#94a3b8', ...opts });
      s.position.copy(pos);
      scene.add(s);
    };
    NAMES.forEach((name, l) => label(name, new THREE.Vector3(-6 + l * 4, nodePos[l][0].y + 1.05, 0), { height: 0.42 }));
    IN_LABELS.forEach((t, i) => label(t, nodePos[0][i].clone().add(new THREE.Vector3(-1.75, 0, 0)), { color: '#e2e8f0' }));
    OUT_LABELS.forEach((t, j) => label(t, nodePos[3][j].clone().add(new THREE.Vector3(1.4, 0.45, 0)), { color: '#e2e8f0' }));
    const valueLabels = new THREE.Group();
    scene.add(valueLabels);

    const stats = {
      country: ui.stat('Training on', { wide: true }),
      pred: ui.stat('Prediction vs actual', { wide: true }),
      loss: ui.stat('Dataset loss (all countries)', { wide: true }),
      cycle: ui.stat('Training step'),
      phase: ui.stat('Phase'),
    };

    function forward(x) {
      const a = [x];
      for (let l = 0; l < W.length; l++) {
        a.push(W[l].map((row, j) => Math.tanh(row.reduce((s, w, i) => s + w * a[l][i], B[l][j]))));
      }
      return a;
    }

    const datasetLoss = () => INPUTS.reduce((s, x, i) => {
      const out = forward(x)[SIZES.length - 1];
      return s + 0.5 * out.reduce((t, o, j) => t + (o - TARGETS[i][j]) ** 2, 0);
    }, 0) / INPUTS.length;

    function compute() {
      if (!W.length) return;
      acts = forward(input);
      const L = SIZES.length - 1;
      deltas = new Array(SIZES.length);
      deltas[L] = acts[L].map((a, j) => (a - target[j]) * (1 - a * a));
      for (let l = L - 1; l >= 1; l--) {
        deltas[l] = acts[l].map((a, i) => W[l].reduce((s, row, j) => s + row[i] * deltas[l + 1][j], 0) * (1 - a * a));
      }
    }

    function pickCountry() {
      sample = Math.floor(rand() * COUNTRIES.length);
      input = INPUTS[sample];
      target = TARGETS[sample];
      compute();
      const c = COUNTRIES[sample];
      stats.country(`${c.name}: ${IN_FEATS.map((f) => `${FIELDS[f].short} ${FIELDS[f].fmt(c[f])}`).join(' · ')}`);
      clearGroup(valueLabels);
      OUT_FEATS.forEach((f, j) => {
        const s = textSprite(`actual ${FIELDS[f].fmt(c[f])}`, { height: 0.32, color: '#ffd166', background: 'rgba(2,6,23,0.7)' });
        s.position.copy(nodePos[3][j]).add(new THREE.Vector3(1.6, -0.3, 0));
        valueLabels.add(s);
      });
    }

    function initWeights() {
      W = [];
      B = [];
      for (let l = 0; l < SIZES.length - 1; l++) {
        W.push(Array.from({ length: SIZES[l + 1] }, () => Array.from({ length: SIZES[l] }, () => gauss(rand) * 0.9)));
        B.push(Array.from({ length: SIZES[l + 1] }, () => gauss(rand) * 0.1));
      }
      cycle = 0;
      startLoss = datasetLoss();
      compute();
      styleEdges();
      restartCycle();
    }

    function applyUpdate() {
      for (let l = 0; l < W.length; l++) {
        for (let j = 0; j < SIZES[l + 1]; j++) {
          for (let i = 0; i < SIZES[l]; i++) W[l][j][i] -= lr * deltas[l + 1][j] * acts[l][i];
          B[l][j] -= lr * deltas[l + 1][j];
        }
      }
      cycle++;
      styleEdges();
    }

    function styleEdges() {
      for (let l = 0; l < edges.length; l++) {
        edges[l].forEach((row, j) => row.forEach((m, i) => {
          const w = W[l][j][i];
          const mag = Math.min(Math.abs(w), 2) / 2;
          m.material.color.copy(w > 0 ? POS : NEG);
          m.material.opacity = 0.12 + 0.6 * mag;
          m.scale.x = m.scale.z = 0.01 + 0.045 * mag;
        }));
      }
    }

    function restartCycle() {
      pi = 0;
      timer = 0;
      shown.fill(null);
      shown[0] = input;
    }

    function trainFast(n) {
      for (let s = 0; s < n; s++) {
        sample = Math.floor(rand() * COUNTRIES.length);
        input = INPUTS[sample];
        target = TARGETS[sample];
        compute();
        applyUpdate();
      }
      lossShown = datasetLoss();
      pickCountry();
      restartCycle();
    }

    ui.slider({ label: 'Learning rate η', min: 0.01, max: 1, step: 0.01, value: lr, format: (v) => v.toFixed(2), onInput: (v) => { lr = v; } });
    ui.button('Train 500 steps instantly', () => trainFast(500), { primary: true });
    ui.button('Re-initialise weights', initWeights);
    ui.html(`<div class="legend"><span style="--c:#4cc9f0">positive</span><span style="--c:#f72585">negative</span><span style="--c:#ff8c42">gradient</span></div>`);

    const tmp = new THREE.Vector3();
    const mtx = new THREE.Matrix4();
    const col = new THREE.Color();

    let lossShown = 0;
    initWeights();
    lossShown = startLoss;
    pickCountry();
    restartCycle();

    return {
      update(dt) {
        timer += dt;
        const ph = PHASES[pi];
        const u = Math.min(timer / (ph.type === 'update' ? DUR * 0.5 : DUR), 1);

        let n = 0;
        if (ph.type !== 'update') {
          const l = ph.l;
          for (let j = 0; j < SIZES[l + 1]; j++) {
            for (let i = 0; i < SIZES[l]; i++) {
              const a = nodePos[l][i];
              const b = nodePos[l + 1][j];
              let sig;
              if (ph.type === 'fwd') {
                sig = acts[l][i] * W[l][j][i];
                tmp.lerpVectors(a, b, u);
                col.copy(sig > 0 ? POS : NEG);
              } else {
                sig = deltas[l + 1][j] * W[l][j][i] * 6;
                tmp.lerpVectors(b, a, u);
                col.copy(GRAD);
              }
              const s = clamp(0.35 + Math.abs(sig), 0.35, 1.6);
              mtx.makeScale(s, s, s).setPosition(tmp);
              pulses.setMatrixAt(n, mtx);
              pulses.setColorAt(n, col);
              n++;
            }
          }
        }
        for (; n < MAXE; n++) pulses.setMatrixAt(n, mtx.makeScale(0, 0, 0));
        pulses.instanceMatrix.needsUpdate = true;
        if (pulses.instanceColor) pulses.instanceColor.needsUpdate = true;

        if (u >= 1) {
          if (ph.type === 'fwd') shown[ph.l + 1] = acts[ph.l + 1];
          if (ph.type === 'bwd') flash[ph.l] = 1;
          if (ph.type === 'update') {
            applyUpdate();
            lossShown = datasetLoss();
            pickCountry();
            restartCycle();
          } else {
            pi++;
            timer = 0;
          }
        }

        nodes.forEach((layer, l) => {
          flash[l] = Math.max(0, flash[l] - dt * 2.5);
          layer.forEach((m, i) => {
            const a = shown[l]?.[i];
            const mat = m.material;
            if (a === undefined) {
              mat.color.copy(IDLE);
              mat.emissive.copy(IDLE);
              mat.emissiveIntensity = 0.2;
            } else {
              mat.color.copy(IDLE).lerp(a > 0 ? POS : NEG, 0.3 + 0.7 * Math.abs(a));
              mat.emissive.copy(mat.color);
              mat.emissiveIntensity = 0.25 + Math.abs(a) * 1.1;
            }
            if (flash[l] > 0) mat.emissive.lerp(GRAD, flash[l]);
            m.scale.setScalar(1 + 0.3 * flash[l]);
          });
        });

        const out = acts[SIZES.length - 1];
        stats.pred(OUT_FEATS.map((f, j) => `${OUT_LABELS[j]} ${FIELDS[f].fmt(Math.max(0, toReal(out[j], j)))} vs ${FIELDS[f].fmt(COUNTRIES[sample][f])}`).join(' · '));
        stats.loss(`${lossShown.toFixed(4)} (started at ${startLoss.toFixed(4)})`);
        stats.cycle(cycle);
        stats.phase(ph.type === 'fwd' ? `Forward ${ph.l}→${ph.l + 1}` : ph.type === 'bwd' ? `Backprop ${ph.l + 1}→${ph.l}` : 'Update weights');
      },
    };
  },
};
