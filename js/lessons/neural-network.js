import * as THREE from 'three';
import { rng, gauss, textSprite, placeCylinder, PALETTE, clamp } from '../helpers.js';

const SIZES = [3, 5, 5, 2];
const DUR = 0.8;
const PHASES = [
  { type: 'fwd', l: 0 }, { type: 'fwd', l: 1 }, { type: 'fwd', l: 2 },
  { type: 'bwd', l: 2 }, { type: 'bwd', l: 1 }, { type: 'bwd', l: 0 },
  { type: 'update' },
];
const NAMES = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];
const POS = new THREE.Color(PALETTE[0]);
const NEG = new THREE.Color(PALETTE[1]);
const GRAD = new THREE.Color(PALETTE[5]);
const IDLE = new THREE.Color(0x334155);

export default {
  id: 'neural-network',
  title: 'Neural Networks & Backprop',
  category: 'Deep Learning',
  camera: { position: [0.9, 0.5, 21], target: [0.9, 0, 0] },
  formula: 'Forward:  aˡ = tanh(Wˡ·aˡ⁻¹ + bˡ)<br>Backward: δᴸ = (a − y)⊙(1 − a²);  δˡ = (Wˡ⁺¹)ᵀδˡ⁺¹ ⊙ (1 − (aˡ)²)<br>Update:  Wˡ ← Wˡ − η·δˡ(aˡ⁻¹)ᵀ',
  hint: 'Blue/pink pulses = signals moving forward · Orange pulses = error flowing backward',
  info: `
    <h3>What it is</h3>
    <p>A neural network is layers of simple units (<strong>neurons</strong>). Each one computes a weighted
    sum of its inputs, adds a bias, and applies a non-linear <strong>activation function</strong> (here
    <code>tanh</code>). Stacking layers lets the network represent very complex functions.</p>
    <h3>Forward pass</h3>
    <p>Inputs flow left to right. Cyan edges are positive weights, pink are negative, and thicker means
    stronger. A neuron's brightness shows how strongly it activates.</p>
    <h3>Backpropagation</h3>
    <p>After the forward pass we compare the output with the <strong>target</strong> and measure the error.
    <strong>Backprop</strong> then sends that error backwards (the orange pulses), using the chain rule to
    work out how much each weight contributed. Each weight is then nudged to reduce the error. That nudge is
    gradient descent again.</p>
    <h3>Watch it learn</h3>
    <p>Every cycle is one training step on the same example, so the <strong>loss</strong> drops cycle by cycle
    and the outputs move toward the target. Press <strong>New sample</strong> to give it a new task.</p>
  `,
  create({ scene, ui }) {
    const rand = rng(81);
    const nodePos = SIZES.map((n, l) => Array.from({ length: n }, (_, i) => new THREE.Vector3(-6 + l * 4, ((n - 1) / 2 - i) * 1.5, 0)));

    let W = [];
    let B = [];
    let input = [];
    let target = [];
    let acts = [];
    let deltas = [];
    let lr = 0.5;
    let cycle = 0;
    let pi = 0;
    let timer = 0;
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

    NAMES.forEach((name, l) => {
      const s = textSprite(name, { height: 0.42, color: '#94a3b8' });
      s.position.set(-6 + l * 4, nodePos[l][0].y + 1.05, 0);
      scene.add(s);
    });
    const targetLabels = new THREE.Group();
    scene.add(targetLabels);

    const stats = {
      cycle: ui.stat('Training step'),
      loss: ui.stat('Loss'),
      out: ui.stat('Output', { wide: true }),
      tgt: ui.stat('Target', { wide: true }),
      phase: ui.stat('Phase', { wide: true }),
    };

    function initWeights() {
      W = [];
      B = [];
      for (let l = 0; l < SIZES.length - 1; l++) {
        W.push(Array.from({ length: SIZES[l + 1] }, () => Array.from({ length: SIZES[l] }, () => gauss(rand) * 0.9)));
        B.push(Array.from({ length: SIZES[l + 1] }, () => gauss(rand) * 0.1));
      }
      cycle = 0;
      compute();
      styleEdges();
      restartCycle();
    }

    function newSample() {
      input = Array.from({ length: SIZES[0] }, () => +(rand() * 2 - 1).toFixed(2));
      target = Array.from({ length: SIZES[3] }, () => +(rand() * 1.6 - 0.8).toFixed(2));
      for (const c of [...targetLabels.children]) {
        targetLabels.remove(c);
        c.material.map.dispose();
        c.material.dispose();
      }
      target.forEach((t, j) => {
        const s = textSprite(`target ${t.toFixed(2)}`, { height: 0.34, color: '#ffd166', background: 'rgba(2,6,23,0.7)' });
        s.position.copy(nodePos[3][j]).add(new THREE.Vector3(1.75, 0, 0));
        targetLabels.add(s);
      });
      cycle = 0;
      compute();
      restartCycle();
    }

    function compute() {
      if (!W.length) return;
      acts = [input];
      for (let l = 0; l < W.length; l++) {
        acts.push(W[l].map((row, j) => Math.tanh(row.reduce((s, w, i) => s + w * acts[l][i], B[l][j]))));
      }
      const L = SIZES.length - 1;
      deltas = new Array(SIZES.length);
      deltas[L] = acts[L].map((a, j) => (a - target[j]) * (1 - a * a));
      for (let l = L - 1; l >= 1; l--) {
        deltas[l] = acts[l].map((a, i) => W[l].reduce((s, row, j) => s + row[i] * deltas[l + 1][j], 0) * (1 - a * a));
      }
    }

    const loss = () => 0.5 * acts[SIZES.length - 1].reduce((s, a, j) => s + (a - target[j]) ** 2, 0);

    function applyUpdate() {
      for (let l = 0; l < W.length; l++) {
        for (let j = 0; j < SIZES[l + 1]; j++) {
          for (let i = 0; i < SIZES[l]; i++) W[l][j][i] -= lr * deltas[l + 1][j] * acts[l][i];
          B[l][j] -= lr * deltas[l + 1][j];
        }
      }
      cycle++;
      compute();
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

    ui.slider({ label: 'Learning rate η', min: 0.05, max: 1.5, step: 0.05, value: lr, format: (v) => v.toFixed(2), onInput: (v) => { lr = v; } });
    ui.button('New sample', newSample, { primary: true });
    ui.button('Re-initialise weights', initWeights);
    ui.html(`<div class="legend"><span style="--c:#4cc9f0">positive</span><span style="--c:#f72585">negative</span><span style="--c:#ff8c42">gradient</span></div>`);

    const tmp = new THREE.Vector3();
    const mtx = new THREE.Matrix4();
    const col = new THREE.Color();

    newSample();
    initWeights();

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
        stats.cycle(cycle);
        stats.loss(loss().toFixed(5));
        stats.out(`[${out.map((v) => v.toFixed(3)).join(', ')}]`);
        stats.tgt(`[${target.map((v) => v.toFixed(2)).join(', ')}]  input [${input.join(', ')}]`);
        stats.phase(ph.type === 'fwd' ? `Forward pass: layer ${ph.l} → ${ph.l + 1}` : ph.type === 'bwd' ? `Backpropagation: layer ${ph.l + 1} → ${ph.l}` : 'Gradient descent: updating weights');
      },
    };
  },
};
