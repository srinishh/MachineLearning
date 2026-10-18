import * as THREE from 'three';
import { rng, gauss, PALETTE, dotTexture, smoothstep } from '../helpers.js';

const N = 2500;

export default {
  id: 'intro',
  title: 'What is Machine Learning?',
  category: 'Start Here',
  camera: { position: [0, 3, 20], target: [0, 0, 0] },
  formula: 'data  →  learning algorithm  →  model  →  predictions',
  hint: 'Watch the same 2,500 data points take on different structures',
  info: `
    <h3>The big idea</h3>
    <p>Traditional programs follow rules a person wrote down. <strong>Machine learning</strong> works the
    other way round: we show an algorithm many <strong>examples</strong> and it finds the rules (a
    <strong>model</strong>) itself, by adjusting numbers called <strong>parameters</strong> until the model
    fits the data.</p>
    <h3>The three main flavours</h3>
    <ul>
      <li><strong>Supervised learning</strong>: every example has a known answer (label). The model learns to map inputs to answers, as in regression and classification.</li>
      <li><strong>Unsupervised learning</strong>: no labels. The model finds hidden structure such as clusters or a lower-dimensional shape.</li>
      <li><strong>Reinforcement learning</strong>: an agent acts in a world, gets rewards, and learns a strategy by trial and error.</li>
    </ul>
    <h3>The animation</h3>
    <p>The particle cloud stands for a dataset. It starts as noise, then shows the kinds of structure
    learning algorithms look for: groups (clustering), a smooth surface (regression), and a complex
    twisted shape (deep learning).</p>
    <h3>How to use this site</h3>
    <p>Every lesson is a live simulation running the real algorithm in your browser. Drag to orbit, scroll to
    zoom, change the sliders, and use the <strong>Speed</strong> control at the bottom. Lessons get more
    advanced as you go down the list.</p>
  `,
  create({ scene, ui }) {
    const rand = rng(3);
    const stages = [
      { name: 'Raw, unlabeled data', build: chaos },
      { name: 'Unsupervised: finding clusters', build: clusters },
      { name: 'Supervised: fitting a smooth function', build: surface },
      { name: 'Deep learning: complex structure', build: helix },
    ].map((s) => ({ ...s, ...s.build(rand) }));

    const pos = new Float32Array(stages[0].pos);
    const col = new Float32Array(stages[0].col);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    const points = new THREE.Points(geo, new THREE.PointsMaterial({
      size: 0.22,
      map: dotTexture(),
      vertexColors: true,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    }));
    scene.add(points);

    const stageStat = ui.stat('Stage', { wide: true });
    let from = 0;
    let to = 1;
    let t = 0;
    const HOLD = 2.6;
    const MORPH = 2.2;

    ui.button('Next stage', () => { t = HOLD; }, { primary: true });

    return {
      update(dt) {
        t += dt;
        points.rotation.y += dt * 0.12;
        const k = smoothstep((t - HOLD) / MORPH);
        const a = stages[from];
        const b = stages[to];
        for (let i = 0; i < pos.length; i++) {
          pos[i] = a.pos[i] + (b.pos[i] - a.pos[i]) * k;
          col[i] = a.col[i] + (b.col[i] - a.col[i]) * k;
        }
        geo.attributes.position.needsUpdate = true;
        geo.attributes.color.needsUpdate = true;
        stageStat(k < 0.5 ? a.name : b.name);
        if (t > HOLD + MORPH) {
          t = 0;
          from = to;
          to = (to + 1) % stages.length;
        }
      },
    };
  },
};

function chaos(rand) {
  const pos = new Float32Array(N * 3);
  const col = new Float32Array(N * 3);
  const c = new THREE.Color();
  for (let i = 0; i < N; i++) {
    const r = 7 * Math.cbrt(rand());
    const th = rand() * Math.PI * 2;
    const ph = Math.acos(2 * rand() - 1);
    pos.set([r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph), r * Math.sin(ph) * Math.sin(th)], i * 3);
    c.setHSL(0.6, 0.2, 0.35 + rand() * 0.2);
    col.set([c.r, c.g, c.b], i * 3);
  }
  return { pos, col };
}

function clusters(rand) {
  const centers = [[-4, 2, 0], [4, 2.5, -1], [0, -3, 3], [-3, -2, -4], [4, -2, 3]];
  const pos = new Float32Array(N * 3);
  const col = new Float32Array(N * 3);
  const c = new THREE.Color();
  for (let i = 0; i < N; i++) {
    const k = i % centers.length;
    const [x, y, z] = centers[k];
    pos.set([x + gauss(rand) * 0.9, y + gauss(rand) * 0.9, z + gauss(rand) * 0.9], i * 3);
    c.set(PALETTE[k]);
    col.set([c.r, c.g, c.b], i * 3);
  }
  return { pos, col };
}

function surface() {
  const side = Math.sqrt(N);
  const pos = new Float32Array(N * 3);
  const col = new Float32Array(N * 3);
  const c = new THREE.Color();
  for (let i = 0; i < N; i++) {
    const x = ((i % side) / (side - 1) - 0.5) * 14;
    const z = (Math.floor(i / side) / (side - 1) - 0.5) * 14;
    const y = 2.2 * Math.sin(x * 0.6) * Math.cos(z * 0.5);
    pos.set([x, y, z], i * 3);
    c.setHSL(0.55 - (y + 2.2) / 4.4 * 0.45, 0.85, 0.55);
    col.set([c.r, c.g, c.b], i * 3);
  }
  return { pos, col };
}

function helix(rand) {
  const pos = new Float32Array(N * 3);
  const col = new Float32Array(N * 3);
  const c = new THREE.Color();
  for (let i = 0; i < N; i++) {
    const strand = i % 2;
    const t = (i / N) * Math.PI * 6;
    const r = 2.6 + gauss(rand) * 0.12;
    const a = t + strand * Math.PI;
    pos.set([r * Math.cos(a), (i / N - 0.5) * 14, r * Math.sin(a)], i * 3);
    c.set(strand ? PALETTE[1] : PALETTE[0]).lerp(new THREE.Color(PALETTE[2]), (i / N) * 0.5);
    col.set([c.r, c.g, c.b], i * 3);
  }
  return { pos, col };
}
