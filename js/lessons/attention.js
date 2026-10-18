import * as THREE from 'three';
import { textSprite, clearGroup, PALETTE, damp } from '../helpers.js';

const TOKENS = ['The', 'vaccine', 'was', 'tested', 'on', 'adults', 'so', 'it', 'reduced', 'deaths'];
const GAP = 1.55;
const X = (i) => (i - (TOKENS.length - 1) / 2) * GAP;

// Hand-set relevance scores (q·k / √d) for one illustrative attention head.
const SCORES = (() => {
  const n = TOKENS.length;
  const S = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : Math.abs(i - j) === 1 ? 0.6 : 0)));
  const rel = [
    [0, 1, 2.2], [1, 3, 1.8], [1, 8, 1.4], [2, 1, 1.6], [2, 3, 2.0], [3, 1, 2.6], [3, 5, 1.8],
    [4, 5, 2.2], [4, 3, 1.2], [5, 3, 1.8], [5, 4, 1.0], [6, 3, 1.2], [6, 8, 1.6], [7, 1, 3.2],
    [7, 8, 1.4], [8, 7, 2.2], [8, 1, 2.0], [8, 9, 1.8], [9, 8, 2.4], [9, 5, 1.2],
  ];
  for (const [i, j, v] of rel) S[i][j] = v;
  return S;
})();

const softmax = (row, temp) => {
  const m = Math.max(...row);
  const e = row.map((v) => Math.exp((v - m) / temp));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
};

export default {
  id: 'attention',
  title: 'Attention & Transformers',
  category: 'Deep Learning',
  camera: { position: [0, 1.4, 20], target: [0, 0.4, 0] },
  formula: 'Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V<br>each token: qᵢ = W_Q·xᵢ,  kⱼ = W_K·xⱼ,  vⱼ = W_V·xⱼ',
  goal: 'For every word, a <strong>set of weights summing to 100%</strong> that says how much it should draw on each other word, so “it” is understood as “the vaccine”.',
  usage: 'In a trained Transformer, every new sentence is turned into queries, keys and values, the attention weights are computed on the fly, and the mixed representations feed the next layer, ending in a prediction such as the next word or a document label.',
  hint: 'Arcs show how much the highlighted word attends to every other word',
  info: `
    <h3>What it is</h3>
    <p><strong>Attention</strong> is the core mechanism of Transformers, the architecture behind modern
    language models. It lets every word look at every other word in the sentence and decide which ones
    matter for understanding it.</p>
    <h3>Queries, keys and values</h3>
    <ul>
      <li>Each token produces a <strong>query</strong> (what am I looking for?), a <strong>key</strong> (what do I contain?), and a <strong>value</strong> (what information do I pass on?).</li>
      <li>The dot product of a query with every key gives a relevance score. <strong>Softmax</strong> turns the scores into weights that sum to 1 (the bars).</li>
      <li>The token's new representation is the weighted average of all the values.</li>
    </ul>
    <h3>Reading the animation</h3>
    <p>The sentence comes from the COVID story: <em>“The vaccine was tested on adults so it reduced deaths.”</em> Watch
    <strong>“it”</strong>: it attends mostly to <strong>“vaccine”</strong>, which is how the model works out that the vaccine (not the
    adults) did the reducing. <strong>“reduced”</strong> looks at both what did the reducing and what was reduced (“deaths”). Thicker, brighter arcs mean more attention. The weights here are hand-set to
    illustrate one attention head. A real Transformer learns them, and runs many heads in parallel, each
    tracking a different kind of relationship.</p>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>In 2020 the <strong>CORD-19</strong> dataset gathered hundreds of thousands of coronavirus research papers, far more than anyone
      could read. Transformer models searched and summarised them to answer questions such as “what is known about transmission?” Today
      the same architecture powers chatbots and translation, and tools that draft summaries of doctors' clinical notes.</p>
    </div>
    <h3>Temperature</h3>
    <p>Lower temperature makes the softmax sharper, so attention focuses on one word. Higher temperature
    spreads attention more evenly.</p>
  `,
  create({ scene, ui }) {
    let query = 7;
    let temp = 1;
    let auto = true;
    let timer = 0;
    let goal = softmax(SCORES[query], temp);
    const shown = goal.slice();

    const boxes = TOKENS.map((tok, i) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(1.35, 0.62, 0.3), new THREE.MeshStandardMaterial({ color: 0x1e293b, emissive: 0x1e293b, emissiveIntensity: 0.4 }));
      m.position.set(X(i), 0, 0);
      const label = textSprite(tok, { height: 0.36 });
      label.position.set(X(i), 0, 0.2);
      scene.add(m, label);
      return m;
    });

    const bars = TOKENS.map((_, i) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(0.9, 1, 0.2), new THREE.MeshStandardMaterial({ color: PALETTE[1], emissive: PALETTE[1], emissiveIntensity: 0.5 }));
      m.position.set(X(i), -0.6, 0);
      scene.add(m);
      return m;
    });

    const arcs = new THREE.Group();
    const pct = new THREE.Group();
    scene.add(arcs, pct);
    const arcMat = TOKENS.map(() => new THREE.MeshBasicMaterial({ color: PALETTE[2], transparent: true, depthWrite: false }));

    const focusStat = ui.stat('Query token');
    const topStat = ui.stat('Attends most to');

    function setQuery(q) {
      query = q;
      goal = softmax(SCORES[query], temp);
      boxes.forEach((b, i) => {
        const c = i === query ? PALETTE[2] : 0x1e293b;
        b.material.color.set(c);
        b.material.emissive.set(c);
        b.material.emissiveIntensity = i === query ? 0.35 : 0.4;
      });
      clearGroup(pct);
      goal.forEach((w, i) => {
        const s = textSprite(`${(w * 100).toFixed(0)}%`, { height: 0.3, color: '#cbd5e1' });
        s.position.set(X(i), -0.6 - 3.2 * w - 0.35, 0.1);
        pct.add(s);
      });
      const order = goal.map((w, i) => [w, i]).filter(([, i]) => i !== query).sort((a, b) => b[0] - a[0]);
      focusStat(`“${TOKENS[query]}”`);
      topStat(`“${TOKENS[order[0][1]]}” (${(order[0][0] * 100).toFixed(0)}%)`);
    }

    function rebuildArcs() {
      for (const c of [...arcs.children]) {
        arcs.remove(c);
        c.geometry.dispose();
      }
      const max = Math.max(...shown);
      shown.forEach((w, i) => {
        if (i === query || w < 0.005) return;
        const a = new THREE.Vector3(X(query), 0.32, 0);
        const b = new THREE.Vector3(X(i), 0.32, 0);
        const mid = new THREE.Vector3((a.x + b.x) / 2, 0.6 + Math.abs(b.x - a.x) * 0.38, 0);
        const curve = new THREE.QuadraticBezierCurve3(a, mid, b);
        arcMat[i].opacity = 0.2 + 0.8 * (w / max);
        arcs.add(new THREE.Mesh(new THREE.TubeGeometry(curve, 48, 0.015 + 0.13 * w, 8), arcMat[i]));
      });
    }

    const qSelect = ui.select({
      label: 'Query token',
      options: TOKENS.map((t, i) => ({ value: String(i), label: `${i}: ${t}` })),
      value: String(query),
      onChange: (v) => { setQuery(Number(v)); auto = false; autoToggle.set(false); },
    });
    ui.slider({ label: 'Softmax temperature', min: 0.3, max: 3, step: 0.1, value: temp, format: (v) => v.toFixed(1), onInput: (v) => { temp = v; setQuery(query); } });
    const autoToggle = ui.toggle({ label: 'Cycle through tokens automatically', value: true, onChange: (v) => { auto = v; } });

    setQuery(query);
    rebuildArcs();

    return {
      update(dt) {
        timer += dt;
        if (auto && timer > 3.2) {
          timer = 0;
          setQuery((query + 1) % TOKENS.length);
          qSelect.set(String(query));
        }
        const k = damp(6, dt);
        let moved = 0;
        shown.forEach((w, i) => {
          const d = goal[i] - w;
          shown[i] += d * k;
          moved = Math.max(moved, Math.abs(d));
          const h = 0.02 + 3.2 * shown[i];
          bars[i].scale.y = h;
          bars[i].position.y = -0.6 - h / 2;
        });
        if (moved > 1e-3 || arcs.children.length === 0) rebuildArcs();
      },
    };
  },
};
