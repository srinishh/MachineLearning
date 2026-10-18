import * as THREE from 'three';
import { textSprite, PALETTE } from '../helpers.js';
import { SERIES } from '../data/covid.js';

const IN = 9;
const KS = 3;
const OUT = IN - KS + 1;
const CELL = 0.62;
const IN_X = -3.9;
const OUT_X = 4.4;
const STEP = 0.22;

const COVID = 'COVID waves (real data)';
const fromFn = (fn) => Array.from({ length: IN }, (_, r) => Array.from({ length: IN }, (_, c) => (fn(r, c) ? 1 : 0)));

// Real data as an image: one row per country, one column per ~4-month period,
// brightness = average weekly cases per million relative to that country's peak.
const WAVES = (() => {
  const n = SERIES.dates.length;
  const edges = Array.from({ length: IN + 1 }, (_, k) => Math.round((k * n) / IN));
  const rows = SERIES.countries.slice(0, IN).map((c) => {
    const avg = edges.slice(0, IN).map((e, k) => {
      const part = c.cases.slice(e, edges[k + 1]);
      return part.reduce((s, v) => s + v, 0) / part.length;
    });
    const peak = Math.max(...avg) || 1;
    return avg.map((v) => v / peak);
  });
  const short = { 'United States': 'USA', 'United Kingdom': 'UK', 'South Korea': 'S. Korea', 'South Africa': 'S. Africa' };
  const fmt = (d) => new Date(d).toLocaleString('en-US', { month: 'short', year: '2-digit', timeZone: 'UTC' });
  return {
    rows,
    names: SERIES.countries.slice(0, IN).map((c) => short[c.name] ?? c.name),
    periods: edges.slice(0, IN).map((e) => fmt(SERIES.dates[e])),
  };
})();

const IMAGES = {
  [COVID]: WAVES.rows,
  'Letter X': fromFn((r, c) => r === c || r + c === 8),
  Square: fromFn((r, c) => ((r === 1 || r === 7) && c >= 1 && c <= 7) || ((c === 1 || c === 7) && r >= 1 && r <= 7)),
  'Digit 7': fromFn((r, c) => (r === 1 && c >= 1 && c <= 7) || (r >= 2 && r + c === 9)),
  Circle: fromFn((r, c) => Math.abs(Math.hypot(r - 4, c - 4) - 3) < 0.6),
};

const KERNELS = {
  'Vertical edges': [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
  'Horizontal edges': [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
  'Diagonal lines': [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]],
  Sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
  Blur: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map((r) => r.map((v) => v / 9)),
};

const cellXY = (r, c, n, cx) => [cx + (c - (n - 1) / 2) * CELL, ((n - 1) / 2 - r) * CELL];

export default {
  id: 'cnn',
  title: 'Convolutional Neural Networks',
  category: 'Deep Learning',
  camera: { position: [0.2, -13, 14.5], target: [0.2, 0, 0] },
  formula: 'out[i, j] = Σₘ Σₙ input[i+m, j+n] · K[m, n]<br>ReLU(z) = max(0, z)',
  goal: 'A <strong>feature map</strong> that lights up wherever a pattern appears. With real COVID data, it marks when each country\'s cases started rising or falling. Stacking many such maps lets a CNN recognise whole images.',
  usage: 'In a trained CNN you pass a new image (an X-ray, or a grid like this one) through the learned kernels, and the final layer turns the feature maps into a label such as “pneumonia: yes/no”. Here you can apply each kernel to any input and read the response.',
  hint: 'The yellow window is the kernel sliding over the image · In the COVID image, rows are countries and columns are time',
  info: `
    <h3>What it is</h3>
    <p>CNNs are the networks behind image recognition. Instead of connecting every pixel to every neuron,
    they slide a small grid of weights, a <strong>kernel</strong> (or filter), across the image.</p>
    <h3>The convolution</h3>
    <p>At each position, multiply the 3×3 patch under the window element by element with the kernel and add
    up the results. That single number becomes one cell of the <strong>feature map</strong> on the right. Tall
    cyan bars are strong positive responses and pink bars below the plane are negative.</p>
    <h3>Why it works</h3>
    <ul>
      <li>The same kernel is reused everywhere (<strong>weight sharing</strong>), so the network finds a feature wherever it appears in the image.</li>
      <li>Different kernels detect different features. Try <em>Vertical edges</em> on the Square, then <em>Horizontal edges</em>.</li>
      <li>In a real CNN the kernel values are <strong>learned</strong> by backprop, and later layers combine edges into shapes, then shapes into objects.</li>
    </ul>
    <h3>On the COVID data</h3>
    <p>Any grid of numbers can be treated as an image. In <em>COVID waves</em>, each <strong>row is a country</strong>, each
    <strong>column is a ~4-month period</strong> from March 2020 to December 2022, and brightness is that country's average weekly
    cases relative to its own peak. The <em>Vertical edges</em> kernel compares later periods with earlier ones, so it turns into a
    <strong>wave detector</strong>: cyan bars mark where cases were rising (a wave arriving) and pink bars mark declines. Notice how
    almost every country lights up at the same time near the end, which is the global Omicron wave of early 2022.</p>
    <h3>ReLU</h3>
    <p>Toggle ReLU to see the activation function remove every negative response, keeping only the “wave arriving” signals.</p>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>In 2020 hundreds of research groups trained CNNs to detect COVID-19 from <strong>chest X-rays and CT scans</strong>. It is
      also a cautionary tale. A 2021 review in <em>Nature Machine Intelligence</em> (Roberts et al.) examined 62 such models and
      found none ready for clinical use. Many had learned shortcuts, such as which hospital's scanner produced the image or patient
      position, instead of signs of disease. Good data matters as much as a good architecture.</p>
    </div>
  `,
  create({ scene, ui }) {
    let pattern = COVID;
    let kernelName = 'Vertical edges';
    let relu = false;
    let image = [];
    let out = [];
    let maxAbs = 1;
    let s = 0;
    let timer = 0;
    let hold = 0;

    const boxGeo = new THREE.BoxGeometry(CELL * 0.9, CELL * 0.9, 1);
    const inMesh = new THREE.InstancedMesh(boxGeo, new THREE.MeshStandardMaterial({ roughness: 0.5 }), IN * IN);
    const outMesh = new THREE.InstancedMesh(boxGeo, new THREE.MeshStandardMaterial({ roughness: 0.4, emissive: 0xffffff, emissiveIntensity: 0.15 }), OUT * OUT);
    inMesh.frustumCulled = outMesh.frustumCulled = false;
    scene.add(inMesh, outMesh);

    const base = (n, cx) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(n * CELL + 0.2, n * CELL + 0.2), new THREE.MeshBasicMaterial({ color: 0x0f172a, transparent: true, opacity: 0.8 }));
      m.position.set(cx, 0, -0.01);
      return m;
    };
    scene.add(base(IN, IN_X), base(OUT, OUT_X));
    const inLabel = textSprite('Input image 9×9', { height: 0.4, color: '#94a3b8' });
    inLabel.position.set(IN_X, (IN * CELL) / 2 + 0.6, 0);
    const outLabel = textSprite('Feature map 7×7', { height: 0.4, color: '#94a3b8' });
    outLabel.position.set(OUT_X, (OUT * CELL) / 2 + 0.6, 0);
    scene.add(inLabel, outLabel);

    const waveLabels = new THREE.Group();
    scene.add(waveLabels);
    WAVES.names.forEach((name, r) => {
      const [, y] = cellXY(r, 0, IN, IN_X);
      const s = textSprite(name, { height: 0.26, color: '#cbd5e1' });
      s.position.set(IN_X - (IN * CELL) / 2 - 0.75, y, 0.1);
      waveLabels.add(s);
    });
    WAVES.periods.forEach((p, c) => {
      const [x] = cellXY(0, c, IN, IN_X);
      const s = textSprite(p, { height: 0.22, color: '#94a3b8' });
      s.position.set(x, (IN * CELL) / 2 + 0.3 + (c % 2) * 0.32, 0.1);
      waveLabels.add(s);
    });
    const waveTitle = textSprite('Countries × time (cases vs. own peak)', { height: 0.34, color: '#e2e8f0' });
    waveTitle.position.set(IN_X, (IN * CELL) / 2 + 1.3, 0);
    waveLabels.add(waveTitle);

    const win = new THREE.Group();
    const winSize = KS * CELL;
    win.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(winSize, winSize, 1.3)), new THREE.LineBasicMaterial({ color: PALETTE[2] })));
    win.add(new THREE.Mesh(new THREE.BoxGeometry(winSize, winSize, 1.3), new THREE.MeshBasicMaterial({ color: PALETTE[2], transparent: true, opacity: 0.08, depthWrite: false })));
    scene.add(win);

    const linkPos = new Float32Array(4 * 6);
    const linkGeo = new THREE.BufferGeometry();
    linkGeo.setAttribute('position', new THREE.BufferAttribute(linkPos, 3));
    const links = new THREE.LineSegments(linkGeo, new THREE.LineBasicMaterial({ color: PALETTE[2], transparent: true, opacity: 0.5 }));
    links.frustumCulled = false;
    scene.add(links);

    const panel = ui.html('');
    const sumStat = ui.stat('Current output', { wide: true });

    function compute() {
      image = IMAGES[pattern];
      waveLabels.visible = pattern === COVID;
      inLabel.visible = pattern !== COVID;
      const K = KERNELS[kernelName];
      out = Array.from({ length: OUT }, (_, r) => Array.from({ length: OUT }, (_, c) => {
        let v = 0;
        for (let m = 0; m < KS; m++) for (let n = 0; n < KS; n++) v += image[r + m][c + n] * K[m][n];
        return relu ? Math.max(0, v) : v;
      }));
      maxAbs = Math.max(1e-6, ...out.flat().map(Math.abs));
      panel.innerHTML = `<div class="field-head"><span>Kernel: ${kernelName}</span></div><table class="kernel-table">${K.map((row) => `<tr>${row.map((v) => `<td style="color:${v > 0 ? '#4cc9f0' : v < 0 ? '#f72585' : '#64748b'}">${Number.isInteger(v) ? v : v.toFixed(2)}</td>`).join('')}</tr>`).join('')}</table>`;
      s = 0;
      timer = 0;
      hold = 0;
    }

    const mtx = new THREE.Matrix4();
    const col = new THREE.Color();
    const white = new THREE.Color(0xe2e8f0);
    const pos = new THREE.Vector3();
    const quat = new THREE.Quaternion();
    const scl = new THREE.Vector3();

    function draw() {
      const wr = Math.floor(s / OUT);
      const wc = s % OUT;
      for (let r = 0; r < IN; r++) {
        for (let c = 0; c < IN; c++) {
          const v = image[r][c];
          const h = 0.08 + v * 0.6;
          const [x, y] = cellXY(r, c, IN, IN_X);
          mtx.compose(pos.set(x, y, h / 2), quat, scl.set(1, 1, h));
          inMesh.setMatrixAt(r * IN + c, mtx);
          const inside = r >= wr && r < wr + KS && c >= wc && c < wc + KS;
          col.set(0x1e293b).lerp(white, v);
          if (inside) col.lerp(new THREE.Color(PALETTE[2]), 0.45);
          inMesh.setColorAt(r * IN + c, col);
        }
      }
      inMesh.instanceMatrix.needsUpdate = true;
      inMesh.instanceColor.needsUpdate = true;

      for (let r = 0; r < OUT; r++) {
        for (let c = 0; c < OUT; c++) {
          const idx = r * OUT + c;
          const v = out[r][c];
          const [x, y] = cellXY(r, c, OUT, OUT_X);
          const revealed = idx <= s;
          const grow = idx === s ? Math.min(1, timer / STEP) : 1;
          const h = (0.05 + 1.6 * (Math.abs(v) / maxAbs)) * (revealed ? grow : 0) + 1e-4;
          mtx.compose(pos.set(x, y, v >= 0 ? h / 2 : -h / 2), quat, scl.set(1, 1, h));
          outMesh.setMatrixAt(idx, mtx);
          col.set(v > 0 ? PALETTE[0] : v < 0 ? PALETTE[1] : 0x475569);
          outMesh.setColorAt(idx, col);
        }
      }
      outMesh.instanceMatrix.needsUpdate = true;
      outMesh.instanceColor.needsUpdate = true;

      const [cx, cy] = cellXY(wr + 1, wc + 1, IN, IN_X);
      win.position.set(cx, cy, 0.65);
      const [ox, oy] = cellXY(wr, wc, OUT, OUT_X);
      const v = out[wr][wc];
      const top = v >= 0 ? 0.05 + 1.6 * (Math.abs(v) / maxAbs) : 0;
      const half = winSize / 2;
      [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(([dx, dy], i) => {
        linkPos.set([cx + dx * half, cy + dy * half, 1.3, ox + (dx * CELL) / 2.2, oy + (dy * CELL) / 2.2, top], i * 6);
      });
      linkGeo.attributes.position.needsUpdate = true;

      const K = KERNELS[kernelName];
      const terms = [];
      for (let m = 0; m < KS; m++) for (let n = 0; n < KS; n++) if (image[wr + m][wc + n] && K[m][n]) terms.push(K[m][n] * image[wr + m][wc + n]);
      const raw = terms.reduce((a, b) => a + b, 0);
      const fmtT = (t) => (Number.isInteger(t) ? String(t) : t.toFixed(2));
      const expr = terms.length > 6 ? `Σ of ${terms.length} products` : terms.length ? terms.map(fmtT).join(' + ').replace(/\+ -/g, '− ') : '0';
      const where = pattern === COVID ? ` (${WAVES.names[wr + 1]}, ${WAVES.periods[wc + 1]})` : '';
      sumStat(`out[${wr},${wc}]${where} = ${expr} = ${raw.toFixed(2)}${relu ? ` → ReLU ${Math.max(0, raw).toFixed(2)}` : ''}`);
    }

    ui.select({ label: 'Input image', options: Object.keys(IMAGES).map((k) => ({ value: k, label: k })), value: pattern, onChange: (v) => { pattern = v; compute(); } });
    ui.select({ label: 'Kernel / filter', options: Object.keys(KERNELS).map((k) => ({ value: k, label: k })), value: kernelName, onChange: (v) => { kernelName = v; compute(); } });
    ui.toggle({ label: 'Apply ReLU activation', value: relu, onChange: (v) => { relu = v; compute(); } });
    ui.button('Restart sweep', () => { s = 0; timer = 0; hold = 0; }, { primary: true });

    compute();

    return {
      update(dt) {
        if (s < OUT * OUT - 1) {
          timer += dt;
          if (timer >= STEP) {
            timer = 0;
            s++;
          }
        } else {
          timer = Math.min(timer + dt, STEP);
          hold += dt;
          if (hold > 2.5) {
            s = 0;
            timer = 0;
            hold = 0;
          }
        }
        draw();
      },
    };
  },
};
