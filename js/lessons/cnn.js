import * as THREE from 'three';
import { textSprite, PALETTE } from '../helpers.js';

const IN = 9;
const KS = 3;
const OUT = IN - KS + 1;
const CELL = 0.62;
const IN_X = -3.9;
const OUT_X = 4.4;
const STEP = 0.22;

const PATTERNS = {
  'Letter X': (r, c) => r === c || r + c === 8,
  Square: (r, c) => ((r === 1 || r === 7) && c >= 1 && c <= 7) || ((c === 1 || c === 7) && r >= 1 && r <= 7),
  'Digit 7': (r, c) => (r === 1 && c >= 1 && c <= 7) || (r >= 2 && r + c === 9),
  Circle: (r, c) => Math.abs(Math.hypot(r - 4, c - 4) - 3) < 0.6,
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
  hint: 'The yellow window is the kernel sliding over the image',
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
    <h3>ReLU</h3>
    <p>Toggle ReLU to see the activation function remove every negative response. This non-linearity lets
    deep networks learn complex patterns.</p>
  `,
  create({ scene, ui }) {
    let pattern = 'Square';
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
      image = Array.from({ length: IN }, (_, r) => Array.from({ length: IN }, (_, c) => (PATTERNS[pattern](r, c) ? 1 : 0)));
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
          col.set(v ? 0xe2e8f0 : 0x1e293b);
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
      for (let m = 0; m < KS; m++) for (let n = 0; n < KS; n++) if (image[wr + m][wc + n] && K[m][n]) terms.push(K[m][n]);
      const raw = terms.reduce((a, b) => a + b, 0);
      sumStat(`out[${wr},${wc}] = ${terms.length ? terms.map((t) => (Number.isInteger(t) ? t : t.toFixed(2))).join(' + ').replace(/\+ -/g, '− ') : '0'} = ${raw.toFixed(2)}${relu ? ` → ReLU ${Math.max(0, raw).toFixed(2)}` : ''}`);
    }

    ui.select({ label: 'Input image', options: Object.keys(PATTERNS).map((k) => ({ value: k, label: k })), value: pattern, onChange: (v) => { pattern = v; compute(); } });
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
