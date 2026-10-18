import * as THREE from 'three';
import { heat, PALETTE, clamp } from '../helpers.js';
import { COUNTRIES, scaler } from '../data/covid.js';

const MAX_TRAIL = 1200;
const LIMIT = 4.9;

// Target: standardised deaths per million. Feature: median age, either raw
// (in decades, uncentred) or standardised.
const Y = scaler('deaths_pm', { log: false }).z;
const AGE_Z = scaler('median_age', { log: false }).z;
const AGE_DECADES = COUNTRIES.map((c) => c.median_age / 10);

function regressionLandscape(xs, { sw, sb, lr, label, desc }) {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b) / n;
  const my = Y.reduce((a, b) => a + b) / n;
  const w0 = xs.reduce((s, x, i) => s + (x - mx) * (Y[i] - my), 0) / xs.reduce((s, x) => s + (x - mx) ** 2, 0);
  const b0 = my - w0 * mx;
  const loss = ([w, b]) => xs.reduce((s, x, i) => s + (w * x + b - Y[i]) ** 2, 0) / n;
  const lmin = loss([w0, b0]);
  return {
    label,
    desc,
    lr,
    loss,
    grad: ([w, b]) => {
      let gw = 0;
      let gb = 0;
      xs.forEach((x, i) => {
        const e = w * x + b - Y[i];
        gw += e * x;
        gb += e;
      });
      return [(2 * gw) / n, (2 * gb) / n];
    },
    toDisplay: ([w, b]) => [(w - w0) / sw, (b - b0) / sb],
    fromDisplay: (X, Z) => [w0 + X * sw, b0 + Z * sb],
    height: (L) => Math.min(0.7 * Math.log(1 + 6 * (L - lmin)) - 2.5, 2),
    start: [-4.4, 2.4],
  };
}

const bumpy = (x, z) =>
  0.045 * (x * x + z * z)
  - 2.2 * Math.exp(-((x - 1.8) ** 2 + (z - 1.4) ** 2) / 1.4)
  - 3.2 * Math.exp(-((x + 2.1) ** 2 + (z + 1.9) ** 2) / 2.4)
  + 0.35 * Math.sin(1.4 * x) * Math.cos(1.2 * z);

const LANDSCAPES = {
  unscaled: regressionLandscape(AGE_DECADES, {
    sw: 0.16, sb: 0.5, lr: 0.05,
    label: 'COVID regression · raw feature (median age in decades)',
    desc: 'A long, narrow, diagonal valley: w and b are tangled because the feature is not centred or scaled.',
  }),
  scaled: regressionLandscape(AGE_Z, {
    sw: 0.35, sb: 0.35, lr: 0.1,
    label: 'COVID regression · standardised feature',
    desc: 'After standardising, the bowl is round, so every step points straight at the minimum.',
  }),
  bumpy: {
    label: 'Neural-network-like bumpy landscape',
    desc: 'Deep models have non-convex losses with local minima, where momentum can help escape.',
    lr: 0.15,
    loss: ([x, z]) => bumpy(x, z),
    grad: ([x, z]) => {
      const h = 1e-4;
      return [(bumpy(x + h, z) - bumpy(x - h, z)) / (2 * h), (bumpy(x, z + h) - bumpy(x, z - h)) / (2 * h)];
    },
    toDisplay: (p) => p,
    fromDisplay: (X, Z) => [X, Z],
    height: (L) => L,
    start: [4.2, 3.6],
  },
};

export default {
  id: 'gradient-descent',
  title: 'Gradient Descent',
  category: 'Foundations',
  camera: { position: [9, 9, 11], target: [0, -0.5, 0] },
  formula: 'θ ← θ − η·∇L(θ)<br>Momentum:  v ← β·v − η·∇L(θ);  θ ← θ + v',
  hint: 'Click anywhere on the surface to drop both balls there',
  info: `
    <h3>What it is</h3>
    <p>Almost every model is trained by <strong>gradient descent</strong>. The surface is a <strong>loss landscape</strong>:
    each position is one choice of parameters and the height is the loss (how wrong the model is). Training means getting to
    the lowest point.</p>
    <h3>On the COVID data</h3>
    <p>The default surface is <strong>real</strong>: the loss of the <em>deaths per million ~ median age</em> regression from the
    previous lesson, plotted over its two parameters (slope <code>w</code> across, intercept <code>b</code> deep). Each ball's
    position is a candidate line through the countries.</p>
    <ul>
      <li><strong>Raw feature</strong> (age in decades): the valley is long and narrow. Plain gradient descent zig-zags across it and crawls along it.</li>
      <li><strong>Standardised feature</strong>: the same data after z-scoring gives a round bowl that is easy to descend. This is why the data lesson standardises features.</li>
    </ul>
    <h3>Two optimisers race</h3>
    <ul>
      <li><span style="color:#f72585">●</span> <strong>Plain gradient descent</strong> follows the local slope.</li>
      <li><span style="color:#4cc9f0">●</span> <strong>Momentum</strong> builds up velocity like a heavy ball, which smooths out zig-zags and rolls through small bumps.</li>
    </ul>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>Every model fitted during the pandemic, from simple regressions to the neural networks that forecast hospital demand, was
      trained by some version of this loop. Optimisers like <strong>Adam</strong> combine momentum with per-parameter step sizes so
      that badly scaled problems like the raw valley still train quickly.</p>
    </div>
    <h3>Try this</h3>
    <ul>
      <li>On the raw landscape, set momentum to 0 and count how many steps plain descent needs. Then switch to the standardised landscape.</li>
      <li>On the bumpy landscape, click near the shallow dip on the right and see which ball gets stuck in the <strong>local minimum</strong>.</li>
    </ul>
  `,
  create({ scene, ui, onClick }) {
    let land = LANDSCAPES.unscaled;
    let surface = null;
    let wire = null;

    function buildSurface() {
      for (const m of [surface, wire]) {
        if (!m) continue;
        scene.remove(m);
        m.material.dispose();
      }
      surface?.geometry.dispose();
      const geo = new THREE.PlaneGeometry(10, 10, 120, 120);
      geo.rotateX(-Math.PI / 2);
      const p = geo.attributes.position;
      let lo = Infinity;
      let hi = -Infinity;
      for (let i = 0; i < p.count; i++) {
        const y = land.height(land.loss(land.fromDisplay(p.getX(i), p.getZ(i))));
        p.setY(i, y);
        lo = Math.min(lo, y);
        hi = Math.max(hi, y);
      }
      const colors = new Float32Array(p.count * 3);
      const c = new THREE.Color();
      for (let i = 0; i < p.count; i++) {
        heat((p.getY(i) - lo) / (hi - lo), c);
        colors.set([c.r, c.g, c.b], i * 3);
      }
      geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      geo.computeVertexNormals();
      surface = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6, side: THREE.DoubleSide }));
      wire = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.05 }));
      scene.add(surface, wire);
    }

    const walkers = [
      { label: 'Gradient descent', color: PALETTE[1], momentum: false },
      { label: 'Momentum', color: PALETTE[0], momentum: true },
    ].map((wk) => {
      const ball = new THREE.Mesh(
        new THREE.SphereGeometry(0.2, 24, 18),
        new THREE.MeshStandardMaterial({ color: wk.color, emissive: wk.color, emissiveIntensity: 0.6 }),
      );
      scene.add(ball);
      const trailPos = new Float32Array(MAX_TRAIL * 3);
      const trailGeo = new THREE.BufferGeometry();
      trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
      const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: wk.color }));
      trail.frustumCulled = false;
      scene.add(trail);
      return { ...wk, ball, trailPos, trailGeo, p: [0, 0], v: [0, 0], n: 0, stat: ui.stat(`${wk.label} loss`) };
    });

    let lr = land.lr;
    let beta = 0.85;
    let steps = 0;
    let acc = 0;
    const stepStat = ui.stat('Steps');
    const descStat = ui.stat('Landscape', { wide: true });

    const display = (p) => {
      const [X, Z] = land.toDisplay(p);
      return [X, land.height(land.loss(p)), Z];
    };

    function pushTrail(wk) {
      if (wk.n >= MAX_TRAIL) return;
      const [X, y, Z] = display(wk.p);
      wk.trailPos.set([X, y + 0.04, Z], wk.n * 3);
      wk.n++;
      wk.trailGeo.setDrawRange(0, wk.n);
      wk.trailGeo.attributes.position.needsUpdate = true;
    }

    function start(X, Z) {
      steps = 0;
      for (const wk of walkers) {
        wk.p = land.fromDisplay(X, Z);
        wk.v = [0, 0];
        wk.n = 0;
        pushTrail(wk);
      }
    }

    function step() {
      steps++;
      for (const wk of walkers) {
        const g = land.grad(wk.p);
        for (let j = 0; j < 2; j++) wk.v[j] = (wk.momentum ? beta * wk.v[j] : 0) - lr * g[j];
        const [X0, Z0] = land.toDisplay(wk.p);
        let next = [wk.p[0] + wk.v[0], wk.p[1] + wk.v[1]];
        const [X, Z] = land.toDisplay(next);
        if (!Number.isFinite(X) || Math.abs(X) > LIMIT || Math.abs(Z) > LIMIT) {
          // Diverging or leaving the plotted region: clamp to the edge.
          next = land.fromDisplay(clamp(X || 0, -LIMIT, LIMIT), clamp(Z || 0, -LIMIT, LIMIT));
          wk.v = [0, 0];
        }
        wk.p = next;
        const [X1, Z1] = land.toDisplay(wk.p);
        if (Math.hypot(X1 - X0, Z1 - Z0) > 0.004) pushTrail(wk);
      }
    }

    function setLandscape(key) {
      land = LANDSCAPES[key];
      lr = land.lr;
      lrSlider.set(Math.log10(lr));
      descStat(land.desc);
      buildSurface();
      start(...land.start);
    }

    ui.select({
      label: 'Loss landscape',
      options: Object.entries(LANDSCAPES).map(([value, l]) => ({ value, label: l.label })),
      value: 'unscaled',
      onChange: setLandscape,
    });
    const lrSlider = ui.slider({
      label: 'Learning rate η', min: -3, max: 0, step: 0.05, value: Math.log10(lr),
      format: (v) => (10 ** v).toFixed(3), onInput: (v) => { lr = 10 ** v; },
    });
    ui.slider({ label: 'Momentum β', min: 0, max: 0.97, step: 0.01, value: beta, format: (v) => v.toFixed(2), onInput: (v) => { beta = v; } });
    ui.button('Restart', () => start(...land.start), { primary: true });
    ui.button('Random start', () => {
      const a = Math.random() * Math.PI * 2;
      start(Math.cos(a) * 4.2, Math.sin(a) * 4.2);
    });
    ui.html(`<div class="legend"><span style="--c:#f72585">Gradient descent</span><span style="--c:#4cc9f0">With momentum</span></div>`);

    onClick((ray) => {
      const hit = surface && ray.intersectObject(surface)[0];
      if (hit) start(clamp(hit.point.x, -LIMIT, LIMIT), clamp(hit.point.z, -LIMIT, LIMIT));
    });

    setLandscape('unscaled');

    return {
      update(dt) {
        acc += dt;
        while (acc >= 1 / 30) {
          acc -= 1 / 30;
          step();
        }
        for (const wk of walkers) {
          const [X, y, Z] = display(wk.p);
          wk.ball.position.set(X, y + 0.2, Z);
          wk.stat(land.loss(wk.p).toFixed(4));
        }
        stepStat(steps);
      },
    };
  },
};
