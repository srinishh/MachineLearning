import * as THREE from 'three';
import { rng, heat, PALETTE, clamp } from '../helpers.js';

// A bumpy loss landscape with a shallow local minimum and a deeper global one.
const f = (x, z) =>
  0.045 * (x * x + z * z)
  - 2.2 * Math.exp(-((x - 1.8) ** 2 + (z - 1.4) ** 2) / 1.4)
  - 3.2 * Math.exp(-((x + 2.1) ** 2 + (z + 1.9) ** 2) / 2.4)
  + 0.35 * Math.sin(1.4 * x) * Math.cos(1.2 * z);

const grad = (x, z) => {
  const h = 1e-4;
  return [(f(x + h, z) - f(x - h, z)) / (2 * h), (f(x, z + h) - f(x, z - h)) / (2 * h)];
};

const MAX_TRAIL = 800;
const LIMIT = 4.9;

export default {
  id: 'gradient-descent',
  title: 'Gradient Descent',
  category: 'Foundations',
  camera: { position: [9, 9, 11], target: [0, -0.5, 0] },
  formula: 'θ ← θ − η·∇L(θ)<br>Momentum:  v ← β·v − η·∇L(θ);  θ ← θ + v',
  hint: 'Click anywhere on the surface to drop both balls there',
  info: `
    <h3>What it is</h3>
    <p>Almost every model is trained by <strong>gradient descent</strong>. The surface is a
    <strong>loss landscape</strong>: each position is one setting of two parameters and the height is the
    loss (how wrong the model is). Training means getting to the lowest point.</p>
    <h3>How it works</h3>
    <p>The <strong>gradient</strong> <code>∇L</code> points uphill. Each step moves the parameters a small
    amount the opposite way. The <strong>learning rate η</strong> sets the step size.</p>
    <h3>Two optimisers race</h3>
    <ul>
      <li><span style="color:#f72585">●</span> <strong>Plain gradient descent</strong> follows the local slope and often gets stuck in the nearest <strong>local minimum</strong>.</li>
      <li><span style="color:#4cc9f0">●</span> <strong>Momentum</strong> builds up velocity like a heavy ball, which lets it roll through small bumps and settle faster in deep valleys.</li>
    </ul>
    <h3>Try this</h3>
    <ul>
      <li>Click near the shallow dip on the right and compare where each ball ends up.</li>
      <li>Set the learning rate very high to see oscillation, or very low to see a crawl.</li>
      <li>Set momentum to 0 so both balls behave identically.</li>
    </ul>
  `,
  create({ scene, ui, onClick }) {
    const rand = rng(5);

    // Surface mesh coloured by height.
    const geo = new THREE.PlaneGeometry(10, 10, 140, 140);
    geo.rotateX(-Math.PI / 2);
    const p = geo.attributes.position;
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < p.count; i++) {
      const y = f(p.getX(i), p.getZ(i));
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
    const surface = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6, side: THREE.DoubleSide }));
    scene.add(surface);
    const wire = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: true, transparent: true, opacity: 0.05 }));
    scene.add(wire);

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
      return { ...wk, ball, trailPos, trailGeo, x: 0, z: 0, vx: 0, vz: 0, n: 0, stat: ui.stat(`${wk.label} loss`) };
    });

    let lr = 0.15;
    let beta = 0.85;
    let steps = 0;
    let acc = 0;
    const stepStat = ui.stat('Steps', { wide: true });

    function start(x, z) {
      steps = 0;
      for (const wk of walkers) {
        Object.assign(wk, { x, z, vx: 0, vz: 0, n: 0 });
        pushTrail(wk);
      }
    }

    function randomStart() {
      const a = rand() * Math.PI * 2;
      start(Math.cos(a) * (3.4 + rand()), Math.sin(a) * (3.4 + rand()));
    }

    function pushTrail(wk) {
      if (wk.n >= MAX_TRAIL) return;
      wk.trailPos.set([wk.x, f(wk.x, wk.z) + 0.04, wk.z], wk.n * 3);
      wk.n++;
      wk.trailGeo.setDrawRange(0, wk.n);
      wk.trailGeo.attributes.position.needsUpdate = true;
    }

    function step() {
      steps++;
      for (const wk of walkers) {
        const [gx, gz] = grad(wk.x, wk.z);
        if (wk.momentum) {
          wk.vx = beta * wk.vx - lr * gx;
          wk.vz = beta * wk.vz - lr * gz;
        } else {
          wk.vx = -lr * gx;
          wk.vz = -lr * gz;
        }
        const px = wk.x;
        const pz = wk.z;
        wk.x = clamp(wk.x + wk.vx, -LIMIT, LIMIT);
        wk.z = clamp(wk.z + wk.vz, -LIMIT, LIMIT);
        if (Math.hypot(wk.x - px, wk.z - pz) > 0.005) pushTrail(wk);
      }
    }

    ui.slider({ label: 'Learning rate η', min: 0.01, max: 0.8, step: 0.01, value: lr, format: (v) => v.toFixed(2), onInput: (v) => { lr = v; } });
    ui.slider({ label: 'Momentum β', min: 0, max: 0.97, step: 0.01, value: beta, format: (v) => v.toFixed(2), onInput: (v) => { beta = v; } });
    ui.button('Random start', randomStart, { primary: true });
    ui.button('Start on the ridge', () => start(4.2, 3.6));
    ui.html(`<div class="legend"><span style="--c:#f72585">Gradient descent</span><span style="--c:#4cc9f0">With momentum</span></div>`);

    onClick((ray) => {
      const hit = ray.intersectObject(surface)[0];
      if (hit) start(clamp(hit.point.x, -LIMIT, LIMIT), clamp(hit.point.z, -LIMIT, LIMIT));
    });

    start(4.2, 3.6);

    return {
      update(dt) {
        acc += dt;
        while (acc >= 1 / 30) {
          acc -= 1 / 30;
          step();
        }
        for (const wk of walkers) {
          wk.ball.position.set(wk.x, f(wk.x, wk.z) + 0.2, wk.z);
          wk.stat(f(wk.x, wk.z).toFixed(3));
        }
        stepStat(steps);
      },
    };
  },
};
