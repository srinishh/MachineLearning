import * as THREE from 'three';
import {
  rng, gauss, grid2D, sphereCloud, setInstance, setInstanceColor, commitInstances,
  unitCylinder, placeCylinder, PALETTE,
} from '../helpers.js';

export default {
  id: 'linear-regression',
  title: 'Linear Regression',
  category: 'Foundations',
  camera: { position: [0, -4, 17], target: [0, 0, 0] },
  formula: 'ŷ = w·x + b<br>Loss = ¹⁄ₙ Σ (ŷᵢ − yᵢ)²<br>w ← w − η·∂Loss/∂w',
  hint: 'Pink lines are the errors (residuals) the model is trying to shrink',
  info: `
    <h3>What it is</h3>
    <p>Linear regression predicts a number by drawing the <strong>best straight line</strong> through data.
    The model has two parameters: the slope <code>w</code> and the intercept <code>b</code>.</p>
    <h3>What “best” means</h3>
    <p>Each pink line is a <strong>residual</strong>, the gap between the true value and the prediction. We
    square the residuals and average them to get the <strong>mean squared error (MSE)</strong>. Training means
    finding the <code>w</code> and <code>b</code> that make this loss as small as possible.</p>
    <h3>How it learns</h3>
    <p>Each frame runs one step of <strong>gradient descent</strong>: compute how the loss changes when
    <code>w</code> and <code>b</code> change, then nudge both a little downhill. The learning rate
    <code>η</code> sets the size of the nudge.</p>
    <h3>Try this</h3>
    <ul>
      <li>Push the learning rate above about <code>0.11</code> and watch training <strong>diverge</strong>, because each step overshoots.</li>
      <li>Set a tiny learning rate and see how slowly it converges.</li>
    </ul>
  `,
  create({ scene, ui }) {
    scene.add(grid2D(14, 14));
    const N = 40;
    const rand = rng(11);

    const pts = sphereCloud(N, 0.15);
    scene.add(pts);
    const line = unitCylinder(PALETTE[2], { emissive: 0.6 });
    scene.add(line);

    const resPos = new Float32Array(N * 6);
    const resGeo = new THREE.BufferGeometry();
    resGeo.setAttribute('position', new THREE.BufferAttribute(resPos, 3));
    const residuals = new THREE.LineSegments(resGeo, new THREE.LineBasicMaterial({ color: PALETTE[1], transparent: true, opacity: 0.85 }));
    residuals.frustumCulled = false;
    scene.add(residuals);

    let xs = [];
    let ys = [];
    let w = 0;
    let b = 0;
    let epoch = 0;
    let lr = 0.02;
    let acc = 0;
    let status = 'Training…';

    const statW = ui.stat('Slope w');
    const statB = ui.stat('Intercept b');
    const statLoss = ui.stat('Loss (MSE)');
    const statEpoch = ui.stat('Epoch');
    const statStatus = ui.stat('Status', { wide: true });

    const loss = () => xs.reduce((s, x, i) => s + (w * x + b - ys[i]) ** 2, 0) / N;

    function restart() {
      w = (rand() * 2 - 1) * 2;
      b = (rand() * 2 - 1) * 3;
      epoch = 0;
      status = 'Training…';
    }

    function newData() {
      const tw = (rand() * 2 - 1) * 0.9;
      const tb = (rand() * 2 - 1) * 1.5;
      xs = [];
      ys = [];
      for (let i = 0; i < N; i++) {
        const x = rand() * 10 - 5;
        const y = tw * x + tb + gauss(rand) * 0.9;
        xs.push(x);
        ys.push(y);
        setInstance(pts, i, x, y, 0);
        setInstanceColor(pts, i, PALETTE[0]);
      }
      commitInstances(pts);
      restart();
    }

    function step() {
      let gw = 0;
      let gb = 0;
      for (let i = 0; i < N; i++) {
        const e = w * xs[i] + b - ys[i];
        gw += e * xs[i];
        gb += e;
      }
      w -= (lr * 2 * gw) / N;
      b -= (lr * 2 * gb) / N;
      epoch++;
      if (!Number.isFinite(w) || Math.abs(w) > 200) {
        status = 'Diverged! Learning rate too high, restarting';
        const msg = status;
        restart();
        status = msg;
      } else if (Math.abs(gw) / N < 1e-4 && Math.abs(gb) / N < 1e-4) {
        status = 'Converged ✓';
      }
    }

    ui.slider({
      label: 'Learning rate η', min: 0.001, max: 0.12, step: 0.001, value: lr,
      format: (v) => v.toFixed(3), onInput: (v) => { lr = v; },
    });
    ui.button('Restart training', restart, { primary: true });
    ui.button('New data', newData);

    newData();

    const a = new THREE.Vector3();
    const c = new THREE.Vector3();
    return {
      update(dt) {
        acc += dt;
        const interval = 1 / 20;
        while (acc >= interval) {
          acc -= interval;
          step();
        }
        a.set(-7, w * -7 + b, 0);
        c.set(7, w * 7 + b, 0);
        placeCylinder(line, a, c, 0.06);
        for (let i = 0; i < N; i++) {
          resPos.set([xs[i], ys[i], 0, xs[i], w * xs[i] + b, 0], i * 6);
        }
        resGeo.attributes.position.needsUpdate = true;
        statW(w.toFixed(3));
        statB(b.toFixed(3));
        statLoss(loss().toFixed(4));
        statEpoch(epoch);
        statStatus(status);
      },
    };
  },
};
