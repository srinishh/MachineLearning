import * as THREE from 'three';
import {
  rng, gauss, grid2D, sphereCloud, setInstance, setInstanceColor, commitInstances,
  unitCylinder, placeCylinder, PALETTE, damp,
} from '../helpers.js';

const LIFT = 0.4; // φ(x) = (x₁, x₂, LIFT·(x₁² + x₂²))
const RADIUS = 2.2;

export default {
  id: 'svm',
  title: 'Support Vector Machines',
  category: 'Supervised Learning',
  camera: { position: [0, -6, 17], target: [0, 0, 0] },
  formula: 'f(x) = w·x + b<br>minimise  ½λ‖w‖² + ¹⁄ₙ Σ max(0, 1 − yᵢ·f(xᵢ))<br>margin width = 2 / ‖w‖',
  hint: 'Ringed points are support vectors, the only ones that decide the boundary',
  info: `
    <h3>What it is</h3>
    <p>Many lines can separate two classes. An SVM picks the one with the <strong>widest margin</strong>,
    the largest empty “street” between the classes. A wide margin tends to generalise better to new data.</p>
    <h3>Support vectors</h3>
    <p>Only the points that touch or cross the edge of the street affect the solution. These are the
    <strong>support vectors</strong> (ringed). You could delete every other point and get the same boundary.</p>
    <h3>Soft margin</h3>
    <p>Real data overlaps, so the <strong>hinge loss</strong> lets some points sit inside the street at a cost.
    The <strong>λ</strong> slider sets the trade-off: larger λ gives a wider, more tolerant margin, and smaller
    λ gives a narrower, stricter one.</p>
    <h3>The kernel trick</h3>
    <p>Switch the mode to <strong>Kernel trick</strong>. The two classes form rings, and no straight line can
    separate them. Press <strong>Lift into 3D</strong>: each point gets a third coordinate <code>x₁² + x₂²</code>.
    In that higher-dimensional space a flat plane separates them easily, and seen back in 2D that plane becomes a
    circle. Kernels such as RBF do this implicitly, without ever computing the new coordinates.</p>
  `,
  create({ scene, ui, setCamera }) {
    scene.add(grid2D(14, 14));
    const rand = rng(51);

    // ---------- Linear mode ----------
    const linear = new THREE.Group();
    scene.add(linear);
    const N = 60;
    let X = [];
    let Y = [];
    let w = [0.1, 0];
    let b = 0;
    let lambda = 0.05;
    let t = 0;
    let acc = 0;

    const pts = sphereCloud(N, 0.14);
    const rings = new THREE.InstancedMesh(new THREE.TorusGeometry(0.3, 0.035, 8, 32), new THREE.MeshBasicMaterial({ color: 0xffffff }), N);
    rings.frustumCulled = false;
    const boundary = unitCylinder(PALETTE[2], { emissive: 0.8 });
    const marginPos = unitCylinder(PALETTE[1], { emissive: 0.5, opacity: 0.6 });
    const marginNeg = unitCylinder(PALETTE[0], { emissive: 0.5, opacity: 0.6 });
    const street = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.06, depthWrite: false }));
    linear.add(pts, rings, boundary, marginPos, marginNeg, street);

    const stats = {
      margin: ui.stat('Margin width'),
      sv: ui.stat('Support vectors'),
      acc: ui.stat('Accuracy'),
      mode: ui.stat('Space'),
    };

    function newLinear() {
      X = [];
      Y = [];
      for (let i = 0; i < N; i++) {
        const y = i % 2 ? 1 : -1;
        X.push([y * 1.8 + gauss(rand) * 1.0, y * 1.6 + gauss(rand) * 1.0]);
        Y.push(y);
        setInstanceColor(pts, i, y > 0 ? PALETTE[1] : PALETTE[0]);
        setInstance(pts, i, X[i][0], X[i][1], 0);
      }
      commitInstances(pts);
      w = [0.1, 0];
      b = 0;
      t = 0;
    }

    function trainStep() {
      t++;
      const lr = Math.max(0.01, 0.3 / Math.sqrt(1 + t / 20));
      let g0 = lambda * w[0];
      let g1 = lambda * w[1];
      let gb = 0;
      for (let i = 0; i < N; i++) {
        if (Y[i] * (w[0] * X[i][0] + w[1] * X[i][1] + b) < 1) {
          g0 -= (Y[i] * X[i][0]) / N;
          g1 -= (Y[i] * X[i][1]) / N;
          gb -= Y[i] / N;
        }
      }
      w[0] -= lr * g0;
      w[1] -= lr * g1;
      b -= lr * gb;
    }

    // ---------- Kernel mode ----------
    const kernel = new THREE.Group();
    kernel.visible = false;
    scene.add(kernel);
    const KN = 90;
    const KX = [];
    const KY = [];
    for (let i = 0; i < KN; i++) {
      const inner = i % 2 === 0;
      const r = inner ? Math.sqrt(rand()) * 1.6 : 2.8 + rand() * 1.4;
      const a = rand() * Math.PI * 2;
      KX.push([r * Math.cos(a), r * Math.sin(a)]);
      KY.push(inner ? 1 : -1);
    }
    const kpts = sphereCloud(KN, 0.14);
    KX.forEach((_, i) => setInstanceColor(kpts, i, KY[i] > 0 ? PALETTE[1] : PALETTE[0]));
    const planeZ = LIFT * RADIUS * RADIUS;
    const sepPlane = new THREE.Mesh(new THREE.PlaneGeometry(11, 11), new THREE.MeshBasicMaterial({ color: PALETTE[2], transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
    sepPlane.position.z = planeZ;
    const bowlGeo = new THREE.PlaneGeometry(10, 10, 40, 40);
    const bp = bowlGeo.attributes.position;
    for (let i = 0; i < bp.count; i++) bp.setZ(i, Math.min(LIFT * (bp.getX(i) ** 2 + bp.getY(i) ** 2), 9));
    const bowl = new THREE.Mesh(bowlGeo, new THREE.MeshBasicMaterial({ color: 0x94a3b8, wireframe: true, transparent: true, opacity: 0 }));
    const circlePts = [];
    for (let i = 0; i <= 96; i++) circlePts.push(new THREE.Vector3(RADIUS * Math.cos((i / 96) * Math.PI * 2), RADIUS * Math.sin((i / 96) * Math.PI * 2), 0.02));
    const circle = new THREE.Line(new THREE.BufferGeometry().setFromPoints(circlePts), new THREE.LineBasicMaterial({ color: PALETTE[2], transparent: true, opacity: 0 }));
    kernel.add(kpts, sepPlane, bowl, circle);
    let lift = 0;
    let liftTarget = 0;

    // ---------- Controls ----------
    let mode = 'linear';
    ui.select({
      label: 'Mode',
      options: [{ value: 'linear', label: 'Linear max-margin classifier' }, { value: 'kernel', label: 'Kernel trick (non-linear data)' }],
      value: mode,
      onChange: (v) => {
        mode = v;
        linear.visible = v === 'linear';
        kernel.visible = v === 'kernel';
        if (v === 'kernel') setCamera([9, -11, 9], [0, 0, 1.4]);
        else setCamera([0, -6, 17]);
      },
    });
    ui.slider({ label: 'Regularization λ (linear mode)', min: 0.005, max: 0.5, step: 0.005, value: lambda, format: (v) => v.toFixed(3), onInput: (v) => { lambda = v; t = 0; } });
    ui.button('New data', newLinear);
    const liftBtn = ui.button('Lift into 3D', () => {
      if (mode !== 'kernel') return;
      liftTarget = liftTarget ? 0 : 1;
      liftBtn.textContent = liftTarget ? 'Flatten to 2D' : 'Lift into 3D';
    }, { primary: true });

    newLinear();
    const p0 = new THREE.Vector3();
    const p1 = new THREE.Vector3();

    function drawLine(mesh, offset, norm, n, d, radius) {
      const cx = (-b * n[0]) / norm + n[0] * offset;
      const cy = (-b * n[1]) / norm + n[1] * offset;
      p0.set(cx - d[0] * 8, cy - d[1] * 8, 0.01);
      p1.set(cx + d[0] * 8, cy + d[1] * 8, 0.01);
      placeCylinder(mesh, p0, p1, radius);
    }

    return {
      update(dt) {
        if (mode === 'linear') {
          acc += dt;
          while (acc >= 1 / 60) {
            acc -= 1 / 60;
            trainStep();
          }
          const norm = Math.hypot(w[0], w[1]) || 1e-6;
          const n = [w[0] / norm, w[1] / norm];
          const d = [-n[1], n[0]];
          drawLine(boundary, 0, norm, n, d, 0.05);
          drawLine(marginPos, 1 / norm, norm, n, d, 0.025);
          drawLine(marginNeg, -1 / norm, norm, n, d, 0.025);
          street.position.set((-b * n[0]) / norm, (-b * n[1]) / norm, 0);
          street.rotation.z = Math.atan2(d[1], d[0]);
          street.scale.set(16, 2 / norm, 1);

          let sv = 0;
          let correct = 0;
          for (let i = 0; i < N; i++) {
            const m = Y[i] * (w[0] * X[i][0] + w[1] * X[i][1] + b);
            if (m > 0) correct++;
            const isSV = m <= 1.02;
            if (isSV) sv++;
            setInstance(rings, i, X[i][0], X[i][1], 0, isSV ? 1 : 0);
          }
          rings.instanceMatrix.needsUpdate = true;
          stats.margin((2 / norm).toFixed(3));
          stats.sv(sv);
          stats.acc(`${((100 * correct) / N).toFixed(0)}%`);
          stats.mode('2D input space');
        } else {
          lift += (liftTarget - lift) * damp(3, dt);
          KX.forEach(([x, y], i) => setInstance(kpts, i, x, y, lift * LIFT * (x * x + y * y)));
          commitInstances(kpts);
          const vis = Math.max(0, (lift - 0.4) / 0.6);
          sepPlane.material.opacity = 0.22 * vis;
          bowl.material.opacity = 0.12 * lift;
          circle.material.opacity = vis;
          stats.margin('–');
          stats.sv('–');
          stats.acc(lift > 0.9 ? '100% (plane)' : 'not linearly separable');
          stats.mode(lift > 0.5 ? '3D feature space φ(x)' : '2D input space');
        }
      },
    };
  },
};
