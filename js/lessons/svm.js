import * as THREE from 'three';
import {
  rng, grid2D, sphereCloud, setInstance, setInstanceColor, commitInstances,
  unitCylinder, placeCylinder, PALETTE, damp,
  textSprite,
} from '../helpers.js';
import { COUNTRIES, FIELDS, TRAIN_IDX, TEST_IDX, IS_TEST, scaler } from '../data/covid.js';
import { splitPoints, SPLIT_LEGEND } from '../ml/split-points.js';
import { predictor } from '../ml/predictor.js';
import { pct } from '../ml/metrics.js';

const VAX_GOAL = 60; // % fully vaccinated
const S = 2.1; // display units per standard deviation
const LIFE = scaler('life_exp');
const HDI = scaler('hdi');
const LIFT = 0.4; // φ(x) = (x₁, x₂, LIFT·(x₁² + x₂²))
const RADIUS = 2.2;

export default {
  id: 'svm',
  title: 'Support Vector Machines',
  category: 'Supervised Learning',
  camera: { position: [0, -6, 17], target: [0, 0, 0] },
  formula: 'f(x) = w·x + b<br>minimise  ½λ‖w‖² + ¹⁄ₙ Σ max(0, 1 − yᵢ·f(xᵢ))<br>margin width = 2 / ‖w‖',
  goal: 'A <strong>yes/no answer</strong> (will this country reach 60% full vaccination?) from the side of a boundary line the point falls on. The line is placed to leave the <strong>widest possible margin</strong> between the classes. The trained model is a weight per feature plus a bias, defined by a few support vectors.',
  usage: 'Compute the score <em>f(x) = w₁·life expectancy + w₂·HDI + b</em> for a new country. A positive score means “yes”, negative means “no”, and the size of the score is how far it sits from the boundary (confidence). Try it in “Use the trained model”.',
  hint: 'Ringed countries are support vectors · Spheres = train, cubes = test · Click a point to name it',
  info: `
    <h3>What it is</h3>
    <p>Many lines can separate two classes. An SVM picks the one with the <strong>widest margin</strong>,
    the largest empty “street” between the classes. A wide margin tends to generalise better to new data.</p>
    <h3>On the COVID data</h3>
    <p>Can a country's development level predict whether it reached <strong>60% full vaccination</strong> by the end of 2022?
    <span style="color:#f72585">Pink</span> countries did and <span style="color:#4cc9f0">cyan</span> countries didn't. They are placed by
    <strong>life expectancy</strong> (x) and the <strong>Human Development Index</strong> (y). The classes overlap (Bulgaria, Romania and Russia
    lagged despite high HDI, while Cambodia, Nepal and Bangladesh vaccinated quickly), so this is a <strong>soft-margin</strong> problem.</p>
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
    circle. Kernels such as RBF do this implicitly, without ever computing the new coordinates. (This mode uses a small toy
    dataset because the trick is easiest to see on rings.)</p>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>Before deep learning, SVMs were the go-to classifier for medical and biological data with many features and few samples.
      Examples include classifying tumours from gene-expression profiles, detecting disease from blood-test panels, and early COVID
      studies that flagged severe cases from routine lab values. Their margins make them robust when data is scarce.</p>
    </div>
  `,
  create({ scene, ui, setCamera, onClick }) {
    scene.add(grid2D(14, 14));
    const rand = rng(51);

    // ---------- Linear mode ----------
    const linear = new THREE.Group();
    scene.add(linear);
    const N = COUNTRIES.length;
    let X = [];
    let Y = [];
    let w = [0.1, 0];
    let b = 0;
    let lambda = 0.05;
    let t = 0;
    let acc = 0;

    const pts = splitPoints(N, 0.14);
    const marker = new THREE.Mesh(new THREE.OctahedronGeometry(0.3), new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.8 }));
    const rings = new THREE.InstancedMesh(new THREE.TorusGeometry(0.3, 0.035, 8, 32), new THREE.MeshBasicMaterial({ color: 0xffffff }), N);
    rings.frustumCulled = false;
    const boundary = unitCylinder(PALETTE[2], { emissive: 0.8 });
    const marginPos = unitCylinder(PALETTE[1], { emissive: 0.5, opacity: 0.6 });
    const marginNeg = unitCylinder(PALETTE[0], { emissive: 0.5, opacity: 0.6 });
    const street = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.06, depthWrite: false }));
    linear.add(pts.group, marker, rings, boundary, marginPos, marginNeg, street);

    const pipe = ui.pipeline([
      ['Prepare data', `X = life expectancy, HDI (standardised) · Y = +1 if ≥ ${VAX_GOAL}% fully vaccinated, else −1`],
      ['Split', `${TRAIN_IDX.length} train / ${TEST_IDX.length} test countries`],
      ['Train & optimise', 'Sub-gradient descent on hinge loss + λ‖w‖²'],
      ['Evaluate', ''],
    ]);
    const stats = {
      margin: ui.stat('Margin width'),
      sv: ui.stat('Support vectors'),
      acc: ui.stat('Accuracy train / test'),
      mode: ui.stat('Space'),
    };

    function newLinear() {
      X = COUNTRIES.map((_, i) => [LIFE.z[i] * S, HDI.z[i] * S]);
      Y = COUNTRIES.map((c) => (c.vax >= VAX_GOAL ? 1 : -1));
      X.forEach((p, i) => pts.set(i, p[0], p[1], 0, 1, Y[i] > 0 ? PALETTE[1] : PALETTE[0]));
      pts.commit();
      w = [0.1, 0];
      b = 0;
      t = 0;
    }

    const ticks = new THREE.Group();
    linear.add(ticks);
    const tick = (text, x, y, color = '#94a3b8', height = 0.3) => {
      const s = textSprite(text, { height, color });
      s.position.set(x, y, 0);
      ticks.add(s);
    };
    for (const zv of [-2, 0, 2]) {
      tick(LIFE.format(LIFE.fromZ(zv)), zv * S, -7.4);
      tick(HDI.format(HDI.fromZ(zv)), -7.8, zv * S);
    }
    tick('Life expectancy →', 4.5, -8.1, '#e2e8f0', 0.36);
    tick('Human Development Index ↑', -4.4, 7.5, '#e2e8f0', 0.36);
    const pickStat = ui.stat('Selected country', { wide: true });
    pickStat(`Pink = ≥ ${VAX_GOAL}% fully vaccinated · click a dot`);
    onClick((ray) => {
      if (mode !== 'linear') return;
      const i = pts.pick(ray);
      if (i < 0) return;
      const c = COUNTRIES[i];
      pickStat(`${c.name}: life exp. ${FIELDS.life_exp.fmt(c.life_exp)}, HDI ${FIELDS.hdi.fmt(c.hdi)} → ${FIELDS.vax.fmt(c.vax)} fully vaccinated`);
    });

    function trainStep() {
      t++;
      const lr = Math.max(0.01, 0.3 / Math.sqrt(1 + t / 20));
      let g0 = lambda * w[0];
      let g1 = lambda * w[1];
      let gb = 0;
      const n = TRAIN_IDX.length;
      for (const i of TRAIN_IDX) {
        if (Y[i] * (w[0] * X[i][0] + w[1] * X[i][1] + b) < 1) {
          g0 -= (Y[i] * X[i][0]) / n;
          g1 -= (Y[i] * X[i][1]) / n;
          gb -= Y[i] / n;
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
    const liftBtn = ui.button('Lift into 3D', () => {
      if (mode !== 'kernel') return;
      liftTarget = liftTarget ? 0 : 1;
      liftBtn.textContent = liftTarget ? 'Flatten to 2D' : 'Lift into 3D';
    }, { primary: true });
    ui.html(SPLIT_LEGEND);
    const use = predictor(ui, ['life_exp', 'hdi'], { title: 'Use the trained model (linear mode)' });

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
          const correct = { train: 0, test: 0 };
          for (let i = 0; i < N; i++) {
            const m = Y[i] * (w[0] * X[i][0] + w[1] * X[i][1] + b);
            if (m > 0) correct[IS_TEST[i] ? 'test' : 'train']++;
            // Support vectors exist only among the training points.
            const isSV = !IS_TEST[i] && m <= 1.02;
            if (isSV) sv++;
            setInstance(rings, i, X[i][0], X[i][1], 0, isSV ? 1 : 0);
          }
          rings.instanceMatrix.needsUpdate = true;
          const trAcc = correct.train / TRAIN_IDX.length;
          const teAcc = correct.test / TEST_IDX.length;
          stats.margin((2 / norm).toFixed(3));
          stats.sv(sv);
          stats.acc(`${pct(trAcc)} / ${pct(teAcc)}`);
          pipe(3, `Test accuracy ${pct(teAcc)} (train ${pct(trAcc)})`);
          const [ql, qh] = use.values();
          const score = w[0] * ql * S + w[1] * qh * S + b;
          marker.position.set(ql * S, qh * S, 0.3);
          marker.rotation.z += dt * 2;
          use.show(`score f(x) = ${score.toFixed(2)} → ${score >= 0 ? 'YES, likely ≥' : 'NO, likely <'} ${VAX_GOAL}% vaccinated${use.actual((c) => FIELDS.vax.fmt(c.vax))}`);
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
