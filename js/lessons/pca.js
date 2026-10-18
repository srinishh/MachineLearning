import * as THREE from 'three';
import { rng, gauss, sphereCloud, setInstance, setInstanceColor, commitInstances, textSprite, heat, damp, clearGroup, PALETTE } from '../helpers.js';

// Jacobi eigen-decomposition of a symmetric 3×3 matrix.
export function eigSym3(A) {
  const a = A.map((r) => r.slice());
  const V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let sweep = 0; sweep < 50; sweep++) {
    let off = 0;
    for (let p = 0; p < 2; p++) for (let q = p + 1; q < 3; q++) off += a[p][q] ** 2;
    if (off < 1e-18) break;
    for (let p = 0; p < 2; p++) {
      for (let q = p + 1; q < 3; q++) {
        if (Math.abs(a[p][q]) < 1e-15) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 3; k++) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 3; k++) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 3; k++) {
          const vkp = V[k][p];
          const vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  return [0, 1, 2]
    .map((i) => ({ value: a[i][i], vector: [V[0][i], V[1][i], V[2][i]] }))
    .sort((x, y) => y.value - x.value);
}

const MODES = ['Original 3D data', 'Projected onto 2 PCs', 'Projected onto 1 PC'];

export default {
  id: 'pca',
  title: 'Principal Component Analysis',
  category: 'Unsupervised Learning',
  camera: { position: [11, 7, 13], target: [0, 0, 0] },
  formula: 'Σ = ¹⁄ₙ Σ (xᵢ − μ)(xᵢ − μ)ᵀ<br>Σ·vₖ = λₖ·vₖ<br>zᵢ = Vₖᵀ (xᵢ − μ)',
  hint: 'Arrows are principal components, scaled by the spread along each one',
  info: `
    <h3>What it is</h3>
    <p>PCA is the classic <strong>dimensionality reduction</strong> method. Real data often has many
    features, but most of the variation lies along a few directions. PCA finds those directions, the
    <strong>principal components</strong>, so you can drop the rest and lose as little information as possible.</p>
    <h3>How it works</h3>
    <ol>
      <li>Centre the data by subtracting the mean.</li>
      <li>Compute the <strong>covariance matrix</strong> Σ.</li>
      <li>Its <strong>eigenvectors</strong> are the principal directions, and the <strong>eigenvalues</strong> are the variance along each one.</li>
      <li>Project onto the top components.</li>
    </ol>
    <h3>Reading the animation</h3>
    <p>The cloud is a tilted, flattened 3D blob. Colour follows each point's PC1 coordinate, so you can see
    the order is kept after projection. Projecting onto 2 PCs flattens the data onto the plane, and almost
    no structure is lost (look at <em>variance retained</em>). Projecting onto 1 PC squeezes everything onto a line.</p>
    <h3>Used for</h3>
    <p>Visualising high-dimensional data, compression, denoising, and speeding up other models.</p>
  `,
  create({ scene, ui }) {
    const rand = rng(71);
    const N = 450;
    const pts = sphereCloud(N, 0.09, { segments: 10 });
    scene.add(pts);
    const extras = new THREE.Group();
    scene.add(extras);

    let X = [];
    let mean = [0, 0, 0];
    let eig = [];
    let cur = [];
    let mode = 0;
    let auto = true;
    let timer = 0;

    const lineGeo = new THREE.BufferGeometry();
    const linePos = new Float32Array(N * 6);
    lineGeo.setAttribute('position', new THREE.BufferAttribute(linePos, 3));
    const projLines = new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({ color: 0x94a3b8, transparent: true, opacity: 0 }));
    projLines.frustumCulled = false;
    scene.add(projLines);

    const plane = new THREE.Mesh(new THREE.PlaneGeometry(16, 7), new THREE.MeshBasicMaterial({ color: PALETTE[2], transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
    scene.add(plane);

    const stats = {
      ev: ui.stat('Explained variance (PC1/2/3)', { wide: true }),
      kept: ui.stat('Variance retained'),
      mode: ui.stat('View'),
    };

    function newData() {
      const rot = new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(rand() * 2 - 1, rand() * 3, rand() * 2 - 1));
      const v = new THREE.Vector3();
      X = [];
      for (let i = 0; i < N; i++) {
        v.set(gauss(rand) * 3.2, gauss(rand) * 1.3, gauss(rand) * 0.35).applyMatrix4(rot);
        X.push([v.x, v.y, v.z]);
      }
      mean = [0, 1, 2].map((d) => X.reduce((s, p) => s + p[d], 0) / N);
      const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
      for (const p of X) {
        for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) C[r][c] += ((p[r] - mean[r]) * (p[c] - mean[c])) / N;
      }
      eig = eigSym3(C);
      cur = X.map((p) => [...p]);

      const lo = -3 * Math.sqrt(eig[0].value);
      const hi = 3 * Math.sqrt(eig[0].value);
      const col = new THREE.Color();
      X.forEach((p, i) => {
        const z1 = dot(sub(p, mean), eig[0].vector);
        setInstanceColor(pts, i, heat(0.1 + 0.9 * ((z1 - lo) / (hi - lo)), col));
      });

      // Arrows and labels for each component.
      clearGroup(extras);
      const total = eig.reduce((s, e) => s + e.value, 0);
      const origin = new THREE.Vector3(...mean);
      eig.forEach((e, i) => {
        const dir = new THREE.Vector3(...e.vector);
        const len = 2.2 * Math.sqrt(e.value) + 0.4;
        const color = [PALETTE[2], PALETTE[3], PALETTE[1]][i];
        extras.add(new THREE.ArrowHelper(dir, origin, len, color, 0.45, 0.28));
        const label = textSprite(`PC${i + 1} · ${((100 * e.value) / total).toFixed(1)}%`, { height: 0.42, background: 'rgba(2,6,23,0.7)', color: '#' + color.toString(16).padStart(6, '0') });
        label.position.copy(origin).addScaledVector(dir, len + 0.6);
        extras.add(label);
      });
      stats.ev(eig.map((e) => `${((100 * e.value) / total).toFixed(1)}%`).join(' / '));

      const basis = new THREE.Matrix4().makeBasis(
        new THREE.Vector3(...eig[0].vector),
        new THREE.Vector3(...eig[1].vector),
        new THREE.Vector3(...eig[2].vector),
      );
      plane.quaternion.setFromRotationMatrix(basis);
      plane.position.copy(origin);
      setMode(0);
    }

    function setMode(m) {
      mode = m;
      timer = 0;
      const total = eig.reduce((s, e) => s + e.value, 0);
      const kept = eig.slice(0, 3 - m).reduce((s, e) => s + e.value, 0);
      stats.kept(`${((100 * kept) / total).toFixed(1)}%`);
      stats.mode(MODES[m]);
    }

    function target(p) {
      if (mode === 0) return p;
      const d = sub(p, mean);
      const out = [...mean];
      for (let k = 0; k < 3 - mode; k++) {
        const z = dot(d, eig[k].vector);
        for (let j = 0; j < 3; j++) out[j] += z * eig[k].vector[j];
      }
      return out;
    }

    ui.button('Original 3D', () => setMode(0), { primary: true });
    ui.button('Project → 2D', () => setMode(1));
    ui.button('Project → 1D', () => setMode(2));
    ui.button('New data', newData);
    ui.toggle({ label: 'Auto-play through projections', value: true, onChange: (v) => { auto = v; } });

    newData();

    return {
      update(dt) {
        timer += dt;
        if (auto && timer > 4) setMode((mode + 1) % 3);
        const k = damp(3.5, dt);
        for (let i = 0; i < N; i++) {
          const tgt = target(X[i]);
          for (let j = 0; j < 3; j++) cur[i][j] += (tgt[j] - cur[i][j]) * k;
          setInstance(pts, i, cur[i][0], cur[i][1], cur[i][2]);
          linePos.set([...X[i], ...cur[i]], i * 6);
        }
        commitInstances(pts);
        lineGeo.attributes.position.needsUpdate = true;
        const lineGoal = mode ? 0.18 : 0;
        projLines.material.opacity += (lineGoal - projLines.material.opacity) * k;
        const planeGoal = mode === 1 ? 0.14 : 0;
        plane.material.opacity += (planeGoal - plane.material.opacity) * k;
      },
    };
  },
};

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
