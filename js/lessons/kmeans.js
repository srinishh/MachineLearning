import * as THREE from 'three';
import { rng, gauss, sphereCloud, setInstance, setInstanceColor, commitInstances, PALETTE, smoothstep } from '../helpers.js';

const STEP = 1.1; // seconds per assign / update phase
const COLORS = [...PALETTE, 0xe2e8f0, 0x22d3ee];

export default {
  id: 'kmeans',
  title: 'k-Means Clustering',
  category: 'Unsupervised Learning',
  camera: { position: [12, 8, 14], target: [0, 0, 0] },
  formula: '1. Assign: cᵢ = argminⱼ ‖xᵢ − μⱼ‖²<br>2. Update: μⱼ = mean of points assigned to j<br>Inertia = Σ ‖xᵢ − μ_cᵢ‖²',
  hint: 'Octahedra are centroids; lines show which centroid each point belongs to',
  info: `
    <h3>What it is</h3>
    <p>k-means groups <strong>unlabelled</strong> data into <strong>k clusters</strong>. Nobody tells it what
    the groups are; it finds them from the geometry of the data alone.</p>
    <h3>The algorithm</h3>
    <ol>
      <li><strong>Initialise</strong>: place k centroids, here at random data points.</li>
      <li><strong>Assign</strong>: colour each point by its nearest centroid.</li>
      <li><strong>Update</strong>: move each centroid to the average position of its points.</li>
      <li>Repeat until no point changes cluster. The algorithm has <strong>converged</strong>.</li>
    </ol>
    <p>Each round can only lower the <strong>inertia</strong> (total squared distance to centroids), so
    k-means always converges, though not always to the best answer.</p>
    <h3>Try this</h3>
    <ul>
      <li>Press <strong>Re-initialise</strong> a few times. Unlucky starting positions can merge two real clusters or split one, which is why k-means is usually run several times.</li>
      <li>Change <strong>k</strong>. The data has 5 natural groups, so see what happens with 3 or 8.</li>
    </ul>
  `,
  create({ scene, ui }) {
    const rand = rng(61);
    const N = 320;
    let X = [];
    let k = 5;
    let centroids = [];
    let from = [];
    let assign = new Int32Array(N).fill(-1);
    let phase = 'assign';
    let timer = 0;
    let iter = 0;
    let converged = false;

    const pts = sphereCloud(N, 0.11);
    scene.add(pts);

    const cGroup = new THREE.Group();
    scene.add(cGroup);
    const cMeshes = [];

    const linePos = new Float32Array(N * 6);
    const lineCol = new Float32Array(N * 6);
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.BufferAttribute(linePos, 3));
    lineGeo.setAttribute('color', new THREE.BufferAttribute(lineCol, 3));
    const lines = new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.22 }));
    lines.frustumCulled = false;
    scene.add(lines);

    const box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(14, 14, 14)), new THREE.LineBasicMaterial({ color: 0x334155 }));
    scene.add(box);

    const stats = {
      iter: ui.stat('Iteration'),
      inertia: ui.stat('Inertia'),
      status: ui.stat('Status', { wide: true }),
    };

    function newData() {
      X = [];
      const centers = Array.from({ length: 5 }, () => [rand() * 9 - 4.5, rand() * 9 - 4.5, rand() * 9 - 4.5]);
      for (let i = 0; i < N; i++) {
        const c = centers[i % 5];
        X.push(c.map((v) => v + gauss(rand) * 0.95));
      }
      init();
    }

    function init() {
      const picks = new Set();
      while (picks.size < k) picks.add(Math.floor(rand() * N));
      centroids = [...picks].map((i) => [...X[i]]);
      from = centroids.map((c) => [...c]);
      assign = new Int32Array(N).fill(-1);
      phase = 'assign';
      timer = 0;
      iter = 0;
      converged = false;
      for (const m of cMeshes) {
        cGroup.remove(m);
        m.geometry.dispose();
        m.material.dispose();
      }
      cMeshes.length = 0;
      for (let j = 0; j < k; j++) {
        const m = new THREE.Mesh(
          new THREE.OctahedronGeometry(0.42),
          new THREE.MeshStandardMaterial({ color: COLORS[j], emissive: COLORS[j], emissiveIntensity: 0.7, flatShading: true }),
        );
        cGroup.add(m);
        cMeshes.push(m);
      }
      X.forEach((p, i) => { setInstance(pts, i, ...p); setInstanceColor(pts, i, 0x64748b); });
      commitInstances(pts);
    }

    function doAssign() {
      let changed = 0;
      for (let i = 0; i < N; i++) {
        let best = 0;
        let bd = Infinity;
        for (let j = 0; j < k; j++) {
          const d = (X[i][0] - centroids[j][0]) ** 2 + (X[i][1] - centroids[j][1]) ** 2 + (X[i][2] - centroids[j][2]) ** 2;
          if (d < bd) { bd = d; best = j; }
        }
        if (assign[i] !== best) changed++;
        assign[i] = best;
        setInstanceColor(pts, i, COLORS[best]);
      }
      commitInstances(pts);
      return changed;
    }

    function doUpdate() {
      from = centroids.map((c) => [...c]);
      const sums = Array.from({ length: k }, () => [0, 0, 0, 0]);
      for (let i = 0; i < N; i++) {
        const s = sums[assign[i]];
        s[0] += X[i][0];
        s[1] += X[i][1];
        s[2] += X[i][2];
        s[3]++;
      }
      // Empty clusters keep their old position.
      centroids = sums.map((s, j) => (s[3] ? [s[0] / s[3], s[1] / s[3], s[2] / s[3]] : centroids[j]));
    }

    const inertia = () => X.reduce((s, p, i) => {
      if (assign[i] < 0) return s;
      const c = centroids[assign[i]];
      return s + (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2;
    }, 0);

    ui.slider({ label: 'Number of clusters k', min: 2, max: 8, step: 1, value: k, onInput: (v) => { k = v; init(); } });
    ui.button('Re-initialise', init, { primary: true });
    ui.button('New data', newData);

    newData();
    const col = new THREE.Color();

    return {
      update(dt, time) {
        timer += dt;
        if (!converged && timer >= STEP) {
          timer = 0;
          if (phase === 'assign') {
            const changed = doAssign();
            if (changed === 0 && iter > 0) converged = true;
            phase = 'update';
          } else {
            doUpdate();
            iter++;
            phase = 'assign';
          }
        }
        if (converged && timer > 4) init();

        // Centroids glide from their previous to their new position.
        const s = phase === 'assign' ? smoothstep(timer / (STEP * 0.8)) : 1;
        cMeshes.forEach((m, j) => {
          const a = from[j] ?? centroids[j];
          const b = centroids[j];
          m.position.set(a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, a[2] + (b[2] - a[2]) * s);
          m.rotation.y = time * 1.5;
          m.rotation.x = time * 0.7;
        });

        for (let i = 0; i < N; i++) {
          if (assign[i] < 0) {
            linePos.fill(0, i * 6, i * 6 + 6);
            continue;
          }
          const m = cMeshes[assign[i]].position;
          linePos.set([X[i][0], X[i][1], X[i][2], m.x, m.y, m.z], i * 6);
          col.set(COLORS[assign[i]]);
          lineCol.set([col.r, col.g, col.b, col.r, col.g, col.b], i * 6);
        }
        lineGeo.attributes.position.needsUpdate = true;
        lineGeo.attributes.color.needsUpdate = true;

        stats.iter(iter);
        stats.inertia(inertia().toFixed(1));
        stats.status(converged ? 'Converged ✓ (assignments stopped changing)' : iter === 0 && phase === 'assign' ? `Initialised ${k} random centroids` : phase === 'assign' ? 'Moving centroids to cluster means…' : 'Assigning points to nearest centroid…');
      },
    };
  },
};
