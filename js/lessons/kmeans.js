import * as THREE from 'three';
import { rng, sphereCloud, setInstance, setInstanceColor, commitInstances, textSprite, PALETTE, smoothstep } from '../helpers.js';
import { COUNTRIES, FIELDS, scaler } from '../data/covid.js';

const STEP = 1.1; // seconds per assign / update phase
const COLORS = [...PALETTE, 0xe2e8f0, 0x22d3ee];
const S = 1.9; // display units per standard deviation
const FEATS = ['cases_pm', 'deaths_pm', 'vax'];
const hex = (c) => '#' + c.toString(16).padStart(6, '0');

export default {
  id: 'kmeans',
  title: 'k-Means Clustering',
  category: 'Unsupervised Learning',
  camera: { position: [12, 8, 14], target: [0, 0, 0] },
  formula: '1. Assign: cᵢ = argminⱼ ‖xᵢ − μⱼ‖²<br>2. Update: μⱼ = mean of points assigned to j<br>Inertia = Σ ‖xᵢ − μ_cᵢ‖²',
  goal: 'A <strong>group number for every country</strong> (no labels needed) plus k centre points that summarise each group, such as “high cases, high deaths, well vaccinated”.',
  usage: 'To place a new country, measure its distance to each centre and assign it to the nearest one. The cluster profiles describe what that group means, which is useful for comparing similar countries or targeting policies.',
  hint: 'Each dot is a country · Octahedra are cluster centres · Click a dot to see its cluster',
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
    <h3>On the COVID data</h3>
    <p>Each country is placed in 3D by its pandemic profile: <strong>cases per million</strong> (x, log),
    <strong>deaths per million</strong> (y, log) and <strong>% fully vaccinated</strong> (z), all standardised. No labels are
    given. k-means discovers groups such as “high cases, high deaths, well vaccinated” (much of Europe and the Americas) or
    “few reported cases, low vaccination” (much of sub-Saharan Africa, where testing was limited). The <em>Cluster profiles</em>
    box describes each group in real units once the algorithm converges.</p>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>Clustering finds structure nobody labelled. During COVID, researchers clustered hospital patients by symptoms and lab
      results and found distinct <strong>clinical phenotypes</strong> with different risks and treatment responses. Public-health
      teams clustered regions with similar epidemic trajectories to compare policies. Businesses use the same method for
      customer segmentation.</p>
    </div>
    <h3>Try this</h3>
    <ul>
      <li>Press <strong>Re-initialise</strong> a few times. Unlucky starting positions can give different groupings, which is why k-means is usually run several times.</li>
      <li>Change <strong>k</strong>. Real data rarely has an obvious “true” number of clusters. Watch how inertia falls as k grows, and look for the “elbow”.</li>
    </ul>
  `,
  create({ scene, ui, onClick }) {
    const rand = rng(61);
    const scalers = FEATS.map((f) => scaler(f));
    const X = COUNTRIES.map((_, i) => scalers.map((s) => s.z[i] * S));
    const N = X.length;
    let k = 4;
    let centroids = [];
    let from = [];
    let assign = new Int32Array(N).fill(-1);
    let phase = 'assign';
    let timer = 0;
    let iter = 0;
    let converged = false;

    const pts = sphereCloud(N, 0.14);
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

    const axisLabel = (text, x, y, z) => {
      const s = textSprite(text, { height: 0.4, color: '#e2e8f0' });
      s.position.set(x, y, z);
      scene.add(s);
    };
    axisLabel('Cases/M (log) →', 7.8, -7, -7);
    axisLabel('Deaths/M (log) ↑', -7, 7.6, -7);
    axisLabel('% vaccinated →', -7, -7, 7.8);
    for (const name of ['United States', 'India', 'Nigeria', 'Japan', 'Peru', 'Germany', 'China']) {
      const i = COUNTRIES.findIndex((c) => c.name === name);
      if (i < 0) continue;
      const s = textSprite(name, { height: 0.3, color: '#cbd5e1' });
      s.position.set(X[i][0], X[i][1] + 0.4, X[i][2]);
      scene.add(s);
    }

    const profiles = ui.html('');
    const pickStat = ui.stat('Selected country', { wide: true });
    pickStat('Click a dot');
    onClick((ray) => {
      const hit = ray.intersectObject(pts)[0];
      if (hit?.instanceId === undefined) return;
      const c = COUNTRIES[hit.instanceId];
      const a = assign[hit.instanceId];
      pickStat(`${c.name} → cluster ${a >= 0 ? a + 1 : '–'} · ${FIELDS.cases_pm.fmt(c.cases_pm)} cases/M · ${FIELDS.deaths_pm.fmt(c.deaths_pm)} deaths/M · ${FIELDS.vax.fmt(c.vax)} vaccinated`);
    });

    function showProfiles() {
      const rows = [];
      for (let j = 0; j < k; j++) {
        const members = COUNTRIES.filter((_, i) => assign[i] === j);
        if (!members.length) continue;
        const avg = (f) => members.reduce((s, c) => s + c[f], 0) / members.length;
        const examples = [...members].sort((a, b) => b.population - a.population).slice(0, 3).map((c) => c.name).join(', ');
        rows.push(`<li><span style="color:${hex(COLORS[j])}">●</span> <strong>Cluster ${j + 1}</strong> (${members.length}): ~${FIELDS.cases_pm.fmt(avg('cases_pm'))} cases/M, ${FIELDS.deaths_pm.fmt(avg('deaths_pm'))} deaths/M, ${FIELDS.vax.fmt(avg('vax'))} vaccinated · e.g. ${examples}</li>`);
      }
      profiles.innerHTML = `<div class="field-head"><span>Cluster profiles (averages)</span></div><ul class="profiles">${rows.join('')}</ul>`;
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

    init();
    const col = new THREE.Color();

    return {
      update(dt, time) {
        timer += dt;
        if (!converged && timer >= STEP) {
          timer = 0;
          if (phase === 'assign') {
            const changed = doAssign();
            showProfiles();
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
