import * as THREE from 'three';
import { grid2D, sphereCloud, setInstance, setInstanceColor, commitInstances, textSprite, PALETTE, damp } from '../helpers.js';
import { COUNTRIES, FIELDS, scaler, values, quantile } from '../data/covid.js';

const RES = 120;
const HALF = 6;
const S = 2.1; // display units per standard deviation
const NAMES = ['low', 'medium', 'high'];
const CLASS_COLORS = [PALETTE[0], PALETTE[2], PALETTE[1]];

export default {
  id: 'knn',
  title: 'k-Nearest Neighbors',
  category: 'Supervised Learning',
  camera: { position: [0, -5, 17], target: [0, 0, 0] },
  formula: 'ŷ(x) = majority vote of the k countries closest to x<br>distance = √(Δage² + Δlog cases²)   (standardised)',
  hint: 'Click anywhere to place a hypothetical country and see its nearest neighbours',
  info: `
    <h3>What it is</h3>
    <p>k-NN is the simplest classifier there is. There's no training step. To label something new, find the <strong>k</strong>
    most <strong>similar</strong> known examples and take a <strong>majority vote</strong>.</p>
    <h3>On the COVID data</h3>
    <p>Countries are split into three equal groups by deaths per million:
    <span style="color:#4cc9f0">low</span>, <span style="color:#ffd166">medium</span> and <span style="color:#f72585">high</span>.
    Each country is placed by <strong>median age</strong> (x) and <strong>cases per million</strong> (y, log), both standardised so
    neither dominates the distance.</p>
    <ul>
      <li>The white-ringed sphere is a <strong>query</strong>: a hypothetical country. Lines connect it to its k nearest real countries, and the panel names them.</li>
      <li>The tinted background is the <strong>decision map</strong>: the label k-NN would predict at every location.</li>
    </ul>
    <h3>Choosing k</h3>
    <ul>
      <li><strong>k = 1</strong>: islands form around single unusual countries, which is overfitting.</li>
      <li><strong>Large k</strong>: smooth regions, but small groups get outvoted, which is underfitting.</li>
    </ul>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>“Find the most similar cases” is everywhere. Health agencies compared their region with <strong>similar regions further
      along the epidemic</strong> to anticipate what was coming. Hospitals use patient-similarity search, and recommender systems
      suggest products bought by similar customers. k-NN is also used to <strong>fill in missing values</strong>: a country with no
      hospital-bed data can borrow the average of its nearest neighbours.</p>
    </div>
  `,
  create({ scene, ui, onClick }) {
    scene.add(grid2D(HALF * 2, 12));
    const age = scaler('median_age');
    const cases = scaler('cases_pm');
    const deaths = values('deaths_pm');
    const t1 = quantile(deaths, 1 / 3);
    const t2 = quantile(deaths, 2 / 3);
    const X = COUNTRIES.map((_, i) => [age.z[i] * S, cases.z[i] * S]);
    const Y = deaths.map((d) => (d < t1 ? 0 : d < t2 ? 1 : 2));
    const N = X.length;
    let k = 5;
    let auto = true;
    const query = [0, 0];
    const shownQuery = new THREE.Vector3();

    const pts = sphereCloud(N, 0.13);
    scene.add(pts);

    const map = new Uint8Array(RES * RES * 4);
    const tex = new THREE.DataTexture(map, RES, RES);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.magFilter = THREE.LinearFilter;
    tex.minFilter = THREE.LinearFilter;
    const bg = new THREE.Mesh(new THREE.PlaneGeometry(HALF * 2, HALF * 2), new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.38, depthWrite: false }));
    bg.position.z = -0.02;
    scene.add(bg);

    const tick = (text, x, y, color = '#94a3b8', height = 0.3) => {
      const s = textSprite(text, { height, color });
      s.position.set(x, y, 0);
      scene.add(s);
    };
    for (const zv of [-2, 0, 2]) {
      tick(age.format(age.fromZ(zv)), zv * S, -HALF - 0.45);
      tick(cases.format(cases.fromZ(zv)), -HALF - 1, zv * S);
    }
    tick('Median age →', 3.5, -HALF - 1.1, '#e2e8f0', 0.36);
    tick('Cases per million (log) ↑', -3.4, HALF + 0.5, '#e2e8f0', 0.36);

    const qMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.5 });
    const qMesh = new THREE.Mesh(new THREE.SphereGeometry(0.26, 24, 18), qMat);
    const qRing = new THREE.Mesh(new THREE.TorusGeometry(0.36, 0.04, 8, 40), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    scene.add(qMesh, qRing);

    const linePos = new Float32Array(15 * 6);
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.BufferAttribute(linePos, 3));
    const lines = new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.7 }));
    lines.frustumCulled = false;
    scene.add(lines);

    const circlePts = [];
    for (let i = 0; i <= 96; i++) circlePts.push(new THREE.Vector3(Math.cos((i / 96) * Math.PI * 2), Math.sin((i / 96) * Math.PI * 2), 0));
    const circle = new THREE.Line(new THREE.BufferGeometry().setFromPoints(circlePts), new THREE.LineDashedMaterial({ color: 0xffffff, dashSize: 0.04, gapSize: 0.04, transparent: true, opacity: 0.6 }));
    circle.computeLineDistances();
    scene.add(circle);

    const queryStat = ui.stat('Hypothetical country', { wide: true });
    const predStat = ui.stat('Predicted mortality');
    const voteStat = ui.stat('Votes (low/med/high)');
    const nbStat = ui.stat('Nearest neighbours', { wide: true });

    function neighbours(qx, qy, kk) {
      const d = X.map((p, i) => [(p[0] - qx) ** 2 + (p[1] - qy) ** 2, i]);
      d.sort((a, b) => a[0] - b[0]);
      return d.slice(0, kk);
    }

    function vote(nb) {
      const votes = [0, 0, 0];
      // Tiny rank bonus breaks ties toward the nearest neighbours.
      nb.forEach(([, i], r) => { votes[Y[i]] += 1 + (nb.length - r) * 1e-6; });
      return { cls: votes.indexOf(Math.max(...votes)), votes: votes.map(Math.floor) };
    }

    const colors = CLASS_COLORS.map((c) => new THREE.Color(c));
    function paintMap() {
      for (let j = 0; j < RES; j++) {
        for (let i = 0; i < RES; i++) {
          const x = -HALF + ((i + 0.5) / RES) * HALF * 2;
          const y = -HALF + ((j + 0.5) / RES) * HALF * 2;
          const c = colors[vote(neighbours(x, y, k)).cls];
          const o = (j * RES + i) * 4;
          map[o] = c.r * 255;
          map[o + 1] = c.g * 255;
          map[o + 2] = c.b * 255;
          map[o + 3] = 255;
        }
      }
      tex.needsUpdate = true;
    }

    X.forEach((_, i) => setInstanceColor(pts, i, CLASS_COLORS[Y[i]]));

    ui.slider({ label: 'k (neighbours)', min: 1, max: 15, step: 1, value: k, onInput: (v) => { k = v; paintMap(); } });
    const autoToggle = ui.toggle({ label: 'Move query automatically', value: true, onChange: (v) => { auto = v; } });
    ui.toggle({ label: 'Show decision map', value: true, onChange: (v) => { bg.visible = v; } });
    ui.html(`<div class="legend"><span style="--c:#4cc9f0">low &lt; ${Math.round(t1)}</span><span style="--c:#ffd166">medium</span><span style="--c:#f72585">high ≥ ${Math.round(t2)} deaths/M</span></div>`);

    onClick((ray) => {
      const hit = ray.intersectObject(bg)[0];
      if (!hit) return;
      query[0] = hit.point.x;
      query[1] = hit.point.y;
      auto = false;
      autoToggle.set(false);
    });

    paintMap();
    let t = 0;
    const qColor = new THREE.Color();
    const goal = new THREE.Vector3();

    return {
      update(dt) {
        t += dt;
        if (auto) {
          query[0] = 4.6 * Math.sin(t * 0.31);
          query[1] = 3.8 * Math.sin(t * 0.47 + 1);
        }
        shownQuery.lerp(goal.set(query[0], query[1], 0.05), damp(8, dt) || 1);
        const qx = shownQuery.x;
        const qy = shownQuery.y;
        qMesh.position.copy(shownQuery);
        qRing.position.copy(shownQuery);

        const nb = neighbours(qx, qy, k);
        const { cls, votes } = vote(nb);
        const near = new Set(nb.map(([, i]) => i));
        X.forEach((p, i) => setInstance(pts, i, p[0], p[1], 0, near.has(i) ? 1.5 : 1));
        commitInstances(pts);

        linePos.fill(0);
        nb.forEach(([, i], r) => linePos.set([qx, qy, 0.02, X[i][0], X[i][1], 0.02], r * 6));
        lineGeo.setDrawRange(0, nb.length * 2);
        lineGeo.attributes.position.needsUpdate = true;

        const radius = Math.sqrt(nb[nb.length - 1][0]);
        circle.position.set(qx, qy, 0.01);
        circle.scale.setScalar(Math.max(radius, 0.01));

        qColor.set(CLASS_COLORS[cls]);
        qMat.color.copy(qColor);
        qMat.emissive.copy(qColor);
        queryStat(`median age ${FIELDS.median_age.fmt(age.fromZ(qx / S))} · ${FIELDS.cases_pm.fmt(cases.fromZ(qy / S))} cases/M`);
        predStat(NAMES[cls]);
        voteStat(votes.join(' / '));
        nbStat(nb.slice(0, 5).map(([, i]) => COUNTRIES[i].name).join(', ') + (nb.length > 5 ? '…' : ''));
      },
    };
  },
};
