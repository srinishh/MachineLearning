import * as THREE from 'three';
import { rng, gauss, grid2D, sphereCloud, setInstance, setInstanceColor, commitInstances, PALETTE, damp } from '../helpers.js';

const RES = 120;
const HALF = 6;
const NAMES = ['cyan', 'pink', 'yellow'];

export default {
  id: 'knn',
  title: 'k-Nearest Neighbors',
  category: 'Supervised Learning',
  camera: { position: [0, -5, 17], target: [0, 0, 0] },
  formula: 'ŷ(x) = majority vote of the k training points closest to x<br>distance = √((x₁−x₁′)² + (x₂−x₂′)²)',
  hint: 'Click anywhere on the plane to place the query point',
  info: `
    <h3>What it is</h3>
    <p>k-NN is the simplest classifier there is. There's no training step. To label a new point, find the
    <strong>k</strong> stored examples <strong>closest</strong> to it and take a <strong>majority
    vote</strong>.</p>
    <h3>Reading the animation</h3>
    <ul>
      <li>The white-ringed sphere is a <strong>query</strong> point. Lines connect it to its k nearest neighbours, and the circle shows how far away the k-th neighbour is.</li>
      <li>The query takes the colour of the winning class.</li>
      <li>The tinted background is the <strong>decision map</strong>: the label k-NN would give at every location.</li>
    </ul>
    <h3>Choosing k</h3>
    <ul>
      <li><strong>k = 1</strong>: very jagged boundaries that react to every noisy point, which is overfitting.</li>
      <li><strong>Large k</strong>: smooth boundaries, but small groups get outvoted, which is underfitting.</li>
    </ul>
    <p>Change k and watch the decision map reshape.</p>
  `,
  create({ scene, ui, onClick }) {
    scene.add(grid2D(HALF * 2, 12));
    const rand = rng(31);
    const centers = [[-2.5, 1.8], [2.6, 1.6], [0.2, -2.6]];
    const X = [];
    const Y = [];
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < 28; i++) {
        X.push([centers[c][0] + gauss(rand) * 1.4, centers[c][1] + gauss(rand) * 1.4]);
        Y.push(c);
      }
    }
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
    const bg = new THREE.Mesh(new THREE.PlaneGeometry(HALF * 2, HALF * 2), new THREE.MeshBasicMaterial({ map: tex, transparent: true, opacity: 0.4, depthWrite: false }));
    bg.position.z = -0.02;
    scene.add(bg);

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

    const predStat = ui.stat('Prediction');
    const voteStat = ui.stat('Votes');

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

    const colors = PALETTE.slice(0, 3).map((c) => new THREE.Color(c));
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

    X.forEach((p, i) => setInstanceColor(pts, i, PALETTE[Y[i]]));

    ui.slider({ label: 'k (neighbours)', min: 1, max: 15, step: 1, value: k, onInput: (v) => { k = v; paintMap(); } });
    const autoToggle = ui.toggle({ label: 'Move query automatically', value: true, onChange: (v) => { auto = v; } });
    ui.toggle({ label: 'Show decision map', value: true, onChange: (v) => { bg.visible = v; } });

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

    return {
      update(dt) {
        t += dt;
        if (auto) {
          query[0] = 4.6 * Math.sin(t * 0.31);
          query[1] = 3.8 * Math.sin(t * 0.47 + 1);
        }
        shownQuery.lerp(new THREE.Vector3(query[0], query[1], 0.05), damp(8, dt) || 1);
        const qx = shownQuery.x;
        const qy = shownQuery.y;
        qMesh.position.copy(shownQuery);
        qRing.position.copy(shownQuery);

        const nb = neighbours(qx, qy, k);
        const { cls, votes } = vote(nb);
        const near = new Set(nb.map(([, i]) => i));
        X.forEach((p, i) => setInstance(pts, i, p[0], p[1], 0, near.has(i) ? 1.45 : 1));
        commitInstances(pts);

        linePos.fill(0);
        nb.forEach(([, i], r) => linePos.set([qx, qy, 0.02, X[i][0], X[i][1], 0.02], r * 6));
        lineGeo.setDrawRange(0, nb.length * 2);
        lineGeo.attributes.position.needsUpdate = true;

        const radius = Math.sqrt(nb[nb.length - 1][0]);
        circle.position.set(qx, qy, 0.01);
        circle.scale.setScalar(Math.max(radius, 0.01));

        qColor.set(PALETTE[cls]);
        qMat.color.copy(qColor);
        qMat.emissive.copy(qColor);
        predStat(NAMES[cls]);
        voteStat(votes.map((v, i) => `${NAMES[i][0].toUpperCase()}:${v}`).join(' '));
      },
    };
  },
};
