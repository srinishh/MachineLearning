import * as THREE from 'three';

// Shared colour palette (cyan, pink, yellow, green, purple, orange).
export const PALETTE = [0x4cc9f0, 0xf72585, 0xffd166, 0x06d6a0, 0x9b5de5, 0xff8c42];
export const CSS_PALETTE = PALETTE.map((c) => '#' + c.toString(16).padStart(6, '0'));

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export const lerp = (a, b, t) => a + (b - a) * t;
export const smoothstep = (t) => {
  const x = clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
};
// Frame-rate independent exponential approach factor.
export const damp = (rate, dt) => 1 - Math.exp(-rate * dt);

// Seeded PRNG (mulberry32) so every lesson starts from reproducible data.
export function rng(seed = 1) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Standard normal sample (Box–Muller).
export function gauss(rand) {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

export function addLights(scene) {
  scene.add(new THREE.HemisphereLight(0xdbeafe, 0x1e1b4b, 1.4));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(6, 10, 8);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x88aaff, 0.6);
  rim.position.set(-8, -4, -6);
  scene.add(rim);
}

// Grid lying in the XY plane (for 2D-style lessons viewed from the front).
export function grid2D(size = 14, divisions = 14) {
  const group = new THREE.Group();
  const grid = new THREE.GridHelper(size, divisions, 0x334155, 0x1e293b);
  grid.rotation.x = Math.PI / 2;
  group.add(grid);
  const half = size / 2;
  const axes = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-half, 0, 0.001), new THREE.Vector3(half, 0, 0.001),
    new THREE.Vector3(0, -half, 0.001), new THREE.Vector3(0, half, 0.001),
  ]);
  group.add(new THREE.LineSegments(axes, new THREE.LineBasicMaterial({ color: 0x64748b })));
  return group;
}

// Grid lying on the XZ floor.
export function floorGrid(size = 14, divisions = 14) {
  return new THREE.GridHelper(size, divisions, 0x334155, 0x1e293b);
}

const _m = new THREE.Matrix4();
const _p = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _s = new THREE.Vector3();
const _c = new THREE.Color();

// Many spheres in one draw call; position/colour/scale them with setInstance*.
export function sphereCloud(count, radius = 0.15, { segments = 16, emissive = 0.15 } = {}) {
  const mesh = new THREE.InstancedMesh(
    new THREE.SphereGeometry(radius, segments, Math.max(8, Math.floor(segments * 0.75))),
    new THREE.MeshStandardMaterial({ roughness: 0.35, metalness: 0.1, emissive: 0xffffff, emissiveIntensity: emissive }),
    count,
  );
  mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  mesh.frustumCulled = false;
  for (let i = 0; i < count; i++) {
    setInstance(mesh, i, 0, 0, 0, 0);
    mesh.setColorAt(i, _c.set(0xffffff));
  }
  return mesh;
}

export function setInstance(mesh, i, x, y, z, scale = 1) {
  _p.set(x, y, z);
  _s.setScalar(scale);
  _m.compose(_p, _q.identity(), _s);
  mesh.setMatrixAt(i, _m);
}

export function setInstanceColor(mesh, i, color) {
  mesh.setColorAt(i, color instanceof THREE.Color ? color : _c.set(color));
}

export function commitInstances(mesh) {
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
}

// A unit cylinder that placeCylinder() can stretch between two points.
const unitCylGeo = () => new THREE.CylinderGeometry(1, 1, 1, 14);
export function unitCylinder(color, { emissive = 0.4, opacity = 1 } = {}) {
  return new THREE.Mesh(
    unitCylGeo(),
    new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: emissive,
      transparent: opacity < 1,
      opacity,
    }),
  );
}

const _up = new THREE.Vector3(0, 1, 0);
const _dir = new THREE.Vector3();
export function placeCylinder(mesh, a, b, radius = 0.05) {
  _dir.subVectors(b, a);
  const len = _dir.length() || 1e-6;
  mesh.position.addVectors(a, b).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(_up, _dir.divideScalar(len));
  mesh.scale.set(radius, len, radius);
}

// Camera-facing text label rendered to a canvas texture.
export function textSprite(text, { height = 0.45, color = '#e2e8f0', background = null, weight = 600 } = {}) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const fontPx = 64;
  const font = `${weight} ${fontPx}px Inter, system-ui, sans-serif`;
  ctx.font = font;
  const pad = background ? 28 : 12;
  const w = Math.ceil(ctx.measureText(text).width) + pad * 2;
  const h = fontPx + 32;
  canvas.width = w;
  canvas.height = h;
  if (background) {
    ctx.fillStyle = background;
    ctx.beginPath();
    ctx.roundRect(0, 0, w, h, 22);
    ctx.fill();
  }
  ctx.font = font;
  ctx.fillStyle = color;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, w / 2, h / 2 + 2);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
  sprite.scale.set((height * w) / h, height, 1);
  return sprite;
}

// Soft round dot texture for THREE.Points.
export function dotTexture() {
  const size = 64;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.35, 'rgba(255,255,255,0.8)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// Map t in [0,1] onto a deep-blue → cyan → yellow → pink ramp.
const RAMP = [0x1e1b4b, 0x2563eb, 0x4cc9f0, 0xffd166, 0xf72585].map((c) => new THREE.Color(c));
export function heat(t, out = new THREE.Color()) {
  const x = clamp(t, 0, 1) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x));
  return out.copy(RAMP[i]).lerp(RAMP[i + 1], x - i);
}

export function disposeObject(root) {
  root.traverse((obj) => {
    obj.geometry?.dispose();
    const mats = Array.isArray(obj.material) ? obj.material : obj.material ? [obj.material] : [];
    for (const m of mats) {
      m.map?.dispose();
      m.dispose();
    }
  });
}

// Remove and dispose every child of a group (used when rebuilding visuals).
export function clearGroup(group) {
  for (const child of [...group.children]) {
    group.remove(child);
    disposeObject(child);
  }
}
