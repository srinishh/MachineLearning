import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { lessons } from './lessons/index.js';
import { addLights, disposeObject } from './helpers.js';

const $ = (sel) => document.querySelector(sel);

const canvas = $('#scene');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;

const BASE_FOV = 45;
const camera = new THREE.PerspectiveCamera(BASE_FOV, 1, 0.1, 500);
const orbit = new OrbitControls(camera, canvas);
orbit.enableDamping = true;
orbit.dampingFactor = 0.08;
orbit.maxDistance = 80;

let scene = new THREE.Scene();
let active = null;
let paused = false;
let speed = 1;
let clickHandler = null;

// The details panel overlaps the canvas, so shift the projection centre into
// the uncovered area (right-hand panel on desktop, bottom sheet on mobile).
function resize() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  const stage = canvas.getBoundingClientRect();
  const panel = $('#panel').getBoundingClientRect();
  const side = panel.height > stage.height * 0.6;
  const px = side ? Math.max(0, stage.right - panel.left) : 0;
  const py = side ? 0 : Math.max(0, stage.bottom - panel.top);
  // Lessons are framed for a desktop-sized uncovered area. When the uncovered
  // area is narrower or shorter, zoom out: widen the FOV a little, then back the
  // camera off for the rest so wide-angle distortion stays mild.
  const base = Math.tan(THREE.MathUtils.degToRad(BASE_FOV / 2));
  const need = Math.max(1, (0.93 * (h + py)) / (w - px), (0.8 * (h + py)) / (h - py));
  const widen = Math.min(need, 1.5);
  camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(base * widen));
  setFit(need / widen);
  camera.aspect = (w + px) / (h + py);
  camera.setViewOffset(w + px, h + py, px, py, w, h);
  camera.updateProjectionMatrix();
}

// Distance multiplier applied on top of each lesson's camera position.
let fit = 1;
function setFit(next) {
  if (Math.abs(next - fit) < 1e-3) return;
  camera.position.sub(orbit.target).multiplyScalar(next / fit).add(orbit.target);
  fit = next;
}
new ResizeObserver(resize).observe(canvas);
new ResizeObserver(resize).observe($('#panel'));

// ---------- Click-vs-drag detection, forwarded to the lesson as a raycaster ----------
const raycaster = new THREE.Raycaster();
let downAt = null;
canvas.addEventListener('pointerdown', (e) => { downAt = [e.clientX, e.clientY]; });
canvas.addEventListener('pointerup', (e) => {
  if (!downAt || !clickHandler) return;
  const moved = Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]);
  downAt = null;
  if (moved > 6) return;
  const rect = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((e.clientX - rect.left) / rect.width) * 2 - 1,
    -((e.clientY - rect.top) / rect.height) * 2 + 1,
  );
  raycaster.setFromCamera(ndc, camera);
  clickHandler(raycaster);
});

// ---------- Lesson UI builder ----------
function createUI(controlsEl, statsEl) {
  const row = () => {
    const last = controlsEl.lastElementChild;
    if (last?.classList.contains('btn-row')) return last;
    const el = document.createElement('div');
    el.className = 'btn-row';
    controlsEl.append(el);
    return el;
  };

  return {
    button(label, onClick, { primary = false } = {}) {
      const btn = document.createElement('button');
      btn.className = 'text-btn' + (primary ? ' primary' : '');
      btn.textContent = label;
      btn.addEventListener('click', onClick);
      row().append(btn);
      return btn;
    },

    slider({ label, min, max, step, value, onInput, format = (v) => v }) {
      const field = document.createElement('label');
      field.className = 'field';
      field.innerHTML = `<span class="field-head"><span>${label}</span><output></output></span>`;
      const input = document.createElement('input');
      Object.assign(input, { type: 'range', min, max, step, value });
      const out = field.querySelector('output');
      const sync = () => { out.textContent = format(Number(input.value)); };
      input.addEventListener('input', () => { sync(); onInput(Number(input.value)); });
      sync();
      field.append(input);
      controlsEl.append(field);
      return { set(v) { input.value = v; sync(); } };
    },

    select({ label, options, value, onChange }) {
      const field = document.createElement('label');
      field.className = 'field';
      field.innerHTML = `<span class="field-head"><span>${label}</span></span>`;
      const sel = document.createElement('select');
      for (const opt of options) {
        const o = document.createElement('option');
        o.value = opt.value;
        o.textContent = opt.label;
        sel.append(o);
      }
      sel.value = value;
      sel.addEventListener('change', () => onChange(sel.value));
      field.append(sel);
      controlsEl.append(field);
      return { set(v) { sel.value = v; } };
    },

    toggle({ label, value, onChange }) {
      const field = document.createElement('label');
      field.className = 'toggle';
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = value;
      input.addEventListener('change', () => onChange(input.checked));
      field.append(input, document.createTextNode(label));
      controlsEl.append(field);
      return { set(v) { input.checked = v; } };
    },

    stat(label, { wide = false } = {}) {
      const el = document.createElement('div');
      el.className = 'stat' + (wide ? ' wide' : '');
      el.innerHTML = `<div class="stat-label">${label}</div><div class="stat-value">–</div>`;
      statsEl.append(el);
      const value = el.querySelector('.stat-value');
      let last = null;
      return (text) => {
        const s = String(text);
        if (s !== last) value.textContent = last = s;
      };
    },

    // Free-form block (tables, small charts) placed under the controls.
    html(markup = '') {
      const el = document.createElement('div');
      el.className = 'extra';
      el.innerHTML = markup;
      controlsEl.append(el);
      return el;
    },
  };
}

// ---------- Navigation ----------
const nav = $('#nav');
const navButtons = new Map();
let lastCategory = null;
lessons.forEach((lesson, i) => {
  if (lesson.category !== lastCategory) {
    const title = document.createElement('div');
    title.className = 'nav-group-title';
    title.textContent = lesson.category;
    nav.append(title);
    lastCategory = lesson.category;
  }
  const btn = document.createElement('button');
  btn.className = 'nav-item';
  btn.innerHTML = `<span class="num">${String(i).padStart(2, '0')}</span><span>${lesson.title}</span>`;
  btn.addEventListener('click', () => {
    location.hash = lesson.id;
    document.body.classList.remove('nav-open');
  });
  nav.append(btn);
  navButtons.set(lesson.id, btn);
});

function setCamera(position, target = [0, 0, 0]) {
  orbit.target.set(...target);
  camera.position.set(...position).sub(orbit.target).multiplyScalar(fit).add(orbit.target);
  orbit.update();
}

function load(lesson) {
  active?.instance?.dispose?.();
  disposeObject(scene);
  scene = new THREE.Scene();
  addLights(scene);
  clickHandler = null;

  $('#lesson-title').textContent = lesson.title;
  $('#lesson-category').textContent = lesson.category;
  $('#lesson-formula').innerHTML = lesson.formula ?? '';
  $('#lesson-info').innerHTML = lesson.info ?? '';
  $('#hint').textContent = lesson.hint ?? '';
  const controlsEl = $('#lesson-controls');
  const statsEl = $('#lesson-stats');
  controlsEl.replaceChildren();
  statsEl.replaceChildren();
  $('.panel-body').scrollTop = 0;
  document.title = `${lesson.title} · ML in Motion`;

  for (const [id, btn] of navButtons) btn.classList.toggle('active', id === lesson.id);
  navButtons.get(lesson.id)?.scrollIntoView({ block: 'nearest' });

  const cam = lesson.camera ?? {};
  setCamera(cam.position ?? [0, 0, 16], cam.target ?? [0, 0, 0]);

  const ctx = {
    scene,
    camera,
    orbit,
    ui: createUI(controlsEl, statsEl),
    setCamera,
    onClick: (fn) => { clickHandler = fn; },
  };
  active = { lesson, instance: lesson.create(ctx) ?? {} };
}

function route() {
  const id = location.hash.slice(1);
  const lesson = lessons.find((l) => l.id === id) ?? lessons[0];
  if (lesson !== active?.lesson) load(lesson);
}

function step(delta) {
  const i = lessons.indexOf(active.lesson);
  const next = lessons[(i + delta + lessons.length) % lessons.length];
  location.hash = next.id;
}

window.addEventListener('hashchange', route);

// ---------- Global controls ----------
const playBtn = $('#play-btn');
function setPaused(p) {
  paused = p;
  playBtn.textContent = p ? '▶' : '❚❚';
  playBtn.setAttribute('aria-label', p ? 'Play' : 'Pause');
}
playBtn.addEventListener('click', () => setPaused(!paused));

$('#speed').addEventListener('input', (e) => {
  speed = Number(e.target.value);
  $('#speed-out').textContent = `${speed}×`;
});

$('#cam-btn').addEventListener('click', () => {
  const cam = active.lesson.camera ?? {};
  setCamera(cam.position ?? [0, 0, 16], cam.target ?? [0, 0, 0]);
});

$('#prev-btn').addEventListener('click', () => step(-1));
$('#next-btn').addEventListener('click', () => step(1));
$('#menu-btn').addEventListener('click', () => document.body.classList.toggle('nav-open'));
$('#panel-toggle').addEventListener('click', () => document.body.classList.toggle('panel-collapsed'));

window.addEventListener('keydown', (e) => {
  if (e.target.closest?.('input, select, textarea')) return;
  if (e.code === 'Space') { e.preventDefault(); setPaused(!paused); }
  if (e.key === 'ArrowRight') step(1);
  if (e.key === 'ArrowLeft') step(-1);
});

// ---------- Render loop ----------
const clock = new THREE.Clock();
let elapsed = 0;
route();
resize();
renderer.setAnimationLoop(() => {
  const raw = Math.min(clock.getDelta(), 0.05);
  const dt = paused ? 0 : raw * speed;
  elapsed += dt;
  active?.instance?.update?.(dt, elapsed);
  orbit.update();
  renderer.render(scene, camera);
});
