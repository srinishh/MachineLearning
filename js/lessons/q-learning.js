import * as THREE from 'three';
import { rng, textSprite, PALETTE, damp } from '../helpers.js';

const MAP = [
  'S..#...',
  '..##.P.',
  '......#',
  'P.#....',
  '..#.P..',
  '...#...',
  '.P....G',
];
const SIZE = MAP.length;
const CELL = 1.25;
const ACTIONS = [[-1, 0], [0, 1], [1, 0], [0, -1]]; // up, right, down, left
const ANGLES = [Math.PI, Math.PI / 2, 0, -Math.PI / 2];
const GAMMA = 0.95;
const MAX_STEPS = 80;

const cell = (r, c) => MAP[r][c];
const toXZ = (r, c) => [(c - (SIZE - 1) / 2) * CELL, (r - (SIZE - 1) / 2) * CELL];

export default {
  id: 'q-learning',
  title: 'Reinforcement Learning (Q-Learning)',
  category: 'Reinforcement Learning',
  camera: { position: [0, 11, 9.5], target: [0, 0, 0.4] },
  formula: 'Q(s,a) ← Q(s,a) + α·[ r + γ·maxₐ′ Q(s′,a′) − Q(s,a) ]<br>action = random with prob. ε, otherwise argmaxₐ Q(s,a)',
  goal: 'A <strong>policy</strong>: the best move in every grid cell (the arrows), learned purely from rewards so the van reaches the clinic quickly and avoids hotspots.',
  usage: 'After training, set ε to 0 and the agent simply looks up the action with the highest Q-value in its current cell. That lookup table is the trained model, used to drive the van from any starting point.',
  hint: 'The delivery van learns a route to the vaccine clinic · Tiles turn green as it learns which streets lead there',
  info: `
    <h3>What it is</h3>
    <p>In <strong>reinforcement learning</strong> nobody gives the agent the right answers. It
    <strong>acts</strong>, receives <strong>rewards</strong>, and learns from experience which actions pay off
    in the long run.</p>
    <h3>The COVID scenario</h3>
    <p>A <strong>vaccine-delivery van</strong> (the blue ball) starts at the depot in the top corner and must find its way across town.
    It has no map, only experience.</p>
    <ul>
      <li>Reaching the gold <strong>vaccine clinic</strong> gives <code>+10</code>.</li>
      <li>Driving into a red <strong>outbreak hotspot</strong> gives <code>−10</code>: the crew must quarantine and the run fails.</li>
      <li>Each step costs <code>−0.1</code> (fuel and time; vaccines must stay cold), so shorter routes are better. Grey blocks are closed roads.</li>
    </ul>
    <h3>Q-learning</h3>
    <p>The agent keeps a table <code>Q(s,a)</code>: its estimate of the total future reward for taking action
    <code>a</code> in state <code>s</code>. After every step it updates the estimate toward the reward it got
    plus the discounted value of the next state. Rewards slowly <strong>propagate backwards</strong> from the
    goal, and you can watch the green spread across the grid.</p>
    <h3>Exploration vs exploitation</h3>
    <p><strong>ε</strong> is the chance of taking a random action. With too little exploration the agent may
    never discover the goal, and with too much it never makes use of what it learned. The arrows show the
    current best action in each cell, which is the <strong>policy</strong>.</p>
    <div class="real-life">
      <h3>Real-life scenario</h3>
      <p>In summer 2020 Greece deployed <strong>“Eva”</strong>, a reinforcement-learning system (a contextual bandit) that decided
      which arriving travellers to test for COVID at the border. It balanced <em>exploring</em> under-sampled countries with
      <em>exploiting</em> what it had learned about high-risk ones. A study in <em>Nature</em> (Bastani et al., 2021) estimated it
      found nearly twice as many infected travellers as random testing would have. RL is also used for routing deliveries, managing
      power grids and controlling robots.</p>
    </div>
    <h3>Try this</h3>
    <p>Press <strong>Train 200 episodes</strong> to fast-forward, then set ε to 0 and watch the van follow its learned route.</p>
  `,
  create({ scene, ui }) {
    const rand = rng(91);
    const Q = new Float32Array(SIZE * SIZE * 4);
    let alpha = 0.4;
    let eps = 0.2;
    let state = [0, 0];
    let steps = 0;
    let episode = 0;
    let ret = 0;
    let lastReturn = 0;
    const history = [];
    let acc = 0;

    const tiles = [];
    const arrows = [];
    const tileGeo = new THREE.BoxGeometry(CELL * 0.94, 0.16, CELL * 0.94);
    const coneGeo = new THREE.ConeGeometry(0.15, 0.42, 12);
    for (let r = 0; r < SIZE; r++) {
      for (let c = 0; c < SIZE; c++) {
        const t = cell(r, c);
        const [x, z] = toXZ(r, c);
        if (t === '#') {
          const wall = new THREE.Mesh(new THREE.BoxGeometry(CELL * 0.94, 0.9, CELL * 0.94), new THREE.MeshStandardMaterial({ color: 0x475569 }));
          wall.position.set(x, 0.45, z);
          scene.add(wall);
          continue;
        }
        const color = t === 'G' ? PALETTE[2] : t === 'P' ? 0x7f1d1d : 0x1e293b;
        const tile = new THREE.Mesh(tileGeo, new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: t === 'G' ? 0.6 : 0.25 }));
        tile.position.set(x, t === 'P' ? -0.15 : 0, z);
        scene.add(tile);
        if (t === 'G' || t === 'P') {
          const s = textSprite(t === 'G' ? 'Clinic +10' : 'Hotspot −10', { height: 0.3, color: t === 'G' ? '#ffd166' : '#fca5a5', background: 'rgba(2,6,23,0.6)' });
          s.position.set(x, 0.6, z);
          scene.add(s);
          continue;
        }
        tiles.push({ r, c, mesh: tile });
        const holder = new THREE.Group();
        const cone = new THREE.Mesh(coneGeo, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0 }));
        cone.rotation.x = Math.PI / 2;
        holder.add(cone);
        holder.position.set(x, 0.2, z);
        scene.add(holder);
        arrows.push({ r, c, holder, cone });
      }
    }

    const agent = new THREE.Mesh(new THREE.SphereGeometry(0.3, 24, 18), new THREE.MeshStandardMaterial({ color: PALETTE[0], emissive: PALETTE[0], emissiveIntensity: 0.8 }));
    scene.add(agent);
    const agentTarget = new THREE.Vector3();

    const stats = {
      episode: ui.stat('Episode'),
      steps: ui.stat('Steps'),
      ret: ui.stat('Last return'),
      success: ui.stat('Success (last 20)'),
    };

    const qi = (r, c, a) => (r * SIZE + c) * 4 + a;
    const maxQ = (r, c) => Math.max(Q[qi(r, c, 0)], Q[qi(r, c, 1)], Q[qi(r, c, 2)], Q[qi(r, c, 3)]);

    function greedy(r, c) {
      const m = maxQ(r, c);
      const best = [0, 1, 2, 3].filter((a) => Q[qi(r, c, a)] >= m - 1e-9);
      return best[Math.floor(rand() * best.length)];
    }

    function envStep([r, c], a) {
      const nr = r + ACTIONS[a][0];
      const nc = c + ACTIONS[a][1];
      if (nr < 0 || nc < 0 || nr >= SIZE || nc >= SIZE || cell(nr, nc) === '#') return { next: [r, c], reward: -0.3, done: false };
      const t = cell(nr, nc);
      if (t === 'G') return { next: [nr, nc], reward: 10, done: true };
      if (t === 'P') return { next: [nr, nc], reward: -10, done: true };
      return { next: [nr, nc], reward: -0.1, done: false };
    }

    function learnStep() {
      const [r, c] = state;
      const a = rand() < eps ? Math.floor(rand() * 4) : greedy(r, c);
      const { next, reward, done } = envStep(state, a);
      const future = done ? 0 : maxQ(next[0], next[1]);
      Q[qi(r, c, a)] += alpha * (reward + GAMMA * future - Q[qi(r, c, a)]);
      state = next;
      steps++;
      ret += reward;
      if (done || steps >= MAX_STEPS) {
        history.push(cell(next[0], next[1]) === 'G');
        if (history.length > 20) history.shift();
        lastReturn = ret;
        episode++;
        state = [0, 0];
        steps = 0;
        ret = 0;
        return true;
      }
      return false;
    }

    function reset() {
      Q.fill(0);
      state = [0, 0];
      steps = episode = 0;
      ret = lastReturn = 0;
      history.length = 0;
    }

    ui.slider({ label: 'Exploration ε', min: 0, max: 1, step: 0.01, value: eps, format: (v) => v.toFixed(2), onInput: (v) => { eps = v; } });
    ui.slider({ label: 'Learning rate α', min: 0.05, max: 1, step: 0.05, value: alpha, format: (v) => v.toFixed(2), onInput: (v) => { alpha = v; } });
    ui.button('Train 200 episodes', () => {
      let done = 0;
      while (done < 200) if (learnStep()) done++;
    }, { primary: true });
    ui.button('Reset Q-table', reset);

    const base = new THREE.Color(0x1e293b);
    const good = new THREE.Color(PALETTE[3]);
    const bad = new THREE.Color(0xef4444);

    return {
      update(dt, time) {
        acc += dt;
        while (acc >= 1 / 9) {
          acc -= 1 / 9;
          learnStep();
        }

        for (const { r, c, mesh } of tiles) {
          const v = maxQ(r, c);
          mesh.material.color.copy(base).lerp(v >= 0 ? good : bad, Math.min(Math.abs(v) / 8, 1));
          mesh.material.emissive.copy(mesh.material.color);
        }
        for (const { r, c, holder, cone } of arrows) {
          const qs = [0, 1, 2, 3].map((a) => Q[qi(r, c, a)]);
          const spread = Math.max(...qs) - Math.min(...qs);
          const best = qs.indexOf(Math.max(...qs));
          holder.rotation.y = ANGLES[best];
          cone.material.opacity = Math.min(spread / 2, 0.9);
        }

        const [x, z] = toXZ(state[0], state[1]);
        agentTarget.set(x, 0.45, z);
        agent.position.lerp(agentTarget, damp(18, dt) || 1);
        agent.position.y = 0.45 + 0.12 * Math.abs(Math.sin(time * 10));

        stats.episode(episode);
        stats.steps(steps);
        stats.ret(lastReturn.toFixed(1));
        stats.success(history.length ? `${Math.round((100 * history.filter(Boolean).length) / history.length)}%` : '–');
      },
    };
  },
};
