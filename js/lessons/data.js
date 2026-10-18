import * as THREE from 'three';
import {
  sphereCloud, setInstance, setInstanceColor, commitInstances, textSprite, clearGroup, damp, rng, PALETTE,
} from '../helpers.js';
import {
  COUNTRIES, SERIES, SOURCE, FIELDS, CONTINENTS, CONTINENT_COLORS, scaler, pearson, values, MEDIAN_DEATHS, HIGH_MORTALITY, IS_TEST,
} from '../data/covid.js';

const N = COUNTRIES.length;
const S = 2.1; // display units per standard deviation
const FEATURED = 'Italy';

const STEPS = [
  {
    title: '1 · Every dot is a country',
    camera: [[9, 6, 13], [0, 0, 0]],
    text: `This dataset holds one <strong>row per country</strong>: ${N} countries, each a snapshot of the pandemic up to 31 Dec 2022.
      Here each country sits in 3D by <strong>median age</strong> (x), <strong>deaths per million</strong> (y, log scale) and
      <strong>% fully vaccinated</strong> (z). Size is population and colour is continent. Drag to orbit and click a dot to read its row.`,
  },
  {
    title: '2 · Features vs. target',
    camera: [[9, 6, 13], [0, 0, 0]],
    text: `Machine learning separates columns into <strong>features</strong> (inputs we know, such as age structure, wealth, hospital beds
      and vaccination) and a <strong>target</strong> (what we want to predict, here <em>deaths per million</em>). Take ${FEATURED}: its
      features describe the country, and its target is the outcome a model would try to predict from them.`,
  },
  {
    title: '3 · Look at the distribution',
    camera: [[0, 0.5, 16], [0, 0.5, 0]],
    text: `Before modelling, look at how the target is distributed. Each column holds the countries in one 500-deaths-per-million band.
      The data is <strong>right-skewed</strong>: most countries are on the left and a few (Peru, Bulgaria, Hungary…) form a long tail.`,
  },
  {
    title: '4 · Transform skewed data',
    camera: [[0, 0.5, 16], [0, 0.5, 0]],
    text: `Taking <strong>log₁₀</strong> spreads out the crowded low values and pulls in the tail, so the shape becomes much more balanced.
      Many models work better on balanced inputs, which is why several lessons use the log of cases or deaths.`,
  },
  {
    title: '5 · Find relationships',
    camera: [[0, 0, 16], [0, 0, 0]],
    text: `Plotting two columns against each other shows relationships. Older populations reported more deaths per million (correlation
      <strong>r = {R}</strong>). But <strong>correlation is not causation</strong>: richer, older countries also tested more and recorded
      deaths more completely. That kind of hidden factor is called a <strong>confounder</strong>.`,
  },
  {
    title: '6 · Time series',
    camera: [[1, 6, 19], [0, 2.2, 0]],
    text: `The dataset is also a <strong>time series</strong>: daily cases for every country. Each ribbon is weekly new cases per million for
      one country, 2020–2022. You can see waves arrive at different times, and the huge Omicron peak in early 2022. Time ordering matters
      here: you can't train on the future to predict the past.`,
  },
  {
    title: '7 · Features live on different scales',
    camera: [[0, 0, 16], [0, 0, 0]],
    text: `Raw units don't match. GDP per capita ranges from about $700 to $115,000, while median age only spans about 15–48 years. Drawn
      with equal units, the cloud is flattened into a thin smear and GDP dominates any distance or gradient calculation.`,
  },
  {
    title: '8 · Standardise (z-scores)',
    camera: [[0, 0, 16], [0, 0, 0]],
    text: `<strong>Standardising</strong> rescales each feature to mean 0 and standard deviation 1 (log first for GDP). Now both axes
      matter equally. k-NN, SVM, k-means, PCA and neural networks all depend on this step.`,
  },
  {
    title: '9 · Train / test split',
    camera: [[0, 0, 16], [0, 0, 0]],
    text: `To judge a model fairly, hide some data. About <strong>80%</strong> of countries (cyan) are used for <strong>training</strong> and
      20% (pink) are held out as a <strong>test set</strong>. A model that scores well on training data but poorly on test data has
      <strong>overfit</strong>.`,
  },
  {
    title: '10 · Labels for classification',
    camera: [[0, 0, 16], [0, 0, 0]],
    text: `For classification we turn the number into a <strong>label</strong>: <span style="color:#f72585">high mortality</span> if deaths per
      million are above the median ({MED}), otherwise <span style="color:#4cc9f0">low</span>. Logistic regression, k-NN and decision trees
      learn to predict this label.`,
  },
];

export default {
  id: 'data',
  title: 'Meet the Data: COVID-19',
  category: 'Start Here',
  camera: { position: [9, 6, 13], target: [0, 0, 0] },
  formula: `${N} countries × ${Object.keys(FIELDS).length} columns · snapshot ${SOURCE.snapshot}<br>+ weekly time series for ${SERIES.countries.length} countries (2020–2022)`,
  hint: 'Use Next / Previous to walk through the data · Click any dot to inspect that country',
  narrateOnLoad: false,
  info: `
    <h3>Why one dataset?</h3>
    <p>Every lesson on this site uses the <strong>same real dataset</strong>: COVID-19 statistics for ${N} countries. Seeing each
    algorithm ask a different question of familiar data makes it easier to compare what each one does.</p>
    <h3>The columns</h3>
    <ul>${Object.values(FIELDS).map((f) => `<li><strong>${f.label}</strong>: ${f.about}</li>`).join('')}</ul>
    <h3>Caveats (real data is messy)</h3>
    <ul>
      <li>Case counts depend on how much each country <strong>tested</strong>, and death counts on how well deaths were <strong>registered</strong>. Some countries under-report heavily.</li>
      <li>Values are country averages, so they can't tell you about individual patients.</li>
      <li>Only countries with over 1 million people and no missing values are included (a simple form of <strong>data cleaning</strong>).</li>
    </ul>
    <h3>Source</h3>
    <p>${SOURCE.name} (<a href="${SOURCE.url}" target="_blank" rel="noopener">${SOURCE.url.replace('https://', '')}</a>), licensed ${SOURCE.license}.
    The site reads <code>js/data/covid-data.js</code>, which is regenerated by <code>scripts/build_covid_data.py</code>.</p>
  `,
  create({ scene, ui, onClick, flyTo, camera, narrate }) {
    const rand = rng(5);
    let step = 0;

    // ---------- Precomputed layouts ----------
    const age = scaler('median_age');
    const logDeaths = scaler('deaths_pm', { log: true });
    const vax = scaler('vax');
    const gdp = scaler('gdp');
    const deaths = values('deaths_pm');
    const maxPop = Math.max(...values('population'));
    const popScale = COUNTRIES.map((c) => 0.55 + 2.1 * Math.sqrt(c.population / maxPop));
    const r = pearson(values('median_age'), deaths);
    const isTest = IS_TEST;
    const featured = COUNTRIES.findIndex((c) => c.name === FEATURED);

    const scatter3D = COUNTRIES.map((_, i) => [age.z[i] * S, logDeaths.z[i] * S, vax.z[i] * S]);
    const histogram = (vals, binWidth, lo) => {
      const bins = vals.map((v) => Math.floor((v - lo) / binWidth));
      const nb = Math.max(...bins) + 1;
      const counts = new Array(nb).fill(0);
      const order = vals.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
      const pos = new Array(N);
      const maxCount = Math.max(...bins.reduce((c, b) => { c[b] = (c[b] || 0) + 1; return c; }, []).filter(Boolean));
      const dy = Math.min(0.32, 8.5 / maxCount);
      const dx = Math.min(0.75, 13 / nb);
      for (const [, i] of order) {
        const b = bins[i];
        pos[i] = [(b - (nb - 1) / 2) * dx, -4 + counts[b] * dy, 0];
        counts[b]++;
      }
      return { pos, nb, dx, lo, binWidth };
    };
    const histRaw = histogram(deaths, 500, 0);
    const histLog = histogram(deaths.map((v) => Math.log10(v + 1)), 0.25, 0);
    const ageVsDeaths = COUNTRIES.map((c, i) => [age.z[i] * S, (c.deaths_pm / 1000) * 1.6 - 4, 0]);
    const gdpRaw = COUNTRIES.map((c) => [(c.gdp / 1000) * 0.12 - 6.5, c.median_age * 0.12 - 3.8, 0]);
    const gdpStd = COUNTRIES.map((_, i) => [gdp.z[i] * S, age.z[i] * S, 0]);

    const LAYOUTS = [scatter3D, scatter3D, histRaw.pos, histLog.pos, ageVsDeaths, null, gdpRaw, gdpStd, gdpStd, gdpStd];

    // ---------- Points ----------
    const pts = sphereCloud(N, 0.12);
    scene.add(pts);
    const cur = scatter3D.map((p) => [...p]);
    const curScale = popScale.slice();
    const curColor = COUNTRIES.map((c) => new THREE.Color(CONTINENT_COLORS[c.continent]));

    function targetFor(i) {
      const c = COUNTRIES[i];
      let color = CONTINENT_COLORS[c.continent];
      let scale = step <= 1 ? popScale[i] : 0.85;
      if (step === 1 && i !== featured) scale *= 0.7;
      if (step === 1 && i === featured) scale = 2.6;
      if (step === 8) color = isTest[i] ? PALETTE[1] : PALETTE[0];
      if (step === 9) color = HIGH_MORTALITY[i] ? PALETTE[1] : PALETTE[0];
      if (step === 5) scale = 0;
      const pos = LAYOUTS[step]?.[i] ?? cur[i];
      const z = step === 8 && isTest[i] ? 0.6 : pos[2];
      return { pos: [pos[0], pos[1], z], scale, color };
    }

    // ---------- Overlays (one group per step) ----------
    const overlays = STEPS.map(() => new THREE.Group());
    overlays.forEach((g) => scene.add(g));
    const lineMat = new THREE.LineBasicMaterial({ color: 0x64748b });
    const label = (text, x, y, z, opts = {}) => {
      const s = textSprite(text, { height: 0.34, color: '#94a3b8', ...opts });
      s.position.set(x, y, z);
      return s;
    };
    const lines = (pairs) => new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pairs.map((p) => new THREE.Vector3(...p))), lineMat);

    function axes3D(g) {
      const L = 6;
      g.add(lines([[-L, -L, -L], [L, -L, -L], [-L, -L, -L], [-L, L, -L], [-L, -L, -L], [-L, -L, L]]));
      g.add(label('Median age →', L + 0.6, -L, -L, { color: '#e2e8f0' }));
      g.add(label('Deaths per million (log) ↑', -L, L + 0.5, -L, { color: '#e2e8f0' }));
      g.add(label('% vaccinated →', -L, -L, L + 0.6, { color: '#e2e8f0' }));
      for (const zv of [-2, 0, 2]) {
        g.add(label(age.format(age.fromZ(zv / 1)), zv * S, -L - 0.45, -L, { height: 0.28 }));
        g.add(label(logDeaths.format(logDeaths.fromZ(zv / 1)), -L - 1.1, zv * S, -L, { height: 0.28 }));
        g.add(label(vax.format(vax.fromZ(zv / 1)), -L, -L - 0.45, zv * S, { height: 0.28 }));
      }
    }
    axes3D(overlays[0]);
    axes3D(overlays[1]);
    const fc = COUNTRIES[featured];
    const card = textSprite(`${fc.name}: age ${FIELDS.median_age.fmt(fc.median_age)} · ${FIELDS.vax.fmt(fc.vax)} vaxed → ${FIELDS.deaths_pm.fmt(fc.deaths_pm)} deaths/M`, { height: 0.42, background: 'rgba(2,6,23,0.85)', color: '#ffd166' });
    card.position.set(scatter3D[featured][0], scatter3D[featured][1] + 1.1, scatter3D[featured][2]);
    overlays[1].add(card);
    overlays[1].add(label('features (inputs)', scatter3D[featured][0] + 2.5, scatter3D[featured][1] + 1.9, scatter3D[featured][2], { color: '#4cc9f0', height: 0.32 }));

    function histAxes(g, h, fmtTick, title) {
      const y0 = -4.25;
      const half = ((h.nb - 1) / 2) * h.dx + h.dx / 2;
      g.add(lines([[-half, y0, 0], [half, y0, 0]]));
      for (let b = 0; b <= h.nb; b += Math.max(1, Math.round(h.nb / 6))) {
        g.add(label(fmtTick(h.lo + b * h.binWidth), (b - (h.nb - 1) / 2 - 0.5) * h.dx, y0 - 0.45, 0, { height: 0.28 }));
      }
      g.add(label(title, 0, y0 - 1.1, 0, { color: '#e2e8f0' }));
    }
    histAxes(overlays[2], histRaw, (v) => Math.round(v).toLocaleString('en-US'), 'Deaths per million');
    histAxes(overlays[3], histLog, (v) => Math.round(10 ** v).toLocaleString('en-US'), 'Deaths per million (log scale)');

    // Step 5: age vs deaths with least-squares line.
    {
      const g = overlays[4];
      g.add(lines([[-6.5, -4.2, 0], [6.5, -4.2, 0], [-6.5, -4.2, 0], [-6.5, 7, 0]]));
      g.add(label('Median age →', 5.2, -4.8, 0, { color: '#e2e8f0' }));
      g.add(label('Deaths per million ↑', -4.6, 7.2, 0, { color: '#e2e8f0' }));
      const xs = ageVsDeaths.map((p) => p[0]);
      const ys = ageVsDeaths.map((p) => p[1]);
      const mx = xs.reduce((a, b) => a + b) / N;
      const my = ys.reduce((a, b) => a + b) / N;
      const slope = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0) / xs.reduce((s, x) => s + (x - mx) ** 2, 0);
      const trend = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-5.5, my + slope * (-5.5 - mx), 0.05), new THREE.Vector3(5.5, my + slope * (5.5 - mx), 0.05)]),
        new THREE.LineBasicMaterial({ color: PALETTE[2] }),
      );
      g.add(trend);
      g.add(label(`r = ${r.toFixed(2)}`, 4.8, my + slope * (4.8 - mx) + 0.7, 0, { color: '#ffd166', height: 0.42 }));
      for (const d of [0, 2000, 4000, 6000]) g.add(label(d.toLocaleString('en-US'), -7.3, (d / 1000) * 1.6 - 4, 0, { height: 0.28 }));
      for (const zv of [-2, 0, 2]) g.add(label(age.format(age.fromZ(zv)), zv * S, -4.65, 0, { height: 0.28 }));
    }

    // Step 6: weekly case ribbons.
    {
      const g = overlays[5];
      const W = 15;
      const n = SERIES.dates.length;
      const peak = Math.max(...SERIES.countries.flatMap((c) => c.cases));
      SERIES.countries.forEach((c, k) => {
        const z = (k - (SERIES.countries.length - 1) / 2) * 1.1;
        const pts3 = c.cases.map((v, t) => new THREE.Vector3((t / (n - 1) - 0.5) * W, (v / peak) * 5, z));
        const color = PALETTE[k % PALETTE.length];
        g.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts3), n * 2, 0.04, 6), new THREE.MeshBasicMaterial({ color })));
        g.add(label(c.name, W / 2 + 1.3, 0.15, z, { color: '#' + color.toString(16).padStart(6, '0'), height: 0.3 }));
      });
      g.add(lines([[-W / 2, 0, 2 * 1.1 * 4.6], [W / 2, 0, 2 * 1.1 * 4.6]]));
      for (const yr of ['2020', '2021', '2022']) {
        const t = SERIES.dates.findIndex((d) => d.startsWith(yr));
        g.add(label(yr, (t / (n - 1) - 0.5) * W, -0.5, 5.6, { height: 0.36 }));
      }
      g.add(label('Weekly new cases per million (smoothed)', 0, 5.8, -5, { color: '#e2e8f0' }));
    }

    // Steps 7–8: GDP vs age, raw and standardised.
    {
      const g = overlays[6];
      g.add(lines([[-6.5, -4, 0], [7.5, -4, 0], [-6.5, -4, 0], [-6.5, 2, 0]]));
      for (const v of [0, 25000, 50000, 75000, 100000]) g.add(label(`$${v / 1000}k`, (v / 1000) * 0.12 - 6.5, -4.45, 0, { height: 0.28 }));
      g.add(label('GDP per capita ($) → (equal units)', 1, -5.2, 0, { color: '#e2e8f0' }));
      g.add(label('Median age', -6.5, 2.4, 0, { color: '#e2e8f0' }));
      const g2 = overlays[7];
      g2.add(lines([[-6.5, -5.5, 0], [6.5, -5.5, 0], [-6.5, -5.5, 0], [-6.5, 6, 0]]));
      for (const zv of [-2, -1, 0, 1, 2]) {
        g2.add(label(`${zv}σ`, zv * S, -5.95, 0, { height: 0.28 }));
        g2.add(label(`${zv}σ`, -7.1, zv * S, 0, { height: 0.28 }));
      }
      g2.add(label('GDP per capita (log, z-score) →', 2.5, -6.7, 0, { color: '#e2e8f0' }));
      g2.add(label('Median age (z-score) ↑', -4.4, 6.4, 0, { color: '#e2e8f0' }));
    }
    const splitCounts = { test: isTest.filter(Boolean).length };
    overlays[8].add(label(`train ${N - splitCounts.test} countries`, -2.5, 4.9, 0, { color: '#4cc9f0', height: 0.42 }));
    overlays[8].add(label(`test ${splitCounts.test} countries`, 2.5, 4.9, 0, { color: '#f72585', height: 0.42 }));
    overlays[9].add(label(`high mortality (≥ ${Math.round(MEDIAN_DEATHS)} deaths/M)`, -2.5, 4.9, 0, { color: '#f72585', height: 0.42 }));
    overlays[9].add(label('low mortality', 3.2, 4.9, 0, { color: '#4cc9f0', height: 0.42 }));

    // Selection ring for clicked countries.
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.32, 0.035, 8, 40), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    ring.visible = false;
    scene.add(ring);
    let selected = -1;
    const selLabel = new THREE.Group();
    scene.add(selLabel);

    // ---------- Panel ----------
    const stepBox = ui.html('');
    ui.button('← Previous', () => go(step - 1));
    ui.button('Next step →', () => go(step + 1), { primary: true });
    const tableBox = ui.html('');
    const legend = ui.html(`<div class="legend">${CONTINENTS.map((c) => `<span style="--c:#${CONTINENT_COLORS[c].toString(16).padStart(6, '0')}">${c}</span>`).join('')}</div>`);
    const selStat = ui.stat('Selected country (click a dot)', { wide: true });
    selStat('–');

    const SAMPLE = ['Italy', 'India', 'United States', 'Brazil', 'Japan', 'Nigeria', 'Peru', 'Germany'];
    const tableHtml = () => {
      const cols = ['median_age', 'gdp', 'vax', 'cases_pm', 'deaths_pm'];
      const rows = SAMPLE.map((n) => COUNTRIES.find((c) => c.name === n)).filter(Boolean);
      return `<div class="field-head"><span>Sample rows (${N} total)</span></div>
        <div class="table-wrap"><table class="data-table"><thead><tr><th>Country</th>${cols.map((k) => `<th${k === 'deaths_pm' ? ' class="target"' : ''}>${FIELDS[k].short}</th>`).join('')}</tr></thead>
        <tbody>${rows.map((c) => `<tr${c.name === FEATURED && step === 1 ? ' class="hl"' : ''}><td>${c.name}</td>${cols.map((k) => `<td>${FIELDS[k].fmt(c[k])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
        <div class="table-note">Features → · <span class="target">target</span></div>`;
    };

    function go(s, animate = true) {
      step = Math.max(0, Math.min(STEPS.length - 1, s));
      const st = STEPS[step];
      stepBox.innerHTML = `<div class="step-card"><div class="step-count">Step ${step + 1} of ${STEPS.length}</div>
        <div class="step-title">${st.title}</div><p>${st.text.replace('{R}', r.toFixed(2)).replace('{MED}', `${Math.round(MEDIAN_DEATHS)} deaths/M`)}</p></div>`;
      tableBox.style.display = step <= 1 ? '' : 'none';
      if (step <= 1) tableBox.innerHTML = tableHtml();
      legend.style.display = step <= 1 ? '' : 'none';
      overlays.forEach((g, k) => { g.visible = k === step; });
      if (animate) flyTo(...st.camera);
      narrate(script());
    }

    function script() {
      const d = document.createElement('div');
      d.innerHTML = STEPS[step].text.replace('{R}', r.toFixed(2)).replace('{MED}', `${Math.round(MEDIAN_DEATHS)} deaths per million`);
      return `${STEPS[step].title.replace(/^\d+ · /, '')}. ${d.textContent.replace(/\s+/g, ' ')}`;
    }

    onClick((ray) => {
      const hit = ray.intersectObject(pts)[0];
      if (!hit || hit.instanceId === undefined) return;
      selected = hit.instanceId;
      const c = COUNTRIES[selected];
      selStat(`${c.name} (${c.continent}) · median age ${FIELDS.median_age.fmt(c.median_age)} · GDP ${FIELDS.gdp.fmt(c.gdp)} · ${FIELDS.vax.fmt(c.vax)} vaccinated · ${FIELDS.cases_pm.fmt(c.cases_pm)} cases/M · ${FIELDS.deaths_pm.fmt(c.deaths_pm)} deaths/M`);
      clearGroup(selLabel);
      const s = textSprite(c.name, { height: 0.36, background: 'rgba(2,6,23,0.85)', color: '#ffffff' });
      selLabel.add(s);
    });

    go(0, false);

    return {
      script,
      update(dt) {
        const k = damp(4, dt);
        for (let i = 0; i < N; i++) {
          const t = targetFor(i);
          for (let j = 0; j < 3; j++) cur[i][j] += (t.pos[j] - cur[i][j]) * k;
          curScale[i] += (t.scale - curScale[i]) * k;
          curColor[i].lerp(new THREE.Color(t.color), k);
          setInstance(pts, i, cur[i][0], cur[i][1], cur[i][2], Math.max(curScale[i], 0.0001));
          setInstanceColor(pts, i, curColor[i]);
        }
        commitInstances(pts);
        ring.visible = selected >= 0 && step !== 5;
        if (selected >= 0) {
          const p = cur[selected];
          ring.position.set(p[0], p[1], p[2]);
          ring.scale.setScalar(Math.max(curScale[selected], 0.6));
          ring.lookAt(camera.position);
          selLabel.position.set(p[0], p[1] + 0.35 + 0.15 * curScale[selected], p[2]);
          selLabel.visible = ring.visible;
        }
      },
    };
  },
};
