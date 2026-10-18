import { COUNTRIES, FIELDS, IS_TEST, scaler } from '../data/covid.js';

// "Use the trained model" panel: one slider per feature (in real units) plus a
// picker that loads a real country's values. `values()` returns standardised
// inputs ready for the model; `onChange` fires whenever an input moves.
export function predictor(ui, keys, { onChange = () => {}, title = 'Use the trained model' } = {}) {
  ui.section(title);
  const scalers = keys.map((k) => scaler(k));
  const state = keys.map((k) => median(COUNTRIES.map((c) => c[k])));
  let country = -1;

  const picker = ui.select({
    label: 'Load a real country (★ = test set, never seen in training)',
    options: [{ value: '-1', label: 'Custom values (use the sliders)' }].concat(
      COUNTRIES.map((c, i) => ({ value: String(i), label: `${IS_TEST[i] ? '★ ' : ''}${c.name}` })),
    ),
    value: '-1',
    onChange: (v) => {
      country = Number(v);
      if (country >= 0) keys.forEach((k, j) => { state[j] = COUNTRIES[country][k]; sliders[j].set(toSlider(j, state[j])); });
      onChange();
    },
  });

  const isLog = (j) => scalers[j].log;
  const toSlider = (j, v) => (isLog(j) ? Math.log10(v + 1) : v);
  const fromSlider = (j, v) => (isLog(j) ? 10 ** v - 1 : v);
  const sliders = keys.map((k, j) => {
    const vals = COUNTRIES.map((c) => toSlider(j, c[k]));
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    return ui.slider({
      label: `${FIELDS[k].label}${isLog(j) ? ' (log slider)' : ''}`,
      min: lo, max: hi, step: (hi - lo) / 200, value: toSlider(j, state[j]),
      format: (v) => FIELDS[k].fmt(fromSlider(j, v)),
      onInput: (v) => {
        state[j] = fromSlider(j, v);
        country = -1;
        picker.set('-1');
        onChange();
      },
    });
  });
  const out = ui.stat('Model prediction', { wide: true });

  return {
    values: () => state.map((v, j) => scalers[j].toZ(v)),
    raw: () => [...state],
    country: () => country,
    // Text noting the real outcome when a real country is loaded.
    actual: (fmt) => (country >= 0 ? ` · actual: ${fmt(COUNTRIES[country])}${IS_TEST[country] ? ' (test country)' : ' (training country)'}` : ''),
    show: (text) => out(text),
  };
}

function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  return s[s.length >> 1];
}
