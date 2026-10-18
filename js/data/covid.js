// Shared access to the COVID-19 dataset used by every lesson.
import { COUNTRIES, SERIES, DAILY, SOURCE } from './covid-data.js';

export { COUNTRIES, SERIES, DAILY, SOURCE };

export const CONTINENTS = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania'];
export const CONTINENT_COLORS = {
  Africa: 0xffd166,
  Asia: 0xf72585,
  Europe: 0x4cc9f0,
  'North America': 0x06d6a0,
  'South America': 0xff8c42,
  Oceania: 0x9b5de5,
};

const int = (v) => Math.round(v).toLocaleString('en-US');

// Column dictionary. `log` marks heavily skewed columns that are log10-scaled
// before standardising.
export const FIELDS = {
  cases_pm: { label: 'Cases per million', short: 'Cases/M', fmt: int, log: true, about: 'Confirmed cases per million people (depends heavily on how much a country tested)' },
  deaths_pm: { label: 'Deaths per million', short: 'Deaths/M', fmt: int, about: 'Confirmed COVID-19 deaths per million people' },
  vax: { label: 'Fully vaccinated (%)', short: 'Vaccinated', fmt: (v) => `${v.toFixed(0)}%`, about: 'Share of people with a full initial vaccine protocol' },
  median_age: { label: 'Median age (years)', short: 'Median age', fmt: (v) => `${v.toFixed(1)} yrs`, about: 'Half the population is older than this' },
  aged_65: { label: 'Aged 65+ (%)', short: 'Aged 65+', fmt: (v) => `${v.toFixed(1)}%`, about: 'Share of population aged 65 or older' },
  gdp: { label: 'GDP per capita ($)', short: 'GDP/capita', fmt: (v) => `$${int(v)}`, log: true, about: 'Economic output per person (PPP, international $)' },
  beds: { label: 'Hospital beds per 1,000', short: 'Beds/1k', fmt: (v) => v.toFixed(1), about: 'Hospital beds per 1,000 people' },
  life_exp: { label: 'Life expectancy (years)', short: 'Life exp.', fmt: (v) => `${v.toFixed(1)} yrs`, about: 'Life expectancy at birth (2019)' },
  hdi: { label: 'Human Development Index', short: 'HDI', fmt: (v) => v.toFixed(3), about: 'UN composite of health, education and income (0–1)' },
  stringency: { label: 'Lockdown stringency (avg.)', short: 'Stringency', fmt: (v) => v.toFixed(0), about: 'Average Oxford policy stringency index, 0–100' },
  population: { label: 'Population', short: 'Population', fmt: (v) => `${(v / 1e6).toFixed(1)}M`, about: 'Population' },
};

export const values = (key) => COUNTRIES.map((c) => c[key]);
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

export function median(arr) {
  const s = [...arr].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function quantile(arr, q) {
  const s = [...arr].sort((a, b) => a - b);
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  return s[lo] + (s[Math.min(lo + 1, s.length - 1)] - s[lo]) * (pos - lo);
}

// Optional log10, then z-score. `z` holds every country's standardised value;
// toZ/fromZ convert between real units and standardised units.
export function scaler(key, { log = FIELDS[key]?.log ?? false } = {}) {
  const fwd = (v) => (log ? Math.log10(v + 1) : v);
  const t = values(key).map(fwd);
  const m = mean(t);
  const sd = Math.sqrt(mean(t.map((v) => (v - m) ** 2))) || 1;
  return {
    key,
    log,
    mean: m,
    sd,
    z: t.map((v) => (v - m) / sd),
    toZ: (v) => (fwd(v) - m) / sd,
    fromZ: (z) => (log ? 10 ** (z * sd + m) - 1 : z * sd + m),
    format: (v) => FIELDS[key].fmt(v),
    label: FIELDS[key].label,
  };
}

export const pearson = (a, b) => {
  const ma = mean(a);
  const mb = mean(b);
  let num = 0;
  let da = 0;
  let db = 0;
  a.forEach((v, i) => {
    num += (v - ma) * (b[i] - mb);
    da += (v - ma) ** 2;
    db += (b[i] - mb) ** 2;
  });
  return num / Math.sqrt(da * db);
};

// Classification target used across lessons: deaths per million above the median.
export const MEDIAN_DEATHS = median(values('deaths_pm'));
export const HIGH_MORTALITY = COUNTRIES.map((c) => (c.deaths_pm >= MEDIAN_DEATHS ? 1 : 0));

export const describe = (c, keys) => keys.map((k) => `${FIELDS[k].short} ${FIELDS[k].fmt(c[k])}`).join(' · ');
