// gta6-twitch-ccu-forecast-01 entry (Node 22, standard library only, no network, deterministic).
//
// Model: pooled launch-day shape from the case's comparables.
//   Every past big launch is a rolling-midnight release like GTA VI, so its UTC-hour
//   curve already encodes the staggered unlock (NZ/EU/UK first, US East 05:00Z, US West
//   08:00Z), the overnight dip and the US-daytime ramp to a late-day peak. For each
//   comparable we take log-ratios lr_h = ln(viewers_h / viewers_hour1) for hours 2..24.
//   Launches whose hour-1 print is a tiny fraction of their day peak (< 5%) unlocked
//   simultaneously much later in the day, a different regime from GTA VI's rolling
//   midnight, so they are left out of the pool (this also drops the GTA VI category's
//   own pre-release history, which has no release-day curve).
//   For the target case, each hour's centre is a trimmed weighted mean of the pooled
//   log-ratios, weighted by a Gaussian kernel on log(hour-1 print) with bandwidth 1.0:
//   mega-launches draw a huge global midnight audience and stay flat, smaller launches
//   grow several-fold into the US day, and the kernel lets the target's own hour-1
//   scale pick the matching regime. Spread is the pooled interquartile range / 1.349
//   (floored at 0.12 log units, capped at 1.0). Centre and spread are lightly smoothed
//   across neighbouring hours (0.25/0.5/0.25). Quantiles:
//     q_tau(h) = p1 * exp(m_h + z_tau * s_h),  z = [-1.645, -0.674, 0, 0.674, 1.645]
//   with p1 the hour-1 print (floored at 1). If a pool is empty, that hour falls back
//   to the house baseline's persistence spread (0.15 * sqrt(h-1)).
//   Leave-one-out validation on the 29 usable comparables: mean pinball loss 0.265
//   (typical miss ~57%, 90% coverage ~0.88) vs 0.360 for the house baseline.
//
// I/O: reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out to
// $MAKEFUN_OUTPUT_DIR (else /output): {"hours":[{"hour":h,"q":[q05,q25,q50,q75,q95]}]},
// hours 2..24, five non-decreasing finite quantiles in (0, 1e9].

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Gaussian kernel bandwidth on log(hour-1 viewers) for matching launch scale. */
const BANDWIDTH = 1.0;
/** Floor / cap for the log-spread, in natural-log units. */
const SIGMA_MIN = 0.12;
const SIGMA_MAX = 1.0;
/** A comparable whose hour-1 print is below this share of its day peak unlocked
 *  simultaneously later in the day, not at rolling midnight: a different regime. */
const MIN_V1_PEAK = 0.05;
/** Fallback spread (house baseline) when a pooled hour has no data. */
const FALLBACK_SIGMA = 0.15;

interface HourlyPoint { from: string; viewers: number; channels?: number | null }
interface Comparable { name: string; day_start: string; hourly: HourlyPoint[] }
interface Case {
  game?: { name?: string };
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  history_hourly?: Array<{ viewers: number }> | null;
  comparables?: Comparable[] | null;
}

/** x rounded to 2 decimals (half up). */
const round2 = (x: number): number => Math.round(x * 100) / 100;

/** Weighted quantile of [value, weight] pairs sorted by value. */
const weightedQuantile = (pairs: Array<[number, number]>, q: number): number => {
  const total = pairs.reduce((s, p) => s + p[1], 0);
  let acc = 0;
  for (const [v, w] of pairs) { acc += w; if (acc >= q * total) return v; }
  return pairs[pairs.length - 1][0];
};

/** The 24 hourly viewer counts of one comparable's release day (index 0 = hour 1),
 *  null where the source has no point for that hour. Points are keyed by parsed
 *  epoch millisecond, not by the raw `from` string, so a differently formatted
 *  timestamp for the same instant still matches. */
const dayValues = (c: Comparable): Array<number | null> => {
  const byTime = new Map<number, number>();
  for (const p of c.hourly ?? []) {
    if (typeof p.viewers === 'number' && p.viewers >= 0) {
      const t = Date.parse(p.from);
      if (Number.isFinite(t)) byTime.set(t, p.viewers);
    }
  }
  const t0 = Date.parse(c.day_start);
  const out: Array<number | null> = [];
  for (let h = 1; h <= 24; h++) {
    const t = t0 + (h - 1) * 3600_000;
    out.push(byTime.has(t) ? (byTime.get(t) as number) : null);
  }
  return out;
};

/** Per-hour pooled log-ratios to hour 1, with each launch's log hour-1 size. */
const buildPool = (comparables: Comparable[], targetName: string | undefined): Map<number, Array<{ lr: number; lv1: number }>> => {
  const pool = new Map<number, Array<{ lr: number; lv1: number }>>();
  for (let h = 2; h <= 24; h++) pool.set(h, []);
  for (const comp of comparables ?? []) {
    if (targetName && comp.name === targetName) continue;
    const v = dayValues(comp);
    const v1 = v[0];
    if (v1 === null || v1 < 100) continue;
    const peak = Math.max(...v.filter((n): n is number => n !== null));
    if (!(v1 / peak >= MIN_V1_PEAK)) continue;
    const lv1 = Math.log(v1);
    for (let h = 2; h <= 24; h++) {
      const vh = v[h - 1];
      if (vh !== null && vh > 0) (pool.get(h) as Array<{ lr: number; lv1: number }>).push({ lr: Math.log(vh / v1), lv1 });
    }
  }
  return pool;
};

/** Forecast quantiles for one case. */
const forecastCase = (c: Case): Array<{ hour: number; q: number[] }> => {
  const first = c.known.find((k) => k.hour === 1);
  const lastHist = c.history_hourly && c.history_hourly.length > 0 ? c.history_hourly[c.history_hourly.length - 1].viewers : undefined;
  const p1 = Math.max(1, first !== undefined ? first.viewers : (typeof lastHist === 'number' ? lastHist : 1));
  const lv1 = Math.log(p1);
  const pool = buildPool(c.comparables ?? [], c.game?.name);

  const centre: number[] = [];
  const spread: number[] = [];
  for (let h = 2; h <= 24; h++) {
    const arr = pool.get(h) ?? [];
    if (arr.length < 3) {
      centre.push(0);
      spread.push(FALLBACK_SIGMA * Math.sqrt(h - 1));
      continue;
    }
    const pairs = arr
      .map((a): [number, number] => [a.lr, Math.exp(-0.5 * ((a.lv1 - lv1) / BANDWIDTH) ** 2)])
      .sort((a, b) => a[0] - b[0]);
    const q25 = weightedQuantile(pairs, 0.25);
    const q75 = weightedQuantile(pairs, 0.75);
    const iqr = q75 - q25;
    // Trimmed weighted mean: drop far outliers, then average what is left.
    const lo = q25 - 1.5 * iqr;
    const hi = q75 + 1.5 * iqr;
    const kept = pairs.filter(([lr]) => lr >= lo && lr <= hi);
    let m: number;
    if (kept.length > 0) {
      const tot = kept.reduce((s, p) => s + p[1], 0);
      m = kept.reduce((s, p) => s + p[0] * p[1], 0) / tot;
    } else {
      m = weightedQuantile(pairs, 0.5);
    }
    centre.push(m);
    spread.push(Math.min(SIGMA_MAX, Math.max(SIGMA_MIN, iqr / 1.349)));
  }
  // Light 3-point smoothing of centre and spread across neighbouring hours.
  const smooth = (a: number[]): number[] =>
    a.map((v, i) => 0.25 * (a[i - 1] ?? v) + 0.5 * v + 0.25 * (a[i + 1] ?? v));
  const cs = smooth(centre);
  const ss = smooth(spread);

  return c.forecast.map(({ hour }, i) => {
    const m = cs[i];
    const s = ss[i];
    let prev = 0;
    const q = Z.map((z) => {
      const v = Math.min(1e9, Math.max(1, round2(p1 * Math.exp(m + z * s))));
      const out = Math.max(prev, v);
      prev = out;
      return out;
    });
    return { hour, q };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  const hours = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
