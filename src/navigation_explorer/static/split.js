// Split analysis page: statistics over every episode of the selected run (or the episodes matching the list
// filters). Clicking a bar, cell or count filters the episode list to that slice. Rendered by app.js.

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (x, d = 0) => (x == null || !isFinite(x) ? '–' : (100 * x).toFixed(d) + '%');
const num = x => Number(x).toLocaleString('en-US');
const mean = values => { const xs = values.filter(x => x != null && Number.isFinite(x)); return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null; };

export const PLACE = {
  'goal room': 'In the goal room, outside the success area',
  'same room type': 'Same room type, wrong room',
  'other room': 'Another kind of room',
  'wrong floor': 'Wrong floor',
};
export const MISTAKE = {
  'wrong move': 'Moved to a worse node',
  'missed stop': 'Kept moving inside the success area',
  'stopped early': 'Stopped outside the success area',
};
const DIMENSIONS = {
  gt_len: { label: 'Reference path length', unit: ' m', kind: 'quantile' },
  gt_hops: { label: 'Reference path hops', unit: ' hops', kind: 'quantile' },
  instr_words: { label: 'Instruction length', unit: ' words', kind: 'quantile' },
  gt_floors: { label: 'Floors on the reference path', unit: '', kind: 'floors' },
  scan: { label: 'Building (scan)', unit: '', kind: 'category' },
};
let dimension = 'gt_len';

// Five bins with about equal numbers of episodes; edges rounded so the labels stay readable.
function bins(rows, key) {
  const dim = DIMENSIONS[key];
  if (dim.kind === 'category') {
    const keys = [...new Set(rows.map(r => r[key]))];
    return keys.map(k => ({ label: k, test: r => r[key] === k }));
  }
  if (dim.kind === 'floors') {
    return [1, 2, 3].map(k => ({ label: k < 3 ? `${k} floor${k > 1 ? 's' : ''}` : '3 or more floors',
      test: r => (k < 3 ? r[key] === k : r[key] >= 3) }));
  }
  const vals = rows.map(r => r[key]).filter(v => v != null).sort((a, b) => a - b);
  if (!vals.length) return [];
  const step = key === 'gt_len' ? 0.5 : 1;
  const round = v => Math.round(v / step) * step;
  const edges = [...new Set([1, 2, 3, 4].map(q => round(vals[Math.floor(q * vals.length / 5)])))].filter(e => e > vals[0]);
  const lo = [-Infinity, ...edges], hi = [...edges, Infinity];
  const f = v => (key === 'gt_len' ? v.toFixed(1) : String(v));
  return lo.map((a, i) => ({
    label: a === -Infinity ? `< ${f(hi[i])}${dim.unit}` : hi[i] === Infinity ? `≥ ${f(a)}${dim.unit}` : `${f(a)}–${f(hi[i])}${dim.unit}`,
    test: r => r[key] != null && r[key] >= a && r[key] < hi[i],
  }));
}

// Horizontal bars: one per item, the value printed at the tip; the whole row filters the list.
function barList(items, ctx, { max = null, mark = null, markLabel = '' } = {}) {
  const top = max ?? Math.max(1e-9, ...items.map(it => it.value));
  const rows = items.map(it => {
    const id = ctx.register(it.slice);
    const w = Math.max(0, Math.min(100, (100 * it.value) / top));
    const m = mark == null ? '' : `<i class="bar-mark" style="left:${(100 * mark) / top}%"></i>`;
    return `<button class="bar-row" data-slice="${id}" title="${esc(it.title || it.label)} · show these episodes in the list"${it.slice ? '' : ' disabled'}>` +
      `<span class="bar-label">${esc(it.label)}</span><span class="bar-track">${m}<i class="bar-fill" style="width:${w}%"></i></span>` +
      `<span class="bar-value">${esc(it.text)}</span></button>`;
  }).join('');
  const legend = mark == null ? '' : `<p class="chart-note"><i class="mark-key"></i>${esc(markLabel)}</p>`;
  return `<div class="bar-list">${rows}</div>${legend}`;
}

function card(title, sub, body, cls = '') {
  return `<section class="split-card ${cls}"><header><h3>${esc(title)}</h3>${sub ? `<p class="hint">${sub}</p>` : ''}</header>${body}</section>`;
}

// ------------------------------------------------------------------------------------------------ cards

function runsCard(ctx) {
  const groups = Object.create(null), order = ['R2R', 'REVERIE', 'RxR', 'R2R-CE', 'RxR-CE'];
  for (const r of [...ctx.runs].sort((a, b) => (order.indexOf(ctx.dataset(a)) + 1 || 9) - (order.indexOf(ctx.dataset(b)) + 1 || 9))) (groups[ctx.dataset(r)] ||= []).push(r);
  const head = '<tr><th scope="col">Split</th><th scope="col">Model</th><th scope="col" class="n">Episodes</th><th scope="col" class="n">SR</th><th scope="col" class="n">SPL</th><th scope="col" class="n">OSR</th><th scope="col" class="n">NE</th><th scope="col" class="n">nDTW</th></tr>';
  let body = '';
  for (const [ds, runs] of Object.entries(groups)) {
    runs.sort((a, b) => ctx.splitRank(a.split) - ctx.splitRank(b.split) || a.model.localeCompare(b.model));
    body += `<tr class="group"><th colspan="8" scope="colgroup">${esc(ds)}</th></tr>`;
    for (const r of runs) {
      const s = r.summary, cur = r.name === ctx.run.name;
      body += `<tr class="${cur ? 'current' : ''}"><td><button class="link" data-run="${esc(r.name)}" ${cur ? 'aria-current="true"' : ''}>${esc(ctx.splitLabel(r.split))}</button>${r.note ? `<span class="hint"> · ${esc(r.note)}</span>` : ''}</td>` +
        `<td>${esc(ctx.modelLabel(r.model))}</td><td class="n">${num(s.n)}</td>` +
        ['sr', 'spl', 'osr'].map(k => `<td class="n" title="${s.metric_counts?.[k] == null ? '' : `${s.metric_counts[k]} of ${s.n} episodes have this metric`}">${s[k] == null ? '–' : s[k].toFixed(1)}</td>`).join('') +
        `<td class="n">${s.ne == null ? '–' : s.ne.toFixed(2)}</td><td class="n">${s.ndtw == null ? '–' : s.ndtw.toFixed(1)}</td></tr>`;
    }
  }
  return card('All runs', 'Click a split to analyse it · SR, SPL, OSR, nDTW in %, NE in metres',
    `<div class="runs-wrap"><table class="runs-table">${head}${body}</table></div>`, 'wide');
}

function successCard(rows, ctx) {
  const scored = rows.filter(r => r.sr != null);
  if (!scored.length) return '';
  const avg = mean(scored.map(r => r.sr));
  let items = bins(scored, dimension).map(b => {
    const sel = scored.filter(b.test);
    return { label: b.label, value: mean(sel.map(r => r.sr)) ?? 0, n: sel.length,
      text: `${pct(mean(sel.map(r => r.sr)))} · ${num(sel.length)}`,
      title: `${b.label}: success rate ${pct(mean(sel.map(r => r.sr)), 1)} over ${sel.length} episodes`,
      slice: sel.length ? { label: `${DIMENSIONS[dimension].label}: ${b.label}`, test: b.test } : null };
  }).filter(it => it.n);
  if (dimension === 'scan') items.sort((a, b) => a.value - b.value);
  const select = `<label class="field inline">By <select id="splitDimension">${Object.entries(DIMENSIONS).map(([k, d]) =>
    `<option value="${k}"${k === dimension ? ' selected' : ''}>${esc(d.label)}</option>`).join('')}</select></label>`;
  return card('Success rate by episode property', `${select}<span>Bars: success rate · number of episodes</span>`,
    items.length ? `<div class="${dimension === 'scan' ? 'scroll-list' : ''}">${barList(items, ctx, { max: 1, mark: avg, markLabel: `Average of ${scored.length} scored episodes: ${pct(avg, 1)}` })}</div>` : '<p class="note">No recorded values for this property. Choose another property above.</p>');
}

function placeCard(rows, ctx) {
  const failed = rows.filter(r => r.sr === 0);
  if (!failed.length || !failed.some(r => r.stop_place)) return '';
  const items = Object.entries(PLACE).map(([k, label]) => {
    const n = failed.filter(r => r.stop_place === k).length;
    return { label, value: n / failed.length, text: `${num(n)} · ${pct(n / failed.length)}`,
      slice: n ? { label: `Stopped: ${label.toLowerCase()}`, test: r => r.sr === 0 && r.stop_place === k } : null };
  });
  return card('Where failed episodes ended', `${num(failed.length)} failed episodes · recorded room labels`, barList(items, ctx, { max: 1 }));
}

function modeCard(rows, ctx) {
  const failed = rows.filter(r => r.sr === 0);
  if (!failed.length) return '';
  const counts = {};
  failed.forEach(r => { counts[r.failure_mode] = (counts[r.failure_mode] || 0) + 1; });
  const items = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([mode, n]) => ({
    label: mode.replace(/^(passed goal|never near goal): /, (_, a) => `${a[0].toUpperCase()}${a.slice(1)} · `),
    value: n / failed.length, text: `${num(n)} · ${pct(n / failed.length)}`,
    slice: { label: `Failure mode: ${mode}`, test: r => r.failure_mode === mode } }));
  return card('Failure modes', '"Passed goal" means the agent was inside the success area at some point', barList(items, ctx, { max: 1 }));
}

function subtaskCard(rows, ctx) {
  const labelled = rows.filter(r => r.subtasks);
  if (!labelled.length) return '';
  const K = Math.max(...labelled.map(r => r.subtasks));
  const items = [];
  for (let k = 1; k <= K; k++) {
    const have = labelled.filter(r => r.subtasks >= k);
    if (have.length < Math.max(10, labelled.length * 0.02)) break;
    const ok = have.filter(r => r.first_missed_subtask == null || r.first_missed_subtask > k).length;
    items.push({ label: `Step ${k}`, value: ok / have.length, text: `${pct(ok / have.length)} · ${num(have.length)}`,
      title: `${ok} of ${have.length} episodes with at least ${k} steps completed steps 1-${k} in order`,
      slice: { label: `First missed sub-task: ${k}`, test: r => r.first_missed_subtask === k } });
  }
  return card('Sub-task completion', 'Share of episodes that completed steps 1…k in order · click a step to list episodes that first failed it',
    barList(items, ctx, { max: 1 }));
}

function compareCard(rows, ctx) {
  if (!ctx.cmp) return '';
  const both = rows.filter(r => ctx.cmp.rows.has(r.instr_id) && r.sr != null && ctx.cmp.rows.get(r.instr_id).sr != null);
  if (!both.length) return '';
  const o = r => ctx.cmp.rows.get(r.instr_id).sr;
  const cells = [[1, 1], [1, 0], [0, 1], [0, 0]].map(([a, b]) => {
    const test = r => r.sr === a && ctx.cmp.rows.has(r.instr_id) && o(r) === b;
    const n = both.filter(test).length;
    const label = a && b ? 'Both succeed' : a ? `Only ${ctx.model} succeeds` : b ? `Only ${ctx.cmp.label} succeeds` : 'Both fail';
    return `<button class="cell${a !== b ? ' differs' : ''}" data-slice="${ctx.register(n ? { label, test } : null)}"${n ? '' : ' disabled'}>` +
      `<b>${num(n)}</b><span>${esc(label)}</span><span class="hint">${pct(n / both.length)}</span></button>`;
  }).join('');
  return card(`${ctx.model} vs ${ctx.cmp.label}`, `${num(both.length)} shared episodes`,
    `<div class="matrix" role="group" aria-label="Outcome agreement">${cells}</div>`);
}

// Columns with the value on each cap; an optional hairline diagonal for "value = x".
function columns(items, ctx, { yMax = 1, diagonal = false, fmt = v => pct(v), xLabel = '' } = {}) {
  const W = 320, H = 150, L = 30, B = 26, T = 14, n = items.length, bw = (W - L) / n, barW = Math.min(24, bw - 4);
  const Y = v => T + (1 - v / yMax) * (H - T - B);
  let s = `<line class="axis" x1="${L}" x2="${W}" y1="${Y(0)}" y2="${Y(0)}"/>`;
  for (const g of [0.5, 1]) s += `<line class="grid" x1="${L}" x2="${W}" y1="${Y(g * yMax)}" y2="${Y(g * yMax)}"/><text class="tick" x="${L - 4}" y="${Y(g * yMax) + 3}" text-anchor="end">${fmt(g * yMax)}</text>`;
  if (diagonal) s += `<line class="diag" x1="${L}" y1="${Y(0)}" x2="${W}" y2="${Y(yMax)}"/>`;
  items.forEach((it, i) => {
    const x = L + i * bw + (bw - barW) / 2, y = Y(it.value), h = Math.max(0, Y(0) - y);
    const id = ctx.register(it.slice);
    const r = Math.min(4, h);
    const path = h ? `M${x},${Y(0)} V${y + r} q0,-${r} ${r},-${r} h${barW - 2 * r} q${r},0 ${r},${r} V${Y(0)} Z` : '';
    s += `<g class="col${it.slice ? ' clickable' : ''}" data-slice="${id}" tabindex="${it.slice ? 0 : -1}" role="${it.slice ? 'button' : 'img'}" aria-label="${esc(it.title)}">` +
      `<rect class="hit" x="${L + i * bw}" y="${T}" width="${bw}" height="${H - T}"/>${path ? `<path class="colfill" d="${path}"/>` : ''}` +
      (it.n ? `<text class="cap" x="${x + barW / 2}" y="${y - 3}" text-anchor="middle">${esc(it.cap ?? fmt(it.value))}</text>` : '') +
      `<text class="tick" x="${x + barW / 2}" y="${H - B + 12}" text-anchor="middle">${esc(it.label)}</text><title>${esc(it.title)}</title></g>`;
  });
  if (xLabel) s += `<text class="tick axis-label" x="${(L + W) / 2}" y="${H - 2}" text-anchor="middle">${esc(xLabel)}</text>`;
  return `<svg class="columns" viewBox="0 0 ${W} ${H}" role="group">${s}</svg>`;
}

function calibrationCard(rows, ctx) {
  const ids = new Set(rows.map(r => r.instr_id));
  const pts = Object.entries(ctx.points || {}).filter(([id]) => ids.has(id)).flatMap(([, v]) => v).filter(p => p[2] != null && p[1] != null);
  if (!pts.length) return '';
  const items = [];
  for (let b = 0; b < 10; b++) {
    const sel = pts.filter(p => Math.min(9, Math.floor(p[1] * 10)) === b);
    const v = sel.length ? mean(sel.map(p => p[2])) : 0;
    items.push({ label: `${b * 10}`, value: v, n: sel.length, cap: sel.length ? pct(v) : '',
      title: `chosen-option probability ${b * 10}–${b * 10 + 10}%: ${sel.length} decisions, ${pct(v, 1)} match the reference-best` });
  }
  const agree = mean(pts.map(p => p[2]));
  const confident = pts.filter(p => p[1] >= 0.9);
  return card('Model confidence vs. reference-best', `${pct(agree, 1)} of ${num(pts.length)} decisions match the reference-best · the hairline marks agreement = confidence`,
    columns(items, ctx, { diagonal: true, xLabel: 'probability of the chosen option (%) →' }) +
    `<p class="chart-note">At ≥ 90% confidence the choice matches ${pct(mean(confident.map(p => p[2])), 1)} of the time (${num(confident.length)} decisions). ${esc(ctx.referenceNote)}</p>`);
}

function firstMistakeCard(rows, ctx) {
  const failed = rows.filter(r => r.sr === 0 && r.first_nonbest != null);
  if (!failed.length) return '';
  const T = Math.max(...failed.map(r => r.first_nonbest));
  const items = [];
  for (let t = 0; t <= Math.min(T, 14); t++) {
    const test = t < 14 ? r => r.sr === 0 && r.first_nonbest === t : r => r.sr === 0 && r.first_nonbest >= 14;
    const n = failed.filter(test).length;
    items.push({ label: t < 14 ? String(t) : '14+', value: n / failed.length, n, cap: n ? num(n) : '',
      title: `${n} failed episodes made their first non-best decision at decision ${t}${t < 14 ? '' : ' or later'}`,
      slice: n ? { label: `First non-best decision: ${t < 14 ? t : '14 or later'}`, test } : null });
  }
  const yMax = Math.max(...items.map(i => i.value)) * 1.15;
  const types = Object.entries(MISTAKE).map(([k, label]) => {
    const n = failed.filter(r => r.first_nonbest_type === k).length;
    return { label, value: n / failed.length, text: `${num(n)} · ${pct(n / failed.length)}`,
      slice: n ? { label: `First mistake: ${label.toLowerCase()}`, test: r => r.sr === 0 && r.first_nonbest_type === k } : null };
  });
  const conf = [['≥ 90% sure', p => p >= 0.9], ['50–90%', p => p >= 0.5 && p < 0.9], ['< 50%', p => p < 0.5]].map(([label, f]) => {
    const test = r => r.sr === 0 && r.first_nonbest_conf != null && f(r.first_nonbest_conf);
    const n = failed.filter(test).length;
    return { label: `Made ${label}`, value: n / failed.length, text: `${num(n)} · ${pct(n / failed.length)}`,
      slice: n ? { label: `First mistake made ${label}`, test } : null };
  });
  return card('First non-best decision in failed episodes', `${num(failed.length)} failed episodes · columns: episodes per decision index`,
    columns(items, ctx, { yMax, fmt: v => pct(v), xLabel: 'decision →' }) +
    '<h4>What went wrong</h4>' + barList(types, ctx, { max: 1 }) + '<h4>How sure the model was</h4>' + barList(conf, ctx, { max: 1 }));
}

function stopCard(rows, ctx) {
  const ids = new Set(rows.map(r => r.instr_id));
  const pts = Object.entries(ctx.points || {}).filter(([id]) => ids.has(id)).flatMap(([, v]) => v).filter(p => p[3] != null);
  if (!pts.length) return '';
  const buckets = [['< 10%', p => p < 0.1], ['10–50%', p => p >= 0.1 && p < 0.5], ['50–90%', p => p >= 0.5 && p < 0.9], ['≥ 90%', p => p >= 0.9]];
  const row = (label, sel) => {
    const segs = buckets.map(([b, f], i) => {
      const n = sel.filter(p => f(p[3])).length, w = (100 * n) / Math.max(1, sel.length);
      return w ? `<i class="stack-seg s${i + 1}" style="flex-basis:${w}%" title="${esc(`${label}: stop probability ${b} at ${n} of ${sel.length} decisions (${pct(n / sel.length, 1)})`)}"></i>` : '';
    }).join('');
    const high = sel.filter(p => p[3] >= 0.5).length;
    return `<div class="stack-row"><span class="bar-label">${esc(label)}</span><span class="stack" role="img" aria-label="${esc(label)}">${segs}</span><span class="bar-value">${pct(high / Math.max(1, sel.length))} ≥ 50% · ${num(sel.length)}</span></div>`;
  };
  const legend = buckets.map(([b], i) => `<span><i class="seg-key s${i + 1}"></i>${esc(b)}</span>`).join('');
  return card('Stop probability at each decision', 'Inside vs. outside the success area · how strongly the model wanted to stop there',
    row('Inside the success area', pts.filter(p => p[4] === 1)) + row('Outside', pts.filter(p => p[4] === 0)) +
    `<div class="legend-row" aria-label="Stop probability">${legend}</div>`);
}

// Continuous (VLN-CE) runs: how the low-level controller carried out the planner's moves, and how the stop
// decision went for episodes that passed through the success area.
// `base` picks `rows` out of the whole population, so a clicked bar lists the same episodes it counts.
function bucketList(rows, buckets, label, ctx, base) {
  const items = buckets.map(([name, test]) => {
    const sel = rows.filter(test), scored = sel.filter(r => r.sr != null), sr = mean(scored.map(r => r.sr));
    return { label: name, value: sel.length / Math.max(1, rows.length), text: `${num(sel.length)} · SR ${pct(sr)}${scored.length < sel.length ? ` (${scored.length} scored)` : ''}`,
      title: `${name}: ${sel.length} episodes, success rate ${pct(sr, 1)}`,
      slice: sel.length ? { label: `${label}: ${name}`, test: r => base(r) && test(r) } : null };
  });
  return barList(items, ctx, { max: 1 });
}

function controlCard(rows, ctx) {
  if (ctx.run.env !== 'continuous') return '';
  const base = r => r.stuck != null, moved = rows.filter(base);
  if (!moved.length) return '';
  const blocked = [['None', r => !r.stuck], ['1 move', r => r.stuck === 1], ['2–4 moves', r => r.stuck >= 2 && r.stuck <= 4], ['5 or more', r => r.stuck >= 5]];
  const miss = [['< 0.5 m', r => r.ctrl_miss_max < 0.5], ['0.5–1 m', r => r.ctrl_miss_max >= 0.5 && r.ctrl_miss_max < 1],
    ['1–2 m', r => r.ctrl_miss_max >= 1 && r.ctrl_miss_max < 2], ['≥ 2 m', r => r.ctrl_miss_max >= 2]];
  const c = r => r.collision_count;
  const coll = [['None', r => !c(r)], ['1–5', r => c(r) >= 1 && c(r) <= 5], ['6–20', r => c(r) >= 6 && c(r) <= 20], ['More than 20', r => c(r) > 20]];
  const col = (title, body) => `<div><h4>${esc(title)}</h4>${body}</div>`;
  return card('Low-level control', 'How the controller carried out the planner\'s moves · bars: share of episodes, with their success rate',
    '<div class="split-cols">' +
    col('Moves that went nowhere (ended < 0.25 m from where the agent decided)', bucketList(moved, blocked, 'Moves that went nowhere', ctx, base)) +
    col('Worst distance between a chosen waypoint and where the agent ended up', bucketList(moved.filter(r => r.ctrl_miss_max != null), miss, 'Worst waypoint miss', ctx, r => base(r) && r.ctrl_miss_max != null)) +
    col('Collisions with the scene during the episode', bucketList(moved.filter(r => c(r) != null), coll, 'Collisions', ctx, r => base(r) && c(r) != null)) +
    '</div>', 'wide');
}

function ceStopCard(rows, ctx) {
  if (ctx.run.adapter !== 'etpnav') return '';
  const base = r => r.sr === 0 && r.osr === 1, passed = rows.filter(base);
  if (!passed.length) return '';
  const z = r => r.zone_stop_score;
  const buckets = [['No map node inside it', r => z(r) == null], ['Best node there < 10%', r => z(r) != null && z(r) < 0.1],
    ['10–50%', r => z(r) >= 0.1 && z(r) < 0.5], ['≥ 50%, another node scored higher', r => z(r) >= 0.5]];
  return card('Passed the goal, stopped elsewhere', `${num(passed.length)} failed episodes that were inside the success area at some point · ETPNav stops at the visited node with the highest stop score`,
    '<p class="chart-note">Highest stop score among map nodes inside the success area</p>' + bucketList(passed, buckets, 'Passed the goal', ctx, base));
}

function whatifCard(rows, ctx) {
  if (!ctx.run.capabilities?.whatif || ctx.gpu?.allowed === false) return '';
  const g = ctx.gpu || {}, results = ctx.whatif || {};
  const byId = new Map(rows.map(r => [r.instr_id, r]));
  let decisions = 0, refDiff = 0, refWins = 0, refLoses = 0;
  const failedBranched = new Set(), recoverable = new Set();
  for (const [iid, byT] of Object.entries(results)) {
    const r = byId.get(iid);
    if (!r) continue;
    for (const opts of Object.values(byT)) {
      const branches = Object.values(opts);
      decisions++;
      if (r.sr === 0) { failedBranched.add(iid); if (branches.some(b => b[0])) recoverable.add(iid); }
      const ref = branches.find(b => b[3]);
      if (ref) {
        refDiff++;
        if (ref[0]) refWins++;
        else if (branches.some(b => b[0]) || r.sr) refLoses++;
      }
    }
  }
  const listedFails = ctx.listed.filter(r => r.sr === 0 && r.first_nonbest != null);
  const on = g.enabled, busy = g.job || (g.queued || []).length;
  const btn = (kind, label, n) => `<button class="whatif-run" data-whatif="${kind}" ${on && n && !busy ? '' : 'disabled'}>${esc(label)} <span class="hint">${num(n)} decisions</span></button>`;
  const stat = (value, label, slice) => `<div class="stat">${slice ? `<button class="link" data-slice="${ctx.register(slice)}">` : ''}<b>${value}</b>${slice ? '</button>' : ''}<span>${esc(label)}</span></div>`;
  let h = '<p class="hint">At a recorded decision, force each other option (the 8 most likely plus the reference-best), let the model continue on its own, and record where it ends.</p>';
  h += `<div class="whatif-actions">${btn('first', 'Branch the first non-best decision of each listed failure', listedFails.length)}` +
    `${btn('all', 'Branch every decision of the listed episodes', ctx.listed.reduce((s, r) => s + (r.decisions || 0), 0))}</div>`;
  h += `<p class="note whatif-gate">${g.allowed === false ? 'This server was started with --no-whatif.' : on ? (busy ? 'A what-if job is running; see the top bar.' : 'GPU switch is on. Jobs run on GPU ' + esc(g.gpu) + '.') : 'Turn on <b>What-if · GPU</b> in the top bar to run branches. Stored results are shown either way.'}</p>`;
  if (decisions) {
    h += '<div class="stats">' +
      stat(num(decisions), 'decisions branched in these episodes') +
      stat(`${num(recoverable.size)} / ${num(failedBranched.size)}`, 'failed episodes where one different decision leads to success',
        recoverable.size ? { label: 'What-if: one decision from success', test: r => recoverable.has(r.instr_id) } : null) +
      stat(refDiff ? pct(refWins / refDiff) : '–', `of ${num(refDiff)} decision${refDiff === 1 ? '' : 's'} where the model did not pick the reference-best: taking it instead succeeds`) +
      stat(num(refLoses), 'decisions where the reference-best fails but another option (or the recorded one) succeeds') + '</div>';
  } else h += '<p class="note">No what-if results for these episodes yet.</p>';
  if (g.last && g.last.run === ctx.run.name && g.last.check?.n) h += `<p class="chart-note">Replay check of the last job: ${g.last.check.match} of ${g.last.check.n} recorded episodes reproduced exactly without forcing.</p>`;
  return card('What-if branches', '', h, 'wide');
}

function tagCard(rows, ctx) {
  const notes = ctx.notes || {}, counts = {};
  let noted = 0;
  for (const r of rows) {
    const e = notes[r.instr_id];
    if (!e) continue;
    if (e.note) noted++;
    for (const t of e.tags || []) counts[t] = (counts[t] || 0) + 1;
  }
  const tagged = rows.filter(r => notes[r.instr_id]?.tags?.length).length;
  const items = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([t, n]) => ({
    label: t, value: n / Math.max(1, tagged), text: num(n),
    slice: { label: `Tag: ${t}`, test: r => (notes[r.instr_id]?.tags || []).includes(t) } }));
  const body = items.length ? barList(items, ctx, { max: Math.max(...items.map(i => i.value)) })
    : '<p class="note">No tags yet. Add them in the episode view (Tags &amp; notes card, below the decision).</p>';
  return card('Your error tags', `${num(tagged)} tagged · ${num(noted)} with notes`, body);
}

// --------------------------------------------------------------------------------------------- render

export function renderSplit(el, ctx) {
  const slices = [];
  ctx.register = slice => { slices.push(slice || null); return slices.length - 1; };
  const rows = ctx.population;
  const scored = rows.filter(r => r.sr != null);
  const head = `<div class="split-scope"><b>${num(rows.length)}</b> episodes ${ctx.useFilters ? 'matching the list filters' : 'in this split'}` +
    (scored.length ? ` · SR ${pct(mean(scored.map(r => r.sr)), 1)}` : ' · success evaluation unavailable') + '</div>';
  el.innerHTML = head + '<div class="split-grid">' + [
    successCard(rows, ctx), placeCard(rows, ctx), modeCard(rows, ctx), controlCard(rows, ctx), ceStopCard(rows, ctx), subtaskCard(rows, ctx),
    compareCard(rows, ctx), calibrationCard(rows, ctx), firstMistakeCard(rows, ctx), stopCard(rows, ctx),
    whatifCard(rows, ctx), tagCard(rows, ctx), runsCard(ctx),
  ].filter(Boolean).join('') + '</div>';
  el.onclick = e => {
    const run = e.target.closest('[data-run]');
    if (run) return ctx.selectRun(run.dataset.run);
    const w = e.target.closest('[data-whatif]');
    if (w) return ctx.runWhatIf(w.dataset.whatif, w);
    const s = e.target.closest('[data-slice]');
    if (s && slices[s.dataset.slice]) ctx.slice(slices[s.dataset.slice]);
  };
  el.onkeydown = e => {
    const s = e.target.closest('g[data-slice]');
    if (s && (e.key === 'Enter' || e.key === ' ') && slices[s.dataset.slice]) { e.preventDefault(); ctx.slice(slices[s.dataset.slice]); }
  };
  el.querySelector('#splitDimension')?.addEventListener('change', e => { dimension = e.target.value; ctx.rerender(); });
}
