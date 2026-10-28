// VLN Navigation Debugger front end: episode browser, 3D building view (map3d.js) and 2D graph with
// playback, agent views (views.js), instruction sub-tasks with model attention, and the current DUET
// decision. Data comes from visualizer_app.py.
import { Map3D } from './map3d.js';
import { Views } from './views.js';
import { subtaskProgress, firstDivergence, episodeEvents, episodeSummary, metricDifference, successRadius, datasetTitle, comparableRun, finalActionLabel, distanceLabel } from './presentation.js';
import { renderSplit, PLACE, MISTAKE } from './split.js';
import { Investigations } from './investigations.js';

const $ = id => document.getElementById(id);
const OUTCOME = {
  success: { label: 'Success', icon: '✓' },
  failure: { label: 'Failure', icon: '✕' },
  unscored: { label: 'Unscored', icon: '–' },
  'no ground truth': { label: 'No reference', icon: '–' },
  wrong_stop: { label: 'Wrong stop', icon: '✕' },
  step_limit: { label: 'Step limit', icon: '!' },
};
const KIND = { sample: 'Recorded position', observe: 'Observation', unknown: 'Recorded position', end: 'Episode end', stop: 'Stop', start: 'Start', local: 'Local move', remote: 'Remote-frontier jump', return: 'Return to best stop node' };
// Continuous (VLN-CE) runs: moves go to predicted waypoints; a remote one first walks back to an earlier map node.
const KIND_CE = { local: 'Move to a predicted waypoint', remote: 'Walk back, then move to a waypoint', return: 'Walk back to the best stop node' };
const kindLabel = k => (S.ep?.adapter !== 'recording' && S.scan?.ce && KIND_CE[k]) || KIND[k] || 'Recorded move';
// Frames of a continuous run are points of the agent's track (every 0.5 m), not graph hops.
const hopWord = () => (S.scan?.ce ? 'frame' : 'hop');
const SPLIT_ORDER = ['val_unseen', 'val_seen', 'val_train_seen', 'train', 'test',
  'reverie_val_unseen', 'reverie_val_seen', 'rxr_val_unseen', 'rxr_val_seen',
  'r2r_ce_val_unseen', 'r2r_ce_val_seen', 'rxr_ce_val_unseen', 'rxr_ce_val_seen'];
// runs on other datasets: detail_<dataset>_<split>.json; continuous runs: <dataset>_ce_<split>
const splitLabel = s => String(s || 'unspecified').replace(/^(reverie|rxr|r2r)(_ce)?_/, '');
const S = {
  meta: null, run: null, rows: [], filtered: [], compare: '', cmpRows: new Map(),
  outcomes: new Set(Object.keys(OUTCOME)),
  ep: null, scan: null, adj: null, frame: 0, playing: false, timer: null, anim: null,
  advance: null, loadSeq: 0, viewReady: null,
  floor: 'auto', shownFloor: null, view: null, tab: '3d',
  page: 'episode', slice: null, points: null, whatif: null, notes: { notes: {}, vocabulary: [] }, gpu: {},
  investigationRows: null,
};
// Offered as tag chips until the user's own tags take over.
const SUGGESTED_TAGS = ['landmark misread', 'wrong room type', 'wrong floor / stairs', 'stopped too early', 'overshot goal',
  'direction error', 'ambiguous instruction', 'annotation error'];
// Inspection state belongs to the interface; recorded episodes and scores remain read-only.
const UI = { frameKey: '', eventsKey: '', events: [], eventId: null, hoverVp: null, selectedVp: null, divergence: false, goalPreview: null, branch: null };
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (x, d = 1) => (x == null || !isFinite(x) ? '–' : Number(x).toFixed(d));
const pct = x => (x == null ? '–' : (100 * x).toFixed(x >= 0.995 || x < 0.1 ? 1 : 0) + '%');
const short = vp => (vp ? vp.slice(0, 6) : 'STOP');
const statusGlyph = o => (OUTCOME[o] ? `<span class="st ${o}" aria-label="${OUTCOME[o].label}">${OUTCOME[o].icon}</span>` : '');
const runInfo = key => S.meta?.runs?.find(r => r.name === key);

// Name the bundled ScaleVLN checkpoints without applying that branding to other models.
function modelLabel(model) {
  const explicit = S.meta?.runs.find(r => r.model === model && r.model_label)?.model_label;
  if (explicit) return explicit;
  switch (model) {
    case 'released_h14': return 'ScaleVLN · H/14';
    case 'released_b16': return 'ScaleVLN · B/16';
    case 'etpnav': return 'ETPNav · VLN-CE';
    default: return model;
  }
}

// Names and evaluation semantics come from the project rather than filenames.
const datasetLabel = datasetTitle;
const margin = () => successRadius(S.ep, runInfo(S.run)?.evaluation?.success_radius_m ?? S.meta?.error_margin);
const assetAvailable = key => S.ep?.capabilities?.[key] ?? S.scan?.capabilities?.[key] ?? runInfo(S.run)?.capabilities?.[key] ?? false;
const hasDecisionTrace = () => S.ep?.capabilities?.decisions !== false && S.ep?.decisions?.some(d => d.source !== 'replay_placeholder');
const decisionAt = frame => S.ep.decisions[frame.decision] || { t: null, action: 'unknown', source: 'replay_placeholder', at: frame.vp, target: frame.vp, route: [], visited: [], frontier: [] };
const decisionContext = frame => hasDecisionTrace() && Number.isInteger(frame.decision) ? `decision ${frame.decision}/${S.ep.decisions.length - 1}` : 'decision trace unavailable';

async function getJSON(url, opts) {
  const r = await fetch(url, { cache: 'no-store', ...opts });
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).error || msg; } catch (_) { /* not JSON */ }
    throw new Error(msg);
  }
  return r.json();
}

const mapHooks = {
  status: msg => { $('status3d').textContent = msg; $('status3d').hidden = !msg || S.tab !== '3d'; },
  mode: m => {
    document.querySelectorAll('#camModes button').forEach(b => b.setAttribute('aria-pressed', b.dataset.cam === m));
    $('jumpGoal').setAttribute('aria-pressed', !!UI.goalPreview);
    updateCameraContext(m);
  },
};
function createMap3D() {
  try { return new Map3D($('map3d'), mapHooks); }
  catch (error) {
    console.warn('3D rendering unavailable; 2D replay remains available.', error.message);
    $('tab3d').disabled = true;
    $('tab3d').title = 'WebGL is unavailable in this browser; use the 2D graph.';
    const fallback = { available: false, mode: 'chase', baseReady: Promise.resolve() };
    for (const name of ['meshVisible', 'recolor', 'setActive', 'setCamera', 'setCut', 'setCutaway', 'setDetail', 'setEpisode', 'setFloorMode', 'setFrame', 'setInspection', 'setNodesVisible', 'setPhoto', 'setPresentation', 'setScan']) fallback[name] = () => {};
    return fallback;
  }
}
const map3d = createMap3D();
const views = new Views({ front: $('fpv'), frontSvg: $('fpvSvg'), pano: $('pano'), panoSvg: $('panoSvg'),
                          msg: $('fpvMsg'), panoMsg: $('panoMsg'), hint: $('fpvHint'), follow: $('fpvFollow') });
const investigations = new Investigations({
  page: $('investigationPage'), card: $('investigationCard'),
  onOpen: async id => {
    const run = S.run;
    S.investigationRows = (investigations.queueData?.cases || []).map(row => ({ instr_id: row.instr_id }));
    setTab('2d');
    const loaded = await loadEpisode(id);
    if (!loaded || S.run !== run || S.ep?.instr_id !== id) return;
    setPage('episode');
    await investigations.loadCase(run, id);
    if (S.run === run && S.ep?.instr_id === id) investigations.focusFirst();
  },
  onSeek: frame => { if (S.ep) { stop(); setFrame(frame); } },
  onQueue: () => setPage('investigation'),
});

// ------------------------------------------------------------------------------------ runs & rows

async function init() {
  S.meta = await getJSON('/api/meta');
  const h = new URLSearchParams(location.hash.slice(1));
  const autoplay = h.get('autoplay') === '1';
  if (autoplay) S.outcomes = new Set(Object.keys(OUTCOME));
  if (!S.meta.runs.length) { $('mapEmpty').textContent = 'No runs loaded. Add a run to the project manifest and restart the explorer.'; return; }
  const models = [...new Set(S.meta.runs.map(r => r.model))];
  $('model').innerHTML = models.map(m => `<option value="${esc(m)}">${esc(modelLabel(m))}</option>`).join('');
  const want = runInfo(h.get('run')) || S.meta.runs.find(r => r.split === 'val_unseen' && r.model === models[models.length - 1]) || S.meta.runs[0];
  $('model').value = want.model;
  fillSplits(want.model, want.name);
  await selectRun(want.name, false, h.get('cmp'));
  if (!S.filtered.length && S.rows.length) { S.outcomes = new Set(Object.keys(OUTCOME)); renderChips(); applyFilters(); }
  const id = h.get('id');
  // A direct review-queue link needs summaries, not an arbitrary episode's meshes and images.
  if (h.get('view') === 'investigation' && !id) {
    setPage('investigation');
    refreshGpu();
    return;
  }
  if (id && S.rows.some(r => r.instr_id === id)) {
    if (!S.filtered.some(r => r.instr_id === id)) { S.outcomes = new Set(Object.keys(OUTCOME)); renderChips(); applyFilters(); }
    await loadEpisode(id, Number(h.get('f') || 0));
  } else if (S.filtered.length) {
    await loadEpisode(S.filtered[0].instr_id);
  }
  if (autoplay && S.ep) play();
  if (['split', 'investigation'].includes(h.get('view'))) setPage(h.get('view'));
  refreshGpu();
}

function fillSplits(model, selected) {
  const runs = S.meta.runs.filter(r => r.model === model)
    .sort((a, b) => (SPLIT_ORDER.indexOf(a.split) + 1 || 99) - (SPLIT_ORDER.indexOf(b.split) + 1 || 99));
  $('split').innerHTML = runs.map(r => `<option value="${esc(r.name)}">${esc(datasetLabel(r))} · ${esc(splitLabel(r.split))} (${r.summary.n})</option>`).join('');
  const sel = runs.find(r => r.name === selected) || runs.find(r => r.split === selected && r.dataset === runInfo(S.run)?.dataset) || runs.find(r => r.split === selected) || runs[0];
  $('split').value = sel.name;
  return sel.name;
}

async function selectRun(key, reload = true, cmp = undefined) {
  stop();
  ++S.loadSeq;
  S.run = key;
  clearInvestigationNavigation();
  investigations.setRun(key);
  const info = runInfo(key);
  const [rows, notes] = await Promise.all([getJSON(`/api/rows?run=${encodeURIComponent(key)}`),
    getJSON(`/api/notes?run=${encodeURIComponent(key)}`)]);
  if (S.run !== key) return;
  S.rows = rows;
  S.notes = notes;
  S.notes.notes = Object.assign(Object.create(null), notes.notes);
  S.points = S.whatif = null;           // loaded when the split page needs them
  S.slice = null;
  const s = info.summary;
  const dataset = datasetLabel(info);
  $('split').title = $('split').selectedOptions[0]?.textContent || '';
  $('datasetName').textContent = dataset;
  const about = info.dataset_description || { R2R: 'Room-to-Room', REVERIE: 'Remote object goals', RxR: 'Room-across-Room',
    'R2R-CE': 'Room-to-Room in continuous environments (Habitat)', 'RxR-CE': 'Room-across-Room in continuous environments (Habitat)' }[dataset];
  $('datasetDescription').textContent = [about, info.note].filter(Boolean).join(' · ');
  $('datasetDescription').hidden = !$('datasetDescription').textContent;
  $('runDetails').innerHTML = `<h2>Recorded rollout</h2><p>Playback of saved predictions and decision logs.</p><dl>` +
    `<dt>Model / checkpoint</dt><dd>${esc(modelLabel(info.model))}</dd><dt>Dataset</dt><dd>${esc(dataset)} · ${esc(splitLabel(info.split))}</dd>` +
    `<dt>Run</dt><dd>${esc(info.name)}</dd><dt>Prediction file</dt><dd>${esc(String(info.predictions || '').split(/[\\/]/).pop() || 'Not provided')}</dd>` +
    `<dt>Decision evidence</dt><dd>${info.steplog ? 'Step log available; scores shown when logged for the selected decision' : 'No step log; action scores unavailable'}</dd>` +
    `<dt>Run timestamp</dt><dd>${esc(info.created_at || 'Not supplied')}</dd></dl>`;
  $('runSummary').innerHTML = (s.sr == null ? `<b>${s.n}</b> episodes · success evaluation unavailable` :
    `<b>${s.n}</b> episodes · <abbr title="Success rate">SR</abbr> <b>${fmt(s.sr, 2)}</b> · <abbr title="Success weighted by path length">SPL</abbr> <b>${fmt(s.spl, 2)}</b> · <abbr title="Oracle success rate: came within the success radius">OSR</abbr> ${fmt(s.osr, 2)} · <abbr title="Final navigation error: route distance to the goal">NE</abbr> ${fmt(s.ne, 2)} m`) +
    (info.steplog ? ' · step log ✓' : ' · <span title="Add recorded decision scores to inspect model choices">no decision scores</span>');
  if (s.metric_counts?.sr != null && s.metric_counts.sr < s.n) $('runSummary').innerHTML += ` · ${s.metric_counts.sr}/${s.n} episodes scored`;
  // Comparison eligibility is established from dataset and coordinate identity by the server.
  const cross = r => r.env !== info.env;
  const others = S.meta.runs.filter(r => comparableRun(info, r));
  $('compare').innerHTML = '<option value="">none</option>' + others.map(r => `<option value="${esc(r.name)}">${esc(modelLabel(r.model))}${cross(r) ? (r.env === 'continuous' ? ' · continuous' : ' · discrete graph') : ''}</option>`).join('');
  const keep = cmp !== undefined && cmp !== null ? cmp : (S.compare && runInfo(S.compare)?.model);
  const pick = others.find(r => r.name === keep || r.model === keep) || (!keep && !S.compare && others[0]) || null;
  $('compare').value = pick ? pick.name : '';
  await selectCompare($('compare').value, false);
  // continuous runs never revisit a viewpoint; their list column, filter and sorts are about the controller
  const ce = info.env === 'continuous';
  document.querySelectorAll('#sort option[data-env]').forEach(o => { o.hidden = o.dataset.env !== (ce ? 'continuous' : 'discrete'); });
  if ($('sort').selectedOptions[0]?.hidden) $('sort').value = 'id';
  $('thLast').textContent = ce ? '⊘' : '↺';
  $('thLast').title = ce ? 'moves that went nowhere (< 0.25 m)' : 'viewpoints visited ≥2×';
  $('fRevLabel').textContent = ce ? 'Stuck moves ≥' : 'Revisited ≥';
  $('fRemoteLabel').textContent = ce ? 'Walked back to an earlier node' : 'Remote-frontier jump';
  const modes = [...new Set(S.rows.map(r => r.failure_mode))].sort();
  $('fMode').innerHTML = '<option value="">any</option>' + modes.map(m => `<option>${esc(m)}</option>`).join('');
  const scans = [...new Set(S.rows.map(r => r.scan))].sort();
  $('fScan').innerHTML = '<option value="">any</option>' + scans.map(m => `<option>${esc(m)}</option>`).join('');
  const subs = [...new Set(S.rows.map(r => r.first_missed_subtask).filter(x => x != null))].sort((a, b) => a - b);
  $('fSub').innerHTML = '<option value="">any</option>' + subs.map(k => `<option value="${k}">sub-task ${k}</option>`).join('');
  const places = Object.keys(PLACE).filter(k => S.rows.some(r => r.stop_place === k));
  $('fPlace').innerHTML = '<option value="">any</option>' + places.map(k => `<option value="${k}">${esc(PLACE[k])}</option>`).join('');
  const firsts = [...new Set(S.rows.map(r => r.first_nonbest).filter(x => x != null))].sort((a, b) => a - b);
  $('fFirst').innerHTML = '<option value="">any</option>' + firsts.map(t => `<option value="${t}">decision ${t}</option>`).join('');
  $('fFirst').disabled = $('fConfident').disabled = !firsts.length;
  renderTagFilter();
  renderChips();
  applyFilters();
  if (S.page === 'split') loadSplitData().then(renderSplitPage);
  if (S.page === 'investigation') investigations.loadQueue();
  if (reload && S.ep) {
    const same = S.rows.some(r => r.instr_id === S.ep.instr_id);
    const id = same ? S.ep.instr_id : S.filtered[0]?.instr_id;
    if (id) await loadEpisode(id, same ? S.frame : 0);
  }
}

async function selectCompare(key, redraw = true) {
  S.compare = key;
  S.cmpRows = new Map();
  if (key) (await getJSON(`/api/rows?run=${encodeURIComponent(key)}`)).forEach(r => S.cmpRows.set(r.instr_id, r));
  $('fDiffWrap').hidden = !key;
  if (!key) $('fDiff').checked = false;
  if (!redraw) return;
  applyFilters();
  if (S.ep) { map3d.setEpisode(S.ep, otherPath()); fitView(); renderStatic(); setFrame(S.frame, false, true); updateMapControls(); renderLegend(); }
}

function renderChips() {
  const counts = {};
  S.rows.forEach(r => { counts[r.outcome] = (counts[r.outcome] || 0) + 1; });
  $('outcomeChips').innerHTML = Object.entries(OUTCOME).map(([k, o]) =>
    `<button class="chip" data-o="${k}" aria-pressed="${S.outcomes.has(k)}">${statusGlyph(k)}${o.label}<span class="n">${counts[k] || 0}</span></button>`).join('');
}

function filterRows(withSlice = true) {
  const mode = $('fMode').value, scan = $('fScan').value, sub = $('fSub').value;
  const place = $('fPlace').value, first = $('fFirst').value, tag = $('fTag').value, confident = $('fConfident').checked;
  const minLen = parseFloat($('fLen').value), minRev = parseInt($('fRev').value, 10);
  const text = $('fText').value.trim().toLowerCase();
  const diff = $('fDiff').checked && S.compare;
  const tags = id => S.notes.notes[id]?.tags || [];
  return S.rows.filter(r =>
    (S.outcomes.has(r.outcome) || !OUTCOME[r.outcome]) &&
    (!mode || r.failure_mode === mode) && (!scan || r.scan === scan) &&
    (!sub || String(r.first_missed_subtask) === sub) &&
    (isNaN(minLen) || r.tl > minLen) && (isNaN(minRev) || (r.stuck ?? r.revisited_nodes) >= minRev) &&
    (!$('fRemote').checked || r.remote_jumps > 0) && (!$('fReturn').checked || r.stop_return) &&
    (!diff || (S.cmpRows.has(r.instr_id) && S.cmpRows.get(r.instr_id).sr !== r.sr)) &&
    (!place || r.stop_place === place) && (!first || String(r.first_nonbest) === first) &&
    (!confident || r.first_nonbest_conf >= 0.9) && (!tag || tags(r.instr_id).includes(tag)) &&
    (!withSlice || !S.slice || S.slice.test(r)) &&
    (!text || r.instr_id.toLowerCase().includes(text) || r.instruction.toLowerCase().includes(text)));
}

function applyFilters() {
  if (S.advance) stop();
  const rows = filterRows();
  const key = $('sort').value;
  const num = id => id.split('_').map(Number);
  if (key === 'id') rows.sort((a, b) => { const x = num(a.instr_id), y = num(b.instr_id); return x[0] - y[0] || x[1] - y[1] || a.instr_id.localeCompare(b.instr_id); });
  else if (key === 'ndtw') rows.sort((a, b) => (a.ndtw ?? 0) - (b.ndtw ?? 0));
  else if (key === 'first_nonbest') rows.sort((a, b) => (a.first_nonbest ?? Infinity) - (b.first_nonbest ?? Infinity));
  else rows.sort((a, b) => (b[key] ?? -1) - (a[key] ?? -1));
  S.filtered = rows;
  $('sliceChip').hidden = !S.slice;
  if (S.slice) $('sliceChip').innerHTML = `<span>Slice: ${esc(S.slice.label)}</span><button id="clearSlice" aria-label="Clear slice" title="Show the list without this slice">✕</button>`;
  $('count').textContent = `${rows.length} of ${S.rows.length}`;
  const keyboardRow = rows.find(r => r.instr_id === S.ep?.instr_id) || rows[0];
  const cmpName = S.compare && modelLabel(runInfo(S.compare)?.model);
  $('rows').innerHTML = rows.map(r => {
    const c = S.compare && S.cmpRows.get(r.instr_id);
    const cmp = c ? ` title="${esc(cmpName)}: ${OUTCOME[c.outcome]?.label || c.outcome}, NE ${fmt(c.ne)} m"` : '';
    return `<tr data-id="${esc(r.instr_id)}"${S.ep && S.ep.instr_id === r.instr_id ? ' class="sel"' : ''}${cmp}>` +
      `<td>${statusGlyph(r.outcome)}</td><td><button class="episode-pick" tabindex="${r === keyboardRow ? 0 : -1}" aria-label="Open episode ${esc(r.instr_id)}" title="${esc(r.instruction)}">${esc(r.instr_id)}</button>${c && c.sr !== r.sr ? ` <span title="${esc(cmpName)} differs">${c.sr ? '◆' : '◇'}</span>` : ''}${tagDot(r.instr_id)}</td>` +
      `<td>${fmt(r.ne)}</td><td>${fmt(r.tl)}</td><td>${r.subtasks ? `${r.subtasks_done}/${r.subtasks}` : ''}</td><td>${(r.stuck ?? r.revisited_nodes) || ''}</td></tr>`;
  }).join('');
  if (S.page === 'split' && $('splitUseFilters').checked) renderSplitPage();
}

function tagDot(id) {
  const e = S.notes.notes[id];
  if (!e) return '';
  const title = [(e.tags || []).join(', '), e.note].filter(Boolean).join(' · ');
  return ` <span class="tag-dot" title="${esc(title)}" aria-label="Tagged: ${esc(title)}">●</span>`;
}

function renderTagFilter() {
  const used = [...new Set(Object.values(S.notes.notes).flatMap(e => e.tags || []))].sort();
  const keep = $('fTag').value;
  $('fTag').innerHTML = '<option value="">any</option>' + used.map(t => `<option>${esc(t)}</option>`).join('');
  $('fTag').value = used.includes(keep) ? keep : '';
  $('fTag').disabled = !used.length;
}

function applySlice(slice) {
  S.slice = slice;
  applyFilters();
  // a slice of successful episodes would otherwise hide behind the default failures-only chips
  if (slice && !S.filtered.length && S.rows.some(slice.test)) { S.outcomes = new Set(Object.keys(OUTCOME)); renderChips(); applyFilters(); }
  document.querySelector('.table-wrap').scrollTop = 0;
}

// ------------------------------------------------------------------------------------- split analysis

function setPage(page) {
  S.page = page;
  $('layout').classList.toggle('split-mode', page === 'split');
  $('layout').classList.toggle('investigation-mode', page === 'investigation');
  $('splitPage').hidden = page !== 'split';
  $('investigationPage').hidden = page !== 'investigation';
  $('viewEpisode').setAttribute('aria-pressed', page === 'episode');
  $('viewSplit').setAttribute('aria-pressed', page === 'split');
  $('viewInvestigation').setAttribute('aria-pressed', page === 'investigation');
  map3d.setActive(page === 'episode' && S.tab === '3d');     // no 3D rendering behind the split page
  if (page === 'split') { stop(); loadSplitData().then(renderSplitPage); }
  else if (page === 'investigation') { stop(); investigations.loadQueue(); }
  else if (!S.ep && S.filtered.length) loadEpisode(S.filtered[0].instr_id);
  updateHash();
}

async function loadSplitData(force = false) {
  const run = S.run, q = `run=${encodeURIComponent(run)}`;
  const [points, whatif] = await Promise.all([S.points && !force ? S.points : getJSON(`/api/points?${q}`),
    S.whatif && !force ? S.whatif : getJSON(`/api/whatif/results?${q}`)]);
  if (S.run !== run) return;
  S.points = points;
  S.whatif = whatif;
}

function renderSplitPage() {
  if (S.page !== 'split' || !S.points) return;
  const info = runInfo(S.run), useFilters = $('splitUseFilters').checked;
  renderSplit($('splitBody'), {
    run: info, runs: S.meta.runs, population: useFilters ? filterRows(false) : S.rows, useFilters, listed: S.filtered,
    cmp: S.compare ? { label: modelLabel(runInfo(S.compare).model), rows: S.cmpRows } : null, model: modelLabel(info.model),
    points: S.points, whatif: S.whatif, notes: S.notes.notes, gpu: S.gpu,
    referenceNote: info.reference === 'none' ? 'Reference-best actions are not inferred for this recording.' : info.reference === 'route' ? 'Reference-best follows the annotated route.' : 'Reference-best uses the supplied evaluation goal distances and success criterion.',
    dataset: datasetLabel, splitLabel, modelLabel, splitRank: s => SPLIT_ORDER.indexOf(s) + 1 || 99,
    slice: applySlice, rerender: renderSplitPage, runWhatIf,
    selectRun: async key => { const r = runInfo(key); $('model').value = r.model; fillSplits(r.model, r.name); await selectRun(key); },
  });
}

// ------------------------------------------------------------------------------------------- what-if

async function postJSON(url, body) {
  return getJSON(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
}

async function runWhatIf(kind, btn) {
  const items = kind === 'first'
    ? S.filtered.filter(r => r.sr === 0 && r.first_nonbest != null).map(r => [r.instr_id, r.first_nonbest])
    : S.filtered.flatMap(r => Array.from({ length: r.decisions || 0 }, (_, t) => [r.instr_id, t]));
  const label = kind === 'first' ? `first non-best decision · ${items.length} episodes` : `every decision · ${S.filtered.length} episodes`;
  if (btn) btn.disabled = true;
  try {
    const plan = await postJSON('/api/whatif/run', { run: S.run, items, dry_run: true });
    if (!plan.branches) { alert(`Nothing to run: ${plan.cached} branches are already stored${plan.skipped ? `, ${plan.skipped} decisions cannot be branched` : ''}.`); return; }
    if (!confirm(`Run ${plan.branches.toLocaleString()} what-if branches on GPU ${S.gpu.gpu}?\n${plan.cached} are already stored and will be reused.`)) return;
    await postJSON('/api/whatif/run', { run: S.run, items, label });
    refreshGpu();
  } catch (e) { alert('What-if: ' + e.message); } finally { if (btn) btn.disabled = false; }
}

let gpuTimer = 0, gpuJob = null;
async function refreshGpu() {
  clearTimeout(gpuTimer);
  try { S.gpu = await getJSON('/api/whatif/status'); } catch (_) { S.gpu = {}; }
  const g = S.gpu, job = g.job;
  $('gpuSwitch').hidden = g.allowed === false;
  $('optGPU').checked = !!g.enabled;
  $('gpuCancel').hidden = !(job || g.queued?.length);
  const last = g.last && g.last.run ? `${g.last.phase} ${g.last.done}/${g.last.total}` : '';
  $('gpuState').textContent = !g.enabled ? 'off' : job ? (job.phase === 'running' ? `running ${job.done}/${job.total}` : `${job.phase}…`) + (g.queued?.length ? ` · ${g.queued.length} queued` : '')
    : `on · idle${g.loaded?.length ? ' (model loaded)' : ''}${last ? ' · last job ' + last : ''}`;
  $('gpuState').dataset.state = g.enabled ? (job ? 'busy' : 'on') : 'off';
  const finished = gpuJob && (!job || job.id !== gpuJob);
  gpuJob = job ? job.id : null;
  if (finished || (job && job.done && job.done % 64 === 0)) await refreshWhatIf();
  else if (S.page === 'split') renderSplitPage();
  gpuTimer = setTimeout(refreshGpu, job || g.queued?.length ? 2000 : g.enabled ? 10000 : 30000);
}

async function refreshWhatIf() {
  if (S.ep) {
    const run = S.run, id = S.ep.instr_id;
    const ep = await getJSON(`/api/episode?run=${encodeURIComponent(run)}&id=${encodeURIComponent(id)}`);
    if (S.run === run && S.ep?.instr_id === id) { S.ep.whatif = ep.whatif; renderDecision(frameState(S.frame)); }
  }
  if (S.whatif || S.page === 'split') { await loadSplitData(true); renderSplitPage(); }
}

async function whatIfDecision(t) {
  try {
    const res = await postJSON('/api/whatif/run', { run: S.run, items: [[S.ep.instr_id, t]], label: `${S.ep.instr_id} · decision ${t}` });
    $('decisionCard').querySelector('.whatif-line').textContent = res.branches ? `Queued ${res.branches} branches…` : 'All options are already branched.';
    refreshGpu();
  } catch (e) { alert('What-if: ' + e.message); }
}

function toggleBranch(key) {
  const [t, force] = key.split('|'), wi = S.ep.whatif?.[t]?.[force];
  UI.branch = !wi || UI.branch?.key === key ? null : { key, path: wi.path, name: `What-if · ${force === 'STOP' ? 'STOP' : short(force)} at decision ${t}` };
  map3d.setEpisode(S.ep, otherPath()); renderStatic(); setFrame(S.frame, false, true); updateMapControls(); renderLegend();
}

// --------------------------------------------------------------------------------------- tags & notes

function renderNotes(force = false) {
  const e = noteDrafts.get(JSON.stringify([S.run, S.ep.instr_id])) || S.notes.notes[S.ep.instr_id] || { tags: [], note: '' };
  const vocab = S.notes.vocabulary.length >= 6 ? S.notes.vocabulary.slice(0, 14) : [...new Set([...S.notes.vocabulary, ...SUGGESTED_TAGS])];
  const tags = [...new Set([...e.tags, ...vocab])];
  $('tagChips').innerHTML = tags.map(t => `<button class="tag-chip" aria-pressed="${e.tags.includes(t)}" data-tag="${esc(t)}">${esc(t)}</button>`).join('');
  $('tagVocabulary').innerHTML = S.notes.vocabulary.map(t => `<option value="${esc(t)}">`).join('');
  if (force || document.activeElement !== $('noteText')) $('noteText').value = e.note || '';
  $('notesStatus').textContent = '';
}

const noteTimers = new Map();
const noteDrafts = new Map();
async function saveNotes(tags, note, id = S.ep.instr_id, run = S.run) {
  const key = JSON.stringify([run, id]), draft = { tags, note };
  clearTimeout(noteTimers.get(key)); noteTimers.delete(key);
  noteDrafts.set(key, draft);
  if (S.run === run && S.ep?.instr_id === id) $('notesStatus').textContent = 'Saving…';
  try {
    const e = await postJSON('/api/notes', { run, id, tags, note });
    if (noteDrafts.get(key) === draft) noteDrafts.delete(key);
    if (S.run !== run) return;
    if (e.tags.length || e.note) S.notes.notes[id] = e; else delete S.notes.notes[id];
    const counts = Object.create(null);
    Object.values(S.notes.notes).forEach(x => (x.tags || []).forEach(t => { counts[t] = (counts[t] || 0) + 1; }));
    S.notes.vocabulary = [...new Set([...Object.keys(counts).sort((a, b) => counts[b] - counts[a]), ...S.notes.vocabulary])];
    const cell = $('rows').querySelector(`tr[data-id="${CSS.escape(id)}"] td:nth-child(2)`);
    if (cell) { cell.querySelector('.tag-dot')?.remove(); cell.insertAdjacentHTML('beforeend', tagDot(id)); }
    renderTagFilter();
    if (S.ep?.instr_id === id) { renderNotes(); $('notesStatus').textContent = noteDrafts.has(key) ? 'Editing…' : 'Saved'; }
  } catch (err) { if (S.run === run && S.ep?.instr_id === id) $('notesStatus').textContent = 'Not saved: ' + err.message; }
}
function currentTags() { return S.notes.notes[S.ep.instr_id]?.tags || []; }

async function exportRows(btn) {
  const body = { run: S.run, ids: S.filtered.map(r => r.instr_id), format: btn.dataset.fmt, full: !!btn.dataset.full,
                 label: S.outcomes.has('success') ? 'filtered' : 'failures' };
  btn.disabled = true;
  try {
    const r = await fetch('/api/export', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!r.ok) throw new Error((await r.json()).error);
    const name = (r.headers.get('Content-Disposition') || '').match(/filename="(.+)"/)?.[1] || 'export';
    const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(await r.blob()), download: name });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  } catch (e) { alert('Export failed: ' + e.message); } finally { btn.disabled = false; }
}

// ---------------------------------------------------------------------------------------- episode

async function loadEpisode(id, frame = 0, advance = null) {
  if (advance && S.advance !== advance) return false;
  if (!advance) stop();
  const run = S.run, request = ++S.loadSeq;
  // A late auto-next response must not replace an episode the user chose in the meantime.
  const current = () => request === S.loadSeq && S.run === run && (!advance || S.advance === advance);
  const ep = await getJSON(`/api/episode?run=${encodeURIComponent(run)}&id=${encodeURIComponent(id)}`);
  if (!current()) return false;
  // continuous (VLN-CE) episodes bring their own graph: map nodes, ghost waypoints, track samples, reference points
  const scan = ep.graph || (S.scan?.scan === ep.scan && !S.scan.ce ? S.scan : await getJSON(`/api/scan?run=${encodeURIComponent(run)}&scan=${encodeURIComponent(ep.scan)}`));
  if (!current()) return false;
  if (!advance) stop();
  if (S.scan !== scan) {
    S.scan = scan;
    S.adj = new Map();
    for (const [a, b] of S.scan.edges) {
      (S.adj.get(a) || S.adj.set(a, new Set()).get(a)).add(b);
      (S.adj.get(b) || S.adj.set(b, new Set()).get(b)).add(a);
    }
  }
  // the same instruction run in the other setting brings its route's own points (kind 'compare')
  for (const o of Object.values(ep.others || {})) if (o.nodes) Object.assign(S.scan.nodes, o.nodes);
  ep.decisionFrame = [];
  ep.frames.forEach((f, i) => { if (Number.isInteger(f.decision) && ep.decisionFrame[f.decision] == null) ep.decisionFrame[f.decision] = i; });
  ep.gtIndex = new Map(ep.gt_path.map((v, i) => [v, i]));
  ep.zone = new Set(ep.goal_zone || []);
  leaveGoalPreview();
  UI.branch = null;
  S.ep = ep;
  if (S.investigationRows && !S.investigationRows.some(row => row.instr_id === id)) S.investigationRows = null;
  updateInvestigationNavigation();
  S.floor = 'auto';
  S.shownFloor = null;
  S.view = null;
  views.look = { yaw: 0, pitch: 0, free: false };
  $('mapEmpty').hidden = true;
  $('jump').max = Math.max(0, ep.decisions.length - 1);
  for (const id of ['jump', 'bPrevD', 'bNextD']) $(id).disabled = !hasDecisionTrace();
  if (!assetAvailable('mesh')) setTab('2d');
  $('tab3d').textContent = assetAvailable('mesh') ? '3D building' : '3D path';
  $('optMesh').disabled = $('optHD').disabled = !assetAvailable('mesh');
  $('optPhoto').disabled = !assetAvailable('mesh') || !assetAvailable('images');
  $('epTitle').innerHTML = `${statusGlyph(ep.outcome)}<span>${esc(ep.instr_id)}</span>`;
  document.querySelectorAll('#rows tr').forEach(tr => {
    tr.classList.toggle('sel', tr.dataset.id === id);
    tr.querySelector('.episode-pick').tabIndex = tr.dataset.id === id ? 0 : -1;
  });
  if (!$('rows').querySelector('.episode-pick[tabindex="0"]')) { const first = $('rows').querySelector('.episode-pick'); if (first) first.tabIndex = 0; }
  document.querySelector('#rows tr.sel')?.scrollIntoView({ block: 'nearest' });
  map3d.setScan(S.scan);      // the building mesh loads in the background
  map3d.setFloorMode('auto');
  if (map3d.mode === 'goal') map3d.setCamera('chase', true);
  map3d.setEpisode(ep, otherPath());
  renderFloors();
  setFrame(Math.min(frame, ep.frames.length - 1));
  updateMapControls();
  renderLegend();
  renderNotes(true);
  investigations.loadCase(run, id);
  return true;
}

function otherPath() {
  if (UI.branch) return { path: UI.branch.path, name: UI.branch.name, endLabel: 'end' };   // a what-if branch replaces the comparison route
  const o = S.compare && S.ep?.others[S.compare];
  return o ? { path: o.path, name: modelLabel(runInfo(S.compare).model), endLabel: finalActionLabel(o).toLowerCase() } : null;
}
function node(vp) { return S.scan.nodes[vp]; }
function curFrame() { return S.ep.frames[S.frame]; }
function curDecision() { return decisionAt(curFrame()); }

function modelState(dec) {
  // Graph memory at a decision: logged DUET global map when available, otherwise reconstructed.
  if (dec.model && Array.isArray(dec.model.gmap) && Array.isArray(dec.model.fused)) {
    const m = dec.model;
    const visited = m.gmap.filter((_, i) => m.visited[i]);
    const frontier = m.gmap.filter((_, i) => !m.visited[i]);
    const score = new Map(m.gmap.map((vp, i) => [vp, m.fused[i + 1]]));
    const global = new Map(m.gmap.map((vp, i) => [vp, (m.global || [])[i + 1]]));
    return { visited, frontier, score, global, stop: m.fused[0], logged: true };
  }
  return { visited: dec.visited || [], frontier: dec.frontier || [], score: null, stop: dec.stop_prob, logged: false };
}

function oracleChoice(dec, st) {
  // Best option by geodesic distance to the goal: stop if inside the success radius, else the frontier
  // node closest to the goal. With a step log the server decides (RxR: along the annotated route).
  const ep = S.ep;
  if (!ep.goal || ep.reference === 'none') return null;
  if (dec.ref) return { vp: dec.ref.best, dist: ep.dist_to_goal[dec.ref.best || dec.at] };
  if (margin() != null && ep.dist_to_goal[dec.at] != null && ep.dist_to_goal[dec.at] < margin()) return { vp: null, dist: ep.dist_to_goal[dec.at] };
  let best = null;
  for (const vp of st.frontier) if (!best || ep.dist_to_goal[vp] < best.dist) best = { vp, dist: ep.dist_to_goal[vp] };
  return best;
}

function frameState(f, t = 1) {
  // Everything the maps, panels and views need at frame f (t < 1 while a hop is animating).
  const ep = S.ep, fr = ep.frames[f], dec = decisionAt(fr), st = modelState(dec);
  const counts = new Map();
  for (let i = 0; i <= f; i++) counts.set(ep.frames[i].vp, (counts.get(ep.frames[i].vp) || 0) + 1);
  if (t < 1) counts.set(fr.vp, counts.get(fr.vp) - 1);
  const ranked = st.score ? [...st.frontier].sort((a, b) => st.score.get(b) - st.score.get(a)) : [];
  const top = new Set(ranked.slice(0, 5).filter(vp => st.score.get(vp) >= 0.01));
  const atDecisionNode = hasDecisionTrace() && ep.decisionFrame[fr.decision] === f;
  const orc = oracleChoice(dec, st);
  const sub = ep.subtasks || {};
  const subIdx = fr.subtask;
  return {
    frame: f, fr, dec, state: st, counts, top, atDecisionNode, orc,
    chosen: ['local', 'remote'].includes(dec.action) && dec.target && (atDecisionNode || fr.kind === 'remote') ? (dec.choice || dec.target) : null,
    // continuous runs render views only at map nodes: between nodes, show the node the agent decided at
    skyVp: ep.adapter === 'recording' || !S.scan.ce || node(fr.vp)?.rot?.length === 9 ? fr.vp : dec.at,
    oracle: orc ? (orc.vp || dec.at) : null,
    lost: ep.diag.lost_hop != null && f >= ep.diag.lost_hop ? ep.path[ep.diag.lost_hop] : null,
    final: f === ep.frames.length - 1 ? { text: `${finalActionLabel(ep)}${ep.metrics.ne != null ? ' · ' + fmt(ep.metrics.ne) + ' m' : ''}`, cls: ep.outcome } : null,
    subIdx, subGoal: sub.goals && subIdx != null ? sub.goals[subIdx] : null, chunk: sub.chunks && subIdx != null ? sub.chunks[subIdx] : null,
    labels: { scores: $('optScores').checked, stop: $('optStop').checked, detailed: $('optLabels').checked,
      selected: UI.hoverVp || UI.selectedVp, divergence: UI.divergence ? comparisonDivergence() : null },
  };
}

function setFrame(f, animate = false, preservePreview = false) {
  const ep = S.ep;
  f = Math.max(0, Math.min(ep.frames.length - 1, f));
  if (!preservePreview && leaveGoalPreview()) {
    S.floor = 'auto'; S.shownFloor = null;
    map3d.setFloorMode('auto'); map3d.setCamera('chase', true); renderFloors();
  }
  const frameKey = `${S.run}/${ep.instr_id}/${f}`;
  if (UI.frameKey !== frameKey) { UI.frameKey = frameKey; UI.hoverVp = UI.selectedVp = null; UI.divergence = false; }
  const from = S.frame;
  S.frame = f;
  const fs = frameState(f);
  const dur = animate && f === from + 1 ? Math.min(800, Number($('speed').value) * 1000 * 0.7) : 0;
  // 2D map
  const floor = S.floor === 'auto' ? ep.frames[f].floor : S.floor;
  if (floor !== S.shownFloor) { S.shownFloor = floor; fitView(); renderStatic(); }
  if (S.anim) cancelAnimationFrame(S.anim.raf);
  S.anim = null;
  if (dur && S.tab === '2d') {
    const t0 = performance.now();
    const tick = now => {
      const t = Math.min(1, (now - t0) / dur);
      renderDynamic(1 - Math.pow(1 - t, 3));
      S.anim = t < 1 ? { raf: requestAnimationFrame(tick) } : null;
    };
    S.anim = { raf: requestAnimationFrame(tick) };
  } else {
    renderDynamic(1);
  }
  // 3D map
  map3d.setFrame({ ...fs, animate: dur });
  renderPanels(fs);
  investigations.setFrame(f);
  renderTimeline();
  showCameraViews(fs);
  const next = ep.frames[f + 1];
  if (assetAvailable('images') && next && node(next.vp)?.sky) views.prefetch(S.scan.scan, next.vp, node(next.vp).sky);
  updateHash();
  return dur;
}

function updateHash() {
  if (!S.run) return;
  const p = new URLSearchParams({ run: S.run });
  if (S.ep) { p.set('id', S.ep.instr_id); p.set('f', S.frame); }
  if (S.compare) p.set('cmp', S.compare);
  if (S.page !== 'episode') p.set('view', S.page);
  history.replaceState(null, '', '#' + p);
}

// ------------------------------------------------------------------------------------- view markers

function updateViewContext() {
  const preview = !!UI.goalPreview;
  const provenance = S.ep?.image_provenance || S.ep?.ce?.image_provenance;
  $('cameraProvenance').hidden = !S.ep;
  $('cameraProvenance').textContent = provenance?.label || 'Image provenance unknown';
  $('cameraProvenance').title = provenance?.description || 'Image origin has not been verified for this recording.';
  const container = document.querySelector('.views');
  container.dataset.preview = preview ? 'goal' : 'agent';
  container.hidden = !$('optViews').checked || !assetAvailable('images');
  $('assetNotice').hidden = !S.ep || assetAvailable('images');
  $('assetNotice').textContent = 'No camera observations supplied for this episode. The recorded trajectory is available for inspection.';
  $('optViews').disabled = preview || !assetAvailable('images');
  $('optViews').title = preview ? 'Camera panels stay visible during goal inspection. Return to agent to restore your layout.' : '';
  $('fpvTitle').textContent = preview ? 'Goal view' : 'Agent view';
  $('panoTitle').textContent = preview ? 'Goal preview · 360°' : '360° panorama';
  $('panoHint').hidden = preview;
  $('returnAgent').hidden = !preview;
  $('fpvFollow').textContent = preview ? '↺ reset look' : '↺ follow agent';
  $('fpv').setAttribute('aria-label', preview ? 'Front view at the goal; drag to inspect the surroundings' : "Front view at the agent's heading; drag to look around");
  $('pano').setAttribute('aria-label', preview ? '360 degree panorama of the reference goal' : '360 degree panorama at the current viewpoint');
  $('goalPreviewStatus').hidden = !preview;
  $('goalPreviewStatus').textContent = preview ? `Agent paused at frame ${S.frame} · click the panorama to look around` : '';
  document.querySelectorAll('.progress-legend').forEach(element => { element.hidden = preview || !$('optNodes').checked; });
  $('viewGoalLegend').hidden = preview || !S.ep?.goal;
  $('jumpGoal').setAttribute('aria-pressed', preview);
  $('centerAgent').textContent = preview ? 'Return to agent' : 'Center agent';
  $('centerAgent').title = preview ? 'Restore the agent camera and panorama at the paused frame' : 'Return the camera to the agent at the current playback frame';
  $('bPlay').title = preview ? 'Return to agent and play (Space)' : 'Play / pause (Space)';
  if (preview) $('autoNextStatus').textContent = 'Paused for goal preview';
}

function showCameraViews(fs = frameState(S.frame)) {
  updateViewContext();
  const preview = !!UI.goalPreview, vp = preview ? S.ep.goal : fs.skyVp;
  const missing = !assetAvailable('images') || !node(vp)?.sky ? 'No camera observations supplied at this position.' : '';
  // A neutral heading lets the user inspect the entire goal panorama without inventing a model pose.
  S.viewReady = views.show({ scan: S.scan.scan, vp, rot: node(vp)?.rot || [], sky: node(vp)?.sky, missing,
    heading: preview ? 0 : fs.fr.heading, elevation: preview ? 0 : fs.fr.elevation || 0,
    note: preview ? null : viewNote(fs), markers: preview ? [] : viewMarkers(S.frame, fs) });
}

function viewNote(fs) {
  // Where the agent view is not simply the agent's camera at this frame, say so next to the heading.
  const deg = h => Math.round(((h * 180 / Math.PI) % 360 + 360) % 360);
  if (S.scan.ce && fs.skyVp !== fs.fr.vp) {
    const a = node(fs.skyVp).pos, b = node(fs.fr.vp).pos, d = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    return { text: `view from ${fs.skyVp}, ${fmt(d)} m away`,
             title: `The continuous run renders camera images only at map nodes: this is the view at ${fs.skyVp}, where the agent decided, ${fmt(d)} m from its position at this frame, turned to its current heading` };
  }
  const annotated = S.ep.start_heading;
  if (S.ep.adapter !== 'recording' && !S.scan.ce && fs.frame === 0 && annotated != null && deg(annotated) !== deg(fs.fr.heading))
    return { text: `annotated ${deg(annotated)}°`,
             title: `The episode's annotated start heading is ${deg(annotated)}°. DUET's simulator turns in 30° steps and starts the agent at the nearest one, ${deg(fs.fr.heading)}°, which is what the model saw.` };
  if (S.ep.adapter !== 'recording' && !S.scan.ce && fs.fr.elevation && fs.atDecisionNode)
    return { text: '', title: `DUET's simulator points the camera at the centre of the 30° view in which the previous node saw this one, here ${Math.round(fs.fr.elevation * 180 / Math.PI)}° from level` };
  return null;
}

function refreshViewMarkers() {
  updateViewContext();
  if (views.ctx && S.ep) { views.ctx.markers = UI.goalPreview ? [] : viewMarkers(S.frame, frameState(S.frame)); views.draw(); }
}

function leaveGoalPreview() {
  if (!UI.goalPreview) return false;
  views.look = UI.goalPreview.look;
  $('optViews').checked = UI.goalPreview.showViews;
  UI.goalPreview = null;
  $('autoNextStatus').textContent = '4 s pause between episodes';
  updateViewContext();
  return true;
}

function viewMarkers(f, fs) {
  const ep = S.ep, fr = fs.fr, cam = fs.skyVp || fr.vp, here = node(cam).pos, st = fs.state, out = [];
  const rel = vp => { const p = node(vp).pos; return Math.atan2(p[0] - here[0], p[1] - here[1]) - fr.heading; };
  const elev = vp => { const p = node(vp).pos; return Math.atan2(p[2] - here[2], Math.hypot(p[0] - here[0], p[1] - here[1])); };
  const dist = vp => Math.hypot(...node(vp).pos.map((c, i) => c - here[i]));
  const room = vp => node(vp).region || '';
  const toGoal = vp => (ep.goal ? ` · ${fmt(ep.dist_to_goal[vp])} m to goal` : '');
  // continuous runs: only the map nodes and ghost waypoints ETPNav had at this decision
  const now = S.scan.ce ? new Set([...st.visited, ...st.frontier]) : null;
  const nbs = [...(S.adj.get(cam) || [])].filter(vp => !now || now.has(vp));
  for (const nb of nbs) {
    const visited = fs.counts.get(nb);
    const p = st.score && st.frontier.includes(nb) ? st.score.get(nb) : null;
    out.push({ rel: rel(nb), elev: elev(nb), kind: visited ? 'visited' : 'cand', label: fs.labels.scores && (fs.labels.detailed || nb === fs.chosen || nb === fs.labels.selected) && p != null && p >= 0.01 ? pct(p) : '',
               title: `${short(nb)} ${room(nb)}${visited ? ' · visited' : ''}${toGoal(nb)}` });
  }
  // best next hop along the shortest path to the goal (RxR: the reference-best option when it is adjacent)
  if (ep.reference === 'route' || S.scan.ce) {
    const best = fs.atDecisionNode && fs.orc?.vp;
    if (best && nbs.includes(best)) out.push({ rel: rel(best), elev: elev(best), kind: 'best', title: `reference-best ${S.scan.ce ? 'waypoint' : 'next hop'}${ep.reference === 'route' ? ' along the annotated route' : ''} ${short(best)}${toGoal(best)}` });
  } else if (ep.reference !== 'none' && ep.goal && margin() != null && ep.dist_to_goal[fr.vp] >= margin() && nbs.length) {
    const best = nbs.reduce((a, b) => (dist(a) + ep.dist_to_goal[a] <= dist(b) + ep.dist_to_goal[b] ? a : b));
    out.push({ rel: rel(best), elev: elev(best), kind: 'best', title: `reference-best next hop ${short(best)}${toGoal(best)}` });
  }
  const next = ep.frames[f + 1];
  if (next) out.push({ rel: rel(next.vp), elev: elev(next.vp), kind: 'chosen', label: next.kind === 'local' ? 'next' : 'next (jump)',
                       title: `${kindLabel(next.kind)} → ${short(next.vp)} ${room(next.vp)}${toGoal(next.vp)}` });
  if (fs.chosen && !S.adj.get(cam)?.has(fs.chosen) && fs.chosen !== next?.vp)
    out.push({ rel: rel(fs.chosen), elev: elev(fs.chosen), kind: 'remote', label: `target ${fmt(dist(fs.chosen))} m`,
               title: `remote-frontier target ${short(fs.chosen)}` });
  if (ep.goal && ep.goal !== fr.vp)
    out.push({ rel: rel(ep.goal), elev: elev(ep.goal), kind: 'goal', label: `goal ${fmt(ep.dist_to_goal[fr.vp])} m`, title: 'goal direction (may be behind walls)' });
  return $('optNodes').checked ? out : out.filter(marker => marker.kind === 'goal');
}

// ------------------------------------------------------------------------------------------ playback

function episodeNavigationRows() { return S.investigationRows || S.filtered; }

function updateInvestigationNavigation() {
  const rows = S.investigationRows;
  $('investigationNavigation').hidden = !rows;
  if (rows) {
    const index = rows.findIndex(row => row.instr_id === S.ep?.instr_id);
    $('investigationNavigationLabel').textContent = `Investigation case ${index + 1} of ${rows.length} on this queue page. Previous, Next, and Auto-next follow these cases.`;
  }
  $('prevEp').title = rows ? 'Previous case on this investigation queue page (K)' : 'Previous episode in the list (K)';
  $('nextEp').title = rows ? 'Next case on this investigation queue page (J)' : 'Next episode in the list (J)';
}

function clearInvestigationNavigation() {
  if (S.investigationRows && S.advance) stop();
  S.investigationRows = null;
  updateInvestigationNavigation();
}

function play() {
  if (!S.ep) return;
  if (UI.goalPreview) centerOnAgent();
  if (S.playing) return stop();
  if (S.frame >= S.ep.frames.length - 1 && $('optAutoNext').checked && nextAutoEpisode()) return queueNextEpisode();
  if (S.frame >= S.ep.frames.length - 1) setFrame(0);
  S.playing = true;
  $('autoNextStatus').textContent = '4 s pause between episodes';
  updatePlayButton();
  const step = () => {
    if (!S.playing) return;
    if (S.frame >= S.ep.frames.length - 1) return queueNextEpisode();
    const duration = setFrame(S.frame + 1, true);
    // Let the final movement finish before starting the four-second pause.
    S.timer = setTimeout(step, S.frame === S.ep.frames.length - 1 ? duration : Number($('speed').value) * 1000);
  };
  S.timer = setTimeout(step, 250);
}

function updatePlayButton() {
  $('bPlay').textContent = S.playing ? '❚❚' : '▶';
  $('bPlay').setAttribute('aria-label', S.playing ? 'Pause' : 'Play');
}

function stop() {
  S.playing = false;
  clearTimeout(S.timer);
  S.timer = null;
  S.advance = null;
  $('autoNextStatus').textContent = '4 s pause between episodes';
  updatePlayButton();
}

function nextAutoEpisode() {
  const rows = episodeNavigationRows();
  if (!S.ep || !rows.length) return null;
  const i = rows.findIndex(row => row.instr_id === S.ep.instr_id);
  const next = rows[(i + 1) % rows.length];
  return next.instr_id !== S.ep.instr_id ? next : null;
}

function queueNextEpisode() {
  if (!$('optAutoNext').checked) return stop();
  const next = nextAutoEpisode();
  if (!next) {
    stop();
    $('autoNextStatus').textContent = 'No other episodes in the current list';
    return;
  }
  clearTimeout(S.timer);
  const pending = { run: S.run, id: next.instr_id, phase: 'waiting', deadline: performance.now() + 4000 };
  S.advance = pending;
  S.playing = true;
  updatePlayButton();
  const countdown = () => {
    if (S.advance !== pending) return;
    const remaining = pending.deadline - performance.now();
    if (remaining > 0) {
      $('autoNextStatus').textContent = `Next episode in ${Math.ceil(remaining / 1000)} s · ${pending.id}`;
      S.timer = setTimeout(countdown, Math.min(remaining, 1000));
    } else {
      S.timer = null;
      pending.phase = 'loading';
      $('autoNextStatus').textContent = `Loading episode ${pending.id}…`;
      advanceEpisode(pending);
    }
  };
  countdown();
}

async function advanceEpisode(pending) {
  try {
    if (!await loadEpisode(pending.id, 0, pending)) return;
    // Give the new episode's building and first camera view time to appear before moving.
    await Promise.allSettled([map3d.baseReady, S.viewReady]);
    if (S.advance !== pending || S.run !== pending.run) return;
    S.advance = null;
    S.playing = false;
    play();
  } catch (error) {
    if (S.advance !== pending) return;
    stop();
    $('autoNextStatus').textContent = 'Could not load the next episode. Press Play to retry.';
    console.warn('Auto-next episode failed:', error);
  }
}

function stepDecision(dir) {
  if (!hasDecisionTrace()) return;
  const ep = S.ep, d = curFrame().decision;
  const atDecisionNode = ep.decisionFrame[d] === S.frame;
  let target = dir > 0 ? d + 1 : (atDecisionNode ? d - 1 : d);
  target = Math.max(0, Math.min(ep.decisions.length - 1, target));
  setFrame(ep.decisionFrame[target] ?? ep.frames.length - 1);
}

// -------------------------------------------------------------------------------------------- 2D map

const mapEl = $('map');

function visibleFloor() { return S.floor === 'all' ? null : S.shownFloor; }
function isVisible(vp) { const n = node(vp), fl = visibleFloor(); return n && (fl == null || n.floor === fl); }

function fitView(whole = false) {
  if (!whole && map3d.mode === 'goal' && S.ep?.goal) return fitGoalView();
  // Default: the episode's paths (agent, ground truth, compared run) on the shown floor, with margin.
  const ep = S.ep, other = otherPath();
  let vps = whole ? [] : [...ep.path, ...ep.gt_path, ...(other ? other.path : []), ...ep.zone].filter(isVisible);
  if (!vps.length) vps = Object.keys(S.scan.nodes).filter(isVisible);
  if (!vps.length) return;
  const W = mapEl.clientWidth || 800, H = mapEl.clientHeight || 500;
  const xs = vps.map(v => node(v).pos[0]), ys = vps.map(v => node(v).pos[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const pad = whole ? 1 : 3;
  const k = Math.min((W - 60) / Math.max(x1 - x0 + 2 * pad, 8), (H - 60) / Math.max(y1 - y0 + 2 * pad, 8));
  S.view = { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, k: Math.min(k, 120), fit: whole ? 'floor' : 'episode' };
  updateCameraContext();
}

function fitGoalView() {
  const p = node(S.ep.goal).pos;
  const width = mapEl.clientWidth || $('mapwrap').clientWidth || 800;
  const height = mapEl.clientHeight || $('mapwrap').clientHeight || 500;
  S.view = { cx: p[0], cy: p[1], k: Math.max(10, Math.min((width - 60) / 10, (height - 60) / 10, 120)), fit: 'goal' };
}

function jumpToGoal() {
  if (!S.ep?.goal || !node(S.ep.goal)) return;
  stop();
  if (!UI.goalPreview) {
    // Finish drawing the already selected frame, without advancing the recorded agent.
    setFrame(S.frame);
    UI.goalPreview = { look: { ...views.look }, showViews: $('optViews').checked };
  }
  views.look = { yaw: 0, pitch: 0, free: false };
  $('optViews').checked = true;
  S.floor = node(S.ep.goal).floor;
  S.shownFloor = S.floor;
  map3d.setFloorMode(S.floor);
  map3d.setCamera('goal', true);
  renderFloors(); fitGoalView(); renderStatic(); renderDynamic(1);
  showCameraViews();
  // Reveal the goal image even when the map has been enlarged or camera panels were hidden.
  requestAnimationFrame(() => { if (UI.goalPreview && assetAvailable('images')) document.querySelector('.view.pano').scrollIntoView({ block: 'nearest' }); });
}

function centerOnAgent() {
  if (!S.ep) return;
  const preview = leaveGoalPreview();
  S.floor = 'auto'; S.shownFloor = curFrame().floor;
  map3d.setFloorMode('auto'); map3d.setCamera('chase', true);
  renderFloors();
  if (S.tab === '2d') {
    if (!S.view) fitView();
    const p = node(curFrame().vp).pos;
    S.view = { ...S.view, cx: p[0], cy: p[1], fit: null };
    $('cameraContext').textContent = 'Centered on agent · current frame';
  }
  renderStatic(); renderDynamic(1);
  if (preview) showCameraViews();
}

function presentationOptions() {
  return { model: $('optModelRoute').checked, reference: $('optReferenceRoute').checked,
    comparison: $('optCompareRoute').checked, detailed: $('optLabels').checked };
}

function refreshPresentation() {
  updateViewContext();
  map3d.setPresentation(presentationOptions());
  updateMapControls();
  if (!S.ep) return;
  renderStatic(); renderDynamic(1); renderLegend();
  refreshViewMarkers();
}

function refreshInspection() {
  if (!S.ep) return;
  map3d.setInspection(UI.hoverVp || UI.selectedVp, UI.divergence ? comparisonDivergence() : null);
  renderDynamic(1);
}

function updateCameraContext(mode = map3d.mode) {
  $('cameraContext').textContent = UI.goalPreview ? `Goal preview · agent paused at frame ${S.frame}` : mode === 'goal' ? 'Viewing goal · playback stays at agent' : S.tab === '2d'
    ? `2D view · ${S.view?.fit === 'floor' ? 'whole floor' : S.view?.fit === 'episode' ? 'episode routes' : 'manual framing'}`
    : mode === 'chase' ? 'Camera follows agent' : `Camera: ${mode}`;
}

function updateMapControls() {
  const show = $('optNodes').checked;
  map3d.setNodesVisible(show);
  for (const id of ['optScores', 'optStop', 'optRooms', 'optFuture', 'optLabels', 'optModelRoute']) $(id).disabled = !show;
  $('optReferenceRoute').disabled = !show || !S.ep?.gt_path.length;
  $('optCompareRoute').disabled = !show || !otherPath();
  $('referenceRouteWrap').hidden = !S.ep?.gt_path.length;
  $('compareRouteWrap').hidden = !otherPath();
  $('compareRouteLabel').textContent = otherPath()?.name || 'Comparison';
  updateViewContext();
  $('jumpGoal').disabled = !S.ep?.goal || !S.scan?.nodes[S.ep.goal];
  $('jumpGoal').title = S.ep?.goal ? "Pause playback and inspect the goal's 360° panorama without moving the agent" : 'The selected episode does not provide a goal node';
}

function P(vp) { const p = node(vp).pos; return PW(p[0], p[1]); }
function PW(x, y) {
  const W = mapEl.clientWidth, H = mapEl.clientHeight;
  return [(x - S.view.cx) * S.view.k + W / 2, -(y - S.view.cy) * S.view.k + H / 2];
}

// Navigable area of the shown floor(s) (Habitat navmesh, build_navmaps.py), under everything else.
function navmapLayer() {
  const nm = S.scan.navmap;
  if (!nm || !$('optNavmap').checked) return '';
  const fl = visibleFloor(), [x0, y0, x1, y1] = nm.bounds, [x, y] = PW(x0, y1);
  const w = (x1 - x0) * S.view.k, hgt = (y1 - y0) * S.view.k, box = `x="${x}" y="${y}" width="${w}" height="${hgt}"`;
  return nm.floors.filter(f => fl == null || f === fl).map(f => {
    const id = `navmask-${f}`;
    return `<mask id="${id}" maskUnits="userSpaceOnUse" ${box}><image href="/api/navmap/${S.scan.scan}_f${f}.png${S.scan.asset_query || ''}" ${box} preserveAspectRatio="none"/></mask>` +
      `<rect class="m-nav" ${box} mask="url(#${id})"/>`;
  }).join('');
}

// Continuous runs: ETPNav's topological map as it stood at this decision (edges between its nodes, dashed
// edges to its ghost waypoints) and, at the decision point, the waypoint predictor's raw output as spokes;
// a filled dot marks a prediction that was merged into an already visited node.
function ceMapLayer(fs) {
  const st = fs.state, known = new Set(st.visited), ghosts = new Set(st.frontier);
  let h = '';
  for (const a of known) {
    for (const b of S.adj.get(a) || []) {
      if (!(ghosts.has(b) || (known.has(b) && a < b)) || !isVisible(a) || !isVisible(b)) continue;
      const [x1, y1] = P(a), [x2, y2] = P(b);
      h += `<line class="m-edge ${ghosts.has(b) ? 'ghost' : 'map'}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`;
    }
  }
  if (fs.atDecisionNode && fs.dec.cands?.length && isVisible(fs.dec.at)) {
    const [ax, ay] = P(fs.dec.at);
    for (const [p, merged] of fs.dec.cands) {
      const [x, y] = PW(p[0], p[1]);
      h += `<line class="m-spoke" x1="${ax}" y1="${ay}" x2="${x}" y2="${y}"/><circle class="m-spoke-end${merged?.[0] === 'n' ? ' merged' : ''}" cx="${x}" cy="${y}" r="2.6"/>`;
    }
  }
  return h;
}

function renderFloors() {
  const floors = S.scan.floors;
  if (floors.length < 2) { $('floors').innerHTML = ''; return; }
  const btn = (v, label) => `<button data-floor="${v}" aria-pressed="${String(S.floor) === String(v)}">${label}</button>`;
  $('floors').innerHTML = btn('auto', 'Follow agent') + btn('all', 'All floors') + floors.map(f => btn(f.id, `Floor ${f.id + 1}`)).join('');
}

function polylines(vps, cls) {
  let out = '', run = [];
  const flush = () => { if (run.length > 1) out += `<polyline class="${cls}" points="${run.map(p => p.join(',')).join(' ')}"/>`; run = []; };
  vps.forEach((vp, i) => {
    if (isVisible(vp) && (i === 0 || isVisible(vps[i - 1]) || run.length === 0)) run.push(P(vp));
    else { flush(); if (isVisible(vp)) run.push(P(vp)); }
  });
  flush();
  return out;
}

function renderStatic() {
  if (S.tab !== '2d' || !S.ep) return;
  const ep = S.ep, sc = S.scan;
  if (!S.view) fitView();
  if (!$('optNodes').checked) {
    let hits = '';
    for (const vp of new Set([ep.start, ep.goal].filter(Boolean))) {
      const [x, y] = P(vp);
      hits += `<circle class="m-hit" data-vp="${esc(vp)}" cx="${x}" cy="${y}" r="12"/>`;
    }
    mapEl.innerHTML = `<g id="gStatic">${navmapLayer()}${startMarker()}${goalMarker()}</g><g id="gDyn"></g><g id="gHits">${hits}</g>`;
    return;
  }
  let h = navmapLayer();
  const display = presentationOptions();
  mapEl.classList.toggle('all-labels', display.detailed);
  const kind = vp => sc.nodes[vp].kind;     // continuous runs: node | ghost | sample | ref; compared routes: compare
  // continuous runs: ETPNav's map grows with every decision, so its nodes and ghosts are drawn per frame
  const fixed = vp => !sc.ce || kind(vp) === 'ref';
  if (display.reference && sc.ce && ep.goal && margin() != null && isVisible(ep.goal)) { const [x, y] = P(ep.goal); h += `<circle class="m-zone" cx="${x}" cy="${y}" r="${margin() * S.view.k}"/>`; }
  else if (display.reference) for (const vp of ep.zone) if (isVisible(vp)) { const [x, y] = P(vp); h += `<circle class="m-zone" cx="${x}" cy="${y}" r="11"/>`; }
  for (const [a, b] of sc.edges) {
    if (!fixed(a) || !fixed(b) || kind(a) === 'sample' || kind(b) === 'sample') continue;
    if (isVisible(a) && isVisible(b)) { const [x1, y1] = P(a), [x2, y2] = P(b); h += `<line class="m-edge${kind(a) === 'ghost' || kind(b) === 'ghost' ? ' ghost' : ''}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"/>`; }
  }
  if ($('optRooms').checked && display.detailed) {
    const groups = new Map();
    for (const vp in sc.nodes) {
      const n = sc.nodes[vp];
      if (n.rid == null || !n.region || !isVisible(vp)) continue;
      const g = groups.get(n.rid) || { label: n.region, pts: [] };
      g.pts.push(P(vp)); groups.set(n.rid, g);
    }
    for (const g of groups.values()) {
      const x = g.pts.reduce((s, p) => s + p[0], 0) / g.pts.length, y = g.pts.reduce((s, p) => s + p[1], 0) / g.pts.length;
      h += `<text class="m-room" x="${x}" y="${y - 12}">${esc(g.label)}</text>`;
    }
  }
  for (const vp in sc.nodes) {
    if (!isVisible(vp) || !fixed(vp) || kind(vp) === 'sample' || kind(vp) === 'compare') continue;
    const [x, y] = P(vp);
    h += kind(vp) === 'ghost' ? `<circle class="m-node ghost" cx="${x}" cy="${y}" r="2.4"/>` : `<circle class="m-node" cx="${x}" cy="${y}" r="3.2"/>`;
  }
  if ($('optFuture').checked && display.model) h += polylines(ep.path, 'm-future');
  if (ep.gt_path.length && display.reference) h += polylines(ep.gt_path, 'm-gt');
  h += '<g id="gSub"></g>';
  const other = otherPath();
  if (other && display.comparison) {
    h += polylines(other.path, 'm-compare');
    const end = other.path[other.path.length - 1];
    if (isVisible(end)) {
      const [x, y] = P(end);
      h += `<rect class="m-stop" style="stroke:var(--compare)" x="${x - 6}" y="${y - 6}" width="12" height="12" rx="2"/>` +
        (display.detailed ? `<text class="m-label muted" x="${x + 9}" y="${y + 16}">${esc(other.name)} ${esc(other.endLabel || 'end')}</text>` : '');
    }
  }
  if (isVisible(ep.start)) h += startMarker();
  if (ep.goal && isVisible(ep.goal)) {
    h += goalMarker();
  }
  let hits = '';
  for (const vp in sc.nodes) if (isVisible(vp) && fixed(vp) && kind(vp) !== 'sample' && kind(vp) !== 'compare') { const [x, y] = P(vp); hits += `<circle class="m-hit" data-vp="${esc(vp)}" cx="${x}" cy="${y}" r="9"/>`; }
  mapEl.innerHTML = `<g id="gStatic">${h}</g><g id="gDyn" style="pointer-events:none"></g><g id="gHits">${hits}</g><g id="gDynHits"></g>`;
}

function startMarker() {
  if (!S.ep.start) return '';
  const [x, y] = P(S.ep.start);
  return `<line class="m-flagpole" x1="${x}" y1="${y}" x2="${x}" y2="${y - 22}"/><path class="m-flag" d="M${x} ${y - 22} l13 4.5 l-13 4.5z"/>` +
    `<text class="m-label" x="${x + 5}" y="${y - 26}">start</text>`;
}

function goalMarker() {
  if (!S.ep.goal) return '';
  const [x, y] = P(S.ep.goal);
  return `<circle class="m-goal-ring" cx="${x}" cy="${y}" r="10"/><circle class="m-goal-ring" cx="${x}" cy="${y}" r="5.5"/>` +
    `<circle class="m-goal-dot" cx="${x}" cy="${y}" r="2.5"/><text class="m-label" x="${x + 13}" y="${y + 4}">goal</text>`;
}

function renderDynamic(t = 1) {
  if (S.tab !== '2d' || !S.ep || !$('optNodes').checked) return;
  const ep = S.ep, f = S.frame, fs = frameState(f, t), fr = fs.fr;
  const g = mapEl.querySelector('#gDyn');
  if (!g) return;
  const gs = mapEl.querySelector('#gSub');    // ground-truth chunk of the current sub-task
  const display = presentationOptions();
  if (gs) gs.innerHTML = fs.chunk && display.reference ? polylines(ep.gt_path.slice(fs.chunk[0] - 1, fs.chunk[1]), 'm-gt-cur') : '';
  let h = '';
  const traversals = new Map();
  for (let i = 1; i <= f; i++) {
    if (!display.model) break;
    const a = ep.frames[i - 1].vp, b = ep.frames[i].vp;
    if (!isVisible(a) && !isVisible(b)) continue;
    const key = JSON.stringify(a < b ? [a, b] : [b, a]), n = traversals.get(key) || 0;
    traversals.set(key, n + 1);
    let [x1, y1] = P(a), [x2, y2] = P(b);
    const len = Math.hypot(x2 - x1, y2 - y1) || 1, ox = -(y2 - y1) / len * n * 4.5, oy = (x2 - x1) / len * n * 4.5;
    if (i === f && t < 1) { x2 = x1 + (x2 - x1) * t; y2 = y1 + (y2 - y1) * t; }
    const kind = ep.frames[i].kind;
    h += `<line class="m-hop ${esc(kind)}" x1="${x1 + ox}" y1="${y1 + oy}" x2="${x2 + ox}" y2="${y2 + oy}"/>`;
    if (len > 26 && (i < f || t === 1)) {
      const mx = (x1 + x2) / 2 + ox, my = (y1 + y2) / 2 + oy, ang = Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI;
      h += `<path class="m-arrow ${esc(kind)}" d="M-4,-4 L4,0 L-4,4z" transform="translate(${mx},${my}) rotate(${ang})"/>`;
    }
  }
  const st = fs.state, ce = S.scan.ce;
  if (ce) h += ceMapLayer(fs);
  for (const vp of st.frontier) {
    if (!isVisible(vp) || fs.counts.get(vp)) continue;
    const [x, y] = P(vp);
    // continuous runs: a ghost waypoint's size follows the planner's score for it
    const s = ce && st.score ? 3.5 + 7 * Math.sqrt(st.score.get(vp) || 0) : 5;
    h += `<rect class="m-frontier${ce ? ' ghost' : ''}" x="${x - s}" y="${y - s}" width="${2 * s}" height="${2 * s}" rx="1.5"/>`;
    if (fs.labels.scores && fs.top.has(vp) && (display.detailed || vp === fs.chosen || vp === fs.labels.selected)) h += `<text class="m-label" x="${x + 8}" y="${y - 6}">${pct(st.score.get(vp))}</text>`;
  }
  const visited = new Set([...st.visited, ...[...fs.counts.keys()].filter(vp => fs.counts.get(vp) && node(vp).kind !== 'sample')]);
  for (const vp of visited) {
    if (!isVisible(vp)) continue;
    const [x, y] = P(vp), c = fs.counts.get(vp) || 1;
    h += `<circle class="${c >= 3 ? 'm-rev3' : c === 2 ? 'm-rev2' : 'm-visited'}" cx="${x}" cy="${y}" r="${c >= 2 ? 6.5 : 5}"/>`;
    if (c >= 2) h += `<text class="m-revlabel" x="${x + 7}" y="${y + 13}">×${c}</text>`;
    const sp = ep.stop_probs[vp];
    if (fs.labels.stop && sp != null && sp >= 0.05) h += `<text class="m-label muted" x="${x - 8}" y="${y - 9}" text-anchor="end">stop ${pct(sp)}</text>`;
  }
  if (t === 1) {
    if (fs.oracle && display.reference && isVisible(fs.oracle)) { const [x, y] = P(fs.oracle); h += `<circle class="m-oracle" cx="${x}" cy="${y}" r="13"/>`; }
    if (fs.chosen && isVisible(fs.chosen)) { const [x, y] = P(fs.chosen); h += `<circle class="m-chosen" cx="${x}" cy="${y}" r="10"/>`; }
  }
  if (fs.lost && isVisible(fs.lost)) {
    const [x, y] = P(fs.lost);
    h += `<g transform="translate(${x - 14},${y + 14})"><circle class="m-lost" r="7"/><path class="m-lost-x" d="M-3,-3L3,3M3,-3L-3,3"/></g>` +
      `<text class="m-label" x="${x - 24}" y="${y + 18}" text-anchor="end">lost</text>`;
  }
  if (fs.final && t === 1 && isVisible(ep.final)) {
    const [x, y] = P(ep.final);
    const col = { success: 'var(--good)', wrong_stop: 'var(--critical)', step_limit: 'var(--serious)' }[ep.outcome] || 'var(--text-primary)';
    h += `<rect class="m-stop" style="stroke:${col}" x="${x - 9}" y="${y - 9}" width="18" height="18" rx="3"/>` +
      `<text class="m-label" x="${x + 13}" y="${y - 10}">${esc(fs.final.text)}</text>`;
  }
  if (isVisible(fr.vp) || (f > 0 && isVisible(ep.frames[f - 1].vp))) {
    let [x, y] = P(fr.vp);
    if (t < 1 && f > 0) { const [px, py] = P(ep.frames[f - 1].vp); x = px + (x - px) * t; y = py + (y - py) * t; }
    const dx = Math.sin(fr.heading), dy = -Math.cos(fr.heading);
    h += `<circle class="m-pulse" cx="${x}" cy="${y}" r="8"/><circle class="m-current" cx="${x}" cy="${y}" r="6.5"/>` +
      `<path class="m-heading" d="M${x + dx * 22},${y + dy * 22} L${x + dx * 10 - dy * 6},${y + dy * 10 + dx * 6} L${x + dx * 10 + dy * 6},${y + dy * 10 - dx * 6}z"/>`;
  }
  const inspect = (vp, name, label, offset = [20, 20]) => {
    if (!vp || !isVisible(vp)) return;
    const [x, y] = P(vp);
    h += `<circle class="m-inspection" style="stroke:var(${name})" cx="${x}" cy="${y}" r="17"/>` +
      `<line class="m-inspection-leader" style="stroke:var(${name})" x1="${x}" y1="${y}" x2="${x + offset[0]}" y2="${y + offset[1] - 4}"/>` +
      `<text class="m-label inspection-label" x="${x + offset[0]}" y="${y + offset[1]}" text-anchor="${offset[0] < 0 ? 'end' : 'start'}">${esc(label)}</text>`;
  };
  if (fs.labels.selected) inspect(fs.labels.selected, '--focus', `${short(fs.labels.selected)} · ${node(fs.labels.selected)?.region || 'viewpoint'}`);
  if (fs.labels.divergence) {
    const d = fs.labels.divergence;
    for (const [vp, name, label] of [[d.model, '--model', 'Model branch'], [d.comparison, '--compare', 'Comparison branch']]) {
      if (vp && d.shared) h += polylines([d.shared, vp], name === '--model' ? 'm-divergence' : 'm-divergence comparison');
      inspect(vp, name, label, name === '--model' ? [-22, -30] : [22, 34]);
    }
    if (!d.model || !d.comparison) inspect(d.shared, '--focus', !d.model ? 'Model route ends here' : 'Comparison route ends here');
  }
  g.innerHTML = h;
  // continuous runs: hover targets for the map as it stands now (static targets cover the reference points)
  const hits = mapEl.querySelector('#gDynHits');
  if (hits) hits.innerHTML = !ce ? '' : [...new Set([...st.visited, ...st.frontier])].filter(isVisible)
    .map(vp => { const [x, y] = P(vp); return `<circle class="m-hit" data-vp="${esc(vp)}" cx="${x}" cy="${y}" r="9"/>`; }).join('');
}

function renderLegend() {
  const sw = (svg, label) => `<span><svg viewBox="0 0 26 12">${svg}</svg>${label}</span>`;
  if (!$('optNodes').checked) {
    $('legend').innerHTML = sw('<line class="m-flagpole" x1="9" y1="1" x2="9" y2="11"/><path class="m-flag" d="M9 1 l10 3 l-10 3z"/>', 'Start') + (S.ep?.goal
      ? sw('<circle class="m-goal-ring" cx="13" cy="6" r="5"/><circle class="m-goal-dot" cx="13" cy="6" r="2"/>', 'Goal')
      : '<span>Goal not provided for this episode</span>');
    return;
  }
  const ep = S.ep, ce = S.scan?.ce, portable = ep?.adapter === 'recording';
  const items = [
    ep?.gt_path?.length && sw('<line class="m-gt" x1="1" y1="6" x2="25" y2="6" style="stroke-dasharray:6 4"/>', portable ? 'Reference path' : 'Ground truth (thick: current sub-task)'),
    sw('<line class="m-hop" x1="1" y1="6" x2="25" y2="6"/>', ce ? 'Agent track' : 'Agent path'),
    !portable && sw('<line class="m-hop remote" x1="1" y1="6" x2="25" y2="6"/>', ce ? 'Walked back over the map' : 'Remote jump / backtrack'),
    sw('<circle class="m-visited" cx="13" cy="6" r="4.5"/>', ce && !portable ? 'Map node (visited)' : 'Visited'),
    (!portable || ep.decisions.some(d => d.frontier?.length)) && (ce && !portable ? sw('<rect class="m-frontier ghost" x="5" y="2.5" width="7" height="7"/><rect class="m-frontier ghost" x="14" y="0.5" width="11" height="11"/>', 'Ghost waypoint (size: planner score)')
      : sw('<rect class="m-frontier" x="8" y="1.5" width="9" height="9"/>', portable ? 'Recorded action candidate' : 'Frontier (unvisited)')),
    ce && !portable && sw('<line class="m-spoke" x1="1" y1="6" x2="21" y2="6"/><circle class="m-spoke-end" cx="21" cy="6" r="2.6"/>', 'Waypoints predicted here'),
    sw('<circle class="m-rev2" cx="9" cy="6" r="4.5"/><circle class="m-rev3" cx="19" cy="6" r="4.5"/>', 'Visited 2× / 3×+'),
    ep?.goal && sw('<circle class="m-goal-ring" cx="13" cy="6" r="5"/><circle class="m-goal-dot" cx="13" cy="6" r="2"/>', margin() != null ? `Goal (declared radius: ${margin()} m)` : 'Reference goal'),
    hasDecisionTrace() && sw('<circle class="m-chosen" cx="13" cy="6" r="5"/>', ce && !portable ? 'Chosen waypoint' : 'Chosen next node'),
    ep?.reference !== 'none' && ep?.goal && sw('<circle class="m-oracle" cx="13" cy="6" r="5"/>', 'Reference-best (ground truth)'),
  ].filter(Boolean);
  if (S.scan?.navmap && $('optNavmap').checked) items.push(sw('<rect class="m-nav" x="1" y="1" width="24" height="10" rx="2"/>', 'Navigable area (2D navmesh projection)'));
  const o = otherPath();
  if (o) items.push(sw('<line class="m-compare" x1="1" y1="6" x2="25" y2="6"/>', esc(o.name)));
  $('legend').innerHTML = items.join('');
}

mapEl.addEventListener('pointermove', e => {
  if (S.drag) return;
  const vp = e.target.dataset?.vp, tip = $('tip');
  if (!vp || !S.ep) { tip.hidden = true; return; }
  const n = node(vp), ep = S.ep, dec = curDecision(), st = modelState(dec);
  let visits = 0;
  for (let i = 0; i <= S.frame; i++) if (ep.frames[i].vp === vp) visits++;
  const rows = [`<b>${esc(vp.slice(0, 8))}</b>${n.region ? ' · ' + esc(n.region) : ''}${S.scan.floors.length > 1 ? ' · floor ' + (n.floor + 1) : ''}`];
  const tags = [];
  if (S.scan.ce) tags.push({ node: S.ep.adapter === 'recording' ? 'recorded node' : `map node (decision ${vp.slice(1)})`, ghost: 'ghost waypoint (predicted, not visited)', ref: 'reference point' }[n.kind] || '');
  if (vp === ep.start) tags.push('start');
  if (vp === ep.goal) tags.push('goal');
  if (ep.gtIndex.has(vp)) tags.push(`GT path #${ep.gtIndex.get(vp)}`);
  if (vp === ep.final) tags.push(finalActionLabel(ep) === 'STOP' ? 'final stop' : 'episode end');
  if (tags.filter(Boolean).length) rows.push(tags.filter(Boolean).map(esc).join(' · '));
  if (ep.goal) rows.push(`<span class="k">to goal</span> ${fmt(ep.dist_to_goal[vp])} m`);
  if (!S.scan.ce || n.kind === 'node') rows.push(`<span class="k">visits so far</span> ${visits}`);
  if (ep.stop_probs[vp] != null && visits) rows.push(`<span class="k">stop prob</span> ${pct(ep.stop_probs[vp])}`);
  if (st.score && st.frontier.includes(vp)) {
    rows.push(`<span class="k">action prob, decision ${dec.t}</span> ${pct(st.score.get(vp))}`);
    if (!S.scan.ce) rows.push(`<span class="k">global branch alone</span> ${pct(st.global.get(vp))}`);   // ETPNav: global only
  }
  tip.innerHTML = rows.join('<br>');
  tip.hidden = false;
  const box = $('mapwrap').getBoundingClientRect();
  tip.style.left = Math.min(e.clientX - box.left + 14, box.width - 220) + 'px';
  tip.style.top = (e.clientY - box.top + 14) + 'px';
});
mapEl.addEventListener('pointerleave', () => { $('tip').hidden = true; });
let raf = 0;
const redraw = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; renderStatic(); renderDynamic(1); updateCameraContext(); }); };
mapEl.addEventListener('wheel', e => {
  if (!S.view) return;
  e.preventDefault();
  const box = mapEl.getBoundingClientRect(), W = box.width, H = box.height;
  const mx = e.clientX - box.left, my = e.clientY - box.top;
  const wx = (mx - W / 2) / S.view.k + S.view.cx, wy = -(my - H / 2) / S.view.k + S.view.cy;
  S.view.k = Math.max(2, Math.min(400, S.view.k * Math.exp(-e.deltaY * 0.0015)));
  S.view.cx = wx - (mx - W / 2) / S.view.k;
  S.view.cy = wy + (my - H / 2) / S.view.k;
  S.view.fit = null;
  redraw();
}, { passive: false });
mapEl.addEventListener('pointerdown', e => {
  if (!S.view) return;
  S.drag = { x: e.clientX, y: e.clientY, cx: S.view.cx, cy: S.view.cy, moved: false };
  mapEl.setPointerCapture(e.pointerId);
});
mapEl.addEventListener('pointermove', e => {
  if (!S.drag) return;
  const dx = e.clientX - S.drag.x, dy = e.clientY - S.drag.y;
  if (Math.abs(dx) + Math.abs(dy) > 3) { S.drag.moved = true; mapEl.classList.add('dragging'); $('tip').hidden = true; }
  S.view.cx = S.drag.cx - dx / S.view.k;
  S.view.cy = S.drag.cy + dy / S.view.k;
  if (S.drag.moved) { S.view.fit = null; redraw(); }
});
mapEl.addEventListener('pointerup', () => { S.drag = null; mapEl.classList.remove('dragging'); });
mapEl.addEventListener('dblclick', () => { if (map3d.mode === 'goal') map3d.setCamera('overview', true); fitView(); redraw(); });
new ResizeObserver(() => {
  if (!S.ep) return;
  if (S.tab === '2d') {
    // Refit automatic views, but retain a manually chosen pan and zoom while resizing panels.
    if (!S.view || S.view.fit === 'episode') fitView();
    else if (S.view.fit === 'floor') fitView(true);
    else if (S.view.fit === 'goal') fitGoalView();
    redraw();
  }
  renderTimeline();
}).observe($('mapwrap'));

function setTab(tab) {
  if (tab === '3d' && map3d.available === false) tab = '2d';
  S.tab = tab;
  $('tab3d').setAttribute('aria-pressed', tab === '3d');
  $('tab2d').setAttribute('aria-pressed', tab === '2d');
  for (const mode of ['2d', '3d']) { $('tab' + mode).setAttribute('aria-selected', tab === mode); $('tab' + mode).tabIndex = tab === mode ? 0 : -1; }
  $('map3d').hidden = tab !== '3d';
  mapEl.toggleAttribute('hidden', tab !== '2d');    // SVG elements have no .hidden property
  document.querySelectorAll('.only3d').forEach(e => { e.hidden = tab !== '3d'; });
  document.querySelectorAll('.only2d').forEach(e => { e.hidden = tab !== '2d'; });
  map3d.setActive(S.page === 'episode' && tab === '3d');
  $('status3d').hidden = tab !== '3d' || !$('status3d').textContent;
  if (S.ep && tab === '2d') { fitView(); renderStatic(); renderDynamic(1); }
  updateCameraContext();
}

// ------------------------------------------------------------------------------------------ timeline

function renderTimeline() {
  const el = $('timeline'), ep = S.ep;
  if (!ep) return;
  const W = el.clientWidth || 600, H = 112, L = 68, R = 8, top = 6, ch = 40;
  const n = ep.frames.length, cw = (W - L - R) / n, hasGoal = ep.goal != null;
  const finiteDistance = frame => frame?.dist != null && Number.isFinite(frame.dist);
  const hasDistances = ep.frames.some(finiteDistance), radius = margin();
  const maxD = Math.max(1, radius || 0, ...ep.frames.filter(finiteDistance).map(f => f.dist)) * 1.08;
  const X = i => L + (i + 0.5) * cw, Y = d => top + (1 - d / maxD) * ch;
  const yKind = 50, ySub = 63, yFoc = 80, rowH = 14;
  let h = '';
  if (hasDistances) {
    h += `<line class="t-grid" x1="${L}" x2="${W - R}" y1="${Y(0)}" y2="${Y(0)}"/>` +
      `<text class="t-axis" x="${L - 4}" y="${top + 8}" text-anchor="end">${Math.round(maxD)} m</text>`;
    if (radius != null) h += `<line class="t-thresh" x1="${L}" x2="${W - R}" y1="${Y(radius)}" y2="${Y(radius)}"/>` +
      `<text class="t-axis" x="${L - 4}" y="${Y(radius) + 3}" text-anchor="end">${radius} m</text>`;
    let continuing = false;
    const path = ep.frames.map((f, i) => {
      if (!finiteDistance(f)) { continuing = false; return ''; }
      const command = continuing ? 'L' : 'M'; continuing = true;
      return `${command}${X(i)},${Y(f.dist)}`;
    }).join(' ');
    h += `<path class="t-line" fill="none" d="${path}"/>`;
  } else {
    h += `<text class="t-axis" x="${L}" y="${top + 20}">Distance to goal was not supplied</text>`;
  }
  const subs = ep.subtasks;
  if (subs.reached) h += `<text class="t-axis" x="${L - 4}" y="${ySub + 10}" text-anchor="end">sub-task</text>`;
  if (ep.decisions.some(d => d.focus)) h += `<text class="t-axis" x="${L - 4}" y="${yFoc + 10}" text-anchor="end">attention</text>`;
  ep.frames.forEach((f, i) => {
    const dec = decisionAt(f);
    const reading = dec.focus ? dec.focus.indexOf(Math.max(...dec.focus)) : null;
    const tip = `frame ${i} · ${decisionContext(f)} · ${kindLabel(f.kind)}${f.nhops > 1 ? ` (${S.scan.ce ? 'step' : 'hop'} ${f.hop}/${f.nhops})` : ''}` +
      `${hasGoal ? ` · ${fmt(f.dist)} m to goal` : ''}${f.visits > 1 ? ` · visit ${f.visits}` : ''}` +
      `${f.subtask != null ? ` · on sub-task ${f.subtask + 1}` : ''}${reading != null ? ` · highest attention on sub-task ${reading + 1}` : ''}`;
    const x = L + i * cw + 1, w = Math.max(1, cw - 2);
    const nonbest = dec.ref?.ok === false && ep.decisionFrame[f.decision] === i;
    const tipKind = nonbest ? `${tip} · decision ${f.decision} was not the reference-best (${MISTAKE[dec.ref.type].toLowerCase()})` : tip;
    h += `<rect class="t-cell ${esc(f.kind)}${nonbest ? ' nonbest' : ''}" x="${x}" y="${yKind}" width="${w}" height="8" rx="2"><title>${esc(tipKind)}</title></rect>`;
    if (f.subtask != null) {
      const done = subs.reached && subs.reached[f.subtask] != null && subs.reached[f.subtask] <= i;
      h += `<rect class="t-sub${done ? ' done' : ''}" x="${x}" y="${ySub}" width="${w}" height="${rowH}" rx="2"><title>${esc(tip)}</title></rect>`;
      if (cw >= 10) h += `<text class="t-subtxt" x="${X(i)}" y="${ySub + 10}">${f.subtask + 1}</text>`;
    }
    if (reading != null) {
      const off = f.subtask != null && reading !== f.subtask;
      h += `<rect class="t-sub${off ? ' off' : ''}" x="${x}" y="${yFoc}" width="${w}" height="${rowH}" rx="2"><title>${esc(tip)}</title></rect>`;
      if (cw >= 10) h += `<text class="t-subtxt" x="${X(i)}" y="${yFoc + 10}">${reading + 1}</text>`;
    }
    if (f.visits > 1) h += `<circle class="t-dot" style="fill:var(${f.visits >= 3 ? '--critical' : '--warning'})" cx="${X(i)}" cy="${yKind - 4}" r="3"><title>${esc(tip)}</title></circle>`;
    if (hasDecisionTrace() && ep.decisionFrame[f.decision] === i && cw >= 14) h += `<text class="t-axis" x="${X(i)}" y="${H - 1}" text-anchor="middle">${f.decision}</text>`;
  });
  const lost = ep.diag.lost_hop;
  if (lost != null) h += `<line style="stroke:var(--critical);stroke-width:1.5" x1="${X(lost)}" x2="${X(lost)}" y1="${top}" y2="${yKind + 8}"><title>left the declared reference corridor</title></line>` +
    `<text class="t-axis" style="fill:var(--critical)" x="${X(lost) + 3}" y="${top + 8}">lost</text>`;
  if (hasDistances && finiteDistance(ep.frames[ep.diag.closest_hop])) h += `<circle class="t-dot" style="fill:var(--goal)" cx="${X(ep.diag.closest_hop)}" cy="${Y(ep.frames[ep.diag.closest_hop].dist)}" r="4"><title>closest ${ep.ce ? 'sampled ' : ''}approach</title></circle>`;
  h += `<line class="t-cursor" x1="${X(S.frame)}" x2="${X(S.frame)}" y1="${top}" y2="${yFoc + rowH + 2}"/>`;
  if (hasDistances && finiteDistance(ep.frames[S.frame])) h += `<circle style="fill:var(--model);stroke:var(--surface-1);stroke-width:2" cx="${X(S.frame)}" cy="${Y(ep.frames[S.frame].dist)}" r="4.5"/>`;
  el.setAttribute('viewBox', `0 0 ${W} ${H}`);
  el.innerHTML = h;
  el.dataset.l = L; el.dataset.cw = cw;
}

const seek = e => {
  if (!S.ep) return;
  const el = $('timeline'), box = el.getBoundingClientRect();
  const i = Math.floor((e.clientX - box.left - Number(el.dataset.l)) / Number(el.dataset.cw));
  if (i >= 0 && i !== S.frame) { stop(); setFrame(i); }
};
$('timeline').addEventListener('pointerdown', e => { stop(); S.seeking = true; $('timeline').setPointerCapture(e.pointerId); seek(e); });
$('timeline').addEventListener('pointermove', e => { if (S.seeking) seek(e); });
$('timeline').addEventListener('pointerup', () => { S.seeking = false; });

// -------------------------------------------------------------------------------------------- panels

function renderPanels(fs) {
  const ep = S.ep, f = S.frame, fr = fs.fr, m = ep.metrics, o = OUTCOME[ep.outcome];
  const finalOpen = $('finalOutcome')?.open;
  const finalFocused = document.activeElement.closest?.('#finalOutcome > summary');
  const walked = ep.frames.slice(1, f + 1).reduce((s, x, i) => s + dist3(ep.frames[i].vp, x.vp), 0);
  const tags = [];
  const tag = (txt, color) => tags.push(`<span class="tag">${color ? `<i style="background:var(${color})"></i>` : ''}${esc(txt)}</span>`);
  tag(kindLabel(fr.kind) + (fr.nhops > 1 ? ` · ${S.scan.ce ? 'step' : 'hop'} ${fr.hop}/${fr.nhops}` : ''), fr.kind === 'remote' || fr.kind === 'return' ? '--remote' : fr.kind === 'start' ? '--gt' : '--model');
  if (fr.visits >= 2) tag(`revisit ×${fr.visits}`, fr.visits >= 3 ? '--critical' : '--warning');
  if (ep.goal) {
    if (margin() != null && fr.dist != null && fr.dist < margin()) tag(`within ${margin()} m of goal`, '--good');
    else if (margin() != null && fr.gt_dist != null && fr.gt_dist > margin()) tag(`>${margin()} m from reference path`, '--critical');
    else if (!S.scan.ce && ep.gt_path.length && !ep.gtIndex.has(fr.vp)) tag('outside reference path');
  }
  const n = node(fr.vp);
  if (n.region) tag(n.region);
  if (S.scan.floors.length > 1) tag(`floor ${n.floor + 1}`);
  let h = `<h2>Current position · frame ${f}</h2><div class="position-strip"><strong>${ep.goal ? fmt(fr.dist, 2) + ' m' : 'No reference goal'}</strong>` +
    `<span>${ep.goal ? `${esc(distanceLabel(ep))}${fr.euclid != null ? `<br>straight-line ${fmt(fr.euclid)} m` : ''}` : 'Recorded route only'}</span></div>` +
    `<p class="position-context">${ep.adapter === 'recording' ? 'Sampled route' : 'Walked'} ${fmt(walked)} m · ${decisionContext(fr)}</p><div class="position-tags">${tags.join('')}</div>` +
    `<details id="finalOutcome" class="final-outcome"${finalOpen ? ' open' : ''}><summary><span>Final outcome</span><strong>${statusGlyph(ep.outcome)}${esc(o?.label || ep.outcome)}</strong></summary>` +
    `<p class="note">Full recorded episode · ${esc(ep.instr_id)}<br>Scan ${esc(ep.scan)}</p><div class="metrics">` +
    `<div><b>${fmt(m.ne, 2)}${m.ne == null ? '' : ' m'}</b>Navigation error</div><div title="${esc(ep.metric_notes?.tl || '')}"><b>${fmt(m.tl)} m</b>${ep.metric_sources?.tl === 'sampled_trace_derived_metric' ? 'Sampled path length' : 'Path length'}</div>` +
    `<div><b>${m.spl == null ? '–' : fmt(m.spl * 100, 0) + '%'}</b><abbr title="Success weighted by path length">SPL</abbr></div><div><b>${m.decisions == null ? '–' : m.decisions}${ep.max_action_len != null ? ' / ' + ep.max_action_len : ''}</b>Decisions</div></div>` +
    `<p class="note">${esc(ep.failure_mode)}${ep.goal ? ` · reference path ${fmt(m.gt_len)} m` : ''}` +
    (S.scan.ce ? `<br>Continuous recording${m.low_level_steps != null ? ` · ${m.low_level_steps} low-level actions` : ''}${ep.diag.collision_count != null ? ` · ${ep.diag.collision_count} collisions` : ''}${m.sp_len != null ? ` · shortest route ${fmt(m.sp_len)} m` : ''}` : '') + '</p></details>';
  $('outcomeCard').innerHTML = h;
  if (finalFocused) $('finalOutcome').querySelector('summary').focus({ preventScroll: true });
  $('frameLabel').textContent = `frame ${f}/${ep.frames.length - 1} · ${decisionContext(fr)}`;
  $('jump').value = hasDecisionTrace() ? fr.decision : '';
  renderSubtasks(fs);
  renderDecision(fs);
  renderComparison();
  renderDiagnostics();
  renderEventShortcuts();
}

function dist3(a, b) {
  const p = node(a).pos, q = node(b).pos;
  return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
}

function renderSubtasks(fs) {
  const ep = S.ep, sub = ep.subtasks, f = S.frame, last = ep.frames.length - 1, dec = fs.dec;
  const w = dec.model?.wattn, words = ep.words, focus = dec.focus;
  const maxW = w ? Math.max(...w, 1e-9) : 0;
  const top3 = w ? new Set([...w.keys()].sort((a, b) => w[b] - w[a]).slice(0, 3)) : new Set();
  const reading = focus ? focus.indexOf(Math.max(...focus)) : -1;
  const current = Math.max(0, Math.min(sub.spans.length - 1, fs.subIdx ?? (reading >= 0 ? reading : 0)));
  const focused = document.activeElement.closest?.('[data-subtask]')?.dataset.subtask;
  const scroll = $('subtasks').scrollTop;
  const undivided = sub.source === 'instruction';
  $('subtasks').hidden = $('subtaskProgress').hidden = undivided;
  document.querySelector('.subinstructions-heading').hidden = undivided;
  const text = ep.instruction;
  $('mainInstruction').textContent = text;
  $('subtasks').classList.toggle('show-all', $('optAllSteps').checked);
  const complete = sub.reached?.filter(hop => hop != null && hop <= f).length;
  $('subtaskProgress').textContent = `${complete == null ? 'No reference completion labels' : `${complete}/${sub.spans.length} completed`} · at frame ${f}`;
  $('subtasks').innerHTML = sub.spans.map(([s, e], k) => {
    let html = '', pos = s;
    if (w && words) {
      words.forEach((wd, i) => {
        if (!wd.span || wd.span[0] < s || wd.span[0] >= e) return;
        const a = w[i] / maxW;
        html += esc(text.slice(pos, wd.span[0])) +
          `<span class="w" style="background:color-mix(in srgb, var(--model) ${Math.round(a * 55)}%, transparent);${top3.has(i) ? 'text-decoration:underline;text-decoration-thickness:2px;' : ''}" title="attention ${(w[i] * 100).toFixed(1)}%">${esc(text.slice(wd.span[0], wd.span[1]))}</span>`;
        pos = wd.span[1];
      });
    }
    html += esc(text.slice(pos, e));
    const progress = subtaskProgress(sub.reached, k, f, last, current, hopWord()), status = progress.state;
    const verdict = `<span class="verdict ${status === 'done' ? 'ok' : status === 'missed' ? 'bad' : ''}">${esc(progress.label)}</span>`;
    const icon = { done: '✓', missed: '✗', current: '▶', pending: k + 1 }[status];
    const foc = focus ? `<span class="attention-share" title="Share of logged attention; this is not an explanation of the model's reasoning">Attention ${pct(focus[k])}${k === reading ? ' · highest' : ''}</span>` : '';
    const target = sub.reached?.[k] ?? ep.frames.findIndex(fr => fr.subtask === k);
    return `<li data-k="${k}" class="${k === current ? 'now' : ''}"><button class="subtask-button" data-subtask="${k}" ${target == null || target < 0 ? 'disabled' : ''} title="Inspect sub-instruction ${k + 1} in the recorded episode">` +
      `<span class="st ${status}" aria-label="${esc(progress.label)}">${icon}</span><span class="txt">${html}</span><span class="meta">${verdict}${foc}</span></button></li>`;
  }).join('');
  $('subtasks').scrollTop = scroll;
  if (focused != null) $('subtasks').querySelector(`[data-subtask="${focused}"]`)?.focus({ preventScroll: true });
  const from = { FGR2R: 'from FGR2R', 'RxR guide timing': 'split into steps and placed on the path by when the RxR guide spoke them' }[sub.source];
  $('subHint').textContent = from
    ? (sub.reached ? `· sub-tasks ${from}, checked against the ground-truth path` : `· sub-tasks ${from} (no ground-truth path in this split)`)
    : undivided ? 'No sub-instruction annotations were supplied.' : '· split by sentence (no sub-task annotation)';
  $('attnHint').textContent = w
    ? `Shading shows logged attention at decision ${dec.t}, not a causal explanation. Sub-instructions and any completion labels come from the configured annotations.`
    : 'Attention was not logged for this decision. Completion labels, when available, come from reference annotations.';
}

function renderDecision(fs) {
  const ep = S.ep, dec = fs.dec, st = fs.state, orc = fs.orc;
  if (dec.source === 'replay_placeholder' || !hasDecisionTrace()) { $('decisionCard').innerHTML = '<h2>Decision trace unavailable</h2><p class="note">This recording contains trajectory samples. Actions, model scores, and stopping decisions were not supplied.</p>'; return; }
  const inTransit = !fs.atDecisionNode;
  const room = vp => (vp && node(vp).region) || '';
  const rawOptions = dec.logged_options?.some(option => option.score != null || option.probability != null);
  let title;
  if (dec.action === 'stop') title = dec.route.length ? `STOP → return to ${short(dec.target)} (best stop score)` : 'STOP here';
  else if (dec.action === 'observe') title = 'Recorded observation';
  else if (dec.action === 'unknown') title = 'Action not supplied';
  else if (ep.adapter === 'recording') title = `MOVE → ${short(dec.target)}${room(dec.target) ? ' · ' + room(dec.target) : ''}`;
  else if (S.scan.ce) title = `${dec.action === 'remote' ? 'WALK BACK, THEN MOVE' : 'MOVE'} → waypoint ${short(dec.choice)} · arrived at ${short(dec.target)}${room(dec.target) ? ' · ' + room(dec.target) : ''}`;
  else title = `${dec.action === 'remote' ? 'REMOTE JUMP' : 'LOCAL MOVE'} → ${short(dec.target)}${room(dec.target) ? ' · ' + room(dec.target) : ''}`;
  let h = `<div class="decision-heading"><h2>Current decision ${dec.t}</h2><span class="evidence-tag">${st.logged ? 'Logged probabilities' : rawOptions ? 'Logged scores' : 'Scores unavailable'}</span></div>` +
    `${inTransit ? '<p class="note">Walking along this recorded decision</p>' : ''}<div class="decision-action">${esc(title)}</div>`;
  if (dec.model && dec.model.forced) h += ep.adapter === 'recording'
    ? '<p class="note">Action marked as externally forced in the recording.</p>'
    : `<p class="note">Stop forced by the ${ep.max_action_len}-decision budget (argmax was a move).</p>`;
  const ce = S.scan.ce ? ceDecisionDetail(dec) : { exec: '', stops: '' };
  h += ce.exec;
  if (st.logged) {
    const w = dec.model.fuse_w;
    if (w != null) h += `<div class="fuse" aria-label="fusion weight"><div class="g" style="flex:${w}"></div><div class="l" style="flex:${1 - w}"></div></div>` +
      `<div class="fuse-lbl"><span>global map ${pct(w)}</span><span>local view ${pct(1 - w)}</span></div>`;
    else if (S.scan.ce && ep.adapter !== 'recording') h += '<p class="note">ETPNav scores every map node and ghost waypoint with one global planner (no local branch).</p>';
    const opts = [{ vp: null, p: st.stop }, ...st.frontier.map(vp => ({ vp, p: st.score.get(vp) }))].filter(o => o.p != null && Number.isFinite(o.p)).sort((a, b) => b.p - a.p);
    const chosenVp = dec.action === 'stop' ? null : dec.choice || dec.target;
    const shown = opts.slice(0, 5);
    const orcVp = orc ? orc.vp : undefined;
    const branches = ep.whatif?.[dec.t] || {};
    if (orc && !shown.some(o => o.vp === orcVp)) { const o = opts.find(o => o.vp === orcVp); if (o) shown.push(o); }
    for (const o of opts) if (branches[o.vp || 'STOP'] && !shown.includes(o)) shown.push(o);   // every branched option
    const branched = Object.keys(branches).length;
    const outcome = (sr, label, ne) => `<span class="whatif-badge ${sr ? 'ok' : 'bad'}">${label} ${sr ? '✓ success' : '✕ fails'} · NE ${fmt(ne)} m</span>`;
    h += '<p class="score-caption">Action scores · higher is preferred</p><div class="decision-options">' + shown.map(o => {
      const marks = [];
      if (o.vp === chosenVp && !(dec.model.forced && o.vp === null)) marks.push('✓ chosen');
      if (orc && o.vp === orcVp) marks.push('★ reference-best');
      if (o.vp && ep.goal) marks.push(`${fmt(ep.dist_to_goal[o.vp])} m to goal`);
      if (o.vp && !S.adj.get(dec.at)?.has(o.vp)) marks.push('remote');
      const label = o.vp ? `${short(o.vp)} ${room(o.vp)}` : 'STOP';
      const key = `${dec.t}|${o.vp || 'STOP'}`, wi = branches[o.vp || 'STOP'];
      let what = '';
      if (wi) what = outcome(wi.sr, 'if taken:', wi.ne) + `<button class="branch-route" data-branch="${esc(key)}" aria-pressed="${UI.branch?.key === key}" title="Show this branch's route on the map in place of the comparison route">route</button>` +
        (wi.prefix_agree === false ? '<span class="hint" title="The model did not reproduce the recorded moves before this decision when re-run">replay differed</span>' : '');
      else if (branched && o.vp === chosenVp && ep.metrics.sr != null) what = outcome(ep.metrics.sr, 'recorded:', ep.metrics.ne);
      return `<div class="dbar${o.vp === chosenVp ? ' chosen-option' : ''}"><button class="lbl inspect-node" data-inspect-vp="${esc(o.vp || dec.at)}" title="Highlight ${esc(label)} on the map">${esc(label)}</button><div class="track"><div class="fill${o.vp ? '' : ' stop'}" style="width:${(o.p * 100).toFixed(1)}%"></div></div>` +
        `<span class="val">${pct(o.p)}</span>${marks.length ? `<span class="marks">${marks.join(' · ')}</span>` : ''}${what ? `<span class="whatif">${what}</span>` : ''}</div>`;
    }).join('') + '</div>' + ce.stops;
    if (dec.ref?.ok === false && !inTransit) h += `<p class="note nonbest-note">✕ Not the reference-best:${esc(MISTAKE[dec.ref.type].toLowerCase())}${dec.ref.gap != null ? ` (${fmt(dec.ref.gap)} m further${ep.reference === 'route' ? ' from the next reference viewpoint' : ' from the goal'})` : ''}.</p>`;
    if (orc && chosenVp !== orcVp && !inTransit) h += `<p class="note">Reference-best action ranked #${opts.findIndex(o => o.vp === orcVp) + 1} of ${opts.length}.</p>`;
    if (orc) h += `<p class="reference-note">${ep.reference === 'route' ? 'Reference-best follows the annotated route (RxR paths are not shortest paths).' : 'Reference-best uses ground-truth goal distance.'}</p>`;
    if (S.gpu.allowed !== false && runInfo(S.run)?.capabilities?.whatif && ep.goal && !dec.model.forced && dec.t < ep.max_action_len - 1) {
      const g = S.gpu, alternatives = Math.min(opts.length - 1, 8) + (orc && chosenVp !== orcVp && opts.findIndex(o => o.vp === orcVp) > 8 ? 1 : 0);
      const missing = alternatives - branched;
      h += `<p class="whatif-line note">${branched ? `What-if: ${branched} option${branched > 1 ? 's' : ''} branched. ` : ''}` +
        (missing <= 0 ? '' : g.enabled ? `<button class="whatif-decision" data-whatif-t="${dec.t}" ${g.job ? 'disabled' : ''}>Branch ${branched ? 'the remaining options' : `this decision · ${alternatives} options`}</button>`
          : `<span title="What-if branches re-run the model on the GPU">Turn on What-if · GPU (top bar) to see where the other options lead.</span>`) + '</p>';
    }
  } else if (rawOptions) {
    h += '<p class="score-caption">Recorded option values · raw scores are not probabilities</p><div class="comparison-table-wrap"><table class="comparison-table"><thead><tr><th scope="col">Action</th><th scope="col">Recorded value</th></tr></thead><tbody>' +
      dec.logged_options.map(option => {
        const vp = option.node_id, label = vp ? `${short(vp)} ${room(vp)}` : 'STOP';
        const values = [option.probability != null ? `probability ${pct(option.probability)}` : '', option.score != null ? `score ${fmt(option.score, 3)}` : ''].filter(Boolean);
        return `<tr><th scope="row"><button class="link inspect-node" data-inspect-vp="${esc(vp || dec.at)}">${esc(label)}</button></th><td>${values.join(' · ') || 'not supplied'}</td></tr>`;
      }).join('') + '</tbody></table></div>';
  } else {
    h += `<p class="note">Available graph context: ${st.visited.length} visited, ${st.frontier.length} frontier nodes.` +
      `${st.stop != null ? ` Stop prob here: ${pct(st.stop)}.` : ''}` +
      `${orc ? ` Reference-best: ${orc.vp ? esc(short(orc.vp)) + ' (' + fmt(orc.dist) + ' m to goal)' : 'stop here'}.` : ''}<br>` +
      'Action scores are not available in this decision log.</p>';
  }
  const focused = document.activeElement.dataset?.inspectVp;
  const optionsScroll = $('decisionCard').querySelector('.decision-options')?.scrollTop || 0;
  $('decisionCard').innerHTML = h;
  const options = $('decisionCard').querySelector('.decision-options');
  if (options) options.scrollTop = optionsScroll;
  if (focused) $('decisionCard').querySelector(`[data-inspect-vp="${CSS.escape(focused)}"]`)?.focus({ preventScroll: true });
}

// Continuous runs: what the waypoint predictor proposed at this node and how the low-level controller carried
// the move out (exec), and for STOP the stop score each visited node got when the agent stood there, which
// ETPNav picks its stop node from, walking back if that is not the current node (stops).
function ceDecisionDetail(dec) {
  const ep = S.ep, ex = dec.exec || {}, rows = [];
  const li = (k, v, title = '') => rows.push(`<li${title ? ` title="${esc(title)}"` : ''}><span>${k}</span><span class="v">${v}</span></li>`);
  if (dec.cands?.length) {
    const before = new Set(dec.t > 0 ? ep.decisions[dec.t - 1].model?.gmap || [] : []);
    const count = test => dec.cands.filter(c => test(c[1])).length;
    const onNode = count(m => m?.[0] === 'n'), known = count(m => m?.[0] === 'g' && before.has(m)), dropped = count(m => !m);
    const fresh = dec.cands.length - onNode - known - dropped;
    const parts = [fresh && `${fresh} new ghost${fresh > 1 ? 's' : ''}`, known && `${known} merged into earlier ghosts`,
      onNode && `${onNode} on visited nodes`, dropped && `${dropped} dropped`].filter(Boolean);
    li('Waypoint predictor', `${dec.cands.length} waypoints · ${parts.join(', ')}`, 'Waypoints predicted from the depth panorama at this node (spokes on the 2D map)');
  }
  if (dec.action !== 'stop' && ex.aim_m != null) {
    const warn = ex.moved_m < 0.25 ? ' · <b class="bad">little net displacement</b>'
      : ex.miss_m >= 1 ? ` · <b class="bad">ended ${fmt(ex.miss_m)} m from the waypoint</b>` : '';
    li('Controller', `${dec.action === 'remote' ? 'walked back, then ' : ''}aimed ${fmt(ex.aim_m)} m · walked ${fmt(ex.walked_m)} m · ended ${fmt(ex.miss_m, 2)} m from it${warn}`,
      'Recorded waypoint execution distances; low net displacement alone does not establish an obstruction');
  }
  const exec = rows.length ? `<ul class="diag ce-exec">${rows.join('')}</ul>` : '';
  const scores = dec.model?.stop_scores;
  if (dec.action !== 'stop' || !scores) return { exec, stops: '' };
  const radius = margin(), inZone = vp => ep.goal && radius != null && ep.dist_to_goal[vp] != null && ep.dist_to_goal[vp] < radius;
  const opts = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const shown = opts.slice(0, 4), zone = opts.find(([vp]) => inZone(vp)), at = dec.model.stop_at;
  if (zone && !shown.includes(zone)) shown.push(zone);
  let stops = '<p class="score-caption">Recorded stop scores at visited nodes</p><div class="decision-options">' +
    shown.map(([vp, p]) => {
      const marks = [vp === at ? '✓ stops here' : '', ep.goal ? `${fmt(ep.dist_to_goal[vp])} m to goal` : '', inZone(vp) ? 'in success area' : ''].filter(Boolean);
      return `<div class="dbar${vp === at ? ' chosen-option' : ''}"><button class="lbl inspect-node" data-inspect-vp="${esc(vp)}" title="Highlight ${esc(vp)} on the map">${esc(vp)} ${esc(node(vp)?.region || '')}</button>` +
        `<div class="track"><div class="fill stop" style="width:${(p * 100).toFixed(1)}%"></div></div><span class="val">${pct(p)}</span><span class="marks">${marks.join(' · ')}</span></div>`;
    }).join('') + '</div>';
  if (zone && zone[0] !== at) stops += `<p class="note nonbest-note">${esc(zone[0])} is inside the success area (${fmt(ep.dist_to_goal[zone[0]])} m from the goal) but its stop score was only ${pct(zone[1])}.</p>`;
  return { exec, stops };
}

function comparisonDivergence() {
  const other = S.ep?.others[S.compare];
  return other?.cross ? null : firstDivergence(S.ep?.path, other?.path);   // other setting: ids never match
}

function renderComparison() {
  const card = $('comparisonCard'), ep = S.ep, other = ep.others[S.compare];
  const restoreFocus = document.activeElement.id === 'firstDivergence';
  card.hidden = !S.compare;
  if (!S.compare) return;
  const left = modelLabel(runInfo(S.run).model), right = modelLabel(runInfo(S.compare).model);
  if (!other) { card.innerHTML = `<h2>Model comparison</h2><p>No matching episode is available for ${esc(right)}.</p>`; return; }
  const a = ep.metrics, b = other.metrics;
  const row = (label, key, digits, unit) => `<tr><th scope="row">${label}</th><td>${fmt(a[key], digits)}${a[key] == null ? '' : unit}</td><td>${fmt(b[key], digits)}${b[key] == null ? '' : unit}</td><td>${metricDifference(a[key], b[key], digits, unit)}</td></tr>`;
  const divergence = comparisonDivergence();
  card.innerHTML = '<h2>Model comparison</h2><p class="note">Final results · same instruction</p>' +
    `<div class="comparison-table-wrap"><table class="comparison-table"><thead><tr><th scope="col">Metric</th><th scope="col">${esc(left)}</th><th scope="col">${esc(right)}</th><th scope="col" title="Selected model minus compared model">Δ</th></tr></thead><tbody>` +
    `<tr><th scope="row">Outcome</th><td>${statusGlyph(ep.outcome)}${esc(OUTCOME[ep.outcome]?.label || ep.outcome)}</td><td>${statusGlyph(other.outcome)}${esc(OUTCOME[other.outcome]?.label || other.outcome)}</td><td>—</td></tr>` +
    row('Goal error', 'ne', 2, ' m') + row('Path length', 'tl', 1, ' m') + row('Decisions', 'decisions', 0, '') + (other.cross ? '' : row('Hops', 'hops', 0, '')) + '</tbody></table></div>' +
    `<p class="comparison-delta">Δ = selected model − comparison</p>` +
    (other.cross ? `<p class="note">Same instruction in the other setting (${S.scan.ce ? 'discrete navigation graph' : 'continuous environment'}); its route is drawn on the map.</p>`
      : divergence ? `<button id="firstDivergence" data-frame="${divergence.frame}" title="Inspect the first different viewpoint after the shared route, or the point where one route ends">First route divergence · hop ${divergence.index}</button>` : '<p class="note">The recorded viewpoint routes match.</p>');
  if (restoreFocus) $('firstDivergence')?.focus({ preventScroll: true });
}

function renderEventShortcuts() {
  const ep = S.ep, key = `${S.run}/${ep.instr_id}`;
  if (UI.eventsKey !== key) {
    UI.eventsKey = key; UI.eventId = null;
    UI.events = episodeEvents(ep, margin());
    $('episodeSummary').textContent = episodeSummary(ep, margin(), hopWord());
    $('eventShortcuts').innerHTML = UI.events.map(event => `<button data-event="${event.id}" title="${esc(event.detail)}">${esc(event.label)}<span>${hopWord()} ${event.frame}</span></button>`).join('');
  }
  for (const button of $('eventShortcuts').children) {
    const event = UI.events.find(event => event.id === button.dataset.event);
    button.setAttribute('aria-pressed', event.frame === S.frame);
  }
  const event = UI.events.find(event => event.id === UI.eventId && event.frame === S.frame);
  $('eventContext').textContent = event ? `${event.label} · ${hopWord()} ${event.frame}. ${event.detail}` : `Selected frame ${S.frame} · ${kindLabel(curFrame().kind)}. Event buttons inspect the full recorded episode.`;
  $('eventContext').hidden = !event;
  $('frameSeek').max = ep.frames.length - 1;
  $('frameSeek').value = S.frame;
  $('frameSeek').setAttribute('aria-valuetext', `Frame ${S.frame} of ${ep.frames.length - 1}, ${decisionContext(curFrame())}`);
  $('frameSeekLabel').textContent = `${S.frame} / ${ep.frames.length - 1}`;
}

function renderDiagnostics() {
  const ep = S.ep, d = ep.diag;
  const known = key => Object.hasOwn(d, key) && d[key] != null;
  const jump = (frame, label) => (!Number.isInteger(frame) || frame < 0 ? '' : `<button data-frame="${frame}" title="jump to frame ${frame}">${label}</button>`);
  const li = (k, v, extra = '') => `<li><span>${k}</span><span class="v">${v}${extra}</span></li>`;
  const rows = [];
  if (known('subtasks')) rows.push(li('Sub-tasks completed', `${d.subtasks_done} / ${d.subtasks}`));
  if (known('first_off_gt_hop')) rows.push(li('First step outside reference path', `${hopWord()} ${d.first_off_gt_hop}`, jump(d.first_off_gt_hop, 'go')));
  if (known('lost_hop')) rows.push(li('Left declared reference corridor', `${hopWord()} ${d.lost_hop}`, jump(d.lost_hop, 'go')));
  if (known('closest_dist')) rows.push(li(ep.ce ? 'Closest sampled approach to goal' : 'Closest approach to goal', `${fmt(d.closest_dist, 2)} m${known('closest_hop') ? ` @ ${hopWord()} ${d.closest_hop}` : ''}`, jump(d.closest_hop, 'go')));
  if (d.goal_visit_unsampled) rows.push(li('Goal visit between playback samples', `${fmt(d.dense_closest_dist, 2)} m in the recorded dense track; no exact replay frame`));
  if (known('stuck')) {
    const first = ep.decisions.findIndex(x => ['local', 'remote'].includes(x.action) && node(x.target) && dist3(x.at, x.target) < 0.25);
    rows.push(li('Moves below 0.25 m displacement', d.stuck, jump(first < 0 ? null : ep.decisionFrame[first], 'first')));
  }
  if (known('ctrl_miss_max')) rows.push(li('Waypoint execution mismatch', `worst ${fmt(d.ctrl_miss_max, 2)} m · mean ${fmt(d.ctrl_miss_mean, 2)} m`));
  if (known('actions_per_decision')) rows.push(li('Low-level actions per decision', fmt(d.actions_per_decision)));
  if (known('ghosts')) rows.push(li('Ghost waypoints predicted', d.ghosts));
  if (known('stop_score')) rows.push(li('Stop score: final / best in success area', `${pct(d.stop_score)} / ${d.zone_stop_score == null ? 'unavailable' : pct(d.zone_stop_score)}`));
  if (known('revisited_nodes')) rows.push(li('Viewpoints revisited', `${d.revisited_nodes}${d.max_visits > 1 ? ` (max ×${d.max_visits})` : ''}`, jump(ep.frames.findIndex(f => f.visits >= 2), 'first')));
  if (known('remote_jumps')) rows.push(li('Remote-frontier jumps', d.remote_jumps, jump(ep.frames.findIndex(f => f.kind === 'remote'), 'first')));
  if (known('backtrack_hops')) rows.push(li('Backtracking', `${d.backtrack_hops} hops${known('backtrack_m') ? `, ${fmt(d.backtrack_m)} m` : ''}`));
  if (known('oscillations')) rows.push(li('A→B→A returns', d.oscillations));
  if (known('stop_return')) rows.push(li('Returned to earlier node to stop', d.stop_return ? 'yes' : 'no', d.stop_return ? jump(ep.frames.findIndex(f => f.kind === 'return'), 'go') : ''));
  if (known('floors_visited') && S.scan.floors.length > 1) rows.push(li('Floors visited', `${d.floors_visited} of ${S.scan.floors.length}`));
  $('diagCard').innerHTML = '<p class="note">Full recorded episode · available diagnostics</p>' + (rows.length ? `<ul class="diag">${rows.join('')}</ul>` : '<p class="note">No diagnostic evidence was supplied for this episode.</p>');
  $('diagSummary').textContent = [d.first_missed_subtask ? `first missed sub-task ${d.first_missed_subtask}` : '', d.lost_hop != null ? `left reference area at ${hopWord()} ${d.lost_hop}` : '', d.revisited_nodes ? `${d.revisited_nodes} revisits` : ''].filter(Boolean).join(' · ');
}

// ---------------------------------------------------------------------------------------------- wiring

$('model').addEventListener('change', e => { const key = fillSplits(e.target.value, runInfo(S.run).split); selectRun(key); });
$('split').addEventListener('change', e => selectRun(e.target.value));
$('compare').addEventListener('change', e => selectCompare(e.target.value));
$('toggleSide').addEventListener('click', () => {
  const hidden = $('layout').classList.toggle('noside');
  $('toggleSide').setAttribute('aria-expanded', String(!hidden));
});
$('prevEp').addEventListener('click', () => nextEpisode(-1));
$('nextEp').addEventListener('click', () => nextEpisode(1));
$('outcomeChips').addEventListener('click', e => {
  const b = e.target.closest('.chip');
  if (!b) return;
  clearInvestigationNavigation();
  const k = b.dataset.o;
  S.outcomes.has(k) ? S.outcomes.delete(k) : S.outcomes.add(k);
  b.setAttribute('aria-pressed', S.outcomes.has(k));
  applyFilters();
});
['fMode', 'fScan', 'fSub', 'sort', 'fRemote', 'fReturn', 'fDiff'].forEach(id => $(id).addEventListener('change', applyFilters));
['fLen', 'fRev', 'fText'].forEach(id => $(id).addEventListener('input', applyFilters));
$('rows').addEventListener('click', e => {
  const tr = e.target.closest('tr');
  if (!tr) return;
  clearInvestigationNavigation();
  if (S.page === 'split') setPage('episode');
  loadEpisode(tr.dataset.id);
});
['fPlace', 'fFirst', 'fConfident', 'fTag'].forEach(id => $(id).addEventListener('change', applyFilters));
$('sliceChip').addEventListener('click', e => { if (e.target.closest('#clearSlice')) applySlice(null); });
$('viewEpisode').addEventListener('click', () => setPage('episode'));
$('viewSplit').addEventListener('click', () => setPage('split'));
$('viewInvestigation').addEventListener('click', () => setPage('investigation'));
$('returnInvestigationQueue').addEventListener('click', () => setPage('investigation'));
for (const type of ['input', 'change']) $('sidebar').addEventListener(type, clearInvestigationNavigation);
$('splitUseFilters').addEventListener('change', renderSplitPage);
$('optGPU').addEventListener('change', async e => {
  const on = e.target.checked;
  if (!on && (S.gpu.job || S.gpu.queued?.length) && !confirm('Turn the GPU switch off? The running what-if job stops; branches already finished are kept.')) { e.target.checked = true; return; }
  try { S.gpu = await postJSON('/api/whatif/switch', { on }); } catch (err) { alert(err.message); }
  refreshGpu();
  if (S.ep) renderDecision(frameState(S.frame));
});
$('gpuCancel').addEventListener('click', async () => { await postJSON('/api/whatif/cancel', {}); refreshGpu(); });
$('tagChips').addEventListener('click', e => {
  const b = e.target.closest('[data-tag]');
  if (!b || !S.ep) return;
  const tags = currentTags(), t = b.dataset.tag;
  saveNotes(tags.includes(t) ? tags.filter(x => x !== t) : [...tags, t], $('noteText').value);
});
const addTag = () => {
  const t = $('tagInput').value.trim();
  if (!t || !S.ep) return;
  $('tagInput').value = '';
  if (!currentTags().includes(t)) saveNotes([...currentTags(), t], $('noteText').value);
};
$('tagAdd').addEventListener('click', addTag);
$('tagInput').addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); addTag(); } });
$('noteText').addEventListener('input', () => {
  const id = S.ep.instr_id, run = S.run, tags = currentTags(), text = $('noteText').value;   // saved even if another episode or run opens first
  const key = JSON.stringify([run, id]);
  clearTimeout(noteTimers.get(key));
  noteDrafts.set(key, { tags, note: text });
  $('notesStatus').textContent = 'Editing…';
  noteTimers.set(key, setTimeout(() => { noteTimers.delete(key); saveNotes(tags, text, id, run); }, 700));
});
$('rows').addEventListener('keydown', e => {
  const button = e.target.closest('.episode-pick');
  if (!button || !['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(e.key)) return;
  e.preventDefault(); e.stopPropagation();
  const buttons = [...$('rows').querySelectorAll('.episode-pick')], index = buttons.indexOf(button);
  const next = e.key === 'Home' ? 0 : e.key === 'End' ? buttons.length - 1 : Math.max(0, Math.min(buttons.length - 1, index + (e.key === 'ArrowDown' ? 1 : -1)));
  buttons.forEach((b, i) => { b.tabIndex = i === next ? 0 : -1; });
  buttons[next].focus();
});
document.querySelectorAll('.export button').forEach(b => b.addEventListener('click', () => exportRows(b)));
$('floors').addEventListener('click', e => {
  const b = e.target.closest('button');
  if (!b) return;
  if (UI.goalPreview) centerOnAgent();
  const v = b.dataset.floor;
  S.floor = v === 'auto' || v === 'all' ? v : Number(v);
  S.shownFloor = S.floor === 'auto' ? curFrame().floor : S.floor;
  map3d.setFloorMode(S.floor);
  if (map3d.mode === 'goal') map3d.setCamera(S.floor === 'auto' ? 'chase' : 'overview', true);
  renderFloors(); fitView(); renderStatic(); renderDynamic(1);
});
$('tab3d').addEventListener('click', () => setTab('3d'));
$('tab2d').addEventListener('click', () => setTab('2d'));
document.querySelector('.map-tabs').addEventListener('keydown', e => {
  if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
  e.preventDefault(); e.stopPropagation();
  const tab = e.key === 'Home' ? '3d' : e.key === 'End' ? '2d' : S.tab === '3d' ? '2d' : '3d';
  setTab(tab); $('tab' + tab).focus();
});
$('camModes').addEventListener('click', e => {
  const b = e.target.closest('button');
  if (!b) return;
  if (UI.goalPreview) centerOnAgent();
  const wasGoal = map3d.mode === 'goal';
  if (wasGoal) {
    S.floor = 'auto'; S.shownFloor = curFrame().floor;
    map3d.setFloorMode('auto'); renderFloors();
  }
  map3d.setCamera(b.dataset.cam, wasGoal);
  if (wasGoal) { fitView(); renderStatic(); renderDynamic(1); }
});
$('jumpGoal').addEventListener('click', jumpToGoal);
$('centerAgent').addEventListener('click', centerOnAgent);
$('returnAgent').addEventListener('click', () => {
  centerOnAgent();
  // This button disappears on return; keep keyboard focus on the persistent camera control.
  $('centerAgent').focus({ preventScroll: true });
});
$('analysisView').addEventListener('click', () => {
  if (UI.goalPreview) centerOnAgent();
  $('optLabels').checked = false;
  $('optViews').checked = false;
  try { localStorage.setItem('vlnviz-optLabels', 'false'); localStorage.setItem('vlnviz-optViews', 'false'); } catch (_) { /* storage blocked */ }
  setTab('2d'); refreshPresentation();
  if (S.ep) { if (map3d.mode === 'goal') map3d.setCamera('overview', true); fitView(); renderStatic(); renderDynamic(1); }
  document.querySelector('.map-settings').open = false;
  $('tab2d').focus({ preventScroll: true });
});
for (const id of ['optLabels', 'optAllSteps', 'optModelRoute', 'optReferenceRoute', 'optCompareRoute', 'optViews']) {
  try { const saved = localStorage.getItem('vlnviz-' + id); if (saved != null) $(id).checked = saved === 'true'; } catch (_) { /* storage blocked */ }
  $(id).addEventListener('change', () => {
    try { localStorage.setItem('vlnviz-' + id, String($(id).checked)); } catch (_) { /* storage blocked */ }
    if (id === 'optAllSteps') { if (S.ep) renderSubtasks(frameState(S.frame)); }
    else refreshPresentation();
  });
}
map3d.setPresentation(presentationOptions());
document.querySelector('.views').hidden = !$('optViews').checked;
$('jumpGoal').addEventListener('keydown', e => { if (e.key === ' ' || e.key === 'Enter') e.stopPropagation(); });
try { $('optNodes').checked = localStorage.getItem('vlnviz-show-nodes') !== 'false'; } catch (_) { /* storage blocked */ }
updateMapControls();
$('optNodes').addEventListener('change', () => {
  try { localStorage.setItem('vlnviz-show-nodes', String($('optNodes').checked)); } catch (_) { /* storage blocked */ }
  updateMapControls();
  $('tip').hidden = true;
  if (!S.ep) return;
  renderStatic(); renderDynamic(1); renderLegend();
  refreshViewMarkers();
});
function updateCutawayControls() {
  const enabled = $('optCutaway').checked;
  $('cut').disabled = !enabled;
  $('cut').title = enabled ? 'Height above the selected floor' : 'Enable Cutaway to adjust the cut height';
  map3d.setCutaway(enabled);
}
function updateCutHeight() {
  const height = Number($('cut').value);
  $('cutLabel').textContent = `${height.toFixed(1)} m`;
  map3d.setCut(height);
}
try {
  $('optCutaway').checked = localStorage.getItem('vlnviz-cutaway') === 'true';
  const height = Number(localStorage.getItem('vlnviz-cut-height'));
  if (Number.isFinite(height) && height >= .6 && height <= 3) $('cut').value = height;
} catch (_) { /* storage blocked */ }
updateCutHeight();
updateCutawayControls();
$('optCutaway').addEventListener('change', () => {
  updateCutawayControls();
  try { localStorage.setItem('vlnviz-cutaway', String($('optCutaway').checked)); } catch (_) { /* storage blocked */ }
});
$('cut').addEventListener('input', () => {
  updateCutHeight();
  try { localStorage.setItem('vlnviz-cut-height', $('cut').value); } catch (_) { /* storage blocked */ }
});
$('optMesh').addEventListener('change', e => map3d.meshVisible(e.target.checked));
try { $('optHD').checked = localStorage.getItem('vlnviz-hd') !== 'false'; } catch (_) { /* storage blocked */ }
map3d.setDetail($('optHD').checked);
try { $('optPhoto').checked = localStorage.getItem('vlnviz-photo') !== 'false'; } catch (_) { /* storage blocked */ }
map3d.setPhoto($('optPhoto').checked);
$('optPhoto').addEventListener('change', e => {
  map3d.setPhoto(e.target.checked);
  try { localStorage.setItem('vlnviz-photo', String(e.target.checked)); } catch (_) { /* storage blocked */ }
});
$('optHD').addEventListener('change', e => {
  map3d.setDetail(e.target.checked);
  try { localStorage.setItem('vlnviz-hd', String(e.target.checked)); } catch (_) { /* storage blocked */ }
});
['optRooms', 'optFuture', 'optNavmap'].forEach(id => $(id).addEventListener('change', () => { renderStatic(); renderDynamic(1); renderLegend(); }));
['optScores', 'optStop'].forEach(id => $(id).addEventListener('change', () => { if (S.ep) setFrame(S.frame, false, true); }));
$('fit').addEventListener('click', () => { if (UI.goalPreview) centerOnAgent(); if (map3d.mode === 'goal') map3d.setCamera('overview', true); fitView(); renderStatic(); renderDynamic(1); });
$('fitFloor').addEventListener('click', () => { if (UI.goalPreview) centerOnAgent(); if (map3d.mode === 'goal') map3d.setCamera('overview', true); fitView(true); renderStatic(); renderDynamic(1); });
$('bPlay').addEventListener('click', play);
try { $('optAutoNext').checked = localStorage.getItem('vlnviz-auto-next') === 'true'; } catch (_) { /* storage blocked */ }
$('optAutoNext').addEventListener('change', () => {
  const enabled = $('optAutoNext').checked;
  try { localStorage.setItem('vlnviz-auto-next', String(enabled)); } catch (_) { /* storage blocked */ }
  if (!enabled && S.advance) stop();
  if (enabled && S.ep && !S.playing && !UI.goalPreview && S.frame === S.ep.frames.length - 1) queueNextEpisode();
  if (UI.goalPreview) updateViewContext();
});
$('bStart').addEventListener('click', () => { stop(); setFrame(0); });
$('bEnd').addEventListener('click', () => { stop(); setFrame(S.ep.frames.length - 1); });
$('bPrev').addEventListener('click', () => { stop(); setFrame(S.frame - 1); });
$('bNext').addEventListener('click', () => { stop(); setFrame(S.frame + 1, true); });
$('bPrevD').addEventListener('click', () => { stop(); stepDecision(-1); });
$('bNextD').addEventListener('click', () => { stop(); stepDecision(1); });
$('speed').addEventListener('input', e => { $('speedLabel').textContent = `${Number(e.target.value)} s/step`; });
$('jump').addEventListener('change', e => { stop(); const d = Number(e.target.value); setFrame(S.ep.decisionFrame[d] ?? S.frame); });
$('frameSeek').addEventListener('input', e => { if (S.ep) { stop(); setFrame(Number(e.target.value)); } });
$('eventShortcuts').addEventListener('click', e => {
  const event = UI.events.find(event => event.id === e.target.closest('[data-event]')?.dataset.event);
  if (event) { UI.eventId = event.id; stop(); setFrame(event.frame); }
});
$('comparisonCard').addEventListener('click', e => {
  if (!e.target.closest('#firstDivergence')) return;
  const divergence = comparisonDivergence();
  if (!divergence) return;
  stop(); setFrame(divergence.frame); UI.divergence = true;
  // Both branches can be on different floors. Show them together for this explicit inspection.
  S.floor = 'all'; S.shownFloor = 'all'; map3d.setFloorMode('all'); map3d.setCamera('overview', true);
  renderFloors(); fitView(); renderStatic(); refreshInspection();
  $('eventContext').textContent = `First route divergence · hop ${divergence.index}. Highlighted branches compare the recorded viewpoint routes; decision numbers may differ between models.`;
  $('eventContext').hidden = false;
});
for (const name of ['pointerover', 'focusin']) $('decisionCard').addEventListener(name, e => {
  const button = e.target.closest('[data-inspect-vp]');
  if (button) { UI.hoverVp = button.dataset.inspectVp; refreshInspection(); }
});
for (const name of ['pointerout', 'focusout']) $('decisionCard').addEventListener(name, e => {
  if (e.target.closest('[data-inspect-vp]') && !e.relatedTarget?.closest?.('[data-inspect-vp]')) { UI.hoverVp = null; refreshInspection(); }
});
$('decisionCard').addEventListener('click', e => {
  const branch = e.target.closest('[data-branch]');
  if (branch) return toggleBranch(branch.dataset.branch);
  const run = e.target.closest('[data-whatif-t]');
  if (run) { run.disabled = true; return whatIfDecision(Number(run.dataset.whatifT)); }
  const button = e.target.closest('[data-inspect-vp]');
  if (!button) return;
  UI.selectedVp = UI.selectedVp === button.dataset.inspectVp ? null : button.dataset.inspectVp;
  $('decisionCard').querySelectorAll('[data-inspect-vp]').forEach(b => b.setAttribute('aria-pressed', b.dataset.inspectVp === UI.selectedVp));
  refreshInspection();
});
$('diagCard').addEventListener('click', e => { const b = e.target.closest('button[data-frame]'); if (b) { stop(); setFrame(Number(b.dataset.frame)); } });
$('subtasks').addEventListener('click', e => {
  const li = e.target.closest('li[data-k]');
  if (!li || !S.ep || e.target.closest('button:disabled')) return;
  const k = Number(li.dataset.k), sub = S.ep.subtasks;
  const f = sub.reached?.[k] ?? S.ep.frames.findIndex(fr => fr.subtask === k);
  if (f != null && f >= 0) { stop(); setFrame(f); }
});
$('theme').addEventListener('click', async () => {
  const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('vlnviz-theme', document.documentElement.dataset.theme); } catch (_) { /* storage blocked */ }
  if (S.ep) { await map3d.recolor(); map3d.setEpisode(S.ep, otherPath()); setFrame(S.frame, false, true); renderStatic(); renderDynamic(1); }
});
try { const t = localStorage.getItem('vlnviz-theme'); if (t) document.documentElement.dataset.theme = t; } catch (_) { /* storage blocked */ }
document.addEventListener('keydown', e => {
  if (S.page !== 'episode' || !S.ep || e.target.closest('input, select, textarea, button, summary, a, [role="separator"], [contenteditable="true"]') || e.metaKey || e.ctrlKey || e.altKey) return;
  const keys = {
    ' ': () => play(),
    ArrowRight: () => { stop(); e.shiftKey ? stepDecision(1) : setFrame(S.frame + 1, true); },
    ArrowLeft: () => { stop(); e.shiftKey ? stepDecision(-1) : setFrame(S.frame - 1); },
    Home: () => { stop(); setFrame(0); },
    End: () => { stop(); setFrame(S.ep.frames.length - 1); },
    j: () => nextEpisode(1), k: () => nextEpisode(-1),
  };
  if (keys[e.key]) { e.preventDefault(); keys[e.key](); }
});
function nextEpisode(dir) {
  stop();
  const rows = episodeNavigationRows();
  if (!S.ep || !rows.length) return;
  const i = rows.findIndex(r => r.instr_id === S.ep.instr_id);
  const next = rows[i < 0 ? 0 : Math.max(0, Math.min(rows.length - 1, i + dir))];
  if (next && next.instr_id !== S.ep.instr_id) loadEpisode(next.instr_id);
}

window.vlnviz = { S, setFrame, map3d, views, investigations };   // for debugging from the console
setTab('3d');
init().catch(e => { $('mapEmpty').hidden = false; $('mapEmpty').textContent = 'Failed to load: ' + e.message; console.error(e); });

// A link pasted into an open tab only changes the fragment; follow it. (updateHash uses replaceState, which
// does not fire this event.)
window.addEventListener('hashchange', async () => {
  if (!S.meta) return;
  const h = new URLSearchParams(location.hash.slice(1)), run = runInfo(h.get('run'));
  if (run && run.name !== S.run) {
    $('model').value = run.model;
    fillSplits(run.model, run.name);
    await selectRun(run.name, false, h.get('cmp'));
  }
  const id = h.get('id'), f = Number(h.get('f') || 0);
  if (id && S.rows.some(r => r.instr_id === id) && id !== S.ep?.instr_id) await loadEpisode(id, f);
  else if (S.ep && f !== S.frame) setFrame(f);
  const page = ['split', 'investigation'].includes(h.get('view')) ? h.get('view') : 'episode';
  if (page !== S.page) setPage(page);
});
