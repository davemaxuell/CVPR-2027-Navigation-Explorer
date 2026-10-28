// Evidence-based review of recorded navigation. Suggested explanations never imply a causal diagnosis.
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const number = value => Number(value || 0).toLocaleString();
const metric = (value, suffix = '', scale = 1) => value == null || !Number.isFinite(Number(value)) ? '—' : `${(Number(value) * scale).toFixed(1)}${suffix}`;
const outcomeLabel = outcome => ({ success: 'Success', wrong_stop: 'Wrong stop', step_limit: 'Step limit', failure: 'Failure', unscored: 'Unscored', 'no ground truth': 'No reference' }[outcome] || outcome || 'Unscored');
// Short display labels only; the exported categories and evidence retain their full wording.
const signalLabel = signal => ({
  missed_stop: 'Goal visit, failed finish',
  blocked_movement: 'Low displacement',
  waypoint_miss: 'Waypoint mismatch',
  oscillation: 'Repeated backtracking',
  inefficient_success: 'Inefficient success',
  confident_reference_disagreement: 'High-score reference mismatch',
  evidence_gap: 'Evidence gaps',
}[signal.id || signal.kind] || signal.label);
const valueText = value => typeof value === 'object' ? JSON.stringify(value) : String(value ?? 'Unavailable');
const evidenceText = item => {
  if (typeof item.value === 'number' && item.value >= 0 && item.value <= 1 && /probability/i.test(item.label)) return `${(item.value * 100).toFixed(1)}%`;
  if (typeof item.value === 'boolean') return item.value ? 'Yes' : 'No';
  if (/\/episode\/metrics\/(sr|osr)$/.test(item.path || '') && [0, 1].includes(item.value)) return item.value ? 'Yes' : 'No';
  return valueText(item.value);
};
const sourceLabel = value => ({
  recorded_metric: 'Recorded evaluation metric',
  reference_derived_metric: 'Calculated using reference annotations',
  trace_derived_metric: 'Calculated from the recorded trajectory',
  sampled_trace_derived_metric: 'Calculated from trajectory samples; travel between samples may be missing',
  unavailable: 'Not supplied',
  trace_derived: 'Derived from the recorded trace',
  reference_derived: 'Derived using reference annotations',
  logged_score: 'Logged model score',
  unspecified_metric: 'Metric source unspecified',
}[value] || value || 'Recorded trace');
const safeLink = value => {
  try { const url = new URL(value, location.href); return url.origin === location.origin && ['http:', 'https:'].includes(url.protocol) ? url.href : '#'; }
  catch (_) { return '#'; }
};

export class Investigations {
  constructor({ page, card, onOpen, onSeek, onQueue }) {
    Object.assign(this, { page, card, onOpen, onSeek, onQueue });
    this.run = null; this.caseKey = null; this.caseData = null; this.findingId = null;
    this.filters = { kind: '', q: '', outcome: '', offset: 0, limit: 50 };
    this.queueRequest = 0; this.caseRequest = 0; this.frame = 0;
    page.innerHTML = `<div class="investigation-heading"><div><span class="eyebrow">Recorded behavior · review workspace</span><h2>Investigation queue</h2><p>Find unusual behavior, inspect its evidence, and choose the next test.</p></div><button type="button" data-investigation-refresh>Refresh queue</button></div>
      <p class="investigation-scope">Flagged successes and failures from the full selected run.</p>
      <div class="investigation-filters"><label>Search episodes<input type="search" data-investigation-search placeholder="Episode ID, scan, or instruction…"></label><label>Outcome<select data-investigation-outcome><option value="">All outcomes</option><option value="success">Success</option><option value="wrong_stop">Wrong stop</option><option value="step_limit">Step limit</option><option value="failure">Failure</option><option value="unscored">Unscored</option><option value="no ground truth">No reference</option></select></label><button type="button" data-investigation-clear hidden>Clear filters</button></div>
      <div class="investigation-categories" role="group" aria-label="Investigation categories"></div>
      <p class="investigation-guidance">Signals are review prompts, not confirmed causes. Category counts cover the full run and can overlap.</p>
      <div class="investigation-results-heading"><div class="investigation-status" role="status" aria-live="polite"></div><div class="investigation-pagination" role="group" aria-label="Queue pagination"><button type="button" data-investigation-prev>Previous</button><span data-investigation-page></span><button type="button" data-investigation-next>Next</button></div></div>
      <div class="investigation-results"></div>
      <div class="investigation-pagination investigation-pagination-bottom" role="group" aria-label="Queue pagination"><button type="button" data-investigation-prev>Previous</button><span data-investigation-page></span><button type="button" data-investigation-next>Next</button></div>`;
    this.status = page.querySelector('.investigation-status');
    this.results = page.querySelector('.investigation-results');
    page.querySelector('[data-investigation-search]').addEventListener('input', e => {
      this.filters.q = e.target.value; this.filters.offset = 0;
      this.invalidateQueue(); clearTimeout(this.searchTimer);
      this.searchTimer = setTimeout(() => this.loadQueue(), 220);
    });
    page.querySelector('[data-investigation-outcome]').addEventListener('change', e => {
      this.filters.outcome = e.target.value; this.filters.offset = 0; this.loadQueue();
    });
    page.addEventListener('click', e => this.queueClick(e));
    card.addEventListener('click', e => this.cardClick(e));
  }

  invalidateQueue() { ++this.queueRequest; this.queueAbort?.abort(); }

  setRun(run) {
    if (run === this.run) return;
    this.run = run; this.invalidateQueue(); ++this.caseRequest; this.caseAbort?.abort();
    clearTimeout(this.searchTimer);
    this.caseKey = null; this.caseData = null; this.casePromise = null;
    this.filters.offset = 0; this.filters.kind = '';
    this.page.querySelector('.investigation-categories').innerHTML = '';
    this.results.innerHTML = ''; this.status.textContent = '';
    this.card.hidden = true;
  }

  async request(path, signal) {
    const response = await fetch(path, { signal });
    if (!response.ok) {
      let message = `Request failed (${response.status})`;
      try { message = (await response.json()).error || message; } catch (_) { /* non-JSON error */ }
      throw new Error(message);
    }
    return response.json();
  }

  async loadQueue() {
    if (!this.run) return;
    this.invalidateQueue(); const request = this.queueRequest, run = this.run;
    this.queueAbort = new AbortController();
    this.status.textContent = 'Loading investigation queue…';
    this.results.setAttribute('aria-busy', 'true');
    this.results.innerHTML = '';
    this.setPagination(null);
    this.page.querySelector('[data-investigation-clear]').hidden = !this.filters.kind && !this.filters.q && !this.filters.outcome;
    try {
      const params = new URLSearchParams({ run, ...this.filters });
      const data = await this.request(`/api/investigations?${params}`, this.queueAbort.signal);
      if (request !== this.queueRequest || run !== this.run) return;
      this.queueData = data;
      const categories = [{ id: '', label: 'All signals', count: data.flagged_episodes, description: 'All episodes with at least one investigation signal.' }, ...(data.categories || [])];
      const categoryGroup = this.page.querySelector('.investigation-categories');
      // Replacing the chips must not lose keyboard focus or steal it from another control.
      const focusedKind = categoryGroup.contains(document.activeElement) ? document.activeElement.dataset.investigationKind : undefined;
      categoryGroup.innerHTML = categories.map(c => `<button type="button" data-investigation-kind="${esc(c.id)}" data-empty="${c.count === 0}" aria-pressed="${c.id === this.filters.kind}" title="${esc(c.description)}"><span>${esc(signalLabel(c))}</span><b>${number(c.count)}</b></button>`).join('');
      if (focusedKind !== undefined) categoryGroup.querySelector(`[data-investigation-kind="${CSS.escape(focusedKind)}"]`)?.focus({ preventScroll: true });
      this.status.textContent = `${number(data.total_matches)} matching cases · ${number(data.flagged_episodes)} of ${number(data.total_episodes)} episodes flagged`;
      this.results.innerHTML = data.cases?.length ? `<div class="investigation-table-wrap"><table class="investigation-table" role="table"><caption>Ordered by review priority, not a probability of failure. Select an episode to inspect it.</caption><thead role="rowgroup"><tr role="row"><th scope="col" role="columnheader">Episode</th><th scope="col" role="columnheader">Signals to inspect</th><th scope="col" role="columnheader">Outcome</th><th scope="col" role="columnheader" title="Final navigation error in meters">NE (m)</th><th scope="col" role="columnheader" title="Success weighted by path length">SPL</th></tr></thead><tbody role="rowgroup">${data.cases.map(c => `<tr role="row"><td role="cell"><button type="button" data-investigation-open="${esc(c.instr_id)}" aria-label="Investigate episode ${esc(c.instr_id)}">${esc(c.instr_id)} <span aria-hidden="true">→</span></button><span class="investigation-scan">${esc(c.scan)}</span></td><td role="cell"><div class="investigation-flags">${(c.flags || []).map(f => `<span title="${esc(f.label + ': ' + f.summary)}">${esc(signalLabel(f))}</span>`).join('')}</div><p>${esc(c.flags?.[0]?.summary || '')}</p></td><td role="cell"><span class="investigation-outcome ${c.outcome === 'success' ? 'success' : ['wrong_stop', 'step_limit', 'failure'].includes(c.outcome) ? 'failure' : 'unscored'}">${esc(outcomeLabel(c.outcome))}</span></td><td role="cell" data-label="NE">${metric(c.metrics?.ne, ' m')}</td><td role="cell" data-label="SPL">${metric(c.metrics?.spl, '%', 100)}</td></tr>`).join('')}</tbody></table></div>` : '<div class="investigation-empty"><strong>No matching cases</strong><p>Try another category or clear the search. Unflagged episodes remain available in Episode view.</p></div>';
      this.setPagination(data);
    } catch (error) {
      if (error.name === 'AbortError' || request !== this.queueRequest) return;
      this.status.textContent = `Could not load the investigation queue: ${error.message}`;
      this.results.innerHTML = '<button type="button" data-investigation-refresh>Retry loading queue</button>';
    } finally {
      if (request === this.queueRequest) this.results.removeAttribute('aria-busy');
    }
  }

  setPagination(data) {
    const offset = data?.offset ?? this.filters.offset, limit = data?.limit ?? this.filters.limit;
    const total = data?.total_matches || 0;
    this.page.querySelectorAll('[data-investigation-prev]').forEach(button => { button.disabled = !data || offset <= 0; });
    this.page.querySelectorAll('[data-investigation-next]').forEach(button => { button.disabled = !data || offset + limit >= total; });
    this.page.querySelectorAll('[data-investigation-page]').forEach(label => { label.textContent = data && total ? `${number(offset + 1)}–${number(Math.min(offset + limit, total))} of ${number(total)}` : ''; });
  }

  async queueClick(event) {
    const button = event.target.closest('button');
    if (!button) return;
    if (button.hasAttribute('data-investigation-kind')) {
      this.filters.kind = button.dataset.investigationKind; this.filters.offset = 0; return this.loadQueue();
    }
    if (button.hasAttribute('data-investigation-refresh')) return this.loadQueue();
    if (button.hasAttribute('data-investigation-clear')) {
      clearTimeout(this.searchTimer);
      Object.assign(this.filters, { kind: '', q: '', outcome: '', offset: 0 });
      this.page.querySelector('[data-investigation-search]').value = '';
      this.page.querySelector('[data-investigation-outcome]').value = '';
      this.page.querySelector('[data-investigation-search]').focus();
      return this.loadQueue();
    }
    if (button.hasAttribute('data-investigation-prev') || button.hasAttribute('data-investigation-next')) {
      this.filters.offset = Math.max(0, this.filters.offset + (button.hasAttribute('data-investigation-prev') ? -1 : 1) * this.filters.limit);
      return this.loadQueue();
    }
    if (button.hasAttribute('data-investigation-open')) {
      button.disabled = true;
      try { await this.onOpen(button.dataset.investigationOpen); }
      catch (error) { this.status.textContent = `Could not open this episode: ${error.message}`; }
      finally { button.disabled = false; }
    }
  }

  async loadCase(run, id, force = false) {
    const key = `${run}\n${id}`;
    if (!force && key === this.caseKey) return this.casePromise || this.caseData;
    this.caseAbort?.abort(); this.caseAbort = new AbortController();
    const request = ++this.caseRequest;
    this.caseKey = key; this.caseData = null; this.findingId = null;
    this.card.hidden = false;
    this.card.innerHTML = '<h2>Investigation</h2><p role="status">Loading evidence for this episode…</p>';
    const promise = this.request(`/api/investigation?${new URLSearchParams({ run, id })}`, this.caseAbort.signal).then(data => {
      if (request !== this.caseRequest || run !== this.run) return null;
      this.caseData = data; this.findingId = data.findings?.[0]?.id || null;
      this.renderCase(); return data;
    }).catch(error => {
      if (error.name === 'AbortError' || request !== this.caseRequest) return null;
      this.card.innerHTML = `<h2>Investigation</h2><p role="status">Could not load evidence: ${esc(error.message)}</p><button type="button" data-case-retry>Retry evidence</button>`;
      return null;
    }).finally(() => { if (request === this.caseRequest) this.casePromise = null; });
    this.casePromise = promise;
    return promise;
  }

  setFrame(frame) {
    this.frame = frame;
    const label = this.card.querySelector('[data-case-frame]');
    if (label) label.textContent = `Viewing frame ${frame} · findings describe the full recorded episode`;
    this.card.querySelectorAll('[data-finding-frame]').forEach(button => {
      const atFrame = Number(button.dataset.findingFrame) === frame;
      button.classList.toggle('at-frame', atFrame);
      button.setAttribute('aria-current', atFrame ? 'step' : 'false');
    });
  }

  focusFirst() {
    const finding = this.caseData?.findings?.find(f => Number.isInteger(f.frame));
    if (finding) this.selectFinding(finding.id, true);
    this.card.querySelector('h2')?.focus({ preventScroll: true });
    this.card.scrollIntoView({ block: 'nearest' });
  }

  selectFinding(id, seek) {
    const finding = this.caseData?.findings?.find(f => f.id === id);
    if (!finding) return;
    this.findingId = id; this.renderCase();
    if (seek && Number.isInteger(finding.frame)) this.onSeek(finding.frame);
    this.card.querySelector(`[data-case-finding="${CSS.escape(id)}"]`)?.focus({ preventScroll: true });
  }

  renderCase() {
    const data = this.caseData;
    if (!data) return;
    const findings = data.findings || [], selected = findings.find(f => f.id === this.findingId) || findings[0];
    const list = (title, values, className = '') => values?.length ? `<section class="case-section ${className}"><h3>${title}</h3><ul>${values.map(item => `<li>${esc(typeof item === 'string' ? item : item.text || item.description || valueText(item))}</li>`).join('')}</ul></section>` : '';
    const images = data.provenance?.images || data.provenance?.image_provenance || {};
    const imageText = typeof images === 'string' ? images : images.description || images.note || images.label || 'See source image references in the case export.';
    const id = data.episode.instr_id;
    const exportUrl = `/api/investigation?${new URLSearchParams({ run: this.run, id, download: '1' })}`;
    let body = `<div class="case-heading"><h2 tabindex="-1">Investigation</h2><button type="button" data-case-queue>Review queue</button></div><p class="case-frame" data-case-frame></p>`;
    body += `<div class="case-actions"><button type="button" data-case-copy>Copy replay link</button><a class="case-download" href="${esc(exportUrl)}" download>Download case JSON</a></div><p class="case-action-status" role="status" aria-live="polite"></p><input class="case-link-fallback" aria-label="Replay link" readonly hidden>`;
    if (findings.length) {
      body += `<div class="case-findings" role="group" aria-label="Evidence findings">${findings.map(f => `<button type="button" data-case-finding="${esc(f.id)}" ${Number.isInteger(f.frame) ? `data-finding-frame="${f.frame}"` : ''} aria-pressed="${f.id === selected.id}" title="${Number.isInteger(f.frame) ? `Inspect frame ${f.frame}` : 'No exact sampled frame available'}">${esc(f.title)}${Number.isInteger(f.frame) ? `<span>Frame ${f.frame}</span>` : ''}</button>`).join('')}</div>`;
      body += `<div class="case-finding-detail"><p class="case-summary">${esc(selected.summary)}</p><p class="case-caution">Possible explanations are untested. A reference disagreement alone does not establish a cause.</p>`;
      if (!Number.isInteger(selected.frame)) body += '<p class="case-frame-gap">No exact replay frame is available for this finding; inspect the recorded evidence below.</p>';
      body += `<section class="case-section"><h3>Observed facts</h3><dl class="case-evidence">${(selected.evidence || []).map(e => `<div${e.value !== null && typeof e.value === 'object' ? ' class="case-evidence-structured"' : ''}><dt>${esc(e.label)}</dt><dd>${esc(evidenceText(e))}</dd><dd class="case-evidence-source"${e.path ? ` title="${esc('Evidence field in exported JSON: ' + e.path)}"` : ''}>${esc(sourceLabel(e.source))}</dd></div>`).join('')}</dl></section>`;
      body += list('Possible explanations · untested', selected.hypotheses) + list('Missing evidence', selected.missing_evidence) + list('Suggested next tests', selected.next_tests, 'case-next-tests') + '</div>';
    } else body += '<p class="case-summary">No investigation signals were detected by the current rules.</p><p class="case-caution">This is not a verification that every decision was correct. You can still inspect the trace and add review notes below.</p>';
    body += `<details class="case-provenance"><summary>Image evidence and detection rules</summary><p>${esc(imageText)}</p><p>Signals use fixed, transparent thresholds. They are review prompts and may include successful episodes.</p>${data.thresholds ? `<pre>${esc(JSON.stringify(data.thresholds, null, 2))}</pre>` : ''}</details>`;
    this.card.innerHTML = body;
    this.setFrame(this.frame);
  }

  async cardClick(event) {
    const button = event.target.closest('button');
    if (!button) return;
    if (button.hasAttribute('data-case-finding')) return this.selectFinding(button.dataset.caseFinding, true);
    if (button.hasAttribute('data-case-queue')) return this.onQueue();
    if (button.hasAttribute('data-case-retry')) {
      const [run, id] = (this.caseKey || '').split('\n');
      if (run && id) return this.loadCase(run, id, true);
    }
    if (button.hasAttribute('data-case-copy') && this.caseData) {
      const replay = new URL(safeLink(this.caseData.replay_url), location.href);
      const params = new URLSearchParams(replay.hash.slice(1));
      params.set('f', String(this.frame)); params.delete('view');
      replay.hash = params.toString();
      const url = replay.href;
      const status = this.card.querySelector('.case-action-status');
      try { await navigator.clipboard.writeText(url); status.textContent = 'Replay link copied.'; }
      catch (_) {
        const input = this.card.querySelector('.case-link-fallback');
        input.hidden = false; input.value = url; input.focus(); input.select();
        status.textContent = 'Copy the selected replay link.';
      }
    }
  }
}
