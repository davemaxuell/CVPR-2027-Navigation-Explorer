// Presentation-only panel resizing. Existing renderers observe their own container sizes.
const $ = selector => document.querySelector(selector);
const layout = $('#layout'), stage = $('.stage'), views = $('.views');
const sidebar = $('#sidebar'), details = $('.details'), front = $('.view.front');
const resetButton = $('#resetLayout');
const STORAGE_KEY = 'vlnviz-layout-v1';
const clamp = (value, min, max) => Math.max(min, Math.min(Math.max(min, max), value));
const desktop = matchMedia('(min-width: 1101px)'), wide = matchMedia('(min-width: 761px)');
const profile = () => desktop.matches ? 'desktop' : wide.matches ? 'tablet' : 'mobile';
const panels = new Map(), handles = [];
let saved = readSaved(), active = null, scheduled = 0;
const preferences = () => saved[profile()] ||= { heights: {} };

function readSaved() {
  const clean = {};
  try {
    const data = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (data?.version !== 1) return clean;
    for (const name of ['desktop', 'tablet', 'mobile']) {
      const source = data[name];
      if (!source || typeof source !== 'object') continue;
      const entry = { heights: {} };
      for (const key of ['sidebar', 'details', 'frontRatio']) {
        if (Number.isFinite(source[key]) && source[key] > 0) entry[key] = source[key];
      }
      for (const [key, height] of Object.entries(source.heights || {})) {
        if (Number.isFinite(height) && height > 0) entry.heights[key] = clamp(height, 80, 1800);
      }
      clean[name] = entry;
    }
  } catch (_) { /* The layout also works when browser storage is unavailable. */ }
  return clean;
}

function persist() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, ...saved })); } catch (_) { /* storage blocked */ }
}

const help = document.createElement('p');
help.id = 'resizeHelp'; help.className = 'resize-help';
help.textContent = 'Drag a grip to resize. When focused, use arrow keys; hold Shift for larger steps. Double-click or press Enter to restore that size. Press Escape to cancel a drag.';
document.body.append(help);
const notice = document.createElement('span');
notice.className = 'resize-help'; notice.setAttribute('aria-live', 'polite');
document.body.append(notice);
const readout = document.createElement('div');
readout.className = 'resize-readout'; readout.hidden = true; readout.setAttribute('aria-hidden', 'true');
document.body.append(readout);
layout.classList.add('resize-enabled');

function setProperty(element, key, value) {
  if (element.style.getPropertyValue(key) !== value) element.style.setProperty(key, value);
}

function availableWidth() {
  const css = getComputedStyle(layout);
  return layout.clientWidth - parseFloat(css.paddingLeft) - parseFloat(css.paddingRight);
}

function columnBounds(which) {
  const gap = parseFloat(getComputedStyle(layout).columnGap);
  const hasSidebar = desktop.matches && !layout.classList.contains('noside');
  const space = availableWidth() - gap * (hasSidebar ? 2 : 1);
  const other = which === 'sidebar' ? details.getBoundingClientRect().width : hasSidebar ? sidebar.getBoundingClientRect().width : 0;
  return [which === 'sidebar' ? 200 : desktop.matches ? 260 : 240, space - other - 360];
}

function applyWidths() {
  const settings = preferences(), root = getComputedStyle(document.documentElement);
  const gap = parseFloat(getComputedStyle(layout).columnGap) || 12;
  const hasSidebar = desktop.matches && !layout.classList.contains('noside');
  const space = availableWidth() - gap * (hasSidebar ? 2 : 1);
  const side = hasSidebar ? clamp(settings.sidebar ?? parseFloat(root.getPropertyValue('--sidebar-width')), 200, space - 620) : 0;
  const right = clamp(settings.details ?? (desktop.matches ? parseFloat(root.getPropertyValue('--details-width')) : 310), desktop.matches ? 260 : 240, space - side - 360);
  setProperty(layout, '--sidebar-width', `${side}px`);
  setProperty(layout, '--details-width', `${right}px`);
  if (wide.matches) {
    const [min, max] = frontBounds();
    const width = settings.frontRatio ? settings.frontRatio * cameraSpace() : (parseFloat(root.getPropertyValue('--view-h')) - 32) * 4 / 3;
    setProperty(views, '--front-width', `${clamp(width, min, max)}px`);
  } else views.style.removeProperty('--front-width');
}

function cameraSpace() { return views.clientWidth - (parseFloat(getComputedStyle(views).columnGap) || 10); }
function frontBounds() {
  const space = cameraSpace();
  return [Math.min(150, space * .4), space - Math.min(190, space * .45)];
}

function apply() {
  applyWidths();
  const heights = preferences().heights;
  for (const [key, panel] of panels) {
    const [min, max] = panel.bounds();
    setProperty(panel.element, 'min-height', `${min}px`);
    if (heights[key]) {
      setProperty(panel.element, 'height', `${clamp(heights[key], min, max)}px`);
      panel.element.dataset.sized = '';
    } else {
      panel.element.style.removeProperty('height');
      delete panel.element.dataset.sized;
    }
  }
  positionHandles();
}

function schedule() {
  if (!scheduled) scheduled = requestAnimationFrame(time => {
    scheduled = 0;
    updateDrag();
    apply();
    showReadout();
    if (scrollDrag(time)) schedule();
  });
}

function positionHandles() {
  const box = layout.getBoundingClientRect(), side = sidebar.getBoundingClientRect(), right = details.getBoundingClientRect();
  const stageBox = stage.getBoundingClientRect(), gap = parseFloat(getComputedStyle(layout).columnGap) || 12;
  const top = stageBox.top - box.top;
  for (const [handle, center] of [[sideHandle, side.right - box.left + gap / 2], [detailsHandle, right.left - box.left - gap / 2]]) {
    setProperty(handle, 'left', `${center - handle.offsetWidth / 2}px`);
    setProperty(handle, 'top', `${top}px`);
    setProperty(handle, 'height', `${stage.clientHeight}px`);
  }
  const cameraGap = parseFloat(getComputedStyle(views).columnGap) || 10;
  setProperty(frontHandle, 'left', `${front.getBoundingClientRect().width + (cameraGap - frontHandle.offsetWidth) / 2}px`);
  for (const { element, config } of handles) {
    const [min, max] = config.bounds();
    const value = Math.round(config.read());
    element.setAttribute('aria-valuemin', Math.round(min));
    element.setAttribute('aria-valuemax', Math.round(Math.max(min, max)));
    element.setAttribute('aria-valuenow', value);
    element.setAttribute('aria-valuetext', `${value} pixels`);
  }
}

function makeHandle(parent, key, label, axis, config, className = '') {
  const element = document.createElement('div');
  element.className = `resize-handle resize-${axis === 'x' ? 'width' : 'height'} ${className}`;
  element.dataset.resize = key; element.tabIndex = 0;
  element.setAttribute('role', 'separator');
  element.setAttribute('aria-label', `Resize ${label}`);
  element.setAttribute('aria-orientation', axis === 'x' ? 'vertical' : 'horizontal');
  element.setAttribute('aria-describedby', help.id);
  element.title = `Drag to resize ${label}. Double-click to reset.`;
  parent.append(element);
  handles.push({ element, config });
  const write = value => {
    const [min, max] = config.bounds();
    config.write(clamp(value, min, max));
  };
  element.addEventListener('pointerdown', event => {
    if (event.button !== 0 || !event.isPrimary || active) return;
    event.preventDefault(); event.stopPropagation();
    element.focus({ preventScroll: true });
    element.setPointerCapture(event.pointerId);
    active = { element, pointerId: event.pointerId, axis, origin: axis === 'x' ? event.clientX : event.clientY,
      x: event.clientX, y: event.clientY, size: config.read(), sign: config.sign || 1, write, config,
      before: structuredClone(saved), scrollers: scrollParents(element), moved: false, time: 0 };
    element.classList.add('is-resizing');
    document.documentElement.dataset.layoutResizing = axis;
    showReadout();
  });
  element.addEventListener('pointermove', event => {
    if (active?.element !== element || active.pointerId !== event.pointerId) return;
    active.x = event.clientX; active.y = event.clientY;
    active.moved ||= Math.abs((axis === 'x' ? event.clientX : event.clientY) - active.origin) > 2;
    schedule();
  });
  element.addEventListener('pointerup', event => { if (active?.pointerId === event.pointerId) finishDrag(false); });
  element.addEventListener('pointercancel', () => { if (active?.element === element) finishDrag(true); });
  element.addEventListener('lostpointercapture', () => { if (active?.element === element) finishDrag(true); });
  element.addEventListener('dblclick', event => { event.preventDefault(); config.reset(); apply(); persist(); });
  element.addEventListener('keydown', event => {
    const decrease = axis === 'x' ? 'ArrowLeft' : 'ArrowUp', increase = axis === 'x' ? 'ArrowRight' : 'ArrowDown';
    if (![decrease, increase, 'Home', 'End', 'Enter', ' '].includes(event.key)) return;
    event.preventDefault(); event.stopPropagation();
    if (event.key === 'Enter') { config.reset(); apply(); }
    else if (event.key !== ' ') {
      const [min, max] = config.bounds();
      const delta = (event.shiftKey ? 32 : 8) * (config.sign || 1);
      write(event.key === 'Home' ? min : event.key === 'End' ? max : config.read() + (event.key === increase ? delta : -delta));
      apply();
    }
    notice.textContent = `${label}: ${Math.round(config.read())} pixels.`;
    persist();
  });
  return element;
}

function scrollParents(element) {
  const result = [];
  for (let parent = element.parentElement; parent && parent !== document.body; parent = parent.parentElement) {
    if (/auto|scroll/.test(getComputedStyle(parent).overflowY)) result.push({ element: parent, start: parent.scrollTop });
  }
  if (getComputedStyle(document.body).overflowY !== 'hidden') {
    result.push({ element: document.scrollingElement, start: document.scrollingElement.scrollTop });
  }
  return result;
}

function updateDrag() {
  if (!active?.moved) return;
  const delta = (active.axis === 'x' ? active.x : active.y) - active.origin;
  const scroll = active.axis === 'y' ? active.scrollers.reduce((sum, item) => sum + item.element.scrollTop - item.start, 0) : 0;
  active.write(active.size + (delta + scroll) * active.sign);
}

// Keep long panels within reach when their lower edge moves beyond the visible pane.
function scrollDrag(time) {
  if (!active?.moved || active.axis !== 'y') return false;
  const elapsed = active.time ? Math.min(32, time - active.time) : 16;
  active.time = time;
  const [min, max] = active.config.bounds(), size = active.config.read();
  for (const { element } of active.scrollers) {
    const box = element === document.scrollingElement ? { top: 0, bottom: innerHeight } : element.getBoundingClientRect();
    const top = Math.max(0, box.top), bottom = Math.min(innerHeight, box.bottom);
    const direction = active.y > bottom - 32 ? 1 : active.y < top + 32 ? -1 : 0;
    if (!direction || (direction > 0 ? size >= max : size <= min)) continue;
    const before = element.scrollTop;
    element.scrollTop += direction * elapsed * .65;
    if (element.scrollTop !== before) return true;
  }
  return false;
}

function showReadout() {
  if (!active) return;
  readout.textContent = `${Math.round(active.config.read())} px ${active.axis === 'x' ? 'wide' : 'high'}`;
  readout.hidden = false;
  setProperty(readout, 'left', `${clamp(active.x + 16, 8, innerWidth - readout.offsetWidth - 8)}px`);
  const y = active.y + 18 + readout.offsetHeight > innerHeight - 8 ? active.y - readout.offsetHeight - 12 : active.y + 18;
  setProperty(readout, 'top', `${clamp(y, 8, innerHeight - readout.offsetHeight - 8)}px`);
}
function finishDrag(cancel) {
  if (!active) return;
  cancelAnimationFrame(scheduled); scheduled = 0;
  if (!cancel) updateDrag();
  const drag = active; active = null;
  if (cancel) saved = drag.before;
  apply();
  if (cancel) { for (const item of drag.scrollers) item.element.scrollTop = item.start; }
  else persist();
  drag.element.classList.remove('is-resizing');
  readout.hidden = true;
  delete document.documentElement.dataset.layoutResizing;
  if (drag.element.hasPointerCapture(drag.pointerId)) drag.element.releasePointerCapture(drag.pointerId);
}

const sideHandle = makeHandle(layout, 'sidebar-width', 'episode list width', 'x', {
  read: () => sidebar.getBoundingClientRect().width, bounds: () => columnBounds('sidebar'),
  write: value => { preferences().sidebar = value; }, reset: () => { delete preferences().sidebar; }
}, 'resize-sidebar');
const detailsHandle = makeHandle(layout, 'details-width', 'details panel width', 'x', {
  sign: -1, read: () => details.getBoundingClientRect().width, bounds: () => columnBounds('details'),
  write: value => { preferences().details = value; }, reset: () => { delete preferences().details; }
}, 'resize-details');
const frontHandle = makeHandle(views, 'camera-width', 'agent view and panorama widths', 'x', {
  read: () => front.getBoundingClientRect().width, bounds: frontBounds,
  write: value => { preferences().frontRatio = value / cameraSpace(); }, reset: () => { delete preferences().frontRatio; }
}, 'resize-front');

function outerHeight(element) {
  if (!element.getClientRects().length) return 0;
  const css = getComputedStyle(element);
  return element.getBoundingClientRect().height + parseFloat(css.marginTop) + parseFloat(css.marginBottom);
}
function verticalChrome(element) {
  const css = getComputedStyle(element);
  return ['paddingTop', 'paddingBottom', 'borderTopWidth', 'borderBottomWidth'].reduce((sum, key) => sum + parseFloat(css[key]), 0);
}
function minimumHeight(key, content, min) {
  if (key === 'map') return Math.ceil(verticalChrome(content) + outerHeight($('.map-heading')) + outerHeight($('.maptools')) + outerHeight($('.route-key')) + 180);
  if (key === 'front' || key === 'panorama') {
    return Math.ceil(verticalChrome(content) + outerHeight(content.querySelector('figcaption')) +
      (key === 'panorama' ? outerHeight(content.querySelector('.vlegend')) + 80 : 96));
  }
  if (key === 'playback') return Math.ceil(verticalChrome(content) + [...content.children].reduce((sum, child) => sum + outerHeight(child), 0));
  if (key === 'diagnostics' && !content.open) return 0;
  return min;
}

function heightPanel(key, selector, label, min, wrap = false) {
  const content = $(selector);
  let element = content;
  // Keep grips outside scrolling content and result cards that the app replaces at each frame.
  if (wrap) {
    element = document.createElement('div');
    element.className = 'resize-shell'; element.dataset.panel = key;
    content.before(element); element.append(content);
  } else element.classList.add('resizable-panel');
  const bounds = () => [minimumHeight(key, content, min), key === 'sidebar' && desktop.matches ? layout.clientHeight - 24 : 1800];
  panels.set(key, { element, bounds });
  makeHandle(element, `${key}-height`, `${label} height`, 'y', {
    read: () => element.getBoundingClientRect().height, bounds,
    write: value => { preferences().heights[key] = value; }, reset: () => { delete preferences().heights[key]; }
  });
}
heightPanel('sidebar', '#sidebar', 'episode list', 330);
heightPanel('map', '.mapbox', 'environment', 240);
heightPanel('front', '.view.front', 'agent view', 140);
heightPanel('panorama', '.view.pano', 'panorama', 120);
heightPanel('playback', '.playback', 'playback and timeline', 120, true);
heightPanel('instruction', '.instruction-card', 'navigation instruction', 100, true);
heightPanel('outcome', '#outcomeCard', 'current position and result', 120, true);
heightPanel('decision', '#decisionCard', 'current decision', 120, true);
heightPanel('comparison', '#comparisonCard', 'model comparison', 120, true);
heightPanel('diagnostics', '#diagWrap', 'diagnostics', 100, true);

resetButton.addEventListener('click', () => {
  finishDrag(true); saved = {}; apply();
  try { localStorage.removeItem(STORAGE_KEY); } catch (_) { /* storage blocked */ }
  notice.textContent = 'Default panel sizes restored.';
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && active) { event.preventDefault(); event.stopPropagation(); finishDrag(true); }
}, true);
window.addEventListener('blur', () => finishDrag(true));
window.addEventListener('resize', schedule);
for (const query of [desktop, wide]) query.addEventListener('change', () => { finishDrag(true); schedule(); });
new MutationObserver(schedule).observe(layout, { attributes: true, attributeFilter: ['class'] });
const observer = new ResizeObserver(schedule);
for (const element of [layout, stage, views, sidebar, details, ...[...panels.values()].map(panel => panel.element)]) observer.observe(element);
// Wrapped controls can change the safe minimum even when the outer panel has a saved size.
for (const element of document.querySelectorAll('.map-heading, .maptools, .route-key, .view figcaption, .vlegend, .playback > *')) observer.observe(element);
$('#diagWrap').addEventListener('toggle', schedule);
apply();
