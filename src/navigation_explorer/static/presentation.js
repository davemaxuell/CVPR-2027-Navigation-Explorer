// Read-only summaries for the viewer. These helpers never modify episodes, scores, or evaluation data.
const number = value => value != null && Number.isFinite(value);
const metres = value => number(value) ? `${value.toFixed(2)} m` : 'unavailable';

// A radius is optional and belongs to the supplied evaluation, not to the viewer.
export function successRadius(ep, fallback = null) {
  const evaluation = ep?.evaluation;
  // Unknown distance semantics cannot support a radius comparison or a success-area overlay.
  if (evaluation?.distance_kind === 'unknown') return null;
  const value = evaluation && Object.hasOwn(evaluation, 'success_radius_m') ? evaluation.success_radius_m : fallback;
  return number(value) && value > 0 ? value : null;
}

export function datasetTitle(run) {
  if (typeof run?.dataset === 'string' && run.dataset.trim()) return run.dataset;
  if (run?.dataset?.name) return run.dataset.name;
  return 'Custom dataset';
}

export function comparableRun(run, other) {
  if (!run || !other || run.name === other.name) return false;
  if (Array.isArray(run.comparable_runs)) return run.comparable_runs.includes(other.name);
  // Legacy split names alone cannot identify the same scene coordinates or episode identities.
  return !!run.dataset && !!run.coordinate_system && run.dataset === other.dataset &&
    run.coordinate_system === other.coordinate_system && run.env === other.env && run.split === other.split;
}

export function finalActionLabel(ep) {
  const frame = ep.frames?.at(-1), decision = ep.decisions?.[frame?.decision];
  return decision?.action === 'stop' ? 'STOP' : 'END';
}

export function distanceLabel(ep) {
  const kind = ep?.evaluation?.distance_kind;
  return kind === 'geodesic' ? 'Geodesic distance to goal' : kind === 'euclidean' ? 'Straight-line distance to goal' : 'Recorded distance to goal';
}

// `unit` names a frame: a graph hop for discrete runs, a track point for continuous ones.
export function subtaskProgress(reached, index, frame, lastFrame, current, unit = 'hop') {
  if (!reached) return { state: index === current ? 'current' : 'pending', label: 'Not evaluated · no reference' };
  const hop = reached[index];
  if (hop != null && hop <= frame) return { state: 'done', label: `Completed · ${unit} ${hop}` };
  if (frame === lastFrame) return { state: 'missed', label: 'Not completed in this episode' };
  return index === current ? { state: 'current', label: 'Active at this frame' } : { state: 'pending', label: 'Pending at this frame' };
}

export function firstDivergence(path, otherPath) {
  if (!path?.length || !otherPath?.length) return null;
  let index = 0;
  while (index < Math.min(path.length, otherPath.length) && path[index] === otherPath[index]) index++;
  if (index === path.length && index === otherPath.length) return null;
  return { index, frame: Math.min(index, path.length - 1), shared: index ? path[index - 1] : null,
    model: path[index] || null, comparison: otherPath[index] || null };
}

export function episodeEvents(ep, errorMargin) {
  const events = [], d = ep.diag || {}, frames = ep.frames || [];
  const add = (id, label, frame, detail) => {
    if (Number.isInteger(frame) && frame >= 0 && frame < frames.length) events.push({ id, label, frame, detail });
  };
  if (ep.goal) {
    const sampled = !!ep.ce;
    add('closest', sampled ? 'Closest sampled approach' : 'Closest approach', d.closest_hop,
      `${sampled ? 'Sampled replay' : 'Route'} distance to goal: ${metres(d.closest_dist)}. ` +
      (d.goal_visit_unsampled ? 'The recorded full trajectory entered the success area, but that visit is absent from these replay frames. ' : '') +
      (number(errorMargin) ? `Declared success radius: ${errorMargin} m.` :
        ep.evaluation?.success_criterion === 'visible_goal_viewpoints' ? 'Success uses annotated visible-goal viewpoints.' :
          ep.evaluation?.distance_kind === 'unknown' ? 'Distance semantics are unspecified; no success-area comparison is shown.' : 'Success criterion not supplied.'));
    add('departure', 'Left reference route', d.first_off_gt_hop, 'First recorded viewpoint outside the annotated reference path. An alternate route can still succeed.');
    add('lost', 'Left reference area', d.lost_hop, number(errorMargin) ? `First viewpoint more than ${errorMargin} m from every reference-path node.` : 'First recorded departure from the declared reference corridor.');
  }
  const revisit = frames.findIndex(f => f.visits >= 2);
  add('revisit', 'First revisit', revisit, `First return to an already visited viewpoint (${frames[revisit]?.visits || 2} visits).`);
  const final = frames.length - 1, stopped = ep.decisions?.[frames[final]?.decision]?.action === 'stop';
  add('final', stopped ? 'Final stop' : 'Episode end', final, ep.goal ? `Recorded final navigation error: ${metres(ep.metrics?.ne)}.` : 'No reference goal was supplied for this episode.');
  return events;
}

export function episodeSummary(ep, errorMargin, unit = 'hop') {
  if (!ep.goal) return 'No reference goal was supplied for this episode. The recorded route is available for inspection.';
  const d = ep.diag || {}, end = ep.metrics?.ne;
  const stopped = finalActionLabel(ep) === 'STOP';
  const radius = number(errorMargin) ? ` Declared success radius: ${errorMargin} m.` : '';
  if (ep.metrics?.sr == null && !['success', 'wrong_stop', 'step_limit', 'failure'].includes(ep.outcome)) return `Outcome not evaluated in this recording.${number(end) ? ` Final navigation error: ${metres(end)}.` : ''}${radius}`;
  if (ep.outcome === 'success') return `Recorded successful outcome${number(end) ? `: ${metres(end)} from the goal` : ''}.${radius}`;
  const ending = ep.outcome === 'step_limit' ? 'The decision limit was reached' : stopped ? 'The agent stopped' : 'The recorded episode ended';
  if (d.goal_visit_unsampled) {
    return `The recorded full trajectory entered the success area, but that visit is absent from the sampled replay. ` +
      `Closest sampled approach: ${metres(d.closest_dist)}${d.closest_hop != null ? ` at ${unit} ${d.closest_hop}` : ''}. ` +
      `${ending} at ${metres(end)} from the goal.`;
  }
  if (number(errorMargin) && number(d.closest_dist) && d.closest_dist < errorMargin) {
    return `Within ${errorMargin} m at ${unit} ${d.closest_hop} (${metres(d.closest_dist)}). ${ending} at ${metres(end)} from the goal.`;
  }
  const closest = number(d.closest_dist) ? `Closest ${ep.ce ? 'sampled ' : ''}approach: ${metres(d.closest_dist)}${d.closest_hop != null ? ` at ${unit} ${d.closest_hop}` : ''}. ` : '';
  return `${closest}${ending}${number(end) ? ` at ${metres(end)} from the goal` : ''}.${radius}`;
}

export function metricDifference(value, other, digits = 1, unit = '') {
  if (!number(value) || !number(other)) return '—';
  const delta = value - other;
  return `${delta > 0 ? '+' : ''}${delta.toFixed(digits)}${unit}`;
}
