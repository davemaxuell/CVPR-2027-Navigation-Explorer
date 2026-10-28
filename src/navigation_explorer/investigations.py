"""Deterministic review cues and portable evidence bundles for recorded navigation.

These rules select cases for inspection. They do not infer a model's reasoning or
establish why an episode failed. No model execution, filesystem access or mutation
of a recorded episode is performed here.
"""
import copy
import hashlib
import json
import math
from urllib.parse import quote, urlencode


SCHEMA_VERSION = 'navigation-investigation/v1'
# Keep this list aligned with CERun._apply_ce_metrics. Other CE fields are still
# computed from replay graphs, reference annotations, or decision/frame arrays.
CE_RECORDED_METRICS = frozenset({
    'ne', 'sr', 'osr', 'spl', 'ndtw', 'sdtw', 'tl', 'sp_len', 'collisions', 'low_level_steps',
})
TRACE_COUNT_METRICS = frozenset({'hops', 'moves', 'decisions'})
REFERENCE_DERIVED_METRICS = frozenset({
    'ne', 'euclid_ne', 'sr', 'osr', 'spl', 'ndtw', 'sdtw', 'cls', 'gt_len', 'gt_hops', 'sp_len',
})
THRESHOLDS = {
    'blocked_movement_m': 0.25,
    'waypoint_miss_m': 1.0,
    'oscillations': 2,
    'inefficient_success_spl_below': 0.5,
    'chosen_probability': 0.9,
}
CATEGORIES = {
    'missed_stop': ('Goal visited, unsuccessful finish', 80,
                    'Failed episodes whose recorded trajectory entered the evaluation success area.'),
    'blocked_movement': ('Movement without displacement', 70,
                         'Movement decisions ending less than 0.25 m from their starting position.'),
    'waypoint_miss': ('Waypoint execution mismatch', 60,
                     'At least one movement ended 1 m or more from its selected waypoint.'),
    'oscillation': ('Repeated A–B–A movement', 45,
                    'At least two immediate returns to the previous viewpoint; useful backtracking is possible.'),
    'inefficient_success': ('Inefficient success', 35,
                           'Successful episodes with SPL below 0.5; inspect route efficiency and recovery.'),
    'confident_reference_disagreement': ('High-score reference disagreement', 25,
                                        'The first reference disagreement chose an option with probability at least 90%; this is not a causal verdict.'),
    'evidence_gap': ('Evidence needs attention', 85,
                     'Missing aligned decision logs or a recorded goal visit absent from sampled replay frames.'),
}
PRIORITY_NOTE = ('Priority is a deterministic review-order heuristic: highest category weight + '
                 '5 per additional flag + 5 for an unsuccessful episode. It is not a probability, '
                 'a causal attribution, or an estimate of recoverable performance.')


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _at_least(value, minimum):
    return _number(value) and value >= minimum


def _metric_source(key, continuous):
    if continuous and key in CE_RECORDED_METRICS:
        return 'recorded_metric'
    if key in TRACE_COUNT_METRICS or key == 'tl':
        return 'trace_derived_metric'
    if key in REFERENCE_DERIVED_METRICS:
        return 'reference_derived_metric'
    return 'unspecified_metric'


def _run_name(info):
    return str(info.get('name') or info.get('run') or '')


def _link(info, instr_id, frame=None):
    fields = {'run': _run_name(info), 'id': str(instr_id)}
    if frame is not None:
        fields['f'] = frame
    return '/#' + urlencode(fields)


def _flags(row):
    flags = []

    def add(kind, summary):
        flags.append({'kind': kind, 'label': CATEGORIES[kind][0], 'summary': summary})

    if row.get('sr') == 0 and row.get('osr') == 1:
        add('missed_stop', 'The trajectory visited the success area, then finished unsuccessfully.')
    if _at_least(row.get('stuck'), 1):
        add('blocked_movement', f"{row['stuck']} movement decisions ended less than 0.25 m from their start.")
    if _at_least(row.get('ctrl_miss_max'), THRESHOLDS['waypoint_miss_m']):
        add('waypoint_miss', f"Worst waypoint execution miss: {row['ctrl_miss_max']:.2f} m.")
    if _at_least(row.get('oscillations'), THRESHOLDS['oscillations']):
        add('oscillation', f"{row['oscillations']} A–B–A returns occurred in the recorded path.")
    spl = row.get('spl')
    if row.get('sr') == 1 and _number(spl) and spl < THRESHOLDS['inefficient_success_spl_below']:
        add('inefficient_success', f"Successful finish with SPL {spl:.3f}; review the extra route length.")
    if row.get('first_nonbest') is not None and _at_least(row.get('first_nonbest_conf'), THRESHOLDS['chosen_probability']):
        add('confident_reference_disagreement',
            f"First reference disagreement at decision {row['first_nonbest']}; chosen-option probability {row['first_nonbest_conf']:.1%}.")
    gaps = []
    if row.get('steplog') is False:
        gaps.append('A fully aligned decision log is unavailable')
    if row.get('goal_visit_unsampled') is True or (row.get('osr') == 1 and row.get('near_goal_hops') == 0):
        gaps.append('The recorded goal visit is absent from sampled replay frames')
    if gaps:
        add('evidence_gap', '. '.join(gaps) + '.')
    return flags


def build_queue(run_info, rows):
    """Build a cheap full-run review index from cached episode summary rows.

    Pagination/filtering belongs to the API so these category counts always have
    the same full-run denominator. Successful episodes remain eligible.
    """
    cases, counts = [], dict.fromkeys(CATEGORIES, 0)
    for row in rows:
        flags = _flags(row)
        if not flags:
            continue
        for flag in flags:
            counts[flag['kind']] += 1
        priority = max(CATEGORIES[f['kind']][1] for f in flags) + 5 * (len(flags) - 1) + 5 * (row.get('sr') == 0)
        cases.append({
            'instr_id': str(row['instr_id']), 'scan': row.get('scan'),
            'instruction': row.get('instruction', ''), 'outcome': row.get('outcome'),
            'priority': priority, 'flags': flags,
            'metrics': {key: row.get(key) for key in ('sr', 'ne', 'spl', 'tl')},
            'replay_url': _link(run_info, row['instr_id']),
        })
    cases.sort(key=lambda case: (-case['priority'], case['instr_id']))
    return {
        'schema_version': SCHEMA_VERSION, 'run': copy.deepcopy(run_info),
        'total_episodes': len(rows), 'flagged_episodes': len(cases),
        'categories': [{'id': kind, 'label': item[0], 'count': counts[kind], 'description': item[2]}
                       for kind, item in CATEGORIES.items()],
        'cases': cases, 'thresholds': dict(THRESHOLDS), 'priority_note': PRIORITY_NOTE,
        'scope': 'Full selected run, including successful episodes; categories can overlap.',
    }


def _image_provenance(ep, continuous):
    supplied = (ep.get('ce') or {}).get('image_provenance') or ep.get('image_provenance')
    if isinstance(supplied, dict):
        return dict({'kind': 'unknown', 'label': 'Camera-image provenance',
                     'description': 'Only the supplied provenance fields are known.'}, **copy.deepcopy(supplied))
    if continuous:
        return {'kind': 'unknown', 'label': 'Camera-image provenance not supplied',
                'description': 'Cube views are associated with logged map-node poses; original versus regenerated status is unknown.',
                'exact_model_input': None}
    return {'kind': 'skybox_derived', 'label': 'Views reconstructed from Matterport skyboxes',
            'description': 'The viewer projects source skyboxes using the replay pose. These are not saved model-input tensors.',
            'exact_model_input': False}


def _images(ep, frames, decisions, provenance, continuous):
    """Image references only. Existence and exact model preprocessing are not inferred."""
    if (ep.get('capabilities') or {}).get('images') is False or provenance.get('kind') == 'unavailable':
        return []
    graph = ep.get('graph') or {}
    nodes = graph.get('nodes') or {}
    images, seen = [], set()
    # Prefer decision observations. Intermediate CE track points have no camera images.
    candidates = [(d.get('t', index), next((f for f, frame in enumerate(frames) if frame.get('decision') == d.get('t', index)), None), d.get('at'))
                  for index, d in enumerate(decisions) if d.get('source') != 'replay_placeholder']
    if ep.get('adapter') == 'recording':
        candidates += [(None, f, frame.get('vp')) for f, frame in enumerate(frames)]
    if not candidates:
        candidates = [(None, f, frame.get('vp')) for f, frame in enumerate(frames)]
    for decision, frame, vp in candidates:
        if not vp or vp in seen:
            continue
        sky = nodes.get(vp, {}).get('sky')
        if not sky and (continuous or provenance.get('kind') != 'skybox_derived'):
            continue
        seen.add(vp)
        urls = ([f'{sky}_{face}.jpg' for face in range(6)] if sky else
                [f'/api/skybox/{quote(str(ep.get("scan", "")), safe="")}/{quote(str(vp), safe="")}/{face}.jpg' for face in range(6)])
        image = {'viewpoint': vp, 'frame': frame, 'decision': decision, 'kind': 'cube_faces',
                 'urls': urls, 'provenance': copy.deepcopy(provenance), 'availability': 'not_checked'}
        if frame is not None:
            image.update({key: frames[frame][key] for key in ('heading', 'elevation') if key in frames[frame]})
        images.append(image)
    return images


def build_case(run_info, ep, note=None):
    """Return a self-contained recorded case with review cues and JSON Pointer evidence.

    All hypotheses are untested. Replay frame numbers are returned only when an
    observed event can be located in the available frames, never from a dense
    metric alone. Arrays remain ordered as in the supplied trace.
    """
    note = note or {}
    metrics, diag = ep.get('metrics') or {}, ep.get('diag') or {}
    frames, decisions = ep.get('frames') or [], ep.get('decisions') or []
    continuous = bool(ep.get('ce') or (ep.get('graph') or {}).get('ce') or run_info.get('env') == 'continuous')
    declared_sources = ep.get('metric_sources') or {}
    metric_source = lambda key: declared_sources.get(key, _metric_source(key, continuous))
    row = dict(metrics, **diag)
    row.update(instr_id=ep['instr_id'], outcome=ep.get('outcome'))
    flags = _flags(row)
    trace_keys = ('frames', 'decisions', 'path', 'gt_path', 'goal', 'goal_zone', 'dist_to_goal',
                  'subtasks', 'stop_probs', 'graph', 'ce', 'words', 'reference', 'start_heading', 'visits',
                  'evaluation', 'termination', 'adapter', 'recording_schema', 'capabilities',
                  'metric_sources', 'metric_notes', 'recording_metadata')
    trace = {key: copy.deepcopy(ep[key]) for key in trace_keys if key in ep}
    trace.setdefault('frames', [])
    trace.setdefault('decisions', [])
    provenance = {
        'images': _image_provenance(ep, continuous),
        'metrics': ('Continuous evaluation fields retain recorded metrics; reference-route geometry and replay counts are derived. '
                    'Per-field sources are listed in metric_sources.' if continuous else
                    'Metrics are derived from recorded trajectories and reference annotations. Per-field sources are listed in metric_sources.'),
        'metric_sources': {key: metric_source(key) for key in metrics},
        'scores': 'Action probabilities and attention are logged model outputs when supplied; neither explains the cause of a decision.',
        'reference': ('Reference-best follows the annotated route.' if ep.get('reference') == 'route' else
                      'Reference-best uses ground-truth goal information; it is an evaluation heuristic.'),
        'source_files': copy.deepcopy(run_info.get('sources', {})),
        'trace_sha256': hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest(),
        'trace_digest_scope': 'Canonical JSON of the exported trace; excludes image contents, model weights, notes, and derived findings.',
    }
    if ep.get('adapter') == 'recording':
        provenance['metrics'] = ('Evaluation values are supplied by the recording. Missing results stay unknown. '
                                 'Counts and sampled-polyline length are derived only where identified in metric_sources; '
                                 'sampled length can underestimate actual travel.')
        provenance['reference'] = ('No benchmark-specific reference-best action is inferred for portable recordings. '
                                   'Goal-area membership requires explicit distances, a distance convention and success radius.')
    tags = note.get('tags') or []
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split(';') if tag.strip()]
    result = {
        'schema_version': SCHEMA_VERSION, 'run': copy.deepcopy(run_info),
        'case_id': f'{_run_name(run_info)}/{ep["instr_id"]}',
        'episode': {'instr_id': str(ep['instr_id']), 'scan': ep.get('scan'), 'instruction': ep.get('instruction', ''),
                    'outcome': ep.get('outcome'), 'failure_mode': ep.get('failure_mode'),
                    'metrics': copy.deepcopy(metrics), 'diagnostics': copy.deepcopy(diag)},
        'findings': [], 'provenance': provenance,
        'review': {'tags': copy.deepcopy(tags), 'note': str(note.get('note') or '')},
        'replay_url': _link(run_info, ep['instr_id']), 'trace': trace,
        'images': _images(ep, frames, decisions, provenance['images'], continuous),
        'thresholds': dict(THRESHOLDS),
        'interpretation': 'Findings are deterministic inspection cues. Hypotheses are untested; no causal diagnosis or counterfactual outcome is asserted.',
    }
    first_frame = {}
    for f, frame in enumerate(frames):
        first_frame.setdefault(frame.get('decision'), f)
    def evidence(label, value, path, source='trace_derived'):
        return {'label': label, 'value': copy.deepcopy(value), 'source': source, 'path': path}

    def metric(key, label):
        return evidence(label, metrics.get(key), f'/episode/metrics/{key}', metric_source(key))

    def diagnostic(key, label):
        return evidence(label, diag.get(key), f'/episode/diagnostics/{key}')

    def finding(kind, summary, items, hypotheses, missing, tests, index=None, frame=None, severity='review'):
        decision = None
        if index is not None:
            decision = decisions[index].get('t', index)
            if frame is None:
                frame = first_frame.get(decision)
        result['findings'].append({
            'id': f'{kind}:{decision if decision is not None else "episode"}', 'kind': kind,
            'title': CATEGORIES[kind][0], 'summary': summary, 'severity': severity,
            'frame': frame, 'decision': decision, 'evidence': items,
            'hypotheses': hypotheses, 'hypothesis_status': 'untested',
            'missing_evidence': missing, 'next_tests': tests,
            'replay_url': _link(run_info, ep['instr_id'], frame),
        })

    flag_kinds = {flag['kind'] for flag in flags}
    if 'missed_stop' in flag_kinds:
        eligible = [(i, d) for i, d in enumerate(decisions)
                    if d.get('action') != 'stop' and not (d.get('model') or {}).get('forced')]
        chosen = next(((i, d) for i, d in eligible if (d.get('ref') or {}).get('type') == 'missed stop'), None)
        chosen = chosen or next(((i, d) for i, d in eligible if (d.get('ref') or {}).get('in_area') is True), None)
        items = [metric('sr', 'Final success'), metric('osr', 'Visited evaluation success area')]
        if 'ne' in metrics:
            items.append(metric('ne', 'Final goal distance (m)'))
        summary = 'The recorded trajectory visited the success area and finished unsuccessfully.'
        index, frame = None, None
        missing = ['Goal recognition and instruction-completion state are not logged; the stopping cause remains untested.']
        if chosen:
            index, dec = chosen
            ref = dec.get('ref') or {}
            summary += f" At decision {dec.get('t', index)}, the agent continued moving from inside the success area."
            if 'in_area' in ref:
                items.append(evidence('Decision was inside success area', ref['in_area'], f'/trace/decisions/{index}/ref/in_area', 'reference_derived'))
            if 'stop' in ref:
                items.append(evidence('STOP probability', ref['stop'], f'/trace/decisions/{index}/ref/stop', 'logged_score'))
            if 'conf' in ref:
                items.append(evidence('Chosen-option probability', ref['conf'], f'/trace/decisions/{index}/ref/conf', 'logged_score'))
            tests = ['Inspect the observation and active instruction at this decision.',
                     'After verifying an unforced replay, force STOP at this decision and record the intervention settings and result.']
        else:
            goal_zone = set(ep.get('goal_zone') or [])
            frame = next((f for f, item in enumerate(frames) if item.get('vp') in goal_zone), None)
            if frame is None:
                missing.append('No sampled replay frame locates the recorded goal visit; no decision index is inferred from oracle success.')
            else:
                items.append(evidence('Sampled position in success area', frames[frame]['vp'], f'/trace/frames/{frame}/vp', 'reference_derived'))
                missing.append('No logged, unforced movement decision inside the success area was identified; the visit can occur between decisions.')
            tests = ['Inspect the original dense trajectory and decision boundaries to locate the goal visit before selecting an intervention.']
        finding('missed_stop', summary, items,
                ['Stopping scores may fail to reflect instruction completion.',
                 'A goal-area visit may occur between decision points, without an opportunity for the high-level planner to stop.'],
                missing, tests, index=index, frame=frame, severity='high')

    if 'blocked_movement' in flag_kinds:
        blocked = [(i, d) for i, d in enumerate(decisions)
                   if d.get('action') != 'stop' and _number((d.get('exec') or {}).get('moved_m'))
                   and d['exec']['moved_m'] < THRESHOLDS['blocked_movement_m']]
        index = blocked[0][0] if blocked else None
        items = [diagnostic('stuck', 'Movement decisions with little displacement')]
        if index is not None:
            items.append(evidence('First flagged move displacement (m)', decisions[index]['exec']['moved_m'],
                                  f'/trace/decisions/{index}/exec/moved_m'))
            if 'walked_m' in decisions[index]['exec']:
                items.append(evidence('Distance walked during that move (m)', decisions[index]['exec']['walked_m'],
                                      f'/trace/decisions/{index}/exec/walked_m'))
            if 'miss_m' in decisions[index]['exec']:
                items.append(evidence('Waypoint miss at that move (m)', decisions[index]['exec']['miss_m'],
                                      f'/trace/decisions/{index}/exec/miss_m'))
        finding('blocked_movement', f"{diag['stuck']} movement decisions ended less than 0.25 m from their start; this measures net displacement, not collision cause.",
                items, ['A deliberate or exploratory round trip can end near its start despite substantial movement.',
                        'Execution may fall short of a waypoint; obstruction, control behavior, and waypoint feasibility require separate evidence.'],
                ['Net displacement alone does not establish obstruction; inspect distance walked, per-action collisions and control traces.'],
                ['Compare the selected waypoint with the actual movement and navigable area.',
                 'Inspect whether later decisions adapt after the first movement without displacement.'],
                index=index, severity='high')

    if 'waypoint_miss' in flag_kinds:
        moves = [(i, d) for i, d in enumerate(decisions) if d.get('action') != 'stop' and _number((d.get('exec') or {}).get('miss_m'))]
        index = max(moves, key=lambda pair: pair[1]['exec']['miss_m'])[0] if moves else None
        items = [diagnostic('ctrl_miss_max', 'Worst waypoint miss (m)')]
        if index is not None:
            if 'target' in decisions[index]:
                items.append(evidence('Actual arrival node', decisions[index]['target'], f'/trace/decisions/{index}/target'))
            items.append(evidence('Movement execution', decisions[index]['exec'], f'/trace/decisions/{index}/exec'))
        finding('waypoint_miss', f"A movement ended {diag['ctrl_miss_max']:.2f} m from its selected waypoint.", items,
                ['Waypoint feasibility, local control, or backtracking execution may contribute to the miss.'],
                ['Waypoint error alone does not identify which component caused it.'],
                ['Inspect the intended waypoint and actual endpoint at the worst move, then check collisions and control actions.'], index=index)

    if 'oscillation' in flag_kinds:
        path = ep.get('path') or []
        end = next((i + 2 for i, (a, b, c) in enumerate(zip(path, path[1:], path[2:])) if a == c), None)
        # The parser's path and frame arrays are aligned; verify that before anchoring.
        frame = end if end is not None and end < len(frames) and frames[end].get('vp') == path[end] else None
        items = [diagnostic('oscillations', 'A–B–A returns')]
        if end is not None:
            items.extend(evidence(f'Path node {i}', path[i], f'/trace/path/{i}') for i in range(end - 2, end + 1))
        finding('oscillation', f"{diag['oscillations']} immediate A–B–A returns occurred; useful exploration and recovery can also produce backtracking.", items,
                ['Repeated choices may reflect ambiguous observations or incomplete use of navigation history.',
                 'Backtracking may be intentional and useful.'],
                ['A repetition pattern does not establish obstruction, a memory failure, or an unsuccessful outcome.'],
                ['Inspect whether each return reveals new information or makes later progress.'], frame=frame)

    if 'inefficient_success' in flag_kinds:
        items = [metric('sr', 'Final success'), metric('spl', 'SPL')]
        if 'tl' in metrics:
            items.append(metric('tl', 'Executed path length (m)'))
        if continuous and 'sp_len' in metrics:
            items.append(metric('sp_len', 'Recorded start-to-goal distance used by CE SPL (m)'))
        if 'gt_len' in metrics:
            items.append(metric('gt_len', 'Reference route length from replay graph (m)' if continuous else 'Reference route length (m)'))
        finding('inefficient_success', f"The episode succeeded with SPL {metrics['spl']:.3f}; inspect route efficiency even though the final outcome was successful.",
                items, ['Exploration, detours, or recovery may explain the extra path length.'],
                ['SPL alone does not show whether extra movement was avoidable under the instruction and available observations.'],
                ['Compare route segments and instruction progress; retain useful recovery as a positive example.'], severity='review')

    if 'confident_reference_disagreement' in flag_kinds:
        t = diag.get('first_nonbest')
        index = next((i for i, d in enumerate(decisions) if d.get('t', i) == t), None)
        # A budget-forced stop must not be described as a model preference.
        if index is None or not (decisions[index].get('model') or {}).get('forced'):
            items = [diagnostic('first_nonbest', 'First reference disagreement decision'),
                     diagnostic('first_nonbest_conf', 'Chosen-option probability at that decision')]
            finding('confident_reference_disagreement',
                    f"The first reference disagreement chose an option with probability {diag['first_nonbest_conf']:.1%}. This is action preference, not calibrated confidence or a demonstrated mistake.",
                    items, ['The reference heuristic and instruction-following action may disagree.',
                            'The model may strongly prefer an unhelpful action; additional evidence is needed.'],
                    ['Reference disagreement does not prove a causal error; successful episodes can have this pattern.'],
                    ['Compare the instruction, available observation, and subsequent recovery before assigning an error label.'], index=index)

    if 'evidence_gap' in flag_kinds:
        items, missing = [], []
        if diag.get('steplog') is False:
            items.append(diagnostic('steplog', 'Fully aligned decision log'))
            missing.append('A complete aligned decision log is unavailable; scores from unlogged steps cannot be inferred.')
        if diag.get('goal_visit_unsampled') is True or (metrics.get('osr') == 1 and diag.get('near_goal_hops') == 0):
            if 'osr' in metrics:
                items.append(metric('osr', 'Recorded visit to success area'))
            if 'near_goal_hops' in diag:
                items.append(diagnostic('near_goal_hops', 'Sampled frames in success area'))
            for key, label in [('first_goal_track_index', 'First goal visit in dense track (not a replay frame)'),
                               ('dense_closest_dist', 'Closest dense-track goal distance (m)')]:
                if key in diag:
                    items.append(diagnostic(key, label))
            missing.append('The dense recorded metric indicates a goal visit, but sampled replay frames contain none.')
        finding('evidence_gap', ' '.join(missing), items, [], missing,
                ['Use the recorded metric for the episode outcome and retain the sampling limitation.',
                 'Recover the original log or collect denser trajectory/decision evidence before attributing the behavior.'], severity='evidence')

    if not frames:
        result['limitations'] = ['Replay frames are unavailable; finding links identify the episode without asserting a frame.']
    else:
        result['limitations'] = []
    if metrics.get('sr') is None:
        result['limitations'].append(('Evaluation success is unavailable' if ep.get('adapter') == 'recording' else
                                      'Ground-truth success is unavailable') + '; goal-based failure findings are not computed.')
    if not result['images']:
        result['limitations'].append('No camera-image references are available in this case.')
    return result
