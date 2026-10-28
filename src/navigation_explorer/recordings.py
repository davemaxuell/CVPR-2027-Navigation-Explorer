"""Model-independent recorded trajectories, with no simulator or dataset dependency.

The public input contract is ``navigation-explorer/episode-v1``. Positions are
meters in a Cartesian frame with z up. Evaluation metrics are imported, never
recomputed with a benchmark's unstated success or geodesic-distance rules.
"""
from collections import Counter
import copy
import gzip
import json
import math
from pathlib import Path
from urllib.parse import quote


EPISODE_SCHEMA = 'navigation-explorer/episode-v1'
COORDINATE_SYSTEM = 'cartesian_z_up_m'
COORDINATE_SYSTEMS = {COORDINATE_SYSTEM, 'matterport_z_up_m'}
_RATE_METRICS = {'sr', 'osr', 'spl', 'ndtw', 'sdtw', 'cls', 'collisions'}
_BINARY_METRICS = {'sr', 'osr'}
_RESERVED_METRICS = {
    'instr_id', 'scan', 'outcome', 'failure_mode', 'instruction', 'steplog', 'stuck',
    'ctrl_miss_max', 'ctrl_miss_mean', 'oscillations', 'near_goal_hops', 'goal_visit_unsampled',
    'first_nonbest', 'first_nonbest_conf', 'first_nonbest_type', 'first_nonbest_subtask',
    'first_goal_track_index', 'dense_closest_dist', 'dense_closest_track_index',
    'closest_hop', 'closest_dist', 'lost_hop', 'first_off_gt_hop', 'revisited_nodes',
    'max_visits', 'revisit_events', 'remote_jumps', 'backtrack_hops', 'backtrack_m',
    'stop_return', 'floors_visited', 'instr_words', 'distance_coverage_complete',
    'sampling_note', 'subtasks', 'subtasks_done', 'first_missed_subtask', 'stop_place',
    'nonbest', 'collision_count',
}


def _fail(location, message):
    raise ValueError(f'{location}: {message}')


def _text(value, location, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        _fail(location, 'must be a nonempty string' if not empty else 'must be a string')
    if any(ord(char) < 32 for char in value) and not empty:
        _fail(location, 'must not contain control characters')
    return value


def _number(value, location, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail(location, 'must be a finite number')
    if minimum is not None and value < minimum:
        _fail(location, f'must be at least {minimum}')
    if maximum is not None and value > maximum:
        _fail(location, f'must be at most {maximum}')
    return value


def _integer(value, location, *, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(location, 'must be an integer')
    return _number(value, location, minimum=minimum, maximum=maximum)


def _vector(value, location, length=3):
    if not isinstance(value, list) or len(value) != length:
        _fail(location, f'must be an array of {length} finite numbers')
    return [_number(item, f'{location}[{i}]') for i, item in enumerate(value)]


def _object(value, location):
    if not isinstance(value, dict):
        _fail(location, 'must be an object')
    return value


def _array(value, location):
    if not isinstance(value, list):
        _fail(location, 'must be an array')
    return value


def _finite_json(value, location):
    """Reject numeric overflow and non-JSON values, including uninterpreted metadata."""
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        _number(value, location)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _finite_json(item, f'{location}[{i}]')
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail(location, 'object keys must be strings')
            _finite_json(item, f'{location}.{key}')
    else:
        _fail(location, 'must contain JSON-compatible values')


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key {key!r}')
        result[key] = value
    return result


def _json(text, location):
    def invalid(value):
        _fail(location, f'{value} is not valid JSON; use null for unknown values')
    try:
        return json.loads(text, object_pairs_hook=_json_object, parse_constant=invalid)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{location}: {exc}') from exc


def read_records(path):
    """Read JSON arrays or JSONL; gzip is supported for either format."""
    path = Path(path)
    if not path.is_file():
        _fail(str(path), 'episode recording file not found')
    opener = gzip.open if path.suffix == '.gz' else open
    plain_suffix = Path(path.stem).suffix if path.suffix == '.gz' else path.suffix
    with opener(path, 'rt', encoding='utf-8') as stream:
        if plain_suffix == '.jsonl':
            records = [_json(line, f'{path}:{index}') for index, line in enumerate(stream, 1) if line.strip()]
        else:
            data = _json(stream.read(), str(path))
            records = data if isinstance(data, list) else [data]
    if not records:
        _fail(str(path), 'must contain at least one episode')
    return records


def _relative_image(value, location):
    _text(value, location)
    path = Path(value)
    if path.is_absolute() or '\\' in value or '..' in path.parts or '?' in value or '#' in value:
        _fail(location, 'must be a relative image prefix inside images_dir, with no traversal or URL components')
    return path.as_posix()


def normalize_episode(record, *, run_name, environment, evaluation=None, images_dir=None,
                      coordinate_system=COORDINATE_SYSTEM):
    """Validate and adapt one recording to the viewer's internal replay contract.

    Returned decisions with ``source=replay_placeholder`` are presentation-only:
    they do not represent an action, a STOP, or an observed model decision.
    """
    _object(record, 'episode')
    _finite_json(record, 'episode')
    if record.get('schema_version') != EPISODE_SCHEMA:
        _fail('episode.schema_version', f'must be {EPISODE_SCHEMA!r}')
    instr_id = _text(record.get('instr_id'), 'episode.instr_id')
    where = f'episode {instr_id!r}'
    scene = _text(record.get('scene'), f'{where}.scene')
    instruction = _text(record.get('instruction', ''), f'{where}.instruction', empty=True)
    if environment not in ('discrete', 'continuous'):
        _fail('run.environment', 'must be discrete or continuous')
    if record.get('environment', environment) != environment:
        _fail(f'{where}.environment', 'must match the run environment')
    if not isinstance(coordinate_system, str) or coordinate_system not in COORDINATE_SYSTEMS:
        _fail('run.coordinate_system', f'must be one of {sorted(COORDINATE_SYSTEMS)}; convert positions before import')
    if record.get('coordinate_system', coordinate_system) != coordinate_system:
        _fail(f'{where}.coordinate_system', 'must match the run coordinate system; no coordinate transformation is inferred')
    camera_height = _number(record.get('camera_height_m', 0.0), f'{where}.camera_height_m', minimum=0)
    evaluation = dict(_object(evaluation or {}, 'run.evaluation'))
    evaluation.update(_object(record.get('evaluation', {}), f'{where}.evaluation'))
    radius = evaluation.get('success_radius_m')
    if radius is not None:
        _number(radius, f'{where}.evaluation.success_radius_m', minimum=0)
        if radius == 0:
            _fail(f'{where}.evaluation.success_radius_m', 'must be positive')
    distance_kind = evaluation.get('distance_kind', 'unknown')
    if distance_kind not in ('geodesic', 'euclidean', 'custom', 'unknown'):
        _fail(f'{where}.evaluation.distance_kind', 'must be geodesic, euclidean, custom, or unknown')
    evaluation.update(success_radius_m=radius, distance_kind=distance_kind)
    termination = record.get('termination', 'unknown')
    if termination not in ('stop', 'step_limit', 'completed', 'unknown'):
        _fail(f'{where}.termination', 'must be stop, step_limit, completed, or unknown')

    nodes, edges, distances = {}, set(), {}
    graph = _object(record.get('graph', {}), f'{where}.graph')
    graph_nodes = _object(graph.get('nodes', {}), f'{where}.graph.nodes')

    def point(value, location, default_id, kind):
        _object(value, location)
        node_id = _text(value.get('node_id', default_id), f'{location}.node_id')
        existing = nodes.get(node_id)
        position = value.get('position', existing['pos'] if existing else None)
        position = _vector(position, f'{location}.position')
        if existing and position != existing['pos']:
            _fail(location, f'node_id {node_id!r} has conflicting positions; use unique IDs for distinct samples')
        floor = _integer(value.get('floor', existing['floor'] if existing else 0), f'{location}.floor')
        height = _number(value.get('height', existing['h'] if existing else camera_height), f'{location}.height', minimum=0)
        if existing and (floor != existing['floor'] or height != existing['h']):
            _fail(location, f'node_id {node_id!r} has conflicting floor/height metadata')
        node = existing or {'pos': position, 'floor': floor, 'h': height, 'rot': [], 'kind': kind,
                            'region': None, 'rid': None}
        if value.get('region') is not None:
            node['region'] = _text(value['region'], f'{location}.region')
        if value.get('image_prefix') is not None:
            prefix = _relative_image(value['image_prefix'], f'{location}.image_prefix')
            if images_dir is None:
                _fail(location, 'image_prefix requires images_dir in the run manifest')
            sky = f'/api/recorded-view/{quote(run_name, safe="")}/{quote(prefix, safe="")}'
            rotation = _vector(value.get('rotation', [1, 0, 0, 0, 1, 0, 0, 0, 1]), f'{location}.rotation', 9)
            if existing and existing.get('sky') and (existing['sky'] != sky or existing['rot'] != rotation):
                _fail(location, f'node_id {node_id!r} has conflicting camera images; use distinct IDs for distinct observations')
            node['sky'], node['rot'] = sky, rotation
        elif 'rotation' in value:
            _vector(value['rotation'], f'{location}.rotation', 9)
        nodes[node_id] = node
        if value.get('distance_to_goal') is not None:
            d = _number(value['distance_to_goal'], f'{location}.distance_to_goal', minimum=0)
            if node_id in distances and distances[node_id] != d:
                _fail(location, f'node_id {node_id!r} has conflicting goal distances')
            distances[node_id] = d
        return node_id

    for node_id, value in graph_nodes.items():
        _text(node_id, f'{where}.graph.nodes key')
        if not isinstance(value, dict):
            _fail(f'{where}.graph.nodes.{node_id}', 'must be an object with position')
        if value.get('node_id', node_id) != node_id:
            _fail(f'{where}.graph.nodes.{node_id}', 'node_id must match the graph key')
        point(dict(value, node_id=node_id), f'{where}.graph.nodes.{node_id}', node_id, 'node')

    trajectory = _array(record.get('trajectory'), f'{where}.trajectory')
    if not trajectory:
        _fail(f'{where}.trajectory', 'must contain at least one sampled position')
    path = [point(value, f'{where}.trajectory[{i}]', f'p{i}', 'sample' if environment == 'continuous' else 'node')
            for i, value in enumerate(trajectory)]
    reference = _array(record.get('reference_path', []), f'{where}.reference_path')
    gt_path = [point(value, f'{where}.reference_path[{i}]', f'r{i}', 'ref') for i, value in enumerate(reference)]
    goal = point(record['goal'], f'{where}.goal', 'reference_goal', 'ref') if record.get('goal') is not None else None
    # A reference route does not necessarily end at a task's goal (e.g. object goals).
    # Only an explicit goal field creates a goal marker.
    for i, edge in enumerate(_array(graph.get('edges', []), f'{where}.graph.edges')):
        if not isinstance(edge, list) or len(edge) != 2 or any(not isinstance(vp, str) or vp not in nodes for vp in edge):
            _fail(f'{where}.graph.edges[{i}]', 'must name two existing node IDs')
        if edge[0] != edge[1]:
            edges.add(tuple(sorted(edge)))
    # Route segments are replay geometry, not evidence of a full navigability graph.
    for route in (path, gt_path):
        edges.update(tuple(sorted((a, b))) for a, b in zip(route, route[1:]) if a != b)

    declared = _array(record.get('decisions', []), f'{where}.decisions')
    decision_frames, decisions = [], []
    for t, item in enumerate(declared):
        loc = f'{where}.decisions[{t}]'
        _object(item, loc)
        frame = _integer(item.get('frame'), f'{loc}.frame', maximum=len(path) - 1)
        if t == 0 and frame != 0:
            _fail(f'{loc}.frame', 'the first logged decision must align with frame 0; trim unaligned leading samples')
        if decision_frames and frame <= decision_frames[-1]:
            _fail(f'{loc}.frame', 'decision frames must be strictly increasing')
        decision_frames.append(frame)
        if item.get('action') not in ('move', 'stop', 'observe', 'unknown'):
            _fail(f'{loc}.action', 'must be move, stop, observe, or unknown')

    for t, item in enumerate(declared):
        loc, frame = f'{where}.decisions[{t}]', decision_frames[t]
        following = decision_frames[t + 1] if t + 1 < len(declared) else len(path) - 1
        end = _integer(item.get('end_frame', following), f'{loc}.end_frame', minimum=frame, maximum=following)
        action = item['action']
        if action == 'stop' and (end != frame or t != len(declared) - 1 or frame != len(path) - 1):
            _fail(loc, 'a STOP must be the final decision, at the final replay position')
        if action == 'stop' and termination not in ('unknown', 'stop'):
            _fail(loc, 'STOP conflicts with the declared termination')
        target = item.get('target')
        if isinstance(target, dict):
            target = point(target, f'{loc}.target', f'target_{t}', 'ghost')
        elif target is not None and (not isinstance(target, str) or target not in nodes):
            _fail(f'{loc}.target', 'must name an existing node or provide a point object')
        visited = list(dict.fromkeys(path[:frame + 1]))
        dec = {'t': t, 'at': path[frame], 'target': path[end], 'choice': target,
               'action': 'local' if action == 'move' else action,
               'route': path[frame + 1:end + 1], 'visited': visited, 'frontier': [],
               'stop_prob': None, 'source': 'recorded', 'frame': frame, 'end_frame': end}
        if action == 'move' and end > frame:
            moved = math.dist(nodes[path[frame]]['pos'], nodes[path[end]]['pos'])
            walked = sum(math.dist(nodes[a]['pos'], nodes[b]['pos']) for a, b in zip(path[frame:end], path[frame + 1:end + 1]))
            dec['exec'] = {'moved_m': moved, 'walked_m': walked}
            if target is not None:
                dec['exec'].update(aim_m=math.dist(nodes[path[frame]]['pos'], nodes[target]['pos']),
                                   miss_m=math.dist(nodes[path[end]]['pos'], nodes[target]['pos']))
        options = _array(item.get('options', []), f'{loc}.options')
        normalized, seen = [], set()
        for j, option in enumerate(options):
            opt_loc = f'{loc}.options[{j}]'
            _object(option, opt_loc)
            if 'node_id' not in option:
                _fail(f'{opt_loc}.node_id', 'is required; use null to explicitly denote STOP')
            if 'probability' not in option and 'score' not in option:
                _fail(opt_loc, 'must supply probability or score; use null when explicitly unavailable')
            vp = option.get('node_id')
            if vp is not None and (not isinstance(vp, str) or vp not in nodes):
                _fail(f'{opt_loc}.node_id', 'must name an existing node; null explicitly denotes STOP')
            if vp in seen:
                _fail(opt_loc, 'duplicate action option')
            seen.add(vp)
            normalized.append(copy.deepcopy(option))
            for name in ('probability', 'score'):
                if option.get(name) is not None:
                    _number(option[name], f'{opt_loc}.{name}', minimum=0 if name == 'probability' else None,
                            maximum=1 if name == 'probability' else None)
        if normalized:
            dec['logged_options'] = normalized
            if sum(option.get('probability') or 0 for option in normalized) > 1.0001:
                _fail(f'{loc}.options', 'recorded action probabilities must sum to at most 1 (partial distributions are allowed)')
            # Scores and logits are not silently normalized into probabilities.
            if all(option.get('probability') is not None for option in normalized):
                gmap = [option['node_id'] for option in normalized if option.get('node_id') is not None]
                score = {option.get('node_id'): option['probability'] for option in normalized}
                probs = [score.get(None), *(score[vp] for vp in gmap)]
                dec['model'] = {'vp': path[frame], 'gmap': gmap, 'visited': [False] * len(gmap),
                                'fused': probs, 'global': list(probs), 'local': [], 'cands': [],
                                'choice': None if action == 'stop' else target, 'forced': bool(item.get('forced', False)),
                                'score_source': 'recorded_probabilities'}
                dec['frontier'] = gmap
                dec['stop_prob'] = score.get(None)
        if 'forced' in item and not isinstance(item['forced'], bool):
            _fail(f'{loc}.forced', 'must be true or false')
        decisions.append(dec)

    has_decisions = bool(decisions)
    if not decisions:
        decisions = [{'t': 0, 'at': path[0], 'target': path[-1], 'action': 'unknown', 'route': [],
                      'visited': [], 'frontier': [], 'stop_prob': None, 'source': 'replay_placeholder'}]
    if has_decisions and decisions[-1]['action'] == 'stop':
        termination = 'stop'

    frames, visits = [], Counter()
    current_decision, previous_time = 0, None
    for i, vp in enumerate(path):
        while current_decision + 1 < len(decision_frames) and decision_frames[current_decision + 1] <= i:
            current_decision += 1
        value = trajectory[i]
        visits[vp] += 1
        heading = _number(value.get('heading', 0.0), f'{where}.trajectory[{i}].heading')
        elevation = _number(value.get('elevation', 0.0), f'{where}.trajectory[{i}].elevation')
        frame = {'vp': vp, 'decision': current_decision, 'kind': 'start' if i == 0 else 'sample',
                 'hop': i, 'nhops': len(path) - 1, 'heading': heading, 'elevation': elevation,
                 'visits': visits[vp], 'floor': nodes[vp]['floor'],
                 'dist': distances.get(vp), 'euclid': math.dist(nodes[vp]['pos'], nodes[goal]['pos']) if goal else None,
                 'gt_dist': None}
        if value.get('time_s') is not None:
            timestamp = _number(value['time_s'], f'{where}.trajectory[{i}].time_s', minimum=0)
            if previous_time is not None and timestamp < previous_time:
                _fail(f'{where}.trajectory[{i}].time_s', 'timestamps must be nondecreasing')
            previous_time = timestamp
            frame['time_s'] = timestamp
        frames.append(frame)

    metrics = copy.deepcopy(_object(record.get('metrics', {}), f'{where}.metrics'))
    for key, value in metrics.items():
        _text(key, f'{where}.metrics key')
        if key in _RESERVED_METRICS:
            _fail(f'{where}.metrics.{key}', 'is reserved for replay diagnostics or episode identity; preserve custom observations in metadata instead')
        if value is not None:
            _number(value, f'{where}.metrics.{key}', minimum=0 if key in _RATE_METRICS or key in ('ne', 'tl', 'sp_len', 'gt_len') else None,
                    maximum=1 if key in _RATE_METRICS else None)
            if key in _BINARY_METRICS and value not in (0, 1):
                _fail(f'{where}.metrics.{key}', 'must be 0, 1, or null for an individual episode')
    sources = {key: 'recorded_metric' if value is not None else 'unavailable' for key, value in metrics.items()}
    trace_length = sum(math.dist(nodes[a]['pos'], nodes[b]['pos']) for a, b in zip(path, path[1:]))
    derived = {'tl': trace_length, 'hops': len(path) - 1,
               'decisions': len(declared) if has_decisions else None,
               'moves': sum(d['action'] == 'local' for d in decisions) if has_decisions else None}
    for key, value in derived.items():
        if key not in metrics:
            metrics[key] = value
            sources[key] = 'sampled_trace_derived_metric' if key == 'tl' else 'trace_derived_metric' if value is not None else 'unavailable'
    sr = metrics.get('sr')
    outcome = ('success' if sr == 1 else 'step_limit' if sr == 0 and termination == 'step_limit' else
               'wrong_stop' if sr == 0 and termination == 'stop' else 'failure' if sr == 0 else 'unscored')
    failure_mode = {'success': 'recorded success', 'step_limit': 'recorded failure: step limit',
                    'wrong_stop': 'recorded failure: stopped', 'failure': 'recorded failure: termination unspecified',
                    'unscored': 'evaluation result not supplied'}[outcome]
    back = [(a, b) for i, (a, b) in enumerate(zip(path, path[1:])) if b in path[:i + 1] and a != b]
    diag = {'steplog': has_decisions, 'revisited_nodes': sum(n >= 2 for n in visits.values()),
            'max_visits': max(visits.values()), 'revisit_events': sum(n - 1 for n in visits.values()),
            'remote_jumps': 0, 'backtrack_hops': len(back),
            'backtrack_m': sum(math.dist(nodes[a]['pos'], nodes[b]['pos']) for a, b in back),
            'stop_return': False, 'oscillations': sum(a == c and a != b for a, b, c in zip(path, path[1:], path[2:])),
            'floors_visited': len({nodes[vp]['floor'] for vp in path}), 'instr_words': len(instruction.split()),
            'distance_coverage_complete': all(frame['dist'] is not None for frame in frames),
            'sampling_note': 'Replay positions are supplied samples; paths between samples and unlogged actions are unknown.'}
    if diag['distance_coverage_complete']:
        closest = min(range(len(frames)), key=lambda i: frames[i]['dist'])
        diag.update(closest_hop=closest, closest_dist=frames[closest]['dist'])
        if radius is not None and distance_kind != 'unknown':
            diag['near_goal_hops'] = sum(frame['dist'] < radius for frame in frames)
            diag['goal_visit_unsampled'] = metrics.get('osr') == 1 and diag['near_goal_hops'] == 0
    movement = [d for d in decisions if d['action'] == 'local' and 'exec' in d]
    if movement:
        diag['stuck'] = sum(d['exec']['moved_m'] < 0.25 for d in movement)
        misses = [d['exec']['miss_m'] for d in movement if 'miss_m' in d['exec']]
        if misses:
            diag.update(ctrl_miss_max=max(misses), ctrl_miss_mean=sum(misses) / len(misses))
    has_images = any(node.get('sky') for node in nodes.values())
    supplied_provenance = _object(record.get('image_provenance', {}), f'{where}.image_provenance')
    if has_images:
        provenance = {'kind': 'unknown', 'label': 'Recorded camera references',
                      'description': 'Image files were supplied with this recording; their relation to original model inputs is not independently verified.',
                      **copy.deepcopy(supplied_provenance), 'source': 'recording_declaration', 'independently_verified': False}
    else:
        provenance = {'kind': 'unavailable', 'label': 'No camera images',
                      'description': 'This episode supplies trajectory geometry without camera images.', 'exact_model_input': None}
    floor_ids = sorted({node['floor'] for node in nodes.values()})
    # Existing map controls use floor IDs as array indices.
    if floor_ids != list(range(len(floor_ids))):
        _fail(f'{where}.graph', 'floor IDs must be contiguous starting at 0')
    floors = [{'id': floor, 'z': min(node['pos'][2] - node['h'] for node in nodes.values() if node['floor'] == floor),
               'label': f'Floor {floor + 1}'} for floor in floor_ids]
    capabilities = {'images': has_images, 'mesh': False, 'whatif': False, 'decisions': has_decisions}
    budget = record.get('max_action_len')
    if budget is not None:
        _integer(budget, f'{where}.max_action_len', minimum=1)
    ep = {'instr_id': instr_id, 'scan': scene, 'instruction': instruction, 'adapter': 'recording',
          'recording_schema': EPISODE_SCHEMA, 'coordinate_system': coordinate_system, 'reference': 'none',
          'start': path[0], 'final': path[-1], 'goal': goal, 'path': path, 'gt_path': gt_path,
          'goal_zone': [vp for vp, d in distances.items() if radius is not None and distance_kind != 'unknown' and d < radius],
          'dist_to_goal': {vp: distances.get(vp) for vp in nodes}, 'frames': frames, 'decisions': decisions,
          'visits': dict(visits), 'stop_probs': {d['at']: d['stop_prob'] for d in decisions if d['stop_prob'] is not None},
          'start_heading': frames[0]['heading'], 'max_action_len': budget,
          'metrics': metrics, 'metric_sources': sources,
          'metric_notes': {'tl': 'Recorded path length when supplied; otherwise the polyline length of the supplied samples, which may underestimate actual travel.'},
          'diag': diag, 'outcome': outcome, 'failure_mode': failure_mode, 'termination': termination,
          'evaluation': evaluation, 'image_provenance': provenance, 'capabilities': capabilities,
          'subtasks': {'source': 'instruction', 'spans': [[0, len(instruction)]] if instruction else [],
                       'reached': None, 'chunks': None, 'goals': None},
          'graph': {'scan': scene, 'ce': environment == 'continuous', 'nodes': nodes, 'edges': [list(edge) for edge in sorted(edges)],
                    'floors': floors, 'navmap': None, 'capabilities': dict(capabilities),
                    'edge_source': 'supplied graph edges and recorded route segments; not an inferred navigability map'}}
    if environment == 'continuous':
        ep['ce'] = {'episode_id': instr_id, 'camera_height': camera_height,
                    'track': [[*nodes[vp]['pos'][:2], nodes[vp]['pos'][2] - camera_height] for vp in path],
                    'gt_dense': [[*nodes[vp]['pos'][:2], nodes[vp]['pos'][2] - camera_height] for vp in gt_path],
                    'image_provenance': copy.deepcopy(provenance)}
    if record.get('metadata') is not None:
        ep['recording_metadata'] = copy.deepcopy(_object(record['metadata'], f'{where}.metadata'))
    return ep


class PortableRun:
    """Run adapter for explicit, versioned model-independent recordings."""

    def __init__(self, spec, project_dir, state_dir=None):
        project_dir = Path(project_dir)
        self.name = _text(spec.get('id'), 'run.id')
        self.model = _text(spec.get('model'), f'run {self.name}.model')
        self.dataset = _text(spec.get('dataset'), f'run {self.name}.dataset')
        self.split = _text(spec.get('split'), f'run {self.name}.split')
        self.environment = spec.get('environment')
        if self.environment not in ('discrete', 'continuous'):
            _fail(f'run {self.name}.environment', 'must be discrete or continuous')
        self.ce = self.environment == 'continuous'
        self.coordinate_system = spec.get('coordinate_system', COORDINATE_SYSTEM)
        if not isinstance(self.coordinate_system, str) or self.coordinate_system not in COORDINATE_SYSTEMS:
            _fail(f'run {self.name}.coordinate_system', f'must be one of {sorted(COORDINATE_SYSTEMS)}; convert positions before import')
        self.comparison_group = spec.get('comparison_group')
        if self.comparison_group is not None:
            _text(self.comparison_group, f'run {self.name}.comparison_group')
        self.adapter, self.reference, self.route_reference = 'recording', 'none', False
        self.note = _text(spec.get('note', ''), f'run {self.name}.note', empty=True)
        self.evaluation = _object(spec.get('evaluation', {}), f'run {self.name}.evaluation')
        self.settings = copy.deepcopy(_object(spec.get('metadata', {}), f'run {self.name}.metadata'))
        _finite_json(self.settings, f'run {self.name}.metadata')
        self.max_action_len = None
        self.images_dir = None
        if spec.get('images_dir') is not None:
            self.images_dir = (project_dir / Path(_text(spec['images_dir'], f'run {self.name}.images_dir')).expanduser()).resolve()
            if not self.images_dir.is_dir():
                _fail(f'run {self.name}.images_dir', f'directory not found: {self.images_dir}')
        path = project_dir / Path(_text(spec.get('episodes'), f'run {self.name}.episodes')).expanduser()
        self.pred_file, self.anno_file = str(path.resolve()), None
        self.preds, self._episodes = {}, {}
        for record in read_records(path):
            ep = normalize_episode(record, run_name=self.name, environment=self.environment,
                                   evaluation=self.evaluation, images_dir=self.images_dir,
                                   coordinate_system=self.coordinate_system)
            instr_id = ep['instr_id']
            if instr_id in self.preds:
                _fail(str(path), f'duplicate instr_id {instr_id!r}')
            self.preds[instr_id], self._episodes[instr_id] = record, ep
        self.ids = list(self.preds)
        self.gt = {instr_id: {'scan': ep['scan'], 'instruction': ep['instruction'], 'path': ep['gt_path']}
                   for instr_id, ep in self._episodes.items()}
        self.steplog = {instr_id: ep['decisions'] for instr_id, ep in self._episodes.items() if ep['capabilities']['decisions']}
        self.steplog_file = self.pred_file if self.steplog else None
        self.graphs = {}
        self.has_images = any(ep['capabilities']['images'] for ep in self._episodes.values())
        self.capabilities = {'images': self.has_images, 'mesh': False, 'whatif': False,
                             'decisions': bool(self.steplog)}
        self._rows = [dict(instr_id=ep['instr_id'], scan=ep['scan'], instruction=ep['instruction'],
                           outcome=ep['outcome'], failure_mode=ep['failure_mode'], **ep['metrics'], **ep['diag'])
                      for ep in self._episodes.values()]

    def episode(self, instr_id, full=True):
        # Returning copies prevents request-specific comparison/asset decoration from
        # changing the underlying recording or affecting concurrent reviewers.
        ep = copy.deepcopy(self._episodes[str(instr_id)])
        if not full:
            for key in ('graph', 'frames', 'decisions', 'ce', 'visits', 'stop_probs', 'subtasks'):
                ep.pop(key, None)
        return ep

    def rows(self):
        return copy.deepcopy(self._rows)

    def points(self):
        # No benchmark-specific reference-best heuristic is inferred for custom runs.
        return {}

    def image_file(self, relative_name):
        if self.images_dir is None:
            return None
        name = Path(relative_name)
        if name.is_absolute() or '..' in name.parts or '\\' in str(relative_name) or name.suffix.lower() != '.jpg':
            return None
        resolved = (self.images_dir / name).resolve()
        if self.images_dir not in resolved.parents:
            return None
        return resolved
