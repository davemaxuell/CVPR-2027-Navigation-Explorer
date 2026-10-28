"""Native ETPNav continuous-environment recordings (R2R-CE, RxR-CE).
Use the portable recording adapter for other continuous-navigation log formats.

The explorer works on Matterport navigation graphs. A continuous episode gets its own small graph (CEGraph)
with the same interface, whose "viewpoints" are
  n<k>  ETPNav's map nodes: where decision k was taken (the agent's actual position)
  g<k>  ghost waypoints: predicted, never visited (position as last estimated)
  p<k>  samples of the agent's continuous track (every 0.5 m and at every arrival)
  r<k>  the reference path: the R2R / RxR viewpoints snapped to the navmesh; the last one is the goal
Positions are Matterport coordinates (habitat x, -z, y + camera height), so building meshes, floors and
rooms line up with the discrete runs. Distances to reference points are habitat geodesics logged with the
episode, so parse_trajectories.analyse_episode (frames, sub-tasks, decision review, diagnostics) applies
unchanged. Headline metrics are ETPNav's own, with the VLN-CE definitions (nDTW over the dense track,
SPL against the geodesic start-goal distance).

Episode keys are the discrete instruction ids where they exist (R2R <path_id>_<j>, RxR instruction_id), so
FGR2R / RxR sub-tasks, tags and comparisons with the discrete runs line up.
"""
import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from . import parse_trajectories as pt

ROWS_VERSION = 5         # bump when rows gain fields
# Habitat cube faces (explorer_env.CUBE_FACES) seen through views.js's skybox math: texture = -R^T world.
CUBE_ROT = [-1, 0, 0, 0, 0, -1, 0, -1, 0]
STUCK_M = 0.25           # a move that changes the agent's position by less than this went nowhere
ARRIVAL_M = 0.05         # the final position counts as the stop node itself within this distance


def _stamp(path):
    try:
        s = path.stat()
        return (s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    except OSError:
        return None


@lru_cache(maxsize=32)
def _checked_image_provenance(episodes_file, workspace_root, log_stamp, status_stamp, validation_stamp):
    """A migration receipt only applies to the input logs whose hashes it records.

    The cache keys include file metadata, so an appended/replaced log loses the provenance claim.
    This is a historical rendering receipt, not a new byte verification of the image directory.
    """
    unknown = {'kind': 'unknown', 'label': 'Image provenance unknown',
               'description': 'No matching rendering receipt was found for this episode log. '
                              'These views cannot be identified as original evaluation frames.'}
    if not all((log_stamp, status_stamp, validation_stamp)):
        return unknown
    root, log = Path(workspace_root), Path(episodes_file)
    status_file = root / 'migration' / 'local_ce_cube_views_status.json'
    validation_file = root / 'migration' / 'local_ce_cube_views_validation.json'
    try:
        relative = log.resolve().relative_to(root.resolve()).as_posix()
        status = json.loads(status_file.read_text(encoding='utf-8'))
        validation = json.loads(validation_file.read_text(encoding='utf-8'))
        if not isinstance(status, dict) or not isinstance(validation, dict):
            return unknown
        provenance = validation.get('provenance') or {}
        if not isinstance(provenance, dict):
            return unknown
        if (status.get('status') != 'complete' or validation.get('status') != 'pass'
                or validation.get('errors') or provenance.get('kind') != 'regenerated_preview'):
            return unknown
        source = next((item for item in status.get('input_logs', [])
                       if isinstance(item, dict) and item.get('path') == relative), None)
        if not source or not source.get('sha256'):
            return unknown
        digest = hashlib.sha256()
        with log.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(block)
        # A live logger might append while the digest is being computed.
        if digest.hexdigest() != source['sha256'] or _stamp(log) != log_stamp:
            return unknown
        return {'kind': 'regenerated_preview', 'label': 'Regenerated preview',
                'description': 'Rendered after evaluation from saved, rounded poses; these are not '
                               'the original evaluation frames. The matching migration receipt '
                               'verified these previews against their render cache.',
                'source': str(validation_file), 'input_log_sha256': source['sha256'],
                'pose_source': provenance.get('pose_source'), 'runtime': provenance.get('runtime')}
    except (OSError, ValueError, TypeError):
        return unknown


def image_provenance(episodes_file, workspace_root=None):
    """Return rendering provenance only when a local receipt covers this exact input log."""
    if workspace_root is None:
        return {'kind': 'unknown', 'label': 'Image provenance unknown',
                'description': 'No explicit validated rendering receipt was supplied.'}
    root = Path(workspace_root)
    log = Path(episodes_file)
    status = root / 'migration' / 'local_ce_cube_views_status.json'
    validation = root / 'migration' / 'local_ce_cube_views_validation.json'
    return dict(_checked_image_provenance(str(log), str(root), _stamp(log), _stamp(status), _stamp(validation)))


def _mp3d(p, cam):
    return (p[0], -p[2], p[1] + cam)


class CEGraph:
    """One episode's points with the ScanGraph interface parse_trajectories needs (see module doc)."""

    def __init__(self, rec, scan_graph, floors):
        cam = rec['camera_height']
        self.scan, self.heights = rec['scene'], {}
        self.pos, self.kind, self.geo, self.rot = {}, {}, {}, {}
        n_refs = len(rec['reference_path'])
        clean = lambda row: [x if x is not None else math.inf for x in row[:n_refs]]

        def add(vp, p, kind, geo):
            self.pos[vp], self.kind[vp], self.heights[vp] = _mp3d(p, cam), kind, cam
            self.geo[vp] = clean(geo)
        for vp, p in rec['nodes'].items():
            add(vp, p, 'node', rec['geo']['node'][vp])
            if rec.get('views'):
                self.rot[vp] = CUBE_ROT
        for vp, p in rec['ghosts'].items():
            add(vp, p, 'ghost', rec['geo']['ghost'][vp])
        for j, k in enumerate(rec['samples']):
            add(f'p{j}', rec['track'][k], 'sample', rec['geo']['sample'][j])
        self.refs = [f'r{j}' for j in range(n_refs)]
        for j, p in enumerate(rec['reference_path']):
            add(self.refs[j], p, 'ref', rec['ref_geo'][j])
        self.ref_index = {r: j for j, r in enumerate(self.refs)}
        # ETPNav's map: a move adds the edge front node -> arrival node; a waypoint predicted on top of a
        # visited node links the two; ghosts hang off the nodes that predicted them.
        self.adj = {}
        link = lambda a, b: (self.adj.setdefault(a, {}).__setitem__(b, self.euclid(a, b)),
                             self.adj.setdefault(b, {}).__setitem__(a, self.euclid(a, b)))
        steps = rec['steps']
        for t, s in enumerate(steps):
            if s.get('choice') and f'n{t + 1}' in self.pos:
                link(s['front'], f'n{t + 1}')
            for _, merged in s.get('cands', []):
                if merged and merged.startswith('n') and merged != s['vp'] and merged in self.pos:
                    link(s['vp'], merged)
        self.node_adj = {a: dict(b) for a, b in self.adj.items()}
        for g, fronts in rec['ghost_fronts'].items():
            for f in fronts:
                if f in self.pos and g in self.pos:
                    link(g, f)
        # Floors: the scan's floor slabs (so per-floor meshes match); rooms: the nearest Matterport viewpoint
        # on the same floor, for map nodes and reference points only (enough for room labels and stop_place).
        self.floors = floors
        self.floor_of = {vp: min(range(len(floors)), key=lambda f: abs(floors[f]['z'] - (p[2] - cam)))
                         if floors else 0 for vp, p in self.pos.items()}
        self.region_of, self.rid_of = {}, {}
        sg = scan_graph
        for vp in self.pos:
            if self.kind[vp] not in ('node', 'ref'):
                continue
            x, y, z = self.pos[vp]
            fl = self.floor_of[vp]
            best = min(((math.hypot(q[0] - x, q[1] - y), v) for v, q in sg.pos.items()
                        if sg.floor_of.get(v, 0) == fl), default=None)
            if best and best[0] < 4.0:
                self.region_of[vp], self.rid_of[vp] = sg.region_of.get(best[1]), sg.rid_of.get(best[1])
        self._sp = {}

    def dist(self, a, b):
        if a == b:
            return 0.0
        j = self.ref_index.get(b)
        if j is not None:
            return self.geo[a][j]
        j = self.ref_index.get(a)
        if j is not None:
            return self.geo[b][j]
        return self.euclid(a, b)    # only between non-reference points (back-tracking lengths)

    def euclid(self, a, b):
        return math.dist(self.pos[a], self.pos[b])

    def heading(self, a, b):
        dx, dy = self.pos[b][0] - self.pos[a][0], self.pos[b][1] - self.pos[a][1]
        return math.atan2(dx, dy) % (2 * math.pi)

    def path(self, a, b):
        return [b]

    def node_dist(self, a, b):
        """Shortest walk a -> b over ETPNav's map (map nodes only)."""
        dist, heap = {a: 0.0}, [(0.0, a)]
        import heapq
        while heap:
            d, u = heapq.heappop(heap)
            if u == b:
                return d
            if d > dist[u]:
                continue
            for v, w in self.node_adj.get(u, {}).items():
                if d + w < dist.get(v, math.inf):
                    dist[v] = d + w
                    heapq.heappush(heap, (d + w, v))
        return self.euclid(a, b)

    def to_json(self, sky=None):
        edges = sorted({tuple(sorted((a, b))) for a in self.adj for b in self.adj[a]})
        return {
            'scan': self.scan, 'ce': True,
            'nodes': {vp: {'pos': [round(c, 3) for c in p], 'h': round(self.heights[vp], 3),
                           'floor': self.floor_of.get(vp, 0), 'region': self.region_of.get(vp),
                           'rid': self.rid_of.get(vp), 'rot': self.rot.get(vp, []), 'kind': self.kind[vp],
                           **({'sky': f'{sky}/{vp}'} if sky and vp in self.rot else {})}
                      for vp, p in self.pos.items()},
            'edges': edges,
            'floors': self.floors,
        }


def ce_decisions(rec, graph):
    """ETPNav's steps as explorer decisions: route = track samples walked, ending at the arrival node.

    Each decision also carries what is specific to the continuous setting:
      cands  the waypoint predictor's output at this node: [[x, y, z] on the floor (Matterport), merged into]
             where merged into is the map node or ghost the prediction was folded into (None: dropped)
      exec   how the low-level controller carried it out: metres walked (the logged track only has points
             where the position changed), and for a move the route length to the aimed waypoint (over the
             map, then straight), how far the agent got from where it decided, and how far from the waypoint
             it ended
    """
    steps, seg, track = rec['steps'], rec['seg_end'], rec['track']
    sample_at = {k: f'p{j}' for j, k in enumerate(rec['samples'])}
    floor = lambda p: [round(p[0], 3), round(-p[2], 3), round(p[1], 3)]
    decisions, passed = [], [steps[0]['vp']]
    for t, s in enumerate(steps):
        lo, hi = seg[t], seg[t + 1]
        walked = [sample_at[k] for k in range(lo, hi) if k in sample_at]
        if s.get('choice') is None:
            route = walked
            if walked and graph.euclid(walked[-1], s['stop_at']) < ARRIVAL_M:
                route = walked[:-1] + [s['stop_at']]
            d = {'at': s['vp'], 'action': 'stop', 'target': route[-1] if route else s['vp'], 'route': route}
        else:
            arrival = f'n{t + 1}'
            route = (walked[:-1] if walked else []) + [arrival]
            d = {'at': s['vp'], 'action': 'local' if s['front'] == s['vp'] else 'remote', 'target': arrival,
                 'route': route, 'revisit': False, 'choice': s['choice']}
            if t + 1 < len(steps):
                d['heading'] = steps[t + 1]['heading']
        ex = {'walked_m': round(sum(math.dist(track[k - 1], track[k]) for k in range(lo, hi)), 3)}
        if s.get('choice') and s['choice'] in graph.pos and d['target'] in graph.pos:
            ex.update(aim_m=round(graph.node_dist(s['vp'], s['front']) + graph.euclid(s['front'], s['choice']), 3),
                      miss_m=round(graph.euclid(d['target'], s['choice']), 3),
                      moved_m=round(graph.euclid(s['vp'], d['target']), 3))
        d['exec'] = ex
        d['cands'] = [[floor(p), merged] for p, merged in s.get('cands', [])]
        d['passed'] = passed
        passed = [s['vp']] + route
        decisions.append(d)
    return decisions


class CERun:
    """Native ETPNav log adapter with explicit paths and per-run configuration."""

    ce = True
    route_reference = False

    def __init__(self, name, model, split, episodes_file, meta, graphs, *, views_dir=None,
                 subtasks=None, instruction_ids=None, cache_dir=None, cache_inputs=None,
                 provenance=None):
        self.name, self.model, self.split = name, model, split
        self.file = Path(episodes_file)
        self.views_dir = Path(views_dir) if views_dir else None
        self.pred_file = str(self.file)
        self.anno_file = str(meta.get('annotations') or self.file)
        self.steplog_file = str(self.file)
        self.steplog = True
        self.max_action_len = int(meta.get('max_action_len', 15))
        self.route_reference = meta.get('reference') == 'route'
        self.note = str(meta.get('note') or 'ETPNav continuous recording')
        self.settings = dict(meta, env='continuous')
        self.graphs = graphs
        self.subtasks = subtasks or {}
        self.instruction_ids = instruction_ids or {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_inputs = list(cache_inputs or [])
        self._image_provenance = provenance or image_provenance(self.file)
        self._load()

    def _load(self):
        self.signature = pt.file_stamp(self.file)
        self.offsets, self.episode_ids, seen_episodes = {}, {}, set()
        with self.file.open('rb') as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    episode_id = str(record['episode_id'])
                    explicit = record.get('instr_id', record.get('instruction_id'))
                    key = (str(record.get('trajectory_id', '')), ' '.join(record['instruction'].lower().split()))
                    instr_id = str(explicit) if explicit is not None and str(explicit) else self.instruction_ids.get(key, episode_id)
                except (ValueError, KeyError, TypeError, AttributeError) as error:
                    raise ValueError(f'{self.file}: invalid ETPNav episode at byte {offset}: {error}') from error
                if instr_id in self.offsets:
                    raise ValueError(f'{self.file}: duplicate instruction ID {instr_id!r}')
                if episode_id in seen_episodes:
                    raise ValueError(f'{self.file}: duplicate episode ID {episode_id!r}')
                required = ('scene', 'steps', 'nodes', 'ghosts', 'ghost_fronts', 'track', 'track_goal_dist',
                            'samples', 'reference_path', 'ref_geo', 'geo', 'seg_end', 'camera_height',
                            'metrics', 'max_steps', 'gt_locations', 'tokens')
                missing = [field for field in required if field not in record]
                if missing:
                    raise ValueError(f'{self.file}: episode {instr_id!r} is missing native ETPNav fields: {", ".join(missing)}')
                if not record['steps'] or not record['track'] or len(record['reference_path']) < 2:
                    raise ValueError(f'{self.file}: episode {instr_id!r} needs steps, a track, and at least two reference points')
                metrics = ('distance_to_goal', 'success', 'oracle_success', 'spl', 'ndtw', 'sdtw',
                           'path_length', 'collisions', 'steps_taken')
                absent = [key for key in metrics if not isinstance(record['metrics'].get(key), (int, float))
                          or not math.isfinite(record['metrics'][key])]
                if absent:
                    raise ValueError(f'{self.file}: episode {instr_id!r} needs finite recorded metrics: {", ".join(absent)}')
                self.offsets[instr_id], self.episode_ids[instr_id] = offset, episode_id
                seen_episodes.add(episode_id)
        if not self.offsets:
            raise ValueError(f'{self.file}: no ETPNav episode records found')
        self.ids = list(self.offsets)
        self.preds = {key: True for key in self.ids}
        self.gt = {}
        self._rows = self._points = None

    def stale(self):
        return pt.file_stamp(self.file) != self.signature

    @property
    def image_provenance(self):
        return dict(self._image_provenance)

    def record(self, instr_id):
        with self.file.open('rb') as source:
            source.seek(self.offsets[instr_id])
            return json.loads(source.readline())

    def graph(self, rec):
        scan = self.graphs[rec['scene']]
        return CEGraph(rec, scan, scan.to_json()['floors'])

    def episode(self, instr_id, full=True):
        rec = self.record(instr_id)
        g = self.graph(rec)
        decisions = ce_decisions(rec, g)
        gt = {'scan': rec['scene'], 'path': g.refs, 'instruction': rec['instruction'],
              'heading': rec['steps'][0]['heading']}
        pred = {'instr_id': instr_id, 'trajectory': None,
                'details': {vp: {'stop_prob': p} for vp, p in (rec['steps'][-1].get('stop_scores') or {}).items()}}
        steplog = {'steps': rec['steps'], 'tokens': rec['tokens']}
        ep = pt.analyse_episode(pred, gt, g, rec['max_steps'], steplog, full=full,
                                subtasks=self.subtasks.get(instr_id), route_reference=self.route_reference,
                                decisions=decisions, ref_near=lambda vp, ref: g.dist(vp, ref) < pt.SUBGOAL_MARGIN,
                                success_for_subtasks=bool(rec['metrics']['success']))
        self._apply_ce_metrics(ep, rec, g, decisions)
        if full:
            sky = f'/api/legacy-view/{quote(self.name, safe="")}/{quote(str(rec["episode_id"]), safe="")}'
            ep['graph'] = g.to_json(sky if rec.get('views') and self.views_dir else None)
            ep['dist_to_goal'] = {vp: round(g.dist(vp, ep['goal']), 2) for vp in g.pos}
            ep['goal_zone'] = [vp for vp, d in ep['dist_to_goal'].items() if d < pt.ERROR_MARGIN]
            ep['ce'] = {'episode_id': rec['episode_id'], 'language': rec.get('language'),
                        'track': [[round(c, 3) for c in _mp3d(p, 0.0)] for p in rec['track']],
                        'gt_dense': [[round(c, 3) for c in _mp3d(p, 0.0)] for p in rec['gt_locations']],
                        'camera_height': rec['camera_height'], 'image_provenance': self.image_provenance}
        else:
            ep['diag']['steplog'] = True
        return ep

    def _apply_ce_metrics(self, ep, rec, g, decisions):
        """ETPNav's own metrics replace the graph-based ones; CE-specific diagnostics."""
        em, m, diag = rec['metrics'], ep['metrics'], ep['diag']
        m.update(ne=em['distance_to_goal'], sr=em['success'], osr=em['oracle_success'], spl=em['spl'],
                 ndtw=em['ndtw'], sdtw=em['sdtw'], tl=round(em['path_length'], 3),
                 sp_len=round(rec['track_goal_dist'][0], 3), collisions=em['collisions'],
                 low_level_steps=em['steps_taken'])
        # The replay samples positions; the evaluator scores the dense execution track. An agent
        # can cross the success area between replay frames without changing the sampled outcome.
        # Always reconcile the category with the recorded metrics, not only when SR disagrees.
        reached_goal, success = bool(em['oracle_success']), bool(em['success'])
        step_limit = len(decisions) >= rec['max_steps']
        ep['outcome'] = 'success' if success else 'step_limit' if step_limit else 'wrong_stop'
        if success:
            ep['failure_mode'] = 'success'
        elif reached_goal:
            ep['failure_mode'] = 'passed goal: did not stop there'
        else:
            ep['failure_mode'] = 'never near goal: ran out of steps' if step_limit else \
                'never near goal: stopped elsewhere'
        # Preserve sampled closest_hop/dist and near_goal_hops: those remain valid replay anchors.
        # Dense indices are exported separately and must not be interpreted as frame indices.
        dense = [(i, value) for i, value in enumerate(rec.get('track_goal_dist', []))
                 if isinstance(value, (int, float)) and math.isfinite(value)]
        first_goal = next((i for i, value in dense if value < pt.ERROR_MARGIN), None)
        closest = min(dense, key=lambda item: item[1]) if dense else None
        diag.update(goal_visit_source='recorded_oracle_success',
                    goal_visit_unsampled=reached_goal and not bool(diag.get('near_goal_hops')),
                    first_goal_track_index=first_goal,
                    dense_closest_track_index=closest[0] if closest else None,
                    dense_closest_dist=closest[1] if closest else None)
        moves = [d for d in decisions if d['action'] != 'stop']
        remote = [d for d in decisions if d['action'] == 'remote']
        steps = rec['steps']
        back = sum(g.node_dist(d['at'], steps[d_t]['front']) for d_t, d in enumerate(decisions) if d['action'] == 'remote')
        misses = [d['exec']['miss_m'] for d in moves if 'miss_m' in d['exec']]
        diag.update(first_off_gt_hop=None, backtrack_hops=len(remote), backtrack_m=round(back, 3),
                    stuck=sum(1 for d in moves if g.euclid(d['at'], d['target']) < STUCK_M),
                    ghosts=len(rec['ghosts']),
                    # controller: how far from the aimed waypoint each move ended (worst move, mean)
                    ctrl_miss_max=round(max(misses), 3) if misses else None,
                    ctrl_miss_mean=round(sum(misses) / len(misses), 3) if misses else None,
                    actions_per_decision=round(em['steps_taken'] / max(1, len(decisions)), 1),
                    # ETPNav's 'collisions' is collisions per logged track point; turn it back into a count
                    collision_count=round(em['collisions'] * len(rec['track'])))
        # stopping: ETPNav stops at the visited node with the highest stop score; was a node inside the
        # success area scored lower?
        last = steps[-1]
        scores = last.get('stop_scores') or {}
        zone = [s for vp, s in scores.items() if vp in g.pos and g.dist(vp, g.refs[-1]) < pt.ERROR_MARGIN]
        diag.update(stop_score=scores.get(last.get('stop_at')),
                    zone_stop_score=max(zone) if zone else None)
        if (not success and reached_goal and zone and diag['stop_score'] is not None
                and diag['stop_score'] > max(zone)):
            ep['failure_mode'] = 'passed goal: stop score preferred a wrong node'

    def rows(self):
        if self._rows is None:
            cache = self.cache_dir / (hashlib.sha256(self.name.encode()).hexdigest()[:24] + '.json') if self.cache_dir else None
            key = self._cache_key()
            try:
                stored = json.loads(cache.read_text(encoding='utf-8')) if cache else {}
                if stored.get('key') == key:
                    self._rows, self._points = stored['rows'], stored['points']
                    return self._rows
            except (OSError, ValueError, KeyError):
                pass
            episodes = [self.episode(instr_id, full=False) for instr_id in self.ids]
            self._points = {ep['instr_id']: ep['review'] for ep in episodes if 'review' in ep}
            self._rows = [pt.summary_row(ep) for ep in episodes]
            if cache:
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps({'key': key, 'rows': self._rows, 'points': self._points}), encoding='utf-8')
                except OSError:
                    pass
        return self._rows

    def _cache_key(self):
        return [pt.file_stamp(path) for path in (self.file, __file__, pt.__file__, *self.cache_inputs)] + [self.route_reference, self.max_action_len, self.settings, getattr(self, 'cache_context', {})]

    def points(self):
        self.rows()
        return self._points

    def view_file(self, episode_id, node, face):
        if self.views_dir is None:
            return None
        if not re.fullmatch(r'[A-Za-z0-9_-]+', str(episode_id)) or not re.fullmatch(r'n\d+', str(node)) or str(face) not in '012345' or len(str(face)) != 1:
            raise ValueError('Invalid native CE image identifier')
        root = self.views_dir.resolve()
        path = (root / str(episode_id) / f'{node}_{face}.jpg').resolve()
        if root not in path.parents:
            raise ValueError('Image path escapes the configured views directory')
        return path
