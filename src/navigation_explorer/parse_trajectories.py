"""Align VLN agent trajectories with ground truth and Matterport3D connectivity, and diagnose failures.

Native discrete analysis used by the r2r adapter, with an optional summary/export CLI:

  python -m navigation_explorer.parse_trajectories summary \
      --preds predictions.json --anno annotations.json --conn-dir /path/to/connectivity
  python -m navigation_explorer.parse_trajectories export --preds ... --anno ... --conn-dir ... --failed --out failed.csv

Trajectory formats accepted (auto-detected per episode):
  DUET/ScaleVLN  {"trajectory": [[start], [vp], [vp, vp, vp], ...]}  one list per decision; a list with
                 more than one viewpoint is a remote-frontier jump walked through the map. Optional
                 "details": {vp: {"stop_prob": p}} from --detailed_output.
  R2R standard   {"trajectory": [[vp, heading, elevation], ...]}  consecutive repeats are turns in place.

Sub-tasks: R2R instructions are split into sub-instructions with Fine-Grained R2R (FGR2R), whose chunk_view
aligns each sub-instruction to a stretch of the ground-truth path. DUET does not decompose instructions
itself; which sub-task it is "working on" is read off its cross-attention (mass per sub-instruction).

Only the standard library is needed. Graph distances match DUET's (Dijkstra over unobstructed, included
viewpoints, Euclidean edge weights), so NE/SR/SPL/nDTW reproduce the evaluation log.
"""
import argparse
import ast
import csv
import gzip
import heapq
import hashlib
import json
import math
import os
import re
import sys
import threading
from collections import Counter
from pathlib import Path

# Legacy analysis uses explicit files supplied by a project manifest. These inert
# compatibility defaults are used only by the optional mesh helper's CLI.
DUET_DATA = Path('datasets')
SCANS_DIR = None
ERROR_MARGIN = 3.0
SUBGOAL_MARGIN = 1.5    # an intermediate sub-goal counts as reached within this geodesic distance
REF_TOLERANCE = 0.5     # metres: a move this close to the reference-best option's distance still matches it
DUET_MAX_ACTION_LEN = 15
# DUET's MatterSim uses discretised viewing angles: 12 headings x 3 elevations, 30 degrees apart. The table of
# the view in which each viewpoint sees each neighbour (env.make_candidate's pointId) ships with ScaleVLN.
VIEW_STEP = math.pi / 6

# Matterport3D region codes (.house "R" lines)
REGION_LABELS = {
    'a': 'bathroom', 'b': 'bedroom', 'c': 'closet', 'd': 'dining room', 'e': 'entryway', 'f': 'family room',
    'g': 'garage', 'h': 'hallway', 'i': 'library', 'j': 'laundry room', 'k': 'kitchen', 'l': 'living room',
    'm': 'meeting room', 'n': 'lounge', 'o': 'office', 'p': 'porch', 'r': 'game room', 's': 'stairs',
    't': 'toilet', 'u': 'utility room', 'v': 'tv room', 'w': 'gym', 'x': 'outdoor', 'y': 'balcony',
    'z': 'other room', 'B': 'bar', 'C': 'classroom', 'D': 'dining booth', 'S': 'spa', 'Z': 'junk',
}


# ---------------------------------------------------------------------------------------------- graphs

class ScanGraph:
    """Navigation graph of one scan with lazily cached single-source shortest paths."""

    def __init__(self, scan, pos, adj, floor_of, region_of, heights, rot=None, rid_of=None):
        self.scan = scan
        self.pos = pos                  # vp -> (x, y, z)
        self.rot = rot or {}            # vp -> 3x3 camera rotation, row-major (skybox orientation)
        self.rid_of = rid_of or {}      # vp -> region index in the .house file
        self.adj = adj                  # vp -> {neighbour: metres}
        self.floor_of = floor_of        # vp -> floor index, 0 = lowest
        self.region_of = region_of      # vp -> room label or None
        self.heights = heights          # vp -> camera height above floor
        self._sp = {}

    @classmethod
    def load(cls, conn_dir, scan, scans_dir=SCANS_DIR):
        data = json.loads((Path(conn_dir) / f'{scan}_connectivity.json').read_text(encoding='utf-8'))
        pos, adj, heights, rot = {}, {}, {}, {}
        for item in data:
            if item['included']:
                p = item['pose']
                pos[item['image_id']] = (p[3], p[7], p[11])
                rot[item['image_id']] = [p[0], p[1], p[2], p[4], p[5], p[6], p[8], p[9], p[10]]
                heights[item['image_id']] = item.get('height', 1.5)
        for i, item in enumerate(data):
            if not item['included']:
                continue
            a = item['image_id']
            adj.setdefault(a, {})
            for j, conn in enumerate(item['unobstructed']):
                if conn and data[j]['included']:
                    b = data[j]['image_id']
                    adj[a][b] = math.dist(pos[a], pos[b])
        house = Path(scans_dir) / scan / 'house_segmentations' / f'{scan}.house' if scans_dir else None
        floor_of, region_of, rid_of = _floors_from_house(house, pos, heights) if house and house.is_file() else ({}, {}, {})
        if not floor_of:
            floor_of = _floors_from_heights(pos, heights)
        return cls(scan, pos, adj, floor_of, region_of, heights, rot, rid_of)

    def _shortest(self, src):
        if src not in self._sp:
            dist, prev, heap = {src: 0.0}, {src: None}, [(0.0, src)]
            while heap:
                d, u = heapq.heappop(heap)
                if d > dist[u]:
                    continue
                for v, w in self.adj.get(u, {}).items():
                    if d + w < dist.get(v, math.inf):
                        dist[v], prev[v] = d + w, u
                        heapq.heappush(heap, (d + w, v))
            self._sp[src] = (dist, prev)
        return self._sp[src]

    def dist(self, a, b):
        return self._shortest(a)[0].get(b, math.inf)

    def path(self, a, b):
        """Shortest path a -> b, excluding a (DUET's FloydGraph.path convention)."""
        dist, prev = self._shortest(a)
        if b not in dist:
            return [b]
        out = []
        while b != a:
            out.append(b)
            b = prev[b]
        return out[::-1]

    def euclid(self, a, b):
        return math.dist(self.pos[a], self.pos[b])

    def heading(self, a, b):
        """MatterSim heading (0 = +y, clockwise towards +x) of the direction a -> b."""
        dx, dy = self.pos[b][0] - self.pos[a][0], self.pos[b][1] - self.pos[a][1]
        return math.atan2(dx, dy) % (2 * math.pi)

    def sim_heading(self, h):
        """The heading DUET's simulator takes when an episode starts at heading h: MatterSim with discretised
        viewing angles rounds it to the nearest 30 degrees (Simulator::setHeadingElevation)."""
        return math.floor((h % (2 * math.pi)) / VIEW_STEP + 0.5) % 12 * VIEW_STEP

    def arrival_pose(self, a, b):
        """(heading, elevation) of DUET's camera after the hop a -> b, or None if the view is unknown: the
        centre of the view in which a sees b (agent.make_equiv_action)."""
        ix = getattr(self, 'view_indices', {}).get(a, {}).get(b)
        return None if ix is None else ((ix % 12) * VIEW_STEP, (ix // 12 - 1) * VIEW_STEP)

    def to_json(self):
        floor_z = {}
        for vp, (x, y, z) in self.pos.items():
            floor_z.setdefault(self.floor_of.get(vp, 0), []).append(z - self.heights[vp])
        return {
            'scan': self.scan,
            'nodes': {vp: {'pos': [round(c, 3) for c in p], 'h': round(self.heights[vp], 3),
                           'floor': self.floor_of.get(vp, 0),
                           'region': self.region_of.get(vp), 'rid': self.rid_of.get(vp),
                           'rot': [round(c, 5) for c in self.rot.get(vp, [])]} for vp, p in self.pos.items()},
            'edges': sorted({tuple(sorted((a, b))) for a in self.adj for b in self.adj[a]}),
            'floors': [{'id': f, 'z': round(sorted(z)[len(z) // 2], 2)} for f, z in sorted(floor_z.items())],
        }


def _floors_from_house(house, pos, heights):
    """Floor and room label per viewpoint from the scan's .house file (panorama -> region -> level)."""
    region_level, region_label, pano_region = {}, {}, {}
    for line in Path(house).read_text(encoding='utf-8').splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == 'R':
            region_level[tok[1]] = int(tok[2])
            region_label[tok[1]] = REGION_LABELS.get(tok[5])
        elif tok[0] == 'P':
            pano_region[tok[1]] = tok[3]
    levels = {vp: region_level[pano_region[vp]] for vp in pos if pano_region.get(vp) in region_level}
    if not levels:
        return {}, {}, {}
    # Order levels bottom-up by floor height, then give unassigned viewpoints the nearest level by height.
    floor_z = {}
    for vp, lv in levels.items():
        floor_z.setdefault(lv, []).append(pos[vp][2] - heights[vp])
    order = sorted(floor_z, key=lambda lv: sorted(floor_z[lv])[len(floor_z[lv]) // 2])
    rank = {lv: i for i, lv in enumerate(order)}
    median = {rank[lv]: sorted(z)[len(z) // 2] for lv, z in floor_z.items()}
    floor_of = {}
    for vp in pos:
        if vp in levels:
            floor_of[vp] = rank[levels[vp]]
        else:
            z = pos[vp][2] - heights[vp]
            floor_of[vp] = min(median, key=lambda f: abs(median[f] - z))
    region_of = {vp: region_label.get(pano_region.get(vp)) for vp in pos}
    rid_of = {vp: int(pano_region[vp]) for vp in pos if pano_region.get(vp) in region_level}
    return floor_of, region_of, rid_of


def _floors_from_heights(pos, heights, window=0.3, spacing=1.5):
    """Fallback floor split by floor height (camera z minus camera height).

    Stairs viewpoints sit between floors, so plain gap clustering chains floors together; instead pick
    dense height levels greedily (>= 5% of viewpoints within +-`window` m, levels >= `spacing` m apart)
    and assign each viewpoint to the nearest level.
    """
    z = {vp: pos[vp][2] - heights[vp] for vp in pos}
    vals = sorted(z.values())
    levels, min_count = [], max(3, 0.05 * len(vals))
    while True:
        free = [v for v in vals if all(abs(v - lv) >= spacing for lv in levels)]
        if not free:
            break
        best = max(free, key=lambda v: sum(1 for u in free if abs(u - v) <= window))
        if levels and sum(1 for u in free if abs(u - best) <= window) < min_count:
            break
        levels.append(sum(u for u in free if abs(u - best) <= window) / sum(1 for u in free if abs(u - best) <= window))
    levels.sort()
    return {vp: min(range(len(levels)), key=lambda i: abs(levels[i] - z[vp])) for vp in pos}


class GraphCache:
    def __init__(self, conn_dir, scans_dir=None, view_indices=None):
        self.conn_dir, self.scans_dir, self._graphs = Path(conn_dir), scans_dir, {}
        self.view_indices = view_indices or {}

    def __getitem__(self, scan):
        if not isinstance(scan, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', scan):
            raise ValueError('Native scan IDs must contain only letters, digits, underscores and hyphens')
        if scan not in self._graphs:
            self._graphs[scan] = ScanGraph.load(self.conn_dir, scan, self.scans_dir)
            self._graphs[scan].view_indices = self.view_indices.get(scan, {})
        return self._graphs[scan]


# ----------------------------------------------------------------------------------------- inputs

def _read_json_or_jsonl(path):
    text = gzip.decompress(Path(path).read_bytes()).decode('utf-8') if str(path).endswith('.gz') else Path(path).read_text(encoding='utf-8')
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_annotations(path, bbox_file=None):
    """instr_id -> episode ground truth. Handles R2R(-style), REVERIE (object goals) and RxR files."""
    items = _read_json_or_jsonl(path)
    obj2vps = None
    if any('objId' in it for it in items[:50]):
        bbox_file = bbox_file or Path(path).parent / 'BBoxes.json'
        if Path(bbox_file).is_file():
            obj2vps = {}
            for scanvp, objs in json.loads(Path(bbox_file).read_text(encoding='utf-8')).items():
                scan, vp = scanvp.split('_')
                for objid, info in objs.items():
                    if info['visible_pos']:
                        obj2vps.setdefault(f'{scan}_{objid}', []).append(vp)
    gt = {}
    for it in items:
        base = {'scan': it['scan'], 'path': it.get('path', []), 'heading': float(it.get('heading', 0.0))}
        if it.get('objId') is not None:
            base['obj_id'] = str(it['objId'])
            if obj2vps is not None:
                base['goal_vps'] = obj2vps.get(f"{it['scan']}_{it['objId']}", [])
        if 'instruction_id' in it:                       # RxR: one instruction per item
            gt[str(it['instruction_id'])] = dict(base, instruction=it['instruction'])
            continue
        key = it.get('id', it.get('path_id'))
        for j, instr in enumerate(it.get('instructions', [])):
            gt[f'{key}_{j}'] = dict(base, instruction=instr)
    return gt


def load_predictions(path):
    preds = _read_json_or_jsonl(path)
    if isinstance(preds, dict):
        preds = [dict(v, instr_id=k) if isinstance(v, dict) else {'instr_id': k, 'trajectory': v}
                 for k, v in preds.items()]
    return {str(p['instr_id']): p for p in preds}


class StepLog:
    """Lazily read per-episode records ({'tokens', 'steps'}) from log_duet_steps.py's JSONL by byte offset."""

    _ID = re.compile(rb'^\{"instr_id": "([^"]+)"')

    def __init__(self, path):
        self.path, self.offsets = str(path), {}
        with open(path, 'rb') as f:
            pos = 0
            for line in f:
                m = self._ID.match(line)
                key = m.group(1).decode() if m else (str(json.loads(line)['instr_id']) if line.strip() else None)
                if key is not None:
                    self.offsets[key] = pos
                pos += len(line)

    def get(self, instr_id):
        if instr_id not in self.offsets:
            return None
        with open(self.path, 'rb') as f:
            f.seek(self.offsets[instr_id])
            return json.loads(f.readline())

    def __len__(self):
        return len(self.offsets)

    def __contains__(self, instr_id):
        return instr_id in self.offsets


def load_steplog(path):
    return StepLog(path)


def load_subtasks(path):
    """FGR2R file -> {instr_id: {'instruction', 'subs': [[token, ...], ...], 'chunks': [[s, e], ...] | None}}.
    rxr_subtasks.py output is already keyed by instruction id, with character 'spans' instead of 'subs'."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(data, dict):
        return {k: dict(v, source='RxR guide timing') for k, v in data.items()}
    out = {}
    for it in data:
        subs = it['new_instructions']
        subs = ast.literal_eval(subs) if isinstance(subs, str) else subs
        chunks = it.get('chunk_view')
        for j, instr in enumerate(it['instructions'][:len(subs)]):
            out[f"{it['path_id']}_{j}"] = {'instruction': instr, 'subs': subs[j],
                                           'chunks': chunks[j] if chunks and j < len(chunks) else None}
    return out


def default_subtasks_for(anno_file):
    """No implicit subtask discovery; manifests supply an optional subtask file."""
    return None


def default_annotation_for(pred_path):
    raise ValueError('The r2r adapter requires an explicit annotations file')


# ------------------------------------------------------------------------------------- trajectories

def _is_vp(x):
    return isinstance(x, str)


def decisions_from_trajectory(traj, graph):
    """Normalise a trajectory into decisions: [{'at', 'action', 'target', 'route', 'heading'}].

    'at' is where the decision is taken, 'route' the viewpoints walked to reach 'target' (excluding
    'at'). The last decision is always 'stop'; its route is non-empty when DUET returns to the visited
    node with the highest stop score.
    """
    if not traj:
        return []
    duet = all(isinstance(seg, list) and seg and all(_is_vp(v) for v in seg) for seg in traj)
    if duet:
        start, moves, headings = traj[0][0], [list(seg) for seg in traj[1:]], None
    else:
        # [vp, heading, elevation] per simulator step; collapse turns in place.
        hops, headings = [], []
        for entry in traj:
            vp = entry[0] if isinstance(entry, (list, tuple)) else entry
            h = float(entry[1]) if isinstance(entry, (list, tuple)) and len(entry) > 1 else None
            if hops and hops[-1] == vp:
                headings[-1] = h if h is not None else headings[-1]
            else:
                hops.append(vp)
                headings.append(h)
        start = hops[0]
        moves = []
        for a, b in zip(hops, hops[1:]):
            moves.append([b] if b in graph.adj.get(a, {}) else graph.path(a, b))

    decisions, visited, cur = [], {start}, start
    for k, route in enumerate(moves):
        target = route[-1]
        if duet and target in visited and k == len(moves) - 1:
            # DUET never moves to a visited node: a final segment ending on one is the stop-score return.
            decisions.append({'at': cur, 'action': 'stop', 'target': target, 'route': route})
            cur = target
            break
        kind = 'local' if len(route) == 1 else 'remote'
        decisions.append({'at': cur, 'action': kind, 'target': target, 'route': route,
                          'revisit': target in visited})
        visited.add(target)
        cur = target
    else:
        decisions.append({'at': cur, 'action': 'stop', 'target': cur, 'route': []})
    if headings is not None:
        heading_at = dict(zip(hops, headings))
        for d in decisions:
            d['heading'] = heading_at.get(d['target'])
    return decisions


def frontier(graph, visited):
    out = set()
    for v in visited:
        out.update(graph.adj.get(v, {}))
    return sorted(out - set(visited))


def review_decisions(decisions, gt_path, in_area, graph, route=False, near=None):
    """Mark each logged decision with the reference-best option and whether the model's choice matched it.

    Goal reference (R2R, REVERIE): STOP inside the success area, else the frontier node nearest the goal
    (the viewer's ★). Route reference (RxR, whose paths are deliberately not shortest paths): the frontier
    node nearest the next reference viewpoint not yet reached, and STOP once the route is done and the agent
    is inside the success area. A move matches when its distance is within REF_TOLERANCE of the best
    option's; inside the success area a move towards the goal also matches (DUET's teacher stops only at
    the goal viewpoint itself). Sets d['ref'] = {'best', 'ok', 'type', 'gap', 'conf', 'stop', 'in_area'}; stops forced by the
    step budget are not judged (ok None).

    near(vp, ref): whether being at vp counts as reaching reference viewpoint ref (default: vp == ref).
    Continuous runs pass a distance test and list the points walked since the previous decision in
    d['passed'].
    """
    goal, progress, seen = gt_path[-1], 0, set()
    reached = (lambda ref: ref in seen) if near is None else (lambda ref: any(near(vp, ref) for vp in seen))
    for d in decisions:
        seen.update(d.get('passed') or [d['at']])
        if route:
            hits = [j for j in range(progress + 1, len(gt_path))
                    if (d['at'] == gt_path[j] if near is None else near(d['at'], gt_path[j]))]
            if hits:
                progress = hits[0]
            while progress + 1 < len(gt_path) and reached(gt_path[progress + 1]):
                progress += 1
        md = d.get('model')
        if not md:
            continue
        done = not route or progress == len(gt_path) - 1
        target = goal if done else gt_path[progress + 1]
        probs = {vp: md['fused'][i + 1] for i, (vp, v) in enumerate(zip(md['gmap'], md['visited']))
                 if not v and i + 1 < len(md['fused'])}
        best = None if (done and in_area(d['at'])) or not probs else min(probs, key=lambda vp: graph.dist(vp, target))
        chosen = None if d['action'] == 'stop' else d.get('choice', d['target'])   # CE: the ghost picked
        ref = {'best': best, 'conf': md['fused'][0] if chosen is None else probs.get(chosen),
               'stop': md['fused'][0], 'in_area': in_area(d['at']), 'ok': None, 'type': None}
        if md.get('forced'):
            pass
        elif chosen is None or best is None:
            ref['ok'] = chosen == best or (chosen is not None and graph.dist(chosen, target) < graph.dist(d['at'], target))
            ref['type'] = None if ref['ok'] else 'stopped early' if chosen is None else 'missed stop'
        else:
            ref['gap'] = round(graph.dist(chosen, target) - graph.dist(best, target), 2)
            ref['ok'] = ref['gap'] <= REF_TOLERANCE
            ref['type'] = None if ref['ok'] else 'wrong move'
        d['ref'] = ref


def _dtw(graph, pred, ref, success):
    inf = math.inf
    m = [[inf] * (len(ref) + 1) for _ in range(len(pred) + 1)]
    m[0][0] = 0.0
    for i in range(1, len(pred) + 1):
        for j in range(1, len(ref) + 1):
            m[i][j] = graph.dist(ref[j - 1], pred[i - 1]) + min(m[i - 1][j], m[i][j - 1], m[i - 1][j - 1])
    ndtw = math.exp(-m[-1][-1] / (ERROR_MARGIN * len(ref)))
    return ndtw, ndtw * success


def _cls(graph, pred, ref):
    length = lambda nodes: sum(graph.dist(a, b) for a, b in zip(nodes, nodes[1:]))
    coverage = sum(math.exp(-min(graph.dist(u, v) for v in pred) / ERROR_MARGIN) for u in ref) / len(ref)
    expected = coverage * length(ref)
    return coverage * expected / (expected + abs(expected - length(pred))) if expected > 0 else 0.0


# ------------------------------------------------------------------------------ instruction text & sub-tasks

_WORD = re.compile(r"[A-Za-z0-9]+")


def _same_word(w, t):
    # FGR2R tokens are lemmatised ("chairs" -> "chair", "railing" -> "rail")
    return w == t or w.startswith(t) or t.startswith(w) or (len(w) > 3 and len(t) > 3 and w[:4] == t[:4])


def subtask_spans(text, subs):
    """Character spans [start, end) of each sub-instruction in `text`, or None if they do not align."""
    words = [(m.start(), m.group().lower()) for m in _WORD.finditer(text)]
    i, firsts, matched, total = 0, [], 0, 0
    for sub in subs:
        first = None
        for tok in sub:
            if not _WORD.fullmatch(tok):
                continue
            total += 1
            for j in range(i, min(i + 4, len(words))):
                if _same_word(words[j][1], tok.lower()):
                    first = j if first is None else first
                    i, matched = j + 1, matched + 1
                    break
        firsts.append(first)
    if not total or matched / total < 0.7 or any(f is None for f in firsts):
        return None
    starts = [0] + [words[f][0] for f in firsts[1:]]
    return [[s, e] for s, e in zip(starts, starts[1:] + [len(text)])]


def heuristic_spans(text):
    """Sentence-level split when no FGR2R annotation exists (no ground-truth alignment)."""
    bounds = [0] + [m.end() for m in re.finditer(r'[.!?;]+\s+', text) if m.end() < len(text.rstrip())]
    return [[s, e] for s, e in zip(bounds, bounds[1:] + [len(text)])]


def token_words(tokens, text):
    """Merge BERT word pieces into words and locate them in `text`: [(token_indices, [start, end] | None)]."""
    words = []
    for i, tok in enumerate(tokens):
        if tok in ('[CLS]', '[SEP]', '[PAD]'):
            continue
        if tok.startswith('##') and words:
            words[-1][0].append(i)
            words[-1][1] += tok[2:]
        else:
            words.append([[i], tok])
    lower, pos, out = text.lower(), 0, []
    for idx, w in words:
        j = lower.find(w, pos)
        if 0 <= j and j - pos <= 12:
            out.append((idx, [j, j + len(w)]))
            pos = j + len(w)
        else:
            out.append((idx, None))
    return out


def subtask_progress(frames, gt_path, chunks, graph, success):
    """Frame at which each sub-goal (end of its ground-truth chunk) is reached, in order; None = never.

    Intermediate sub-goals count when the agent comes within SUBGOAL_MARGIN of them; the last sub-task
    is the episode's own success criterion (stopping within 3 m).
    """
    if not chunks or not gt_path or max(e for _, e in chunks) > len(gt_path):
        return None
    reached, i = [], 0
    for k, (_, e) in enumerate(chunks):
        if k == len(chunks) - 1:
            reached.append(len(frames) - 1 if success else None)
            break
        goal = gt_path[e - 1]
        hit = next((j for j in range(i, len(frames)) if graph.dist(frames[j]['vp'], goal) < SUBGOAL_MARGIN), None)
        reached.append(hit)
        if hit is None:
            reached.extend([None] * (len(chunks) - k - 1))
            break
        i = hit
    return reached


def analyse_episode(pred, gt, graph, max_action_len=DUET_MAX_ACTION_LEN, steplog=None, full=True, subtasks=None,
                    route_reference=False, decisions=None, ref_near=None, success_for_subtasks=None):
    """Align one prediction with its ground truth and compute metrics, diagnostics and playback frames.

    route_reference: judge decisions against the annotated route rather than the goal (RxR; see
    review_decisions). decisions / ref_near: ready-made decisions and the reference test for continuous runs
    (ce_runs.py); by default decisions come from pred['trajectory']. success_for_subtasks lets continuous
    runs use their recorded success criterion for the final sub-task, rather than sampled graph positions."""
    if decisions is None:
        decisions = decisions_from_trajectory(pred['trajectory'], graph)
    stop_probs = {k: v.get('stop_prob') for k, v in (pred.get('details') or {}).items()}
    gt_path = gt.get('path') or []
    if len(gt_path) < 2:        # test split: only the start viewpoint is published
        gt_path = []
    goal = gt_path[-1] if gt_path else None
    goal_vps = set(gt.get('goal_vps') or ([goal] if goal else []))
    start = decisions[0]['at']

    # Hop-level path and frames. A frame shows the decision being executed (in transit) or, on arrival
    # at a decision node, the decision taken there. The camera pose is the simulator's where known (logged
    # headings; DUET's 30-degree views, see ScanGraph.sim_heading / arrival_pose), else the direction of travel.
    path, frames, visits = [start], [], Counter({start: 1})
    heading, elevation = gt.get('heading', 0.0), 0.0
    if hasattr(graph, 'sim_heading'):
        heading = graph.sim_heading(heading)

    def frame(vp, show, kind, hop, nhops, prev):
        nonlocal heading, elevation
        if prev is not None and prev != vp:
            heading, elevation = graph.heading(prev, vp), 0.0
        f = {'vp': vp, 'decision': show, 'kind': kind, 'hop': hop, 'nhops': nhops,
             'heading': round(heading, 4), 'visits': visits[vp], 'floor': graph.floor_of.get(vp, 0)}
        if elevation:
            f['elevation'] = round(elevation, 4)
        if goal:
            f['dist'] = round(graph.dist(vp, goal), 3)
            f['euclid'] = round(graph.euclid(vp, goal), 3)
            f['gt_dist'] = round(min(graph.dist(vp, g) for g in gt_path), 3)
        frames.append(f)

    frame(start, 0, 'start', 0, 0, None)
    for t, d in enumerate(decisions):
        for k, vp in enumerate(d['route']):
            prev = path[-1]
            path.append(vp)
            visits[vp] += 1
            last = k == len(d['route']) - 1
            show = t + 1 if last and t + 1 < len(decisions) else t
            frame(vp, show, 'return' if d['action'] == 'stop' else d['action'], k + 1, len(d['route']), prev)
            pose = None
            if last and d.get('heading') is not None:
                pose = (d['heading'], 0.0)
            elif last and d['action'] != 'stop' and hasattr(graph, 'arrival_pose'):
                pose = graph.arrival_pose(prev, vp)
            if pose:
                heading, elevation = pose
                frames[-1]['heading'] = round(heading, 4)
                if elevation:
                    frames[-1]['elevation'] = round(elevation, 4)

    # Graph-memory state per decision (reconstructed; exact model scores come from the step log).
    visited = []
    for t, d in enumerate(decisions):
        if d['at'] not in visited:
            visited.append(d['at'])
        d['t'] = t
        d['visited'] = list(visited)
        d['frontier'] = frontier(graph, visited)
        d['stop_prob'] = stop_probs.get(d['at'])
    log_steps = (steplog or {}).get('steps') or []
    for d, s in zip(decisions, log_steps):
        if s.get('vp') == d['at']:
            d['model'] = s
    steplog_ok = bool(log_steps) and len(log_steps) == len(decisions) and all('model' in d for d in decisions)

    final = path[-1]
    moves = sum(1 for d in decisions if d['action'] != 'stop')
    tl = sum(graph.dist(a, b) for a, b in zip(path, path[1:]))
    ep = {'instr_id': pred['instr_id'], 'scan': gt['scan'], 'instruction': gt.get('instruction', ''),
          'gt_path': gt_path, 'goal': goal, 'start': start, 'final': final, 'path': path,
          'max_action_len': max_action_len}
    if gt.get('goal_vps') is not None:
        ep['goal_vps'] = sorted(goal_vps)
    m = {'tl': round(tl, 3), 'hops': len(path) - 1, 'moves': moves, 'decisions': len(decisions)}
    diag = {}
    if goal:
        ne = graph.dist(final, goal)
        success = float(final in goal_vps) if gt.get('goal_vps') is not None else float(ne < ERROR_MARGIN)
        gt_len = sum(graph.dist(a, b) for a, b in zip(gt_path, gt_path[1:]))
        near = [vp for vp in path if (vp in goal_vps if gt.get('goal_vps') is not None
                                      else graph.dist(vp, goal) < ERROR_MARGIN)]
        ndtw, sdtw = _dtw(graph, path, gt_path, success)
        m.update(ne=round(ne, 3), euclid_ne=round(graph.euclid(final, goal), 3), sr=success,
                 osr=float(bool(near)), spl=round(success * gt_len / max(tl, gt_len, 0.01), 4),
                 ndtw=round(ndtw, 4), sdtw=round(sdtw, 4), cls=round(_cls(graph, path, gt_path), 4),
                 gt_len=round(gt_len, 3), gt_hops=len(gt_path) - 1)

        step_limit = len(decisions) >= max_action_len
        ep['outcome'] = 'success' if success else ('step_limit' if step_limit else 'wrong_stop')
        if success:
            mode = 'success'
        elif near:
            best_near = max((stop_probs.get(vp) or 0.0) for vp in near)
            mode = ('passed goal: stop score preferred a wrong node'
                    if stop_probs and (stop_probs.get(final) or 0.0) > best_near else 'passed goal: did not stop there')
        else:
            mode = 'never near goal: ran out of steps' if step_limit else 'never near goal: stopped elsewhere'
        ep['failure_mode'] = mode

        gt_set = set(gt_path)
        off = next((i for i, vp in enumerate(path) if vp not in gt_set), None)
        lost = next((i for i, vp in enumerate(path) if min(graph.dist(vp, g) for g in gt_path) > ERROR_MARGIN), None)
        closest = min(range(len(path)), key=lambda i: graph.dist(path[i], goal))
        diag.update(first_off_gt_hop=off, lost_hop=lost, closest_hop=closest,
                    closest_dist=round(graph.dist(path[closest], goal), 3), near_goal_hops=len(near),
                    gt_floors=len({graph.floor_of.get(vp, 0) for vp in gt_path}),
                    stop_place=None if success or not graph.region_of else stop_place(graph, final, goal))
    else:
        ep['outcome'] = ep['failure_mode'] = 'no ground truth'

    revisited = {vp: n for vp, n in visits.items() if n >= 2}
    # Backtracking: hops onto viewpoints the agent had already been at.
    seen, back = {start}, []
    for a, b in zip(path, path[1:]):
        if b in seen:
            back.append((a, b))
        seen.add(b)
    oscillations = sum(1 for a, b, c in zip(path, path[1:], path[2:]) if a == c)
    diag.update(revisited_nodes=len(revisited), max_visits=max(visits.values()),
                revisit_events=sum(n - 1 for n in visits.values()),
                remote_jumps=sum(1 for d in decisions if d['action'] == 'remote'),
                backtrack_hops=len(back), backtrack_m=round(sum(graph.dist(a, b) for a, b in back), 3),
                stop_return=bool(decisions[-1]['route']), oscillations=oscillations,
                floors_visited=len({graph.floor_of.get(vp, 0) for vp in path}),
                steplog=steplog_ok)

    # Sub-tasks: FGR2R sub-instructions (or RxR steps aligned by guide timing) when they match this
    # instruction, else sentences without a path alignment.
    text, sub = ep['instruction'], subtasks or {}
    spans = None
    if sub.get('instruction', '').split() == text.split():
        spans = sub.get('spans') or (subtask_spans(text, sub['subs']) if sub.get('subs') else None)
    source = sub.get('source', 'FGR2R') if spans else 'sentences'
    spans = spans or heuristic_spans(text)
    chunks = sub.get('chunks') if source != 'sentences' else None
    completion_success = bool(m.get('sr')) if success_for_subtasks is None else bool(success_for_subtasks)
    reached = subtask_progress(frames, gt_path, chunks, graph, completion_success) if goal else None
    if reached is not None:
        missed = next((k for k, r in enumerate(reached) if r is None), None)
        diag.update(subtasks=len(reached), subtasks_done=sum(r is not None for r in reached),
                    first_missed_subtask=None if missed is None else missed + 1)
        for i, f in enumerate(frames):
            f['subtask'] = min(sum(1 for r in reached if r is not None and r <= i), len(reached) - 1)
    if goal and steplog_ok:
        in_area = (lambda vp: vp in goal_vps) if gt.get('goal_vps') is not None else \
            (lambda vp: graph.dist(vp, goal) < ERROR_MARGIN)
        review_decisions(decisions, gt_path, in_area, graph, route_reference, ref_near)
        first_frame = {}
        for i, f in enumerate(frames):
            first_frame.setdefault(f['decision'], i)
        judged = [d for d in decisions if d['ref']['ok'] is not None]
        first = next((dict(d['ref'], t=d['t']) for d in judged if not d['ref']['ok']), None)
        diag['nonbest'] = sum(not d['ref']['ok'] for d in judged)
        if first:
            sub = frames[first_frame[first['t']]].get('subtask')
            diag.update(first_nonbest=first['t'], first_nonbest_type=first['type'],
                        first_nonbest_conf=None if first['conf'] is None else round(first['conf'], 4),
                        first_nonbest_subtask=None if sub is None else sub + 1)
        # one row per decision for the split charts:
        # [t, probability of the chosen option, matched the reference (1/0, None if forced), stop probability,
        #  inside the success area (1/0)]
        r4 = lambda x: None if x is None else round(x, 4)
        ep['review'] = [[d['t'], r4(d['ref']['conf']), None if d['ref']['ok'] is None else int(d['ref']['ok']),
                         r4(d['ref']['stop']), int(d['ref']['in_area'])] for d in decisions]
    diag['instr_words'] = len(ep['instruction'].split())
    ep['metrics'], ep['diag'] = m, diag
    if full:
        ep['decisions'], ep['frames'], ep['visits'] = decisions, frames, dict(visits)
        ep['stop_probs'] = stop_probs
        ep['start_heading'] = gt.get('heading', 0.0)
        ep['subtasks'] = {'source': source, 'spans': spans, 'chunks': chunks if reached is not None else None,
                          'reached': reached,
                          'goals': [gt_path[e - 1] for _, e in chunks] if reached is not None else None}
        if steplog and steplog.get('tokens'):
            # Attention per instruction word and per sub-task ("which part the model is reading now").
            words = token_words(steplog['tokens'], text)
            ep['words'] = [{'span': span, 'text': ''.join(steplog['tokens'][i].lstrip('#') for i in idx)}
                           for idx, span in words]
            for d in decisions:
                md = d.get('model')
                if not md or not md.get('attn'):
                    continue
                a = md.pop('attn')
                w = [sum(a[i] for i in idx if i < len(a)) for idx, _ in words]
                total = sum(w) or 1.0
                md['wattn'] = [round(x / total, 4) for x in w]
                focus = [0.0] * len(spans)
                for (_, span), x in zip(words, md['wattn']):
                    if span:
                        k = next((k for k, (s, e) in enumerate(spans) if s <= span[0] < e), len(spans) - 1)
                        focus[k] += x
                d['focus'] = [round(x, 4) for x in focus]
    return ep


def stop_place(graph, final, goal):
    """Where a failed episode ended relative to the goal: its room, another room of the same type, another
    room, or another floor (Matterport3D region annotations)."""
    if graph.floor_of.get(final) != graph.floor_of.get(goal):
        return 'wrong floor'
    if graph.rid_of.get(final) is not None and graph.rid_of.get(final) == graph.rid_of.get(goal):
        return 'goal room'
    if graph.region_of.get(final) and graph.region_of.get(final) == graph.region_of.get(goal):
        return 'same room type'
    return 'other room'


def summary_row(ep):
    """Flat record for tables and CSV export."""
    row = {'instr_id': ep['instr_id'], 'scan': ep['scan'], 'outcome': ep['outcome'],
           'failure_mode': ep['failure_mode']}
    row.update(ep['metrics'])
    row.update({k: v for k, v in ep['diag'].items()})
    row['instruction'] = ep['instruction']
    return row


class Run:
    """One model's predictions aligned with ground truth."""

    def __init__(self, name, pred_file, anno_file=None, conn_dir=None, scans_dir=SCANS_DIR, steplog_file=None,
                 max_action_len=DUET_MAX_ACTION_LEN, graphs=None, annotations=None, subtasks=None,
                 model=None, split=None, cache_dir=None, cache_inputs=None, route_reference=None):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_inputs = list(cache_inputs or [])
        self.name, self.pred_file = name, str(pred_file)
        self.model, self.split = model or name, split or re.sub(r'^(detail|submit)_', '', Path(pred_file).stem)
        self.anno_file = str(anno_file or default_annotation_for(pred_file))
        if conn_dir is None and graphs is None:
            raise ValueError('The r2r adapter requires an explicit connectivity directory')
        self.graphs = graphs if graphs is not None else GraphCache(conn_dir, scans_dir)
        self.gt = annotations if annotations is not None else load_annotations(self.anno_file)
        if subtasks is None:
            f = default_subtasks_for(self.anno_file)
            subtasks = load_subtasks(f) if f else {}
        self.subtasks = subtasks
        self.preds = load_predictions(pred_file)
        self.steplog = load_steplog(steplog_file) if steplog_file else {}
        self.steplog_file = str(steplog_file) if steplog_file else None
        self.max_action_len = max_action_len
        missing = [k for k in self.preds if k not in self.gt]
        if missing:
            print(f'[{name}] {len(missing)} predictions have no ground truth in {self.anno_file}, '
                  f'e.g. {missing[0]}; skipped', file=sys.stderr)
        self.ids = [k for k in self.preds if k in self.gt]
        # RxR paths are not shortest paths: judge decisions against the annotated route instead of the goal
        self.route_reference = Path(self.anno_file).name.startswith('rxr_') if route_reference is None else bool(route_reference)
        self._rows = self._points = None

    def episode(self, instr_id, full=True):
        gt = self.gt[instr_id]
        ep = analyse_episode(self.preds[instr_id], gt, self.graphs[gt['scan']], self.max_action_len,
                             self.steplog.get(instr_id), full=full, subtasks=self.subtasks.get(instr_id),
                             route_reference=self.route_reference)
        if not full:
            ep['diag']['steplog'] = instr_id in self.steplog
        return ep

    def rows(self):
        if self._rows is None:
            cache = self.cache_dir / (hashlib.sha256(self.name.encode()).hexdigest()[:24] + '.json') if self.cache_dir else None
            key = self._cache_key()
            try:
                c = json.loads(cache.read_text(encoding='utf-8')) if cache else {}
                if c.get('key') == key:
                    self._rows, self._points = c['rows'], c['points']
                    return self._rows
            except (OSError, ValueError):
                pass
            eps = [self.episode(i, full=False) for i in self.ids]
            self._points = {ep['instr_id']: ep['review'] for ep in eps if 'review' in ep}
            self._rows = [summary_row(ep) for ep in eps]
            try:
                if cache:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(json.dumps({'key': key, 'rows': self._rows, 'points': self._points}), encoding='utf-8')
            except OSError:
                pass
        return self._rows

    def _cache_key(self):
        """Everything the rows depend on: the input files and this module (the analysis), by size and mtime."""
        files = [self.pred_file, self.steplog_file, self.anno_file, __file__, *self.cache_inputs]
        return [file_stamp(f) for f in files] + [self.max_action_len, self.route_reference, len(self.ids),
                                               getattr(self, 'cache_context', {})]

    def points(self):
        """Per-decision review rows by episode (see analyse_episode's ep['review']); empty without a step log."""
        self.rows()
        return self._points


def file_stamp(path):
    """[path, size, mtime_ns] of a file, or [path] when there is none (cache keys)."""
    try:
        s = Path(path).stat()
        return [str(path), s.st_size, s.st_mtime_ns]
    except (OSError, TypeError):
        return [str(path)]


def aggregate(rows):
    """Aggregate each available metric over its own finite observations.

    Missing values stay absent; metric_counts makes differing denominators explicit.
    """
    result = {'n': len(rows), 'metric_counts': {},
              'outcomes': dict(Counter(r.get('outcome', 'unscored') for r in rows)),
              'failure_modes': dict(Counter(r.get('failure_mode', 'unknown') for r in rows))}
    percentages = {'sr', 'osr', 'spl', 'ndtw', 'sdtw'}
    for key in ('sr', 'osr', 'spl', 'ndtw', 'sdtw', 'ne', 'tl'):
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and math.isfinite(r[key])]
        if values:
            result[key] = round(sum(values) / len(values) * (100 if key in percentages else 1), 2)
            result['metric_counts'][key] = len(values)
    return result


def filter_rows(rows, failed=False, outcome=None, failure_mode=None, min_length=None, min_revisits=None,
                min_decisions=None, scan=None, ids=None):
    out = []
    for r in rows:
        if failed and r.get('sr', 0) == 1:
            continue
        if outcome and r['outcome'] not in outcome:
            continue
        if failure_mode and failure_mode.lower() not in r['failure_mode'].lower():
            continue
        if min_length is not None and r['tl'] <= min_length:
            continue
        if min_revisits is not None and r['revisited_nodes'] < min_revisits:
            continue
        if min_decisions is not None and r['decisions'] < min_decisions:
            continue
        if scan and r['scan'] != scan:
            continue
        if ids is not None and r['instr_id'] not in ids:
            continue
        out.append(r)
    return out


def write_export(run, rows, out, fmt=None, full=False):
    """Write rows as CSV, or JSON/JSONL episode records (with frames and decisions when full=True)."""
    out = Path(out)
    fmt = fmt or out.suffix.lstrip('.') or 'json'
    if fmt == 'csv':
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        return out
    # Full episode reconstruction contains trajectory evidence, but annotations added by the caller
    # (the browser's notes/tags) live on the summary rows. Preserve only those review fields so stale
    # summaries cannot overwrite the reconstructed metrics, decisions or instruction.
    records = [dict(run.episode(r['instr_id'], full=True),
                    **{k: r[k] for k in ('tags', 'note') if k in r}) if full else r for r in rows]
    with open(out, 'w', encoding='utf-8') as f:
        if fmt == 'jsonl':
            for rec in records:
                f.write(json.dumps(rec) + '\n')
        else:
            json.dump({'model': run.name, 'predictions': run.pred_file, 'annotations': run.anno_file,
                       'summary': aggregate(rows), 'episodes': records}, f, indent=1)
    return out


# ---------------------------------------------------------------------------------------------- CLI

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['summary', 'export'])
    parser.add_argument('--preds', required=True, help='prediction JSON (detail_<split>.json or submit_<split>.json)')
    parser.add_argument('--anno', required=True, help='annotations JSON/JSONL')
    parser.add_argument('--conn-dir', required=True, help='Matterport-format connectivity directory')
    parser.add_argument('--scans-dir', help='Matterport scans (for .house floors/rooms)')
    parser.add_argument('--steplog', help='per-step DUET log from log_duet_steps.py (JSONL)')
    parser.add_argument('--max-action-len', type=int, default=DUET_MAX_ACTION_LEN,
                        help='decision budget; episodes that use all of it count as "step_limit" (DUET: 15)')
    f = parser.add_argument_group('export filters (combined with AND)')
    f.add_argument('--failed', action='store_true', help='only unsuccessful episodes')
    f.add_argument('--outcome', nargs='+', choices=['success', 'wrong_stop', 'step_limit'])
    f.add_argument('--failure-mode', help='substring, e.g. "passed goal"')
    f.add_argument('--min-length', type=float, help='trajectory length strictly above this many metres')
    f.add_argument('--min-revisits', type=int, help='at least this many viewpoints visited twice or more')
    f.add_argument('--min-decisions', type=int)
    f.add_argument('--scan')
    parser.add_argument('--out', help='export path (.csv, .json or .jsonl)')
    parser.add_argument('--full', action='store_true', help='JSON/JSONL: include per-hop frames and decisions')
    args = parser.parse_args()

    run = Run(Path(args.preds).stem, args.preds, args.anno, args.conn_dir, args.scans_dir, args.steplog,
              args.max_action_len)
    rows = run.rows()
    if args.command == 'summary':
        s = aggregate(rows)
        print(f"{run.pred_file}\n  vs {run.anno_file}\n  {s['n']} episodes")
        if 'sr' in s:
            print('  SR %.2f  SPL %.2f  OSR %.2f  NE %.2f  TL %.2f  nDTW %.2f  SDTW %.2f'
                  % (s['sr'], s['spl'], s['osr'], s['ne'], s['tl'], s['ndtw'], s['sdtw']))
            print('  outcome:')
            for k, v in Counter(r['outcome'] for r in rows).most_common():
                print('    %-50s %5d  %5.1f%%' % (k, v, 100 * v / len(rows)))
            print('  failure mode:')
            for k, v in Counter(r['failure_mode'] for r in rows).most_common():
                print('    %-50s %5d  %5.1f%%' % (k, v, 100 * v / len(rows)))
            for key, label in [('revisited_nodes', 'episodes revisiting a viewpoint'),
                               ('remote_jumps', 'episodes with a remote-frontier jump'),
                               ('stop_return', 'episodes returning to an earlier node to stop')]:
                n = sum(1 for r in rows if r[key])
                print('  %-52s %5d  %5.1f%%' % (label, n, 100 * n / len(rows)))
        return
    if not args.out:
        parser.error('export needs --out')
    sel = filter_rows(rows, args.failed, args.outcome, args.failure_mode, args.min_length, args.min_revisits,
                      args.min_decisions, args.scan)
    path = write_export(run, sel, args.out, full=args.full)
    print(f'{len(sel)} of {len(rows)} episodes -> {path}')


if __name__ == '__main__':
    main()
