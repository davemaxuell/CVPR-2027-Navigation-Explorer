"""Local HTTP viewer and agent API for explicitly configured navigation recordings."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import mimetypes
import os
import re
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlencode, urlparse

from . import investigations
from . import parse_trajectories as pt
from .configuration import load_project, read_json, resolve_path

STATIC = Path(__file__).parent / 'static'
SAFE_SCENE = re.compile(r'^[A-Za-z0-9_-]{1,128}$')


def json_safe(value):
    """Unavailable legacy distances become null, never nonstandard JSON Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


class Notes:
    def __init__(self, path):
        self.path, self.lock = Path(path), threading.Lock()
        self.data = read_json(self.path) if self.path.is_file() else {}
        if not isinstance(self.data, dict):
            raise ValueError(f'{self.path}: notes must be an object')

    def for_run(self, run):
        with self.lock:
            tags = {}
            for entries in self.data.values():
                for entry in entries.values():
                    for tag in entry.get('tags', []):
                        tags[tag] = tags.get(tag, 0) + 1
            return {'notes': copy.deepcopy(self.data.get(run, {})),
                    'vocabulary': sorted(tags, key=lambda tag: (-tags[tag], tag.lower()))}

    def save(self, run, instr_id, tags, note):
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError('tags must be a list of strings')
        if note is not None and not isinstance(note, str):
            raise ValueError('note must be a string')
        tags = list(dict.fromkeys(tag.strip()[:40] for tag in tags if tag.strip()))[:20]
        note = (note or '')[:4000]
        with self.lock:
            updated = copy.deepcopy(self.data)
            entries = updated.setdefault(run, {})
            if tags or note.strip():
                entries[instr_id] = {'tags': tags, 'note': note}
            else:
                entries.pop(instr_id, None)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix('.tmp')
            tmp.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding='utf-8')
            os.replace(tmp, self.path)
            self.data = updated
            return copy.deepcopy(entries.get(instr_id, {'tags': [], 'note': ''}))


class RecordedOnly:
    """Replay does not assume access to a model, simulator, or inference worker."""
    def status(self):
        return {'allowed': False, 'enabled': False, 'state': 'off', 'job': None, 'queued': [],
                'reason': 'This server reviews recordings; no inference runner is configured.'}

    def for_episode(self, run, instr_id):
        return {}

    def summary(self, run):
        return {'episodes': {}, 'branches': 0, 'total': 0, 'verification': []}


def compatible(a, b):
    if a.name == b.name:
        return False
    coordinates = getattr(a, 'coordinate_system', None)
    if not coordinates or coordinates != getattr(b, 'coordinate_system', None):
        return False
    group = getattr(a, 'comparison_group', None)
    other_group = getattr(b, 'comparison_group', None)
    if group or other_group:
        return bool(group and group == other_group)
    return (getattr(a, 'dataset', None) == getattr(b, 'dataset', None)
            and a.split == b.split and bool(getattr(a, 'ce', False)) == bool(getattr(b, 'ce', False)))


class App:
    def __init__(self, project, state_dir=None):
        self.project = load_project(project, state_dir)
        self.state_dir = self.project.state_dir
        self.runs, self.assets, self.generated_assets = {}, {}, {}
        self.mesh_lock = threading.Lock()
        self.mesh_building = {}
        from .legacy import load_legacy_run
        from .recordings import PortableRun
        for spec in self.project.runs:
            try:
                run = (PortableRun(spec, self.project.directory, self.state_dir)
                       if spec['adapter'] == 'recording' else
                       load_legacy_run(spec, self.project.directory, self.state_dir))
                run.dataset = spec['dataset']
                run.model_label = spec.get('model_label', spec['model'])
                run.dataset_description = spec.get('dataset_description', '')
                run.comparison_group = spec.get('comparison_group')
                run.coordinate_system = spec.get('coordinate_system', getattr(run, 'coordinate_system', 'matterport_z_up_m'))
                run.adapter = spec['adapter']
                self.runs[spec['id']] = run
                key = hashlib.sha256((spec['id'] + str(getattr(run, 'scans_dir', ''))).encode()).hexdigest()[:16]
                cache = self.state_dir / 'cache' / 'assets' / key
                configured = spec.get('assets', {})
                self.generated_assets[spec['id']] = {kind: cache / kind for kind in ('meshes', 'meshes_hd', 'navmaps')}
                self.assets[spec['id']] = {kind: resolve_path(configured[kind], self.project.directory)
                                          if kind in configured else cache / kind
                                          for kind in ('meshes', 'meshes_hd', 'navmaps')}
                print(f'  {spec["id"]}: {len(run.rows())} episodes ({spec["environment"]}, {spec["adapter"]})', flush=True)
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ValueError(f'Run {spec["id"]!r}: {error}') from error
        self.notes = Notes(self.state_dir / 'episode_notes.json')
        self.whatif = RecordedOnly()

    def meta(self):
        result = []
        for name, run in self.runs.items():
            assets = self.assets[name]
            result.append({'name': name, 'model': run.model, 'model_label': run.model_label,
                           'dataset': run.dataset, 'dataset_description': run.dataset_description,
                           'split': run.split, 'env': 'continuous' if getattr(run, 'ce', False) else 'discrete',
                           'adapter': run.adapter, 'predictions': run.pred_file,
                           'annotations': run.anno_file, 'steplog': run.steplog_file,
                           'max_action_len': run.max_action_len, 'note': run.note,
                           'reference': getattr(run, 'reference', 'route' if run.route_reference else 'goal'),
                           'coordinate_system': run.coordinate_system,
                           'comparable_runs': [other.name for other in self.runs.values() if compatible(run, other)],
                           'capabilities': {'images': bool(getattr(run, 'has_images', getattr(run, 'capabilities', {}).get('images', False))),
                                            'mesh': assets['meshes'].is_dir() or bool(getattr(run, 'scans_dir', None)),
                                            'whatif': False},
                           'summary': pt.aggregate(run.rows())})
        return {'schema_version': 'navigation-explorer/api-v1', 'project': self.project.name,
                'runs': result, 'error_margin': None}

    def decorate_graph(self, run, graph):
        graph = copy.deepcopy(graph)
        scan, assets = graph['scan'], self.assets[run.name]
        source = getattr(run, 'scans_dir', None)
        valid = bool(SAFE_SCENE.fullmatch(scan))
        source_file = Path(source) / scan / 'house_segmentations' / f'{scan}.ply' if source and valid else None
        mesh = valid and ((assets['meshes'] / f'{scan}.bin.gz').is_file()
                          or (self.generated_assets[run.name]['meshes'] / f'{scan}.bin.gz').is_file()
                          or bool(source_file and source_file.is_file()))
        graph['capabilities'] = dict(graph.get('capabilities') or {}, mesh=mesh,
                                     images=any(node.get('sky') for node in graph['nodes'].values()))
        graph['asset_query'] = '?' + urlencode({'run': run.name})
        graph['mesh_url'] = f'/api/mesh/{quote(scan, safe="")}' + graph['asset_query'] if mesh else None
        path = assets['navmaps'] / f'{scan}.json' if valid else None
        graph['navmap'] = read_json(path) if path and path.is_file() else None
        return graph

    def episode(self, name, instr_id):
        run = self.runs[name]
        if instr_id not in run.ids:
            raise KeyError(instr_id)
        ep = copy.deepcopy(run.episode(instr_id))
        if not ep.get('graph'):
            ep['graph'] = run.graphs[ep['scan']].to_json()
        ep['graph'] = self.decorate_graph(run, ep['graph'])
        ep.setdefault('evaluation', getattr(run, 'evaluation', {}))
        ep.setdefault('reference', getattr(run, 'reference', 'route' if run.route_reference else 'goal'))
        ep.setdefault('image_provenance', getattr(run, 'image_provenance',
                      {'kind': 'unavailable', 'label': 'No recorded camera images'}))
        ep['capabilities'] = dict(ep.get('capabilities') or {}, **ep['graph']['capabilities'])
        ep['others'] = {}
        for other, candidate in self.runs.items():
            if not compatible(run, candidate) or instr_id not in candidate.ids:
                continue
            value = candidate.episode(instr_id)
            if value['scan'] != ep['scan']:
                continue
            graph = value.get('graph') or candidate.graphs[value['scan']].to_json()
            ids = {vp: f'cmp:{other}:{index}' for index, vp in enumerate(dict.fromkeys(value['path']))}
            ep['others'][other] = {'path': [ids[vp] for vp in value['path']], 'cross': True,
                                   'nodes': {ids[vp]: dict(graph['nodes'][vp], kind='compare') for vp in ids},
                                   'outcome': value['outcome'], 'metrics': value['metrics'],
                                   'failure_mode': value['failure_mode']}
        ep['whatif'] = {}
        return ep

    def mesh(self, name, scan, floor=None):
        run, assets = self.runs[name], self.assets[name]
        kind = 'meshes' if floor is None else 'meshes_hd'
        filename = f'{scan}.bin.gz' if floor is None else f'{scan}_f{floor}.bin.gz'
        configured = assets[kind] / filename
        path = self.generated_assets[name][kind] / filename
        for cached in (configured, path):
            if cached.is_file():
                return cached.read_bytes()
        source = getattr(run, 'scans_dir', None)
        if not source or not (Path(source) / scan / 'house_segmentations' / f'{scan}.ply').is_file():
            raise FileNotFoundError('No building mesh configured for this scene')
        with self.mesh_lock:
            lock = self.mesh_building.setdefault(str(path), threading.Lock())
        with lock:
            if not path.is_file():
                from . import build_meshes
                path.parent.mkdir(parents=True, exist_ok=True)
                if floor is None:
                    build_meshes.build(scan, source, path.parent)
                else:
                    graph = run.graphs[scan].to_json()
                    heights = {level['id']: level['z'] for level in graph['floors']}
                    if floor not in heights:
                        raise KeyError(f'floor {floor}')
                    build_meshes.build_floor(scan, floor, heights[floor], source, path.parent)
        return path.read_bytes()


    def investigation_run(self, name, sources=False):
        """Recording identity for investigation queries; file stamps are not content hashes."""
        run = self.runs[name]
        info = {'name': name, 'model': run.model, 'split': run.split,
                'env': 'continuous' if getattr(run, 'ce', False) else 'discrete',
                'predictions': str(run.pred_file), 'annotations': str(run.anno_file) if run.anno_file else None,
                'steplog': str(run.steplog_file) if run.steplog_file else None,
                'reference': getattr(run, 'reference', 'route' if run.route_reference else 'goal'),
                'dataset': getattr(run, 'dataset', None), 'adapter': getattr(run, 'adapter', None),
                'max_action_len': run.max_action_len}
        if sources:
            settings = getattr(run, 'settings', {}) or {}
            info['settings'] = {k: settings[k] for k in
                                ('checkpoint', 'model', 'seed', 'max_instr_len', 'max_action_len',
                                 'languages', 'note', 'env') if k in settings}
            info['sources'] = []
            for role in ('predictions', 'annotations', 'steplog'):
                path = info[role]
                record = {'role': role, 'path': path, 'available': False}
                if path:
                    try:
                        stat = Path(path).stat()
                        record.update(available=True, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
                    except OSError:
                        pass
                info['sources'].append(record)
        return info

    def investigation_queue(self, name, query):
        """Filter inexpensive episode summaries before returning a bounded review queue."""
        try:
            offset, limit = int(query.get('offset', '0')), int(query.get('limit', '50'))
        except (TypeError, ValueError):
            raise ValueError('offset and limit must be integers') from None
        if offset < 0 or not 1 <= limit <= 200:
            raise ValueError('offset must be nonnegative and limit must be between 1 and 200')
        outcome = query.get('outcome', '')
        if outcome not in ('', 'success', 'wrong_stop', 'step_limit', 'no ground truth', 'failure', 'unscored'):
            raise ValueError('unknown outcome filter')
        rows = self.runs[name].rows()
        result = investigations.build_queue(self.investigation_run(name), rows)
        kind, search = query.get('kind', ''), query.get('q', '').strip().lower()[:300]
        if kind and kind not in {c['id'] for c in result['categories']}:
            raise ValueError('unknown investigation category')
        by_id = {r['instr_id']: r for r in rows}
        cases = []
        for case in result['cases']:
            if kind and not any(flag['kind'] == kind for flag in case['flags']):
                continue
            if outcome and case['outcome'] != outcome:
                continue
            row = by_id[case['instr_id']]
            if search and search not in ' '.join(str(row.get(k) or '') for k in
                                                ('instr_id', 'scan', 'instruction')).lower():
                continue
            cases.append(case)
        result.update(cases=cases[offset:offset + limit], total_matches=len(cases),
                      offset=offset, limit=limit,
                      filters={'kind': kind, 'outcome': outcome, 'q': search})
        return result

    def investigation_case(self, name, instr_id):
        run = self.runs[name]
        if instr_id not in run.ids:
            raise KeyError(instr_id)
        ep = self.episode(name, instr_id)
        note = self.notes.for_run(name)['notes'].get(instr_id, {'tags': [], 'note': ''})
        case = investigations.build_case(self.investigation_run(name, sources=True), ep, note)
        case['api_url'] = '/api/investigation?' + urlencode({'run': name, 'id': instr_id})
        return case


class Handler(BaseHTTPRequestHandler):
    app = None
    server_version = 'NavigationExplorer/0.1'

    def log_message(self, fmt, *args):
        pass

    def send(self, body, ctype='application/json', status=200, headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(json_safe(body), ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        if ctype == 'application/json' and 'Cache-Control' not in (headers or {}):
            self.send_header('Cache-Control', 'no-store')
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def fail(self, status, message):
        self.send({'error': message}, status=status)

    def do_GET(self):
        url = urlparse(self.path)
        q = {key: values[0] for key, values in parse_qs(url.query).items()}
        app = self.app
        try:
            if url.path in ('/', '/index.html'):
                return self.send((STATIC / 'index.html').read_bytes(), 'text/html; charset=utf-8')
            if url.path.startswith('/static/'):
                path = (STATIC / unquote(url.path[len('/static/'):])).resolve()
                if STATIC.resolve() not in path.parents or not path.is_file():
                    return self.fail(404, 'static file not found')
                ctype = 'text/javascript' if path.suffix == '.js' else mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
                return self.send(path.read_bytes(), ctype, headers={'Cache-Control': 'no-cache'})
            if url.path == '/api/meta':
                return self.send(app.meta())
            if url.path == '/api/rows':
                return self.send(app.runs[q['run']].rows())
            if url.path == '/api/episode':
                return self.send(app.episode(q['run'], q['id']))
            if url.path == '/api/investigations':
                return self.send(app.investigation_queue(q['run'], q), headers={'Cache-Control': 'no-store'})
            if url.path == '/api/investigation':
                case = app.investigation_case(q['run'], q['id'])
                headers = {'Cache-Control': 'no-store'}
                if q.get('download') == '1':
                    filename = re.sub(r'[^A-Za-z0-9_.-]', '_', f'{q["run"]}_{q["id"]}_investigation.json')
                    headers['Content-Disposition'] = f'attachment; filename="{filename}"'
                return self.send(case, headers=headers)
            if url.path == '/api/points':
                return self.send(app.runs[q['run']].points())
            if url.path == '/api/notes':
                return self.send(app.notes.for_run(q['run']))
            if url.path == '/api/whatif/status':
                return self.send(app.whatif.status())
            if url.path == '/api/whatif/results':
                return self.send(app.whatif.summary(app.runs[q['run']]))
            if url.path == '/api/scan':
                if not SAFE_SCENE.fullmatch(q.get('scan', '')):
                    raise ValueError('invalid scene identifier')
                run = app.runs[q['run']]
                return self.send(app.decorate_graph(run, run.graphs[q['scan']].to_json()))
            match = re.fullmatch(r'/api/mesh/([A-Za-z0-9_-]{1,128})(?:/f(\d{1,3}))?', url.path)
            if match:
                floor = int(match[2]) if match[2] is not None else None
                try:
                    body = app.mesh(q['run'], match[1], floor)
                except ImportError:
                    return self.fail(404, 'Mesh conversion requires the optional meshes dependency')
                return self.send(body, 'application/octet-stream', headers={'Content-Encoding': 'gzip'})
            match = re.fullmatch(r'/api/legacy-view/([^/]+)/([^/]+)/([^/]+)_([0-5])\.jpg', url.path)
            if match:
                run = app.runs[unquote(match[1])]
                if not hasattr(run, 'view_file'):
                    return self.fail(404, 'native camera images not configured')
                path = run.view_file(unquote(match[2]), unquote(match[3]), match[4])
                if not path or not path.is_file():
                    return self.fail(404, 'camera image unavailable')
                return self.send(path.read_bytes(), 'image/jpeg')
            match = re.fullmatch(r'/api/recorded-view/([^/]+)/(.+\.jpg)', url.path)
            if match:
                run = app.runs[unquote(match[1])]
                if not hasattr(run, 'image_file'):
                    return self.fail(404, 'recorded images not configured')
                path = run.image_file(unquote(match[2]))
                if not path or not path.is_file():
                    return self.fail(404, 'camera image unavailable')
                return self.send(path.read_bytes(), 'image/jpeg')
            match = re.fullmatch(r'/api/navmap/([A-Za-z0-9_-]{1,128})_f(\d{1,3})\.png', url.path)
            if match:
                path = app.assets[q['run']]['navmaps'] / f'{match[1]}_f{match[2]}.png'
                return self.send(path.read_bytes(), 'image/png')
            return self.fail(404, 'not found')
        except KeyError as error:
            return self.fail(404, f'unknown or missing {error}')
        except (ValueError, TypeError) as error:
            return self.fail(400, str(error))
        except (FileNotFoundError, PermissionError) as error:
            return self.fail(404, str(error))

    def do_POST(self):
        origin = self.headers.get('Origin')
        if origin and urlparse(origin).netloc != self.headers.get('Host'):
            return self.fail(403, 'cross-origin writes are not supported')
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 <= size <= 8 * 1024 * 1024:
                return self.fail(413, 'request body exceeds 8 MiB')
            req = json.loads(self.rfile.read(size) or b'{}')
            if not isinstance(req, dict):
                raise ValueError('request must be a JSON object')
            path, app = urlparse(self.path).path, self.app
            if path.startswith('/api/whatif/'):
                return self.fail(409, 'No inference runner is configured; this server reviews recordings')
            run = app.runs[req.get('run')]
            if path == '/api/notes':
                instr_id = str(req.get('id'))
                if instr_id not in run.ids:
                    raise KeyError(instr_id)
                return self.send(app.notes.save(run.name, instr_id, req.get('tags') or [], req.get('note')))
            if path != '/api/export':
                return self.fail(404, 'not found')
            fmt = req.get('format', 'csv')
            if fmt not in ('csv', 'json', 'jsonl'):
                raise ValueError('format must be csv, json or jsonl')
            requested = req.get('ids') or []
            if not isinstance(requested, list):
                raise ValueError('ids must be a list')
            ids = set(requested)
            notes = app.notes.for_run(run.name)['notes']
            rows = [dict(row, tags='; '.join(notes.get(row['instr_id'], {}).get('tags', [])),
                         note=notes.get(row['instr_id'], {}).get('note', ''))
                    for row in run.rows() if row['instr_id'] in ids]
            with tempfile.TemporaryDirectory() as tmp:
                out = pt.write_export(run, rows, Path(tmp) / f'export.{fmt}', fmt, full=bool(req.get('full')))
                body = out.read_bytes()
            filename = re.sub(r'[^A-Za-z0-9_.-]', '_', f'{run.name}_{len(rows)}.{fmt}')
            return self.send(body, 'text/csv' if fmt == 'csv' else 'application/json',
                             headers={'Content-Disposition': f'attachment; filename="{filename}"'})
        except KeyError as error:
            return self.fail(404, f'unknown {error}')
        except (ValueError, TypeError) as error:
            return self.fail(400, str(error))
        except OSError as error:
            return self.fail(500, f'Could not save local review data: {error}')


def create_server(app, host='127.0.0.1', port=8765):
    handler = type('ProjectHandler', (Handler,), {'app': app})
    return ThreadingHTTPServer((host, port), handler)
