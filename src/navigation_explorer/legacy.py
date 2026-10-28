"""Explicit adapters for the existing R2R/DUET and ETPNav recording formats.

These adapters retain their benchmark's three-metre geodesic criterion. They do
not infer simulator installations, datasets, or writable directories from the
location where Navigation Explorer was installed.
"""
import hashlib
import json
import math
import re
from pathlib import Path
from urllib.parse import quote

from . import ce_runs
from . import parse_trajectories as pt


def _path(spec, key, project_dir, *, required=False, directory=False):
    value = spec.get(key)
    if value is None:
        if required:
            raise ValueError(f"Run {spec.get('id', '<unnamed>')!r}: '{key}' is required for the {spec.get('adapter')} adapter")
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Run {spec.get('id')!r}: '{key}' must be a nonempty path string")
    path = Path(value).expanduser()
    path = (project_dir / path).resolve() if not path.is_absolute() else path.resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Run {spec.get('id')!r}: '{key}' {'directory' if directory else 'file'} does not exist: {path}")
    return path


def _object(value, project_dir, label):
    if value is None:
        return {}, None
    path = None
    if isinstance(value, str):
        path = Path(value).expanduser()
        path = (project_dir / path).resolve() if not path.is_absolute() else path.resolve()
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise ValueError(f'{label}: cannot read JSON object from {path}: {error}') from error
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be an object or a path to a JSON object')
    return dict(value), path


def _finite(value):
    """Unavailable geodesics must not produce invalid Infinity/NaN JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _provenance(value, project_dir, source):
    unknown = {'kind': 'unknown', 'label': 'Image provenance unknown',
               'description': 'No explicit provenance receipt matching this recording was supplied.'}
    if value is None:
        return unknown, None
    receipt, path = _object(value, project_dir, 'image_provenance')
    if receipt.get('kind') not in ('recorded_observation', 'regenerated_preview', 'skybox_derived'):
        raise ValueError('image_provenance.kind must be recorded_observation, regenerated_preview, or skybox_derived')
    expected = receipt.get('input_log_sha256')
    if not isinstance(expected, str) or not re.fullmatch('[0-9a-fA-F]{64}', expected):
        raise ValueError('image_provenance needs input_log_sha256 identifying the supplied recording')
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != expected.lower():
        raise ValueError('image_provenance.input_log_sha256 does not match this recording')
    return {**receipt, 'source': str(path) if path else 'project manifest',
            'verification_scope': 'Source recording hash matched; image contents and model inputs were not independently verified.'}, path


class R2RRun(pt.Run):
    ce = False

    def episode(self, instr_id, full=True):
        try:
            episode = super().episode(instr_id, full)
            if full:
                graph = self.graphs[episode['scan']]
                episode['graph'] = graph.to_json()
                episode['image_provenance'] = self.image_provenance
                if self.scans_dir:
                    base = f'/api/legacy-view/{quote(self.name, safe="")}/{quote(episode["scan"], safe="")}'
                    for vp, node in episode['graph']['nodes'].items():
                        node['sky'] = f'{base}/{quote(vp, safe="")}'
                else:
                    for node in episode['graph']['nodes'].values():
                        node['rot'] = []
                if episode.get('goal'):
                    episode['dist_to_goal'] = {vp: round(graph.dist(vp, episode['goal']), 3) for vp in graph.pos}
                    episode['goal_zone'] = episode.get('goal_vps') if episode.get('goal_vps') is not None else [
                        vp for vp in graph.pos if graph.dist(vp, episode['goal']) < pt.ERROR_MARGIN]
            return _finite(episode)
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError(f'Run {self.name!r}, episode {instr_id!r}: invalid R2R prediction, annotation, or connectivity: {error}') from error

    def view_file(self, scan, viewpoint, face):
        if self.scans_dir is None:
            return None
        if not re.fullmatch(r'[A-Za-z0-9_-]+', str(scan)) or not re.fullmatch(r'[A-Za-z0-9_-]+', str(viewpoint)) or str(face) not in '012345' or len(str(face)) != 1:
            raise ValueError('Invalid native panorama image identifier')
        root = self.scans_dir.resolve()
        path = (root / scan / 'matterport_skybox_images' / f'{viewpoint}_skybox{face}_sami.jpg').resolve()
        if root not in path.parents:
            raise ValueError('Image path escapes the configured scans directory')
        return path


class ETPNavRun(ce_runs.CERun):
    def episode(self, instr_id, full=True):
        try:
            return _finite(super().episode(instr_id, full))
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError(f'Run {self.name!r}, episode {instr_id!r}: invalid native ETPNav recording or connectivity: {error}') from error


def load_legacy_run(spec, project_dir, state_dir):
    """Load one explicit native run; paths resolve relative to the manifest."""
    project_dir, state_dir = Path(project_dir).resolve(), Path(state_dir).resolve()
    adapter = spec.get('adapter')
    if adapter not in ('r2r', 'etpnav'):
        raise ValueError(f'Unsupported legacy adapter {adapter!r}; use r2r or etpnav')
    name = spec.get('id')
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Every run requires a nonempty string id')
    expected_environment = 'continuous' if adapter == 'etpnav' else 'discrete'
    if spec.get('environment', expected_environment) != expected_environment:
        raise ValueError(f'{adapter} requires environment={expected_environment!r}')
    if spec.get('coordinate_system', 'matterport_z_up_m') != 'matterport_z_up_m':
        raise ValueError('Native R2R/ETPNav adapters output matterport_z_up_m coordinates; coordinate_system cannot relabel their positions')
    model, split = str(spec.get('model', name)), str(spec.get('split', 'unspecified'))
    dataset = str(spec.get('dataset', 'R2R' if adapter == 'r2r' else 'VLN-CE'))
    metadata, metadata_file = _object(spec.get('metadata'), project_dir, 'metadata')
    connectivity = _path(spec, 'connectivity', project_dir, required=True, directory=True)
    scans = _path(spec, 'scans', project_dir, directory=True)
    annotations = _path(spec, 'annotations', project_dir, required=adapter == 'r2r')
    subtasks_file = _path(spec, 'subtasks', project_dir)
    bboxes = _path(spec, 'bboxes', project_dir)
    view_index = _path(spec, 'view_index', project_dir)
    subtasks = pt.load_subtasks(subtasks_file) if subtasks_file else {}
    inputs = [path for path in (__file__, metadata_file, annotations, subtasks_file, bboxes, view_index) if path]
    if annotations and bboxes is None:
        inputs.append(annotations.parent / 'BBoxes.json')
    connectivity_files = sorted(connectivity.glob('*_connectivity.json'))
    if not connectivity_files:
        raise ValueError(f'No *_connectivity.json files found in {connectivity}')
    inputs.extend(connectivity_files)
    if scans:
        inputs.extend(scans / f.name[:-len('_connectivity.json')] / 'house_segmentations' /
                      (f.name[:-len('_connectivity.json')] + '.house') for f in connectivity_files)
    indices = {}
    if view_index:
        raw = json.loads(view_index.read_text(encoding='utf-8'))
        for key, candidates in raw.items():
            if key in {f.name[:-len('_connectivity.json')] for f in connectivity_files}:
                indices[key] = candidates
            else:
                scan, vp = key.rsplit('_', 1)
                indices.setdefault(scan, {})[vp] = {nb: values[0] for nb, values in candidates.items()}
    graphs = pt.GraphCache(connectivity, scans, indices)
    reference = spec.get('reference', metadata.get('reference', 'route' if dataset.lower().startswith('rxr') else 'goal'))
    if reference not in ('goal', 'route'):
        raise ValueError('Legacy reference must be goal or route')
    budget = spec.get('max_action_len', metadata.get('max_action_len'))
    if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int) or budget < 1):
        raise ValueError('max_action_len must be a positive integer')
    # Separate projects with the same run ID also have separate cache namespaces.
    identity = hashlib.sha256((str(project_dir) + '\n' + name).encode()).hexdigest()[:24]
    cache_dir = state_dir / 'cache' / adapter / identity
    if adapter == 'r2r':
        predictions = _path(spec, 'predictions', project_dir, required=True)
        steplog = _path(spec, 'steplog', project_dir)
        annotation_rows = pt.load_annotations(annotations, bboxes)
        if any('obj_id' in row and 'goal_vps' not in row for row in annotation_rows.values()):
            raise ValueError('Object-goal annotations require a bboxes file with visible goal viewpoints; path-end distance alone is not object-navigation success')
        run = R2RRun(name, predictions, annotations, connectivity, scans, steplog,
                     budget or 15, graphs=graphs, annotations=annotation_rows, subtasks=subtasks,
                     model=model, split=split, cache_dir=cache_dir, cache_inputs=inputs,
                     route_reference=reference == 'route')
        if not run.ids:
            raise ValueError(f'Run {name!r}: no prediction instruction IDs match the annotations')
        run.image_provenance, provenance_file = _provenance(spec.get('image_provenance'), project_dir, predictions)
        if provenance_file:
            run.cache_inputs.append(provenance_file)
    else:
        episodes = _path(spec, 'episodes', project_dir, required=True)
        views = _path(spec, 'views', project_dir, directory=True)
        provenance, provenance_file = _provenance(spec.get('image_provenance'), project_dir, episodes)
        if provenance_file:
            inputs.append(provenance_file)
        identity_map = {}
        if annotations:
            for item in pt._read_json_or_jsonl(annotations):
                for index, text in enumerate(item.get('instructions', [])):
                    identity_map[(str(item.get('path_id', item.get('id', ''))), ' '.join(text.lower().split()))] = f"{item.get('path_id', item.get('id'))}_{index}"
        config = dict(metadata, reference=reference)
        if annotations:
            config['annotations'] = str(annotations)
        if budget:
            config['max_action_len'] = budget
        run = ETPNavRun(name, model, split, episodes, config, graphs, views_dir=views,
                        subtasks=subtasks, instruction_ids=identity_map, cache_dir=cache_dir,
                        cache_inputs=inputs, provenance=provenance)
        if budget is None:
            run.max_action_len = int(run.record(run.ids[0])['max_steps'])
    run.id, run.dataset, run.scans_dir = name, dataset, scans
    run.cache_context = json.loads(json.dumps(spec))
    run.coordinate_system = 'matterport_z_up_m'
    run.comparison_group = spec.get('comparison_group')
    run.note = str(spec.get('note', metadata.get('note', '')))
    object_goals = adapter == 'r2r' and any('goal_vps' in row for row in run.gt.values())
    run.evaluation = {'success_radius_m': None if object_goals else 3.0, 'distance_kind': 'geodesic',
                      'success_criterion': 'visible_goal_viewpoints' if object_goals else 'geodesic_distance_lt'}
    run.settings = {**metadata, **{key: spec[key] for key in ('checkpoint', 'seed', 'languages') if key in spec},
                    'reference': reference, 'max_action_len': run.max_action_len, 'env': expected_environment,
                    'metric_semantics': 'Native benchmark metrics; three-metre geodesic success, except explicit REVERIE visible-goal viewpoints.'}
    run.capabilities = {'images': bool(scans if adapter == 'r2r' else run.views_dir), 'mesh': bool(scans)}
    return run
