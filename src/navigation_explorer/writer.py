"""Small, simulator-independent helpers for exporting recorded model runs."""
import json
import os
from pathlib import Path
import tempfile

from .recordings import COORDINATE_SYSTEM, EPISODE_SCHEMA, normalize_episode


def episode_from_positions(instr_id, scene, instruction, positions, *, node_ids=None, **fields):
    """Create a portable episode from positions in meters, with Cartesian z up.

    ``positions`` contains three-element lists/tuples. Use ``node_ids`` for stable
    discrete viewpoint identities; omitted IDs represent distinct sampled poses.
    Evaluation results, decisions, and termination must be supplied explicitly.
    """
    positions = list(positions)
    ids = list(node_ids) if node_ids is not None else [None] * len(positions)
    if len(ids) != len(positions):
        raise ValueError('node_ids must contain one ID per position')
    trajectory = [dict(position=list(position), **({'node_id': node_id} if node_id is not None else {}))
                  for position, node_id in zip(positions, ids)]
    reserved = {'schema_version', 'instr_id', 'scene', 'instruction', 'trajectory'}
    if reserved.intersection(fields):
        raise ValueError(f'reserved episode fields: {sorted(reserved.intersection(fields))}')
    return {'schema_version': EPISODE_SCHEMA, 'instr_id': str(instr_id), 'scene': scene,
            'instruction': instruction, 'trajectory': trajectory, **fields}


def write_recordings(path, episodes, *, environment, evaluation=None, images_dir=None, overwrite=False,
                     coordinate_system=COORDINATE_SYSTEM):
    """Validate all episodes, then write portable JSONL atomically.

    Existing files require ``overwrite=True``. This writes recordings only; a
    project manifest chooses their model, dataset, split, and environment labels.
    """
    path = Path(path)
    records, ids = list(episodes), set()
    if not records:
        raise ValueError('at least one episode is required')
    for record in records:
        normalized = normalize_episode(record, run_name='export', environment=environment,
                                       evaluation=evaluation, images_dir=images_dir,
                                       coordinate_system=coordinate_system)
        if normalized['instr_id'] in ids:
            raise ValueError(f'duplicate instr_id {normalized["instr_id"]!r}')
        ids.add(normalized['instr_id'])
    # allow_nan=False also validates caller metadata not otherwise interpreted.
    text = ''.join(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'
                   for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent,
                                         prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic exclusive publication; another writer cannot be overwritten.
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
