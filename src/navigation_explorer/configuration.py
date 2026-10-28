"""Explicit project configuration; paths are relative to the manifest, never the checkout."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_VERSION = 'navigation-explorer/project-v1'
ADAPTERS = {'recording', 'r2r', 'etpnav'}


def read_json(path: Path):
    def reject(value):
        raise ValueError(f'non-finite JSON number {value}')
    try:
        return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)
    except (OSError, ValueError) as error:
        raise ValueError(f'{path}: {error}') from error


def resolve_path(value, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError('expected a nonempty file or directory path')
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


@dataclass(frozen=True)
class Project:
    path: Path
    name: str
    runs: list[dict]
    state_dir: Path

    @property
    def directory(self):
        return self.path.parent


def load_project(path, state_dir=None) -> Project:
    path = Path(path).expanduser().resolve()
    data = read_json(path)
    if not isinstance(data, dict) or data.get('schema_version') != PROJECT_VERSION:
        raise ValueError(f'{path}: schema_version must be {PROJECT_VERSION!r}')
    runs = data.get('runs')
    if not isinstance(runs, list) or not runs:
        raise ValueError(f'{path}: runs must be a nonempty list')
    seen = set()
    for index, spec in enumerate(runs):
        prefix = f'{path}: runs[{index}]'
        if not isinstance(spec, dict):
            raise ValueError(f'{prefix} must be an object')
        for field in ('id', 'adapter', 'model', 'dataset', 'split', 'environment'):
            value = spec.get(field)
            if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
                raise ValueError(f'{prefix}.{field} must be a nonempty string without control characters')
        if spec['id'] in seen:
            raise ValueError(f'{prefix}: duplicate run id {spec["id"]!r}')
        seen.add(spec['id'])
        if spec['adapter'] not in ADAPTERS:
            raise ValueError(f'{prefix}.adapter must be one of {sorted(ADAPTERS)}')
        if spec['environment'] not in ('discrete', 'continuous'):
            raise ValueError(f'{prefix}.environment must be discrete or continuous')
        expected = {'r2r': 'discrete', 'etpnav': 'continuous'}.get(spec['adapter'])
        if expected and spec['environment'] != expected:
            raise ValueError(f'{prefix}: {spec["adapter"]} requires environment={expected!r}')
        if not isinstance(spec.get('assets', {}), dict):
            raise ValueError(f'{prefix}.assets must be an object')
        for key in ('meshes', 'meshes_hd', 'navmaps'):
            if key in spec.get('assets', {}):
                resolve_path(spec['assets'][key], path.parent)
    if state_dir is None:
        root = Path(os.environ.get('XDG_DATA_HOME', str(Path.home() / '.local/share')))
        key = hashlib.sha256(str(path).encode()).hexdigest()[:16]
        state_dir = root / 'navigation-explorer' / 'projects' / key
    return Project(path, str(data.get('name') or path.stem), runs, Path(state_dir).expanduser().resolve())
