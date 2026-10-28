"""Decimate Matterport3D house meshes for the visualizer's 3D building view.

  python -m navigation_explorer.build_meshes --scans-dir /path/to/scans                 # every scan with a house mesh, 8 workers
  python -m navigation_explorer.build_meshes --scans-dir /path/to/scans --scans 8194nk5LbLH zsNo4HB9uLZ
  python -m navigation_explorer.build_meshes --scans-dir /path/to/scans --detail        # also the high-detail mesh of every floor

Source: <scans>/<scan>/house_segmentations/<scan>.ply (vertex-coloured reconstruction, up to ~26M faces,
vertices ~1.5 cm apart). Ceiling faces are dropped (mpcat40 "ceiling") so the building can be seen from
above, then vertices are clustered on a grid, coarsening until the mesh has at most --max-faces triangles.
Colours live on the vertices, so the face budget sets how sharp the building looks.

Two levels of detail:
  whole building  <cache>/meshes/<scan>.bin.gz          --max-faces (450k): cells of 5-20 cm, shown first
  one floor       <cache>/meshes_hd/<scan>_f<id>.bin.gz --detail-faces (2M): cells of 1.2-10 cm (median 2), covering
                  floor z - 0.5 m to z + 3.1 m; the viewer swaps it in for the floor being shown
visualizer_app.py serves them as /api/mesh/<scan> and /api/mesh/<scan>/f<id>, building missing ones on
request. Layout (little endian):
  b'VLNM' | uint32 n_vertices | uint32 n_faces | float32[3] bbox_min | float32[3] bbox_max
  | (b'VLNS' only: float32 z_lo, z_hi of the floor slab)
  | uint16[n_vertices*3] positions quantised to the bbox | pad to 4 bytes | uint8[n_vertices*4] RGBA
  | uint32[n_faces*3]
Requires the optional meshes dependency (numpy).
"""
import argparse
import gzip
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .parse_trajectories import ScanGraph

SCANS_DIR = None  # noqa: E402

CACHE = Path.home() / '.cache' / 'navigation-explorer' / 'meshes'
HD_CACHE = CACHE.parent / 'meshes_hd'
SLAB = (-0.5, 3.1)          # floor slab around a floor's z; the viewer's cut-away reaches 3 m
VERTEX = np.dtype([('p', '<f4', 3), ('n', '<f4', 3), ('t', '<f4', 2), ('rgb', 'u1', 3)])
FACE = np.dtype([('k', 'u1'), ('v', '<i4', 3), ('material', '<i4'), ('segment', '<i4'), ('category', '<i4')])


def read_ply(path):
    raw = path.read_bytes()
    end = raw.index(b'end_header\n') + len(b'end_header\n')
    header = raw[:end].decode('ascii').splitlines()
    nv = int(next(l for l in header if l.startswith('element vertex')).split()[-1])
    nf = int(next(l for l in header if l.startswith('element face')).split()[-1])
    props = [l.split()[-1] for l in header if l.startswith('property')]
    expected = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'tx', 'ty', 'red', 'green', 'blue',
                'vertex_indices', 'material_id', 'segment_id', 'category_id']
    if props != expected:
        raise ValueError(f'{path}: unexpected PLY layout {props}')
    verts = np.frombuffer(raw, VERTEX, nv, end)
    faces = np.frombuffer(raw, FACE, nf, end + nv * VERTEX.itemsize)
    if not (faces['k'] == 3).all():
        raise ValueError(f'{path}: non-triangle faces')
    return verts, faces


def ceiling_categories(house):
    out = set()
    for line in open(house):
        tok = line.split()
        if tok and tok[0] == 'C' and tok[-6] == 'ceiling':   # ... mpcat40_index mpcat40_name 0 0 0 0 0
            out.add(int(tok[1]))
    return out


def cluster(pos, rgb, faces, cell):
    q = np.floor((pos - pos.min(0)) / cell).astype(np.int64)
    key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
    _, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    n = len(counts)
    new_pos = np.stack([np.bincount(inv, pos[:, i], n) for i in range(3)], 1) / counts[:, None]
    new_rgb = np.stack([np.bincount(inv, rgb[:, i], n) for i in range(3)], 1) / counts[:, None]
    f = inv[faces]
    f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
    _, keep = np.unique(np.sort(f, 1), axis=0, return_index=True)
    f = f[np.sort(keep)]
    used, f = np.unique(f, return_inverse=True)
    return new_pos[used], new_rgb[used], f.reshape(-1, 3)


def load_source(scan, scans_dir=SCANS_DIR):
    """Full-resolution house mesh without ceilings: float64 positions, float64 RGB, int64 triangles."""
    seg = Path(scans_dir) / scan / 'house_segmentations'
    verts, faces = read_ply(seg / f'{scan}.ply')
    ceil = ceiling_categories(seg / f'{scan}.house')
    tri = faces['v'][~np.isin(faces['category'], list(ceil))].astype(np.int64)
    return verts['p'].astype(np.float64), verts['rgb'].astype(np.float64), tri


def decimate(pos, rgb, tri, max_faces, cell, step):
    while True:
        p, c, f = cluster(pos, rgb, tri, cell)
        if len(f) <= max_faces:
            return p, c, f, cell
        cell *= step


def write_mesh(out, p, c, f, slab=None):
    lo, hi = p.min(0).astype(np.float32), p.max(0).astype(np.float32)
    qp = np.round((p - lo) / np.maximum(hi - lo, 1e-6) * 65535).astype('<u2')
    rgba = np.concatenate([np.clip(c, 0, 255).astype(np.uint8), np.full((len(c), 1), 255, np.uint8)], 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    with gzip.open(tmp, 'wb', compresslevel=6) as g:
        g.write((b'VLNM' if slab is None else b'VLNS') + struct.pack('<II', len(p), len(f)) + lo.tobytes() + hi.tobytes())
        if slab is not None:
            g.write(struct.pack('<2f', *slab))
        g.write(qp.tobytes())
        if len(qp) % 2:
            g.write(b'\0\0')   # keep the uint8/uint32 blocks 4-byte aligned
        g.write(rgba.tobytes())
        g.write(f.astype('<u4').tobytes())
    tmp.replace(out)
    return out.stat().st_size


def build(scan, scans_dir=SCANS_DIR, out_dir=CACHE, max_faces=450_000, force=False):
    out = Path(out_dir) / f'{scan}.bin.gz'
    if out.is_file() and not force:
        return scan, 'cached', 0
    pos, rgb, tri = load_source(scan, scans_dir)
    p, c, f, cell = decimate(pos, rgb, tri, max_faces, 0.04, 1.3)
    return scan, f'{len(tri)} -> {len(f)} faces (cell {cell * 100:.1f} cm)', write_mesh(out, p, c, f)


def floors_of(scan, scans_dir=SCANS_DIR, conn_dir=None):
    """{floor id: floor z} as the viewer sees them (from the navigation graph and the .house levels)."""
    if conn_dir is None:
        raise ValueError('Floor conversion requires an explicit connectivity directory')
    return {fl['id']: fl['z'] for fl in ScanGraph.load(conn_dir, scan, scans_dir).to_json()['floors']}


def build_floor(scan, floor, z, scans_dir=SCANS_DIR, out_dir=HD_CACHE, max_faces=2_000_000, force=False, source=None):
    """High-detail mesh of one floor: every face with a vertex between z + SLAB[0] and z + SLAB[1]."""
    out = Path(out_dir) / f'{scan}_f{floor}.bin.gz'
    if out.is_file() and not force:
        return scan, f'floor {floor}: cached', 0
    pos, rgb, tri = source or load_source(scan, scans_dir)
    lo, hi = z + SLAB[0], z + SLAB[1]
    inside = (pos[:, 2] >= lo) & (pos[:, 2] <= hi)
    sel = tri[inside[tri].any(1)]
    # Smallest cell within the budget: start at ~the source's own resolution (median triangle edge ~3 cm),
    # grow until under budget (faces fall roughly as cell^-0.85), then bisect until within 10% of it.
    cell, too_fine, fits, best = 0.012, None, None, None
    for _ in range(10):
        p, c, f = cluster(pos, rgb, sel, cell)
        if len(f) <= max_faces:
            fits, best = cell, (p, c, f)
            if too_fine is None or len(f) >= 0.9 * max_faces:
                break
        else:
            too_fine = cell
        cell = (too_fine * fits) ** 0.5 if fits else cell * min(2.0, max(1.05, (len(f) / max_faces) ** 1.2))
    p, c, f = best if best else (p, c, f)
    cell = fits or cell
    size = write_mesh(out, p, c, f, (lo, hi))
    return scan, f'floor {floor}: {len(sel)} -> {len(f)} faces (cell {cell * 100:.1f} cm)', size


def build_floors(scan, scans_dir=SCANS_DIR, out_dir=HD_CACHE, max_faces=2_000_000, force=False, conn_dir=None):
    floors = floors_of(scan, scans_dir, conn_dir)
    todo = {fl: z for fl, z in floors.items() if force or not (Path(out_dir) / f'{scan}_f{fl}.bin.gz').is_file()}
    if not todo:
        return scan, f'{len(floors)} floors cached', 0
    source = load_source(scan, scans_dir)
    done = [build_floor(scan, fl, z, scans_dir, out_dir, max_faces, True, source) for fl, z in todo.items()]
    return scan, '; '.join(m for _, m, _ in done), sum(n for _, _, n in done)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scans', nargs='+', help='default: every scan under --scans-dir with a house mesh')
    ap.add_argument('--scans-dir', required=True)
    ap.add_argument('--connectivity', help='Required for floor conversion')
    ap.add_argument('--out', default=str(CACHE))
    ap.add_argument('--max-faces', type=int, default=450_000, help='whole-building mesh budget')
    ap.add_argument('--detail', action='store_true', help='also build the high-detail mesh of every floor')
    ap.add_argument('--detail-only', action='store_true', help='only the high-detail floor meshes')
    ap.add_argument('--detail-faces', type=int, default=2_000_000, help='per-floor mesh budget')
    ap.add_argument('--detail-out', default=str(HD_CACHE))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if (args.detail or args.detail_only) and not args.connectivity:
        ap.error('--connectivity is required for floor conversion')
    scans = args.scans or sorted(p.parents[1].name for p in Path(args.scans_dir).glob('*/house_segmentations/*.ply'))
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = [] if args.detail_only else \
            [pool.submit(build, s, args.scans_dir, args.out, args.max_faces, args.force) for s in scans]
        if args.detail or args.detail_only:
            jobs += [pool.submit(build_floors, s, args.scans_dir, args.detail_out, args.detail_faces, args.force, args.connectivity)
                     for s in scans]
        failed = 0
        for j in jobs:
            try:
                scan, msg, size = j.result()
            except Exception as e:   # report and keep going; the other scans still build
                failed += 1
                print(f'FAILED: {type(e).__name__}: {e}', flush=True)
                continue
            print(f'{scan}: {msg}{f", {size / 1e6:.1f} MB" if size else ""}', flush=True)
    if failed:
        sys.exit(f'{failed} job(s) failed')


if __name__ == '__main__':
    main()
