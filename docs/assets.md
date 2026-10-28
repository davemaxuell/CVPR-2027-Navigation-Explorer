# Optional scene assets and provenance

All core replay and investigation features work without panoramas or building
meshes. Missing assets are shown as unavailable, and graph/trajectory replay
remains usable.

## Camera panoramas

Portable recordings can supply six JPEG cube faces per observation using
`images_dir` in the run manifest and an `image_prefix` on a trajectory point or
graph node. The files are `<prefix>_0.jpg` through `<prefix>_5.jpg`; nested prefixes
are relative to `images_dir`. Supply the 3×3 row-major camera `rotation` used by
the images. See the episode schema and examples for the point fields.

The renderer's face ordering is `+x=2`, `-x=4`, `+y=0`, `-y=5`, `+z=1`, `-z=3`;
its texture direction is `-Rᵀ × world_direction`. Convert your simulator's camera
convention explicitly. A single RGB frame or equirectangular image must first be
converted to this supported cube format; it should not be relabeled as six views.

Native R2R runs use the configured `scans` directory's `matterport_skybox_images`.
Native ETPNav runs use their explicit `views` directory. Image routes include the
run identity, so identical scene names in separate datasets do not share images.

Provenance should distinguish original recorded observations, regenerated
previews, and images projected from dataset skyboxes. The explorer cannot infer
the original model inputs from a rendered preview. Portable provenance is
producer-supplied; native source-hash receipts validate the named recording only.

## Building meshes and navigable-area maps

For native Matterport PLY reconstruction files, install the optional converter:

```bash
python -m pip install '.[meshes]'
```

Configured `scans/<scene>/house_segmentations/<scene>.ply` files can be converted
on demand. Generated meshes go to the project state directory. No simulator or
GPU is needed for this CPU conversion, but large PLY files can take time and RAM.

To reuse prepared assets, a run may set:

```json
"assets": {
  "meshes": "prepared/meshes",
  "meshes_hd": "prepared/meshes_hd",
  "navmaps": "prepared/navmaps"
}
```

The mesh format is the viewer's gzip-compressed `VLNM`/`VLNS` binary format,
documented in `build_meshes.py`. Navigable-area maps use the existing per-scene
JSON metadata and `<scene>_f<floor>.png` files. This release reads prepared navmaps;
building them from a simulator navmesh remains an external preprocessing step.
Do not mix assets and trajectories from different coordinate frames.

Assets remain external because they can be large and subject to dataset access
terms. The repository includes only synthetic trajectory examples and UI assets.

Configure the canonical image directory when a convenience directory contains
symlinks to files outside it. Image requests must resolve within the configured
`scans`, `views`, or `images_dir` root; symlinks that escape that root are rejected.
