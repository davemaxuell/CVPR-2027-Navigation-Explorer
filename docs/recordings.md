# Importing a model's recordings

The `recording` adapter accepts any model that can export positions and optional
evaluation or decision evidence. It works with discrete viewpoint environments and
continuous environments. The viewer replays these files; it does not run the model
or require its simulator.

After installing the project, create a small synthetic export:

```bash
python examples/export_custom_run.py /tmp/my-navigation-run --environment continuous
navigation-explorer --project /tmp/my-navigation-run/project.json
```

Use `--environment discrete` for a graph-based example. These examples contain
invented coordinates and instructions, not dataset samples. The bundled `--demo`
project includes scored discrete and continuous examples as well as an unscored
recording.

## Project manifest

```json
{
  "schema_version": "navigation-explorer/project-v1",
  "name": "My experiment",
  "runs": [{
    "id": "policy-a/validation",
    "adapter": "recording",
    "model": "Policy A",
    "dataset": "My dataset",
    "split": "validation",
    "environment": "continuous",
    "episodes": "episodes.jsonl"
  }]
}
```

Paths resolve relative to the manifest. Absolute paths and `~` are also accepted.
Keep dataset files and large recordings outside the source repository. A manifest
can mix models, datasets, adapter types, and environments. Run IDs must be unique.
Each recording's `instr_id` must be unique within its run. Matching IDs alone do not
prove that two runs describe the same task or coordinate frame.

## Portable episode format

Write one episode per JSONL line, or use a JSON array. Both formats also support
gzip. Each episode declares `schema_version`, `instr_id`, `scene`, and a nonempty
`trajectory`. `instruction` defaults to an empty string.

```json
{
  "schema_version": "navigation-explorer/episode-v1",
  "instr_id": "task-42",
  "scene": "building-7",
  "instruction": "Go through the doorway.",
  "trajectory": [
    {"position": [0, 0, 0], "time_s": 0},
    {"position": [1.2, 0, 0], "time_s": 1.8},
    {"position": [1.2, 1, 0], "time_s": 3.1}
  ]
}
```

Coordinates use meters, with x/y horizontal and **z up**. The default coordinate-system
identifier is `cartesian_z_up_m`. For recordings already expressed in the Matterport
scene/camera frame, explicitly set the run's `coordinate_system` to
`matterport_z_up_m`. This enables intentionally aligned comparisons with the native
adapters when an appropriate `comparison_group` is supplied. Scene and episode IDs
must also describe the same task. An episode may repeat the coordinate-system
declaration, but it must match its run; omitted declarations inherit the run value.
Neither identifier performs a coordinate transformation. Convert other conventions before exporting;
the viewer never guesses axis order or scale. Habitat `[x, y, z]` can be represented
as `[x, -z, y]` in this convention. State whether samples are camera or agent-base
positions in your metadata; `camera_height_m` describes the offset used to draw a
continuous track beneath camera positions and defaults to zero. Heading is in
radians, zero along +y and increasing clockwise toward +x; elevation is in radians.
An omitted heading/elevation uses zero as a display orientation, not recovered
model-pose evidence.

For discrete environments, add a stable `node_id` to each point. A repeated node ID
must have the same position. If no ID is supplied, each trajectory point gets a
distinct sample ID; identical coordinates alone do not imply a graph revisit.
You can supply graph nodes once and refer to them from the trajectory:

```json
{
  "graph": {
    "nodes": {
      "a": {"position": [0, 0, 0]},
      "b": {"position": [2, 0, 0]}
    },
    "edges": [["a", "b"]]
  },
  "trajectory": [{"node_id": "a"}, {"node_id": "b"}]
}
```

This fragment belongs inside an episode with its required identity fields. Graph
edges and recorded route segments are rendered as geometry; the importer does not
infer an environment's traversability or geodesic distances. Optional `floor`
indices must be contiguous starting at zero. `height` is the point's height above
its floor; `region` is an optional readable room name.

Optional `reference_path` contains point objects using the same structure as the
trajectory. A separate `goal` point marks a published goal. The last reference point
is not automatically promoted to a goal because some tasks use object visibility
or other completion criteria.

## Evaluation evidence

Put evaluator outputs in `metrics`. Standard fields include `sr`, `osr`, `spl`,
`ne`, `ndtw`, `sdtw`, `tl`, and `sp_len`; additional finite numeric metrics are
preserved. Rates use 0–1, not percentages. Individual-episode `sr` and `osr` are
0, 1, or null. Missing fields and null values remain unknown.
Episode identity and replay-diagnostic names, such as `instr_id`, `stuck`, and
`first_nonbest`, are reserved and cannot be supplied as metrics. Put additional
uninterpreted observations in `metadata`; this prevents a custom metric from
silently becoming a diagnostic claim.

The importer does not calculate SR, SPL, oracle success, or geodesics from arbitrary
coordinates. It derives sample counts and sampled-polyline length when those fields
are absent, and labels their sources separately. Sampled-polyline length can
underestimate distance traveled between sparse observations. A missing decision log
produces null move/decision counts, not zero model actions.

If your logger provides distances to the goal, place `distance_to_goal` on each
trajectory point and declare:

```json
{
  "evaluation": {"distance_kind": "geodesic", "success_radius_m": 0.5},
  "metrics": {"sr": 0, "osr": 1, "ne": 1.4, "spl": 0},
  "termination": "stop"
}
```

`distance_kind` is `geodesic`, `euclidean`, `custom`, or `unknown`. The viewer uses
explicit distances and a positive radius for goal-area diagnostics; it never
substitutes straight-line distance for geodesic distance. Radius membership uses
the strict comparison `distance < success_radius_m`. Set `distance_kind` to
`custom` and describe the convention in metadata if needed. A threshold alone
does not define success for tasks with extra conditions; `metrics.sr` remains the
recorded evaluator result.

`termination` is `stop`, `step_limit`, `completed`, or `unknown`. A failed metric
without a known ending is labeled **Failure**, not “wrong stop.” Missing success is
**Unscored**, even if the trace has a goal or an ending condition. The final sample
never implies that the model selected STOP.

## Decision evidence

Each logged decision specifies its starting replay `frame` and an `action`:
`move`, `stop`, `observe`, or `unknown`. Frames must increase strictly and begin at
zero. Trim leading samples if they cannot be aligned to the first logged decision.
`end_frame` defaults to the next decision's frame or the final sample. A STOP must
be the final decision at the final sample.

```json
{
  "decisions": [{
    "frame": 0,
    "end_frame": 2,
    "action": "move",
    "target": {"node_id": "waypoint-0", "position": [2, 1, 0]},
    "options": [
      {"node_id": "waypoint-0", "probability": 0.91},
      {"node_id": null, "probability": 0.09}
    ]
  }]
}
```

`target` is the intended waypoint, either an existing node ID or a point object.
The arrival position comes from `end_frame`; a waypoint miss compares those two
positions. Net displacement and sample-polyline distance do not prove a collision.
Use `node_id: null` to explicitly represent STOP in an option list. You may export
`score` instead of `probability` for logits or other scores. Raw scores are retained
in the case JSON and are never normalized into probabilities. Probability subsets
are allowed, but their sum must be at most one. Missing STOP probability stays
unknown. Portable recordings do not automatically apply a benchmark-specific
reference-best action heuristic.

If decisions are absent, replay still works. Internally the renderer uses a tagged
`replay_placeholder` with action `unknown`; it is not counted as a logged decision.

## Optional camera images

Add an `images_dir` directory to the run manifest. A point's `image_prefix`, such as
`episode-42/frame-0`, refers to six JPEG files inside it:

```text
images/episode-42/frame-0_0.jpg
images/episode-42/frame-0_1.jpg
...
images/episode-42/frame-0_5.jpg
```

The cube-face indices use the explorer's camera convention: +x = 2, −x = 4,
+y = 0, −y = 5, +z = 1, −z = 3. `rotation` is the nine-number row-major camera
rotation used by the cube projection (`texture = -Rᵀ world`), defaulting to the
identity matrix. This is a cube-view format, not a single RGB frame or equirectangular
panorama; convert other image formats explicitly before using this path. Prefixes
must be relative and cannot escape `images_dir`, including through symlinks.

An optional episode `image_provenance` object records the supplied kind, label, and
description. Its source is marked `recording_declaration`, with
`independently_verified: false`; the viewer does not certify that files match model
inputs. Without images, the case explicitly says that camera evidence is unavailable
and generates no fictional Matterport image URLs.

## Existing native formats

- [r2r.project.json](../examples/r2r.project.json) is a template for R2R-style/DUET/ScaleVLN
  predictions, explicit annotations, and connectivity files.
- [etpnav.project.json](../examples/etpnav.project.json) is a template for the native ETPNav
  explorer JSONL format. It is not a loader for every Habitat or CE log format.
  Convert other continuous loggers to the portable episode contract above.

These templates intentionally refer to external files you provide. They are not
working demo datasets. Native adapter options and benchmark assumptions are
documented in the main project documentation.

The JSON schemas in [`schemas/`](../schemas/) describe structural fields. The
importer additionally checks references, duplicate IDs, finite values, decision
alignment, and image-path containment. The exporter in `navigation_explorer.writer`
validates before writing and requires `overwrite=True` to replace an existing file.
