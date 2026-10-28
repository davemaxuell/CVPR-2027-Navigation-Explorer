# Native R2R/DUET and ETPNav imports

These adapters support existing recording formats. Use the portable `recording`
adapter for other simulators or custom logs instead of renaming their files.

## R2R-style predictions and DUET/ScaleVLN

```json
{
  "schema_version": "navigation-explorer/project-v1",
  "runs": [{
    "id": "baseline/validation",
    "adapter": "r2r",
    "model": "My R2R model",
    "dataset": "R2R",
    "split": "val_unseen",
    "environment": "discrete",
    "predictions": "recordings/detail_val_unseen.json",
    "annotations": "datasets/R2R_val_unseen_enc.json",
    "connectivity": "datasets/connectivity",
    "steplog": "recordings/steps_val_unseen.jsonl",
    "max_action_len": 15
  }]
}
```

`steplog` is optional. Without it, trajectories and benchmark metrics remain
available; model scores and attention are unavailable. Supported trajectories
include DUET lists of walked viewpoints per decision and R2R
`[viewpoint, heading, elevation]` traces. Supply the model's actual action budget;
the legacy DUET default is 15.

Connectivity uses the Matterport/R2R `<scene>_connectivity.json` format. Optional
fields are `scans` (Matterport scene assets), `subtasks` (FGR2R or precomputed RxR
alignment), `view_index` (camera candidate angles), and `bboxes` (REVERIE visible
goal viewpoints). For object-navigation annotations containing `objId`, visible-goal
annotations are required; provide the matching BBoxes rather than treating the path
endpoint as the object's success condition. RxR guide annotation JSONL/GZIP is accepted; use `dataset: "RxR"`
and `reference: "route"`. Other native imports can explicitly choose `reference:
"goal"` or `"route"`.

This adapter computes the existing benchmark metrics using graph geodesics and
the three-meter success criterion, except REVERIE's supplied visible-goal
viewpoints. It is not a general evaluator for a new dataset's success rules. The
native R2R format also represents completion by the end of its submitted route;
use explicit portable decisions when a distinction between STOP, truncation, and
an unlogged action matters.

## ETPNav episode logs

```json
{
  "schema_version": "navigation-explorer/project-v1",
  "runs": [{
    "id": "etpnav/custom-validation",
    "adapter": "etpnav",
    "model": "ETPNav",
    "dataset": "R2R-CE",
    "split": "custom-validation",
    "environment": "continuous",
    "episodes": "recordings/episodes_val_unseen.jsonl",
    "connectivity": "datasets/connectivity",
    "views": "recordings/views_val_unseen"
  }]
}
```

The project can contain only CE runs. Split names are arbitrary. `views` is
optional and contains `<episode_id>/n<decision>_<face>.jpg`. Optional `annotations`
map original R2R instructions to their discrete instruction IDs for comparisons;
without that mapping, the logged episode identity is retained. Optional `metadata`
can be an inline object or a JSON file path. `subtasks`, `scans`, and `reference`
are also supported.

This adapter requires the native ETPNav explorer recording fields: planner nodes
and ghost waypoints, high-level steps, dense track and samples, reference positions,
logged geodesics, and evaluation metrics. A generic Habitat trajectory is not the
same format. No ETPNav or Habitat installation is needed to inspect an existing
complete log.

Positions are converted from Habitat coordinates to the Matterport display frame.
Headline success/OSR/SPL and other evaluator metrics come from the original log.
Dense goal visits that are absent from sampled replay frames are reported without
inventing a frame anchor. Preview imagery does not prove what the model saw.

## Image provenance

Native imports may provide `image_provenance` as an object or a JSON path, with
`kind` (`recorded_observation`, `regenerated_preview`, or `skybox_derived`) and
`input_log_sha256` matching the supplied prediction/episode file. Add a human-readable
`label` and `description`. The loader checks the recording hash; it does not verify
image contents or the model's inputs. Without a receipt, provenance remains unknown.

No model checkpoints, external inference code, private paths, or dataset assets
are needed in the public repository. Refer to them from your local manifest.
