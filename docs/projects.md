# Project configuration

A project is a UTF-8 JSON manifest with `schema_version` set to
`navigation-explorer/project-v1`, an optional `name`, and a nonempty `runs` list.

Every run requires `id`, `adapter`, `model`, `dataset`, `split`, and `environment`.
Run IDs must be unique within the project. `environment` is `discrete` or
`continuous`; the adapter decides how to read the recording, independently of
the model's display name.

| Field | Purpose |
| --- | --- |
| `id` | Stable run identity used by links, API requests, and review notes |
| `adapter` | `recording`, `r2r`, or `etpnav` |
| `model`, `dataset`, `split` | Explicit identities; no filename-based dataset inference |
| `model_label`, `dataset_description`, `note` | Optional readable UI descriptions |
| `coordinate_system` | Coordinate identity used to gate model comparisons |
| `comparison_group` | Optional explicit declaration that runs share episode identities, scenes, and aligned coordinates |
| `assets` | Optional directories `meshes`, `meshes_hd`, and `navmaps` |

Input paths are relative to the manifest directory, not to the process working
directory. Absolute paths are accepted. Dataset directories can live outside this
repository. Prefer relative paths for shared project files and keep machine-specific
paths in a file named `project.local.json`, which Git ignores.

For portable `recording` runs, provide `episodes`, optionally `evaluation`,
`images_dir`, and `metadata`. See [recordings](recordings.md). Native adapters use
the fields documented in [native adapters](native-adapters.md).

Two runs are offered for comparison when their dataset, split, environment, and
coordinate system match. An explicit matching `comparison_group` permits an
aligned cross-setting comparison, still requiring the same coordinate system.
Runs with different explicit groups, or with a group on only one side, are not compared.
An individual episode comparison also requires a matching instruction ID and
scene ID. The compared trajectory carries its own positions, so its internal node
IDs do not have to match the selected model's node IDs.

Portable inputs support `cartesian_z_up_m` and explicit `matterport_z_up_m`;
native adapters use `matterport_z_up_m`. Both require meters with z up, and neither
performs a coordinate transform. Keep independently scaled or transformed scene
frames in distinct datasets or comparison groups until their coordinates are aligned.

`--check` validates the manifest, loads every configured run, and reconstructs each
episode. It does not run model inference or download data. Large projects can take
time to check. Restart the server after replacing a recording or changing the manifest.

Review notes and caches live in the project's state directory. Its default name
is derived from the absolute manifest path; moving a manifest therefore creates a
different default review workspace. Set `--state-dir` to an existing review
directory when deliberately retaining the same review history.
