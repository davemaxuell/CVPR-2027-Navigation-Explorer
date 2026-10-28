# Example imports

The [portable recording guide](../docs/recordings.md) explains how to export trajectories,
evaluation results, decisions, and optional images from your own model.

- `export_custom_run.py` writes a tiny synthetic run using the public Python writer.
- `r2r.project.json` is a native R2R/DUET/ScaleVLN import template.
- `etpnav.project.json` is a native ETPNav explorer-log import template.

After installing Navigation Explorer, try:

```bash
python examples/export_custom_run.py /tmp/my-navigation-run --environment continuous
navigation-explorer --project /tmp/my-navigation-run/project.json
```

Use `--environment discrete` for the graph-based example. The JSON templates refer to
external data paths you must edit; real dataset assets are not included. Run the bundled
synthetic demonstrations directly with `navigation-explorer --demo`.
