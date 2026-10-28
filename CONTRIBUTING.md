# Contributing

Install an editable checkout with Python 3.10+ and run the checks in the README.
The demo and automated tests use synthetic data and must work without access to
the original research workspace, model weights, or a simulator.

## Structure

```text
src/navigation_explorer/
  cli.py                 command line and validation
  configuration.py       explicit project manifests
  server.py              HTTP API, review state, run-scoped assets
  recordings.py          portable recording validation and normalization
  writer.py              helpers for custom-model exporters
  legacy.py              native R2R and ETPNav adapter configuration
  parse_trajectories.py   existing discrete benchmark analysis
  ce_runs.py             native continuous log analysis
  investigations.py      deterministic evidence-based review rules
  static/                browser interface and vendored UI dependencies
  demo/                  synthetic examples shipped in the wheel
schemas/                 published input contracts
examples/                exporter and manifest examples
docs/                    use, formats, assets, and agent API
tests/                   dataset-independent regressions
```

Prefer exporting a new model to the portable format before adding an adapter.
A new adapter should preserve recorded values, declare coordinate and metric
semantics, and return the common replay interface. Add a named adapter to the
configuration allowlist and server loader; include a small synthetic fixture and
malformed-input tests. Do not add automatic model imports or checkpoint loading
to the replay server.

Keep these boundaries explicit:

- A missing result is not a failed episode or a zero metric.
- A trajectory ending is not evidence of a model STOP decision.
- Straight-line displacement is not geodesic distance or distance walked.
- Regenerated imagery is not the original model observation.
- A reference disagreement is not proof of causality.
- Scenes from separate runs do not share geometry merely because IDs match.

Keep datasets, private paths, notes, caches, and credentials out of commits.
Machine-specific manifests should use the ignored `*.local.json` suffix. Preserve
the bundled dependency notices when updating browser dependencies.

To check distribution contents:

```bash
python -m pip wheel --no-deps --wheel-dir dist .
```

Install the wheel into a clean virtual environment and run the demo from outside
the checkout before proposing a release. A frontend change should also be checked
in a browser with missing images, unscored episodes, narrow panels, and both themes.
