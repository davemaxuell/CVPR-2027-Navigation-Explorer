# Local API for agents and integrations

The viewer serves JSON over HTTP, normally at `http://127.0.0.1:8765`.
URL-encode run and episode IDs. The server does not depend on an OpenAI SDK or a
particular agent framework.

| Method and route | Result |
| --- | --- |
| `GET /api/meta` | Project, run IDs, dataset/model metadata, summaries, capabilities, compatible comparisons |
| `GET /api/rows?run=…` | Episode summaries for one run |
| `GET /api/episode?run=…&id=…` | Replay frames, decisions, graph, diagnostics, and compatible comparison paths |
| `GET /api/points?run=…` | Available per-decision reference review points; empty when not supported |
| `GET /api/investigations?run=…` | Paginated heuristic review queue |
| `GET /api/investigation?run=…&id=…` | Evidence bundle for any episode, including unflagged episodes |
| `GET /api/investigation?run=…&id=…&download=1` | The same bundle as a JSON attachment |
| `GET /api/notes?run=…` | Saved local review notes and tag vocabulary |
| `POST /api/notes` | Save `{run, id, tags: [], note: "…"}` locally |
| `POST /api/export` | Export `{run, ids: [], format: "jsonl", full: true}`; formats are CSV, JSON, JSONL |

The queue accepts `kind`, `q`, `outcome`, `offset` (nonnegative), and `limit`
(1–200, default 50). Category IDs are returned in `categories[].id`; category
counts cover the full run and may overlap. Outcome values include `success`,
`wrong_stop`, `step_limit`, `failure`, `unscored`, and the native compatibility
value `no ground truth`.

The investigation bundle's schema version is `navigation-investigation/v1`.
Observations include JSON Pointer references into the bundle; hypotheses remain
explicitly untested. Frame and decision anchors can be null when the required
evidence is unavailable. Relative replay/image URLs use the server's origin.
Source file stamps identify files, not their complete contents. The trace digest
does not certify model weights or camera pixels.

The input recording schema and exported investigation schema serve different
purposes. A downloaded investigation bundle is an analysis artifact, not an
input episode file to feed directly to the recording adapter.

Invalid filters or malformed JSON return 400, missing runs/episodes return 404,
and unavailable inference endpoints return 409. Cross-origin writes are rejected.
The server is intended for a trusted local workspace. Use SSH forwarding for
remote access instead of making an unauthenticated server publicly reachable.

Example agent instruction:

> Read `/api/meta`, select a run, and fetch its investigation queue. Inspect case
> bundles for recurring behaviors. Cite episode IDs and decision/frame evidence.
> Separate observations from hypotheses; state when logs or images are missing.
> Do not interpret review priority as a causal diagnosis or calibrated risk.
