# Desktop validation

## Executed locally

- Baseline: 17 existing tests passed before edits. Final local suite: 24 passed, including one pre-existing ignored local test; published suite contains 23 tests. New samples are synthetic.
- Real Tk event loop: invalid input, duplicate start, cancel twice, restart, completion, close during computation, search/no match, pagination and 44,850 result rows with at most 100 Treeview rows.
- Pythonw hidden Tk demo completed with stdout absent, 66 relationships returned, child joined and app closed (one observation: 1.11 s).
- Synthetic SQLite with Chinese/spaced path: read-only connection rejects CREATE TABLE; SHA-256 unchanged; nonexistent database not created. Limit flag survives system-account filtering.
- JSON cache round-trip/corruption; CPU mock loads one model. Compile/AST and Ruff critical-error rules passed. Original entrypoint and heatmap tests remain passing.
- Separate read-only review phase found private-query exception connection cleanup and raw-limit masking after filtering. Fixed both and reran full suite. This was a separate pass by the implementing agent, not an independent human reviewer.

## Performance

Windows/Python 3.13, seeded synthetic data, three trials, median. 20,000 messages, 12 participants, 16-dimensional precomputed random embeddings, 30-second spacing, 300-second window. Compared with backed-up pre-change source; all matrices matched with NumPy allclose.

| Stage | Before | After |
| --- | ---: | ---: |
| Semantic window | 0.6754 s | 0.4612 s |
| Behavior window | 0.8296 s | 0.4119 s |

44,850 synthetic pairs: filter plus 100 visible rows and Tk update 0.0120 s; list search 0.0085 s (single observations, not guarantees).

Window loops now use itertuples instead of allocating Series per message; nickname fallback is vectorized; CPU inference loads one model and batches. No real-data throughput, cold model speedup, GPU speedup or end-to-end gain is claimed.

## Limitations

- No real chat database or authentication material read for testing. SQLCipher read-only flag reviewed, but no encrypted private database tested.
- Native-window screenshot/visual inspection was unavailable. Real Tk controls/event behavior tested; pixel layout, high-DPI and accessibility need visual review on target display.
- Remote PostgreSQL, third-party AI, GPU and private exports not exercised; CLI capabilities retained.
- Dense traffic in one time window can still be costly. Cancellation terminates only this app's worker. Atomic cache writes protect the prior cache but cancellation may leave an ignored temporary cache file.

Offline real-model verification passed with two synthetic sentences (2 x 384 embeddings). A 36-message synthetic SQLite database with Chinese/spaced path then completed real CPU encoding and produced 3 relationships without changing its SHA-256. Public fixed-revision safetensors were fetched separately with token=False and no Python model code; runtime remained offline. Model assets stay ignored and are not shipped in Git.
