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
- Initial delivery lacked screenshots; the visual follow-up below now covers the current Windows display. Other DPI scales and accessibility remain unverified.
- Remote PostgreSQL, third-party AI, GPU and private exports not exercised; CLI capabilities retained.
- Dense traffic in one time window can still be costly. Cancellation terminates only this app's worker. Atomic cache writes protect the prior cache but cancellation may leave an ignored temporary cache file.

Offline real-model verification passed with two synthetic sentences (2 x 384 embeddings). A 36-message synthetic SQLite database with Chinese/spaced path then completed real CPU encoding and produced 3 relationships without changing its SHA-256. Public fixed-revision safetensors were fetched separately with token=False and no Python model code; runtime remained offline. Model assets stay ignored and are not shipped in Git.

## Independent review follow-up

The parent task provided an independent review finding two private-chat defects. Actual synthetic SQLite extraction first reproduced empty results from swapped SQL parameters; correcting that exposed KeyError(text_embedding) in desktop private metrics. Both are fixed. Added single/multiple-peer extraction tests and a desktop integration test that runs the real embedding attachment path with only the model encoder mocked; no embedding column is prefilled. Full local suite now has 32 tests; published suite has 31. Previous counts above describe the initial revision.

The independent review also raised worker-exit event loss and artificial semantic scores. Both were reproduced before fixes: deterministic queue/process stubs lost result and error events, and three messages spaced 601 seconds apart produced nonzero semantic scores; repeated tied inputs changed ranking. New tests cover both terminal-event races, all-zero no-interaction scores, repeatability and observed-zero-cosine versus unobserved pairs. Normal worker exit now drains final events after join. Semantic scaling uses observed masks and deterministic midranks. CLI signatures remain compatible; scores for missing/tied observations intentionally change. The performance comparison above belongs to the initial revision, before this semantic correction; it is not an equivalence claim for the corrected scoring rule.

Private-chat real-model check also passed: 12 synthetic alternating messages, actual SQLite extraction with no embedding column, full offline CPU encoding, one relationship result and unchanged source SHA-256 (5.121 s in one observation). Temporary test data was removed normally.

## Real-window visual follow-up

Captured the actual Tk Desktop instance through Pillow ImageGrab.grab(window=Tk frame HWND), not HTML/design mockups or a full-screen crop. Used only the built-in synthetic demo; selected-file/account fields stayed blank. Captures were inspected at 1120x780 and 900x700 for chart labels, text contrast, form controls, result table, search/pagination, empty state, validation, progress and cancellation. Screenshot files are private deliverables, not repository assets.

The first capture exposed a collapsed table at default size and clipped pagination at compact size. Reduced form whitespace, moved exclusion input into the options row, reserved table height, compacted chart spacing and colored the progress bar. Added a persistent synthetic-data label. Final screenshots show table and footer fully within both window sizes, with no overlapping chart text or controls. Added geometry regressions for both sizes. Current full local test count: 34; published suite: 33. Test windows close after capture. DPI/accessibility coverage remains limited to this display/session.

The visual refinement replaces raised panels/buttons and table borders with flat white cards, neutral text, a single purple accent, separate section/statistics hierarchy and striped rows. Completed/error progress is hidden, with a regression assertion. Fresh actual-window captures cover default results, compact search and empty state; validation/progress/cancellation were also inspected. All 34 local tests pass (33 published); screenshots contain only synthetic anonymous members.
