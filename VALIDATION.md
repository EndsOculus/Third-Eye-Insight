# Current v2 validation

2026-10-19, Windows / Python 3.13. 93 tests passed locally; critical Ruff checks passed. All new fixtures and screenshots are synthetic. The published tests directory excludes an unrelated ignored local test.

Coverage includes real Tk event loops, duplicate start, cancel/restart/close, terminal worker race, invalid dates/paths, empty results, search and 100-row pagination with 44,850 synthetic pair rows, default and 900x700 geometry. Application-owned screenshots verify default, unknown coverage and insufficient evidence; keyboard Enter opens a scrolling details dialog. Hidden pythonw completes and closes owned windows.

Read-only synthetic SQLite tests cover Chinese/spaced paths, unchanged file hash, date-before-limit, missing paths, hostile text metadata, oversized IDs/payloads, malformed sender/time, SQL bounds and missing fields. Engine tests cover ID conflicts, equal timestamps and reordered IDs, reply resolution, repeated text, multiple conversations, per-pair private isolation, session gaps, sparse evidence and holdout-generated counterexamples. Evaluation includes over-20-example regression so UI sample limits do not bias metrics. UI/CLI boundaries replace untrusted native IDs with row aliases.

An independent read-only reviewer found ordering, pair-total leakage, evaluator sampling, native-ID disclosure and extraction-size issues. Fixes have dedicated regression tests. Actual offline encoder evaluation is documented in AI_ASSISTANCE.md; it includes a false attribution.

No real private database, credentials, external model, PostgreSQL, GPU, encrypted private DB or broader DPI/accessibility validation was performed. Synthetic validation does not prove validity on real human relationships. Raw NT binary recipient structure remains unsupported. Current performance measurements are in BENCHMARK_V2.md; historical v1 speed figures do not describe this changed algorithm.
