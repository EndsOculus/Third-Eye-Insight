# v2 synthetic performance measurement

Windows / Python 3.13, 2026-10-19. Three in-process runs, 50,000 generated messages / 300 members / one group, one-second spacing, known-empty recipient metadata. Median engine-only elapsed: **0.6931 s**; individual runs: 0.6931, 0.6740, 0.7388 s. This includes aggregation and pair construction, excludes fixture generation, SQLite I/O, model loading, AI, process startup and GUI rendering. Output pairs: 44,850.

All direct response counts remain zero in this no-target fixture despite dense coactivity. The fixture tests bounded aggregation, not worst-case recipient fan-out or real-data performance. No before/after speedup is claimed because v2 changes the algorithm and meaning of results.

GUI regression separately exercises 44,850 preconstructed pairs with a 100-row visible page. Current screenshots use 1,200 synthetic messages / 12 members / 66 pairs; the insufficient private example uses 220 selected rows / 11 pairs in a single day. AI evaluation is separate: 12 synthetic labeled cases and actual existing local model, not an end-to-end speed benchmark. See AI_ASSISTANCE.md.

The primary performance protections are SQL date filtering before limits, bounded selection and payloads, pair-local aggregation, per-process cancellation and paginated widget updates. An unindexed source scan and large fan-out may still be slow. Original databases are never indexed or modified by this application.
