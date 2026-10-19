# v2 synthetic performance measurement

Windows / Python 3.13, 2026-10-19. Three in-process runs, 50,000 generated messages / 300 members / one group, one-second spacing, known-empty recipient metadata. Median engine-only elapsed: **0.8275 s**; individual runs: 0.7644, 0.8794, 0.8275 s. This includes aggregation and pair construction, excludes fixture generation, SQLite I/O, model loading, AI, process startup and GUI rendering. Output pairs: 44,850.

All direct response counts remain zero in this no-target fixture despite dense coactivity. The fixture tests bounded aggregation, not worst-case recipient fan-out or real-data performance. No before/after speedup is claimed because v2 changes the algorithm and meaning of results.

GUI regression separately exercises 44,850 preconstructed pairs with a 100-row visible page. Current screenshots use 1,200 synthetic messages / 12 members / 66 pairs; the insufficient private example uses 220 selected rows / 11 pairs in a single day. AI evaluation is separate: 12 synthetic labeled cases and actual existing local model, not an end-to-end speed benchmark. See AI_ASSISTANCE.md.

The primary performance protections are SQL date filtering before limits, bounded selection and payloads, pair-local aggregation, per-process cancellation and paginated widget updates. An unindexed source scan and large fan-out may still be slow. Original databases are never indexed or modified by this application.


## Streaming follow-up — synthetic measurements, 2026-10-19

Windows / Python 3.13, local SSD, CPU analysis, AI off. Two participants alternate once per second, unique IDs, short synthetic text. Fixture creation is measured separately; analysis includes read, decode, disk-backed deduplication/indexing, incremental statistics and cleanup, excludes GUI startup. RSS is sampled every 50 ms for the Python process; this is an observed peak, not an allocation bound. These are two sequential runs in one process, not cold-cache benchmarks.

| Selected rows | Source bytes | Fixture creation | Complete analysis | Sampled peak RSS |
|---:|---:|---:|---:|---:|
| 100,001 | 4,440,064 | 0.086 s | 7.136 s | 72.95 MiB |
| 300,001 | 13,565,952 | 0.229 s | 21.806 s | 126.00 MiB |

Both completed with all rows and exactly rows minus one observed private transitions. No prior-version speedup is claimed: the earlier default rejected selections over 50,000 rows. This synthetic shape does not predict every real database, encrypted prefix-copy cost, many-member group, long body, conflict pattern or optional model workload. Resource budgets may reject expensive selections with an explicit message and no partial result.
