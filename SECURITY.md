# Security and privacy scope

Desktop stays offline. SQLite/SQLCipher handles are read-only. Startup header stripping/deletion removed. Pickle caches never deserialized. JSON caches and fixed-event rotating logs remain ignored. Added tests contain no actual messages, names, secrets or private screenshots.

## Dependency verification — 2026-10-19 UTC

Queried repository Dependabot API and GitHub Advisory Database. Old emails were not treated as current facts. Installed anyio 4.13.0, sentence-transformers 5.3.0, Pillow 12.1.1 and transformers 5.4.0 were still covered by open alerts.

Constraints now require anyio >=4.14.2,<4.15; sentence-transformers >=5.6,<6; Pillow >=12.3,<13; transformers >=5.10,<5.11; setuptools >=83,<84; idna >=3.15. PyTorch >=2.8,<2.9 replaces an unshippable local 2.6 CUDA wheel reference. Lock changes include required transitives. Only project environment changed, not system Python/drivers.

References:

- https://github.com/advisories/GHSA-jhr6-gm9c-rqjv (local model loading, fixed 5.6.0)
- https://github.com/advisories/GHSA-82r6-8w77-94w6 (AnyIO TLS hostname, fixed 4.14.2)
- https://github.com/advisories/GHSA-jjj6-mw9f-p565 (Pillow PDF decompression, fixed 12.3.0)
- https://github.com/advisories/GHSA-xrqw-3rrv-vx5w (Transformers path traversal, fixed 5.10.0)
- https://github.com/advisories/GHSA-x9r9-c232-4q39 lists affected 4.49.0 through 5.8.1 and no formally named patched version. Selected version is outside that range; this is not a claim of a designated patch. Desktop forces offline loading with trust_remote_code=False and does not use custom generation.

Versions alone are not an absolute guarantee. Default-branch alerts can stay open until merge/rescan; this change does not dismiss them.

## Publication checks

Explicit file allowlist, staged diff and outgoing commits reviewed. Exclude databases/WAL/SHM, caches, logs, exports, archives, backups, config.json, credentials, user paths and private samples. Historical tracked snapshots scanned without printing secret candidates; no history rewriting. Pattern scans cannot prove zero sensitive content.

CLI exports may contain private data; optional AI reporting sends it externally. They are outside the desktop offline boundary. Treat configuration and all output as private.

Local scan covered the staged snapshot plus both existing local historical commits (44 unique file versions after CI was added). Seven historical README path hits were generic placeholder examples; staged credential/path/prohibited-file scan had no findings. No history was rewritten. This coverage is pattern-based and does not extend to every remote ref or all possible secret formats.
