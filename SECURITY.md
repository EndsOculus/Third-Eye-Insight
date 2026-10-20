# Security and privacy scope

Normal desktop/CLI analysis uses read-only source access. No original DB writes, forced unlocks or user application termination. Normal operation does not call external AI, download models or persist message embeddings. Optional local AI uses only an existing model. SQLCipher credentials are read automatically only for an encrypted candidate; never publish private config.

Default CLI/public results suppress body text and native account/message identifiers in favor of anonymous members and selected-row aliases. The explicitly authorized local GUI receives a separate in-memory identity map and displays verified local names/QQ; privacy mode masks inputs and display names, clears search and closes existing details. No identity map is added to default public results. Dates, counts and aliases can remain identifying in context; anonymization is not a guarantee. Logs contain fixed event codes; private caches, outputs and logs remain ignored. Legacy modules are retained but are outside the default v2 route.

Release review scans the explicit staged allowlist and reachable local Git snapshots for database/archive/cache/log/private-config filenames, common token/private-key patterns, hard-coded secrets and personal paths. Synthetic fixtures are reviewed as synthetic. This is bounded automated and manual inspection, not proof that zero sensitive information exists. Backups, model files, private screenshots and raw results are not committed. The explicitly reviewed docs/images/desktop-demo.png is a public synthetic application screenshot with empty input fields and no metadata; it contains no real conversation. Other screenshots stay outside Git. Only synthetic application-owned screenshots are eligible for Library delivery.

If sensitive history is found, stop publication and report; do not rewrite history or force push without separate authorization. Recovery source bundle remains outside the repository. Dependencies and inherited legacy modules can contain unrelated risks; this work does not certify all third-party packages.

The PyTorch requirement is restricted to `>=2.10.0,<2.11` to address
CVE-2025-2999 / GHSA-vgrw-7cvw-pwgx. An empty `PackedSequence` passed to
`unpack_sequence` could cause a native crash; the upstream fix rejects empty
batch sizes. Although the advisory lists 2.9.1 as patched, the official Windows
2.9.1 CPU wheel still crashed in the child-process regression, and the
[v2.9.1 source](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/PackedSequence.cpp)
lacks that check. The
[v2.10.0 source](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/PackedSequence.cpp)
contains it. Windows CPU behavior is verified locally; other platforms are not
certified by this check. The project does
not directly use these RNN APIs, and the approved
optional sentence encoder uses BERT on CPU, but the dependency is updated rather
than treating that limited exposure as a fix. The regression runs in a child
process and also checks a valid nonempty sequence. This update does not fix the
separate low-severity CVE-2025-3000 advisory; closing or auto-dismissing its
alerts is a triage decision, not a vulnerability repair.

Streaming analysis uses a private, parent-owned temporary SQLite working database containing decoded source records and identity-related state. It is removed on normal completion/error/cancel; process/system crashes or persistent file locks may leave it behind. There is no secure-erasure guarantee. The source stays read-only. Only explicit contact schemas, the ignored same-project contacts.local.json map, or the existing verified sender nickname field 40093 are used. The native nickname fallback is local-GUI-only and uses the latest valid selected-period sender name, rejecting conflicting names at the same timestamp; no body-based identity guesses or online lookup. Working disk, sampled process memory, time and participants remain bounded; sampled budgets are not operating-system hard isolation.
