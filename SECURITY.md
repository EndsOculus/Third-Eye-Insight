# Security and privacy scope

Normal desktop/CLI analysis uses read-only source access. No original DB writes, forced unlocks or user application termination. Normal operation does not call external AI, download models or persist message embeddings. Optional local AI uses only an existing model. SQLCipher credentials are read only when explicitly enabled; never publish private config.

Results suppress body text and native account/message identifiers in favor of anonymous members and selected-row aliases. Input controls still show user-entered paths and identifiers. Dates, counts and aliases can remain identifying in context; anonymization is not a guarantee. Logs contain fixed event codes; private caches, outputs and logs remain ignored. Legacy modules are retained but are outside the default v2 route.

Release review scans the explicit staged allowlist and reachable local Git snapshots for database/archive/cache/log/private-config filenames, common token/private-key patterns, hard-coded secrets and personal paths. Synthetic fixtures are reviewed as synthetic. This is bounded automated and manual inspection, not proof that zero sensitive information exists. Backups, model files, screenshots and raw results are not committed. Only synthetic application-owned screenshots are eligible for Library delivery.

If sensitive history is found, stop publication and report; do not rewrite history or force push without separate authorization. Recovery source bundle remains outside the repository. Dependencies and inherited legacy modules can contain unrelated risks; this work does not certify all third-party packages.
