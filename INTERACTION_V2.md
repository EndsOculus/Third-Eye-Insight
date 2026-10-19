# Interaction v2 definitions

This release reports observations, not intimacy, confidence, emotional labels or causal relationships. There is no combined score, night-time bonus, variance-selected weight or semantic rank. Desktop and CLI share the same engine.

## Direct evidence

- Group: only valid structured mentions or references to a unique earlier source in the same conversation establish a directed event. Plain-text @, opaque native payloads, missing/future/same-time source references do not. A reply and mention to the same target do not duplicate the event.
- Private: the peer identifies direction; sender changes strictly later within a session provide response rounds. Same-time ambiguity clears adjacency. A gap over six hours starts another session; no response delay or response is inferred across that boundary.
- A directed round is a consecutive sender turn for one pair/thread/session; changing metadata type does not create another turn. Simultaneous group events are batched so message-ID ordering cannot create extra turns.
- Bidirectional sessions contain both attributed directions. Common active days require both attributed directions on the date. One-way fraction is absolute directional-round difference divided by total rounds; undefined stays null.
- Displayed messages A/B are deduplicated source messages: group totals per member, private totals restricted to that pair. These differ from attributed message counts in details.
- Median response seconds includes only eligible within-session private transitions. Lexical overlap is character/token Jaccard, an auxiliary measure, not semantic understanding. Common neighbors and fixed five-minute same-thread activity bins are auxiliary and never establish direct replies.

Stable IDs deduplicate only within their conversation namespace. Conflicting IDs are excluded and reported; equal text is not duplicate evidence. Repetition/forward counts are descriptive only. Evidence is unknown when structure coverage is incomplete or relevant references are unresolved. Fewer than three responses or two bidirectional sessions is marked insufficient; these are display thresholds, not statistical confidence. No observed response is distinct from unknown.

Evidence uses anonymous member labels and input-row aliases such as R000001, not native message IDs or body text. Aliases refer to the selected input ordering and are not stable across analyses. Detail samples are capped at 20 per pair; the offline evaluator requests a larger bounded sample explicitly. AI candidates are separate from direct evidence.

## Read-only import contract

Default `nt` reads the existing numeric NT schema conservatively. All message types are considered. Native binary decoding is unsupported; JSON in message content is never trusted as metadata.

Explicit `tei` selection reads a separate SQLite table `tei_messages_v1` with columns `sender_id`, `peer_id`, `message_id`, `timestamp`, `conversation_id`, `mode`, `content`, `metadata`. Mode is group or c2c; conversation_id is the group/owner identifier, peer_id identifies a private peer, timestamp is Unix seconds. IDs for people are ASCII decimal strings. Content is UTF-8 TEXT. Metadata is separate JSON, for example:

```json
{"schema":"tei.message.v1","reply_to":"earlier-message-id","mentions":["2"],"forwarded":false}
```

Absent reply_to and an empty mentions list are valid known-empty metadata. Do not manufacture metadata from body text. The caller is responsible for provenance and consent of imports; no import writer modifies the source database.

Dates are inclusive Asia/Shanghai calendar dates, translated to a half-open SQL interval before limits. Plain SQLite uses a read-only transaction snapshot. SQLCipher preflight and extraction use separate read-only connections; concurrent changes may produce differing counts, but extraction independently checks rows and callback byte budgets. This encrypted path was not exercised with private data.

Limits: 50,000 rows, 300 participants, 100,000 recipient entries, 32 MiB selected payload budget, 8,192 bytes per content/metadata field, bounded IDs. Oversized fields become unknown, invalid sender/time rows are counted and excluded. Exceeding selection budgets fails rather than returning a partial result. Full-table date scans may still be slow when the original schema lacks indexes; this tool never adds indexes to original data.
