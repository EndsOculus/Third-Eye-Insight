# Interaction v2 definitions

This release reports observations, not intimacy, confidence, emotional labels or causal relationships. There is no combined score, night-time bonus, variance-selected weight or semantic rank. Desktop and CLI share the same engine.

## Direct evidence

- Group: only valid structured mentions or references to a unique earlier source in the same conversation establish a directed event. Plain-text @, opaque native payloads, missing/future/same-time source references do not. A reply and mention to the same target do not duplicate the event.
- Private: the peer identifies direction; sender changes strictly later within a session provide response rounds. Same-time ambiguity clears adjacency. A gap over six hours starts another session; no response delay or response is inferred across that boundary.
- A directed round is a consecutive sender turn for one pair/thread/session; changing metadata type does not create another turn. Simultaneous group and private events compare every sender against the pair/thread state before the timestamp batch. Renumbering sender IDs or message IDs cannot impose a turn order; adjacency is cleared after ambiguous batches.
- Bidirectional sessions count group response rounds from both directions, or eligible private sender transitions; mentions alone do not create them. Common active days require both attributed directions on the date. One-way fraction is absolute directional-round difference divided by total rounds; undefined stays null.
- Displayed messages A/B are deduplicated source messages: group totals per member, private totals restricted to that pair. These differ from attributed message counts in details.
- Median response seconds covers every eligible group explicit-reference link or private sender transitions; the `latency_kind` field distinguishes `explicit_group_reply` from `private_sender_transition`. Group link delay/lexical samples are independent of response-round deduplication, so same-time reply-ID ordering cannot select one source arbitrarily. Only positive delays at most six hours are included; cross-session explicit links may still count as links, but never as latency samples. Private lexical samples are omitted when the source or target timestamp batch contains multiple messages, because no unique text-to-text correspondence exists; the timestamp can still support a delay, and no specific source-message ID is claimed. Lexical overlap is character/token Jaccard, an auxiliary measure, not semantic understanding. Common neighbors and fixed five-minute same-thread activity bins are auxiliary and never establish direct replies.

Stable IDs deduplicate only within their conversation namespace. Conflicting IDs are excluded and reported globally; their evidence-quality impact is restricted to senders and explicit mention/quoted-source participants across conflicting versions, rather than unrelated member pairs; equal text is not duplicate evidence. Repetition/forward counts are descriptive only. Evidence is unknown when structure coverage is incomplete or relevant references are unresolved. Fewer than three responses or two bidirectional sessions is marked insufficient; these are display thresholds, not statistical confidence. No observed response is distinct from unknown.

Evidence uses anonymous member labels and input-row aliases such as R000001, not native message IDs or body text. Aliases refer to the selected input ordering and are not stable across analyses. Detail samples are capped at 20 per pair; the offline evaluator requests a larger bounded sample explicitly. AI candidates are separate from direct evidence.

## Read-only import contract

Default `nt` reads the existing numeric NT schema conservatively. All message types are considered. Native BLOB body extraction supports only the observed protobuf body fields described below. Native directional recipients remain unresolved; JSON in message content is never trusted as metadata.

Explicit `tei` selection reads a separate SQLite table `tei_messages_v1` with columns `sender_id`, `peer_id`, `message_id`, `timestamp`, `conversation_id`, `mode`, `content`, `metadata`. Mode is group or c2c; conversation_id is the group/owner identifier, peer_id identifies a private peer, timestamp is Unix seconds. IDs for people are ASCII decimal strings. Content is UTF-8 TEXT. Metadata is separate JSON, for example:

```json
{"schema":"tei.message.v1","reply_to":"earlier-message-id","mentions":["2"],"forwarded":false}
```

Absent reply_to and an empty mentions list are valid known-empty metadata. Do not manufacture metadata from body text. The caller is responsible for provenance and consent of imports; no import writer modifies the source database.

Dates are inclusive Asia/Shanghai calendar dates, translated to a half-open SQL interval before limits. Plain SQLite uses a read-only transaction snapshot. SQLCipher preflight and extraction use separate read-only connections; concurrent changes may produce differing counts, but extraction independently checks rows and callback byte budgets. This encrypted path was not exercised with private data.

Limits: 50,000 rows, 300 participants, 100,000 recipient entries, 32 MiB selected payload budget, 8,192 bytes per content/metadata field, bounded IDs. Oversized fields become unknown, invalid sender/time rows are counted and excluded. Exceeding selection budgets fails rather than returning a partial result. Full-table date scans may still be slow when the original schema lacks indexes; this tool never adds indexes to original data.

## Direction and unknown values

`reply_links_a/b` count explicit group reference links, `response_rounds_a/b` count eligible responses by direction, and `mention_rounds_a/b` count directed @ rounds. General `rounds_a/b` include all attributed direction events, so their balance is not reply reciprocity. Private response rounds are observed speaker transitions, not confirmed answers. Member A/B follow the displayed anonymous pair order.

With incomplete structure, zero direction-derived values display unknown; positive counts display "observed N" with the incomplete-structure status, never a lower-bound claim. These statistics are not all monotonic: events at 00:00 and 08:00 form two sessions/turns when a 04:00 event is unresolved, but one session/turn when that bridge resolves to the same direction. Ratios and delays are limited to observed samples, not bounds on the complete population. Raw deduplicated message counts remain raw counts. CLI numeric counters retain observed values with the mandatory `evidence`/`structure_coverage` status; clients must not interpret unknown zero counters as proven absence.

## Limited native BLOB support and provenance

Read-only inspection of the separate PriChat generic `prichat/messages.py`, `prichat/protobuf.py` and synthetic parser tests established the implemented field contract: a 40800 container holds 45101 UTF-8 body elements; a 47423 subtree caches quoted content. code3 has its own small parser, with no runtime dependency on PriChat and no copied private configuration, keys, messages or databases.

The parser accepts direct 45101 elements and 40800 nesting, preserves repeated body elements, and excludes the entire 47423 cache from the sender's body. Cache presence is reported, not decoded into a recipient. Unknown fields are opaque, never recursively searched for the longest string. Inputs are bounded to 8,192 bytes, 256 fields and depth 8. Invalid/truncated known structures yield no partial body. Body text stays internal and can feed the existing local encoder; it does not appear in normal UI/CLI/logs. A detected quoted-cache message abstains from AI recipient attribution because its actual quoted target remains unknown.

No verified mapping from native quote-cache fields to unique message ID/sender, or native @ elements to recipient IDs, was found in this generic renderer. A cached body and optional display name are insufficient: equal text is not unique identity. Therefore no native `reply_to`, mentions or forward flag is synthesized. Other container layouts, compression, media cards and unknown versions may remain without text; no fallback GBK conversion, string matching or network lookup is attempted. Constructed binary fixtures validate the supported paths, not complete compatibility with all real NT versions.
