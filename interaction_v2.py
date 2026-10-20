"""Interaction v2: counts and coverage, not an intimacy score or probability."""
from collections import Counter, defaultdict
from datetime import timedelta
from statistics import median
import re

VERSION = 'interaction-v2.0'
SESSION_GAP = 6 * 3600
COACTIVE_BIN = 300


def lexical_overlap(a, b):
    """Auxiliary character/token Jaccard, not semantic reply recognition."""
    tokens = lambda s: set(re.findall(r'[\u4e00-\u9fff]|[a-z0-9_]+', s[:4096].lower()))
    x, y = tokens(a), tokens(b)
    return len(x & y) / len(x | y) if x | y else None


def analyze_records(records, mode='group', self_id='1', session_gap=SESSION_GAP, coactive_bin=COACTIVE_BIN, observation=None, evidence_limit=20):
    if mode not in ('group', 'c2c') or session_gap <= 0 or coactive_bin <= 0 or not 0 <= evidence_limit <= 100000:
        raise ValueError('Invalid interaction parameters')
    streamed = getattr(records, 'prepared', False)
    if streamed:
        raw_rows = records.raw_rows
        conflicts = range(records.conflicts)
        duplicate_rows = records.duplicate_rows
        conflict_users = records.conflict_users
        conflict_private = records.conflict_private
    else:
        if sum(len(r.get('mentions', []))+bool(r.get('reply_to')) for r in records)>100000:
            raise ValueError('Selection exceeds record/recipient evidence budget; narrow the dates')
        raw_rows = len(records)
        # Deduplicate only stable IDs in their conversation namespace. Conflicting IDs
        # cannot provide directional evidence, nor resolve references.
        identities, conflicts, duplicate_rows = {}, set(), 0
        for r in records:
            if not r.get('message_id'):
                continue
            key = (r.get('thread', r.get('peer', 'group')), r['message_id'])
            signature = (r['sender'], r['timestamp'], r['text'], r.get('reply_to'), tuple(r['mentions']), r.get('forwarded', False), r['structure'])
            if key in identities:
                if identities[key] != signature:
                    conflicts.add(key)
            else:
                identities[key] = signature
        # Excluded conflicting rows affect their participants, not every group pair.
        # Include every version's sender/explicit targets and possible quoted authors;
        # the global conflict count remains available separately as a quality warning.
        source_authors = defaultdict(set)
        for row in records:
            if row.get('message_id'):
                source_authors[(row.get('thread', row.get('peer', 'group')), row['message_id'])].add(row['sender'])
        conflict_users = set()
        conflict_private = set()
        for row in records:
            thread = row.get('thread', row.get('peer', 'group'))
            if (thread, row.get('message_id')) in conflicts:
                conflict_users.add((thread, row['sender']))
                if mode == 'c2c' and row['peer'] != str(self_id) and row['sender'] in (str(self_id), row['peer']):
                    conflict_private.add(row['peer'])
                if row['structure'] == 'known':
                    conflict_users.update((thread, u) for u in row['mentions'])
                    if row.get('reply_to'):
                        conflict_users.update((thread, u) for u in source_authors.get((thread, row['reply_to']), ()))
        unique, seen = [], set()
        for r in records:
            key = (r.get('thread', r.get('peer', 'group')), r.get('message_id'))
            if r.get('message_id') and key in conflicts:
                continue
            if r.get('message_id') and key in seen:
                duplicate_rows += 1
                continue
            seen.add(key)
            unique.append(r)
        records = sorted(unique, key=lambda r: (r['timestamp'], r.get('message_id') or '', r.get('order', 0)))
    self_id = str(self_id)
    users = {r['sender'] for r in records}
    if mode == 'c2c':
        users.update(r['peer'] for r in records)
        users.add(self_id)
        users.update(conflict_private)
    else:
        users.update(t for r in records if r['structure'] == 'known' for t in r['mentions'])
    if len(users) > 300:
        raise ValueError('Selected records/targets exceed 300 participants')
    users = sorted(users, key=lambda u: (len(u), u))
    pairs = {}
    def pair(a, b):
        key = tuple(sorted((a, b), key=lambda u: (len(u), u)))
        if key not in pairs:
            pairs[key] = dict(a=key[0], b=key[1], messages=Counter(), rounds=Counter(),
                              response_rounds=0, mention_rounds=0, reply_links=0, bidirectional_sessions=0,
                              sessions=0, response_seconds=records.new_list() if streamed else [], overlap=records.new_list() if streamed else [], days=defaultdict(records.new_set if streamed else set),
                              replies=Counter(), responses=Counter(), mentions=Counter(),
                              session_directions=set(), last=None, last_sender=None, last_kind=None,
                              last_response=None, duplicates=0, texts=set(), forwarded=0, states={}, examples=[])
        return pairs[key]
    def event(a, b, row, kind, source=None, turn_snapshot=None):
        if streamed: records.check()
        if a == b:
            return
        p = pair(a, b)
        state = p['states'].setdefault(row.get('thread', row.get('peer','group')), dict(last=None, last_sender=None, texts=records.new_set() if streamed else set(), session_directions=set()))
        ts = row['timestamp']
        new_session = state['last'] is None or (ts-state['last']).total_seconds() > session_gap
        if new_session:
            p['sessions'] += 1
            state['session_directions'] = set()
            state['last_sender'] = None
            state['texts'] = records.new_set() if streamed else set()
        new_round = new_session or state['last_sender'] != a
        if turn_snapshot is not None:
            new_round = (turn_snapshot.get('last') is None
                         or (ts-turn_snapshot['last']).total_seconds() > session_gap
                         or turn_snapshot.get('last_sender') != a)
            if not new_round:
                state['replied_in_turn'] = turn_snapshot.get('replied_in_turn', False)
                state['mentioned_in_turn'] = turn_snapshot.get('mentioned_in_turn', False)
        if new_round:
            p['rounds'][a] += 1
            state['replied_in_turn'] = False
            state['mentioned_in_turn'] = False
        p['messages'][a] += 1
        if len(p['examples']) < evidence_limit:
            p['examples'].append(dict(thread=row.get('thread', row.get('peer','group')), message_row=row.get('order',0)+1, message_id=row.get('message_id'), source_message_id=source.get('message_id') if source else None, sender=a, target=b, kind=kind))
        p['days'][a].add(ts.date())
        if row['text'] and (a, row['text']) in state['texts']:
            p['duplicates'] += 1
        state['texts'].add((a, row['text']))
        p['forwarded'] += int(row.get('forwarded', False))
        if kind == 'mention' and not state.get('mentioned_in_turn',False):
            p['mention_rounds'] += 1
            p['mentions'][a] += 1
            state['mentioned_in_turn'] = True
        if kind == 'reply':
            p['reply_links'] += 1
            p['replies'][a] += 1
        response = (kind == 'reply' and not state.get('replied_in_turn',False)) or (new_round and (kind == 'private' and not new_session and state['last_sender'] is not None and state['last_sender'] != a and ts > state['last']))
        if response:
            state['replied_in_turn'] = True
            p['response_rounds'] += 1
            p['responses'][a] += 1
            before = len(state['session_directions'])
            state['session_directions'].add(a)
            # Private conversation's prior turn is evidence of the other side,
            # unlike an unreciprocated group mention/reference.
            if kind == 'private':
                state['session_directions'].add(b)
            if before < 2 and len(state['session_directions']) == 2:
                p['bidirectional_sessions'] += 1
        # Explicit-link samples are independent of turn deduplication/order.
        if source is not None and (kind == 'reply' or response):
            delta = (ts-source['timestamp']).total_seconds()
            # Explicit cross-session references remain observed reply links,
            # but never contribute an across-session latency.
            if 0 < delta <= session_gap:
                p['response_seconds'].append(delta)
                if (not row.get('forwarded') and not source.get('forwarded')
                            and not row.get('lexical_ambiguous') and not source.get('lexical_ambiguous')):
                    overlap = lexical_overlap(row['text'], source['text'])
                    if overlap is not None:
                        p['overlap'].append(overlap)
        state['last'], state['last_sender'] = ts, a

    def dispatch_batch(entries):
        # Each sender is compared with the pair/thread state BEFORE this time
        # batch. Sender IDs never impose a causal order within simultaneous data.
        grouped = defaultdict(list)
        for entry in entries:
            a, b, row, kind, source = entry
            key = (tuple(sorted((a,b), key=lambda u:(len(u),u))), row.get('thread',row.get('peer','group')))
            grouped[key].append(entry)
        for (key, thread), batch in grouped.items():
            p = pair(*key)
            snapshot = dict(p['states'].get(thread, {}))
            senders = defaultdict(list)
            for entry in batch:
                senders[entry[0]].append(entry)
            for sender_entries in senders.values():
                for index, entry in enumerate(sender_entries):
                    event(*entry, turn_snapshot=snapshot if index == 0 else None)
            if len(senders) > 1:
                p['states'][thread]['last_sender'] = None

    unresolved = 0
    unresolved_users = set()
    if mode == 'c2c':
        if streamed:
            thread_items = records.iter_threads()
        else:
            threads = defaultdict(list)
            for r in records:
                if r['peer'] != self_id and r['sender'] in (self_id, r['peer']):
                    threads[(r['peer'],r.get('thread',r['peer']))].append(r)
            thread_items = threads.items()
        from itertools import groupby
        for (peer, thread_key), thread in thread_items:
            previous = None
            for ts, batch_iter in groupby(thread,key=lambda r:r['timestamp']):
                batch = sorted(records.bounded_batch(batch_iter) if streamed else batch_iter,key=lambda r:(r['sender'],r.get('message_id') or '',r.get('order',0)))
                ambiguous = len({r['sender'] for r in batch}) > 1
                entries = []
                for r in batch:
                    if len(batch) > 1:
                        r = dict(r, lexical_ambiguous=True)
                    entries.append((r['sender'], peer if r['sender'] == self_id else self_id, r,
                                    'private_ambiguous' if ambiguous else 'private', previous))
                    previous = r
                dispatch_batch(entries)
                if len(batch) > 1:
                    # Timestamp is usable, but no unique prior message/body exists.
                    previous = dict(previous, message_id=None)
                if ambiguous:
                    previous = None
    else:
        # Duplicate source IDs are ambiguous; never guess which message was quoted.
        if streamed:
            from streaming_store import Lookup
            lookup = Lookup(records)
        else:
            lookup = {(r.get('thread', r.get('peer', 'group')), r['message_id']): r for r in records if r.get('message_id')}
        pending=records.pending() if streamed else []
        for r in records:
            if r['structure'] != 'known':
                continue
            targets = {}
            if r.get('reply_to'):
                source = lookup.get((r.get('thread', r.get('peer', 'group')), r['reply_to']))
                if source is None or source['timestamp'] >= r['timestamp'] or source['sender'] == r['sender'] or source is r:
                    unresolved += 1
                    unresolved_users.add(r['sender'])
                else:
                    targets[source['sender']] = ('reply', source)
            for target in r['mentions']:
                if target != r['sender'] and target not in targets:
                    targets[target] = ('mention', None)
            for target, (kind, source) in targets.items():
                pending.append((r['sender'], target, r, kind, source))
        from itertools import groupby
        if streamed:
            for batch in pending.batches():
                dispatch_batch(batch)
        else:
            pending.sort(key=lambda e:(e[2]['timestamp'],e[2].get('thread',e[2].get('peer','group')),e[0],e[1],e[2].get('message_id') or ''))
            for timestamp, batch in groupby(pending,key=lambda e:e[2]['timestamp']):
                dispatch_batch(list(batch))

    bins, days, source_counts, known_counts = defaultdict(records.new_set if streamed else set), defaultdict(records.new_set if streamed else set), Counter(), Counter()
    epoch = __import__('datetime').datetime(1970, 1, 1)
    for r in records:
        u = r['sender']
        bins[u].add((r.get('thread',r.get('peer','group')), int((r['timestamp']-epoch).total_seconds()) // coactive_bin))
        days[u].add(r['timestamp'].date())
        source_counts[u] += 1
        known_counts[u] += r['structure'] == 'known'
    neighbors = defaultdict(set)
    for (a, b), p in pairs.items():
        if sum(p['rounds'].values()):
            neighbors[a].add(b); neighbors[b].add(a)
    if mode == 'group':
        for i, a in enumerate(users):
            for b in users[i+1:]:
                pair(a, b)
    user_threads = defaultdict(set)
    for r in records:
        thread = r.get('thread', r.get('peer', 'group'))
        user_threads[r['sender']].add(thread)
        if r['structure'] == 'known':
            for target in r['mentions']:
                user_threads[target].add(thread)
    for thread, user in conflict_users:
        user_threads[user].add(thread)
    # Private zero/one-way threads are present even without a reply.
    private_known=Counter()
    if mode=='c2c':
        for peer in conflict_private:
            pair(self_id, peer)
        for r in records:
            private_known[r['peer']] += r['structure']=='known'
    out = []
    for (a, b), p in pairs.items():
        na, nb = p['messages'][a], p['messages'][b]
        ra, rb = p['rounds'][a], p['rounds'][b]
        covered = known_counts[a]+known_counts[b]
        available = source_counts[a]+source_counts[b]
        if mode=='c2c':
            covered=private_known[b if a==self_id else a]
            available=na+nb
        coverage = covered/available if available else 0.0
        conflict_affected = (b if a == self_id else a) in conflict_private if mode == 'c2c' else any(
            (thread, a) in conflict_users or (thread, b) in conflict_users
            for thread in user_threads[a] & user_threads[b])
        if conflict_affected or (mode == 'group' and (coverage < 1 or a in unresolved_users or b in unresolved_users)):
            evidence = 'coverage_unknown'
        elif not p['response_rounds']:
            evidence = 'no_observed_response'
        elif p['response_rounds'] < 3 or p['bidirectional_sessions'] < 2:
            evidence = 'insufficient'
        else:
            evidence = 'observed'
        pair_days = (p['days'][a] & p['days'][b])
        out.append(dict(a=a, b=b, evidence_examples=p['examples'], response_rounds=p['response_rounds'],
                        directed_rounds=ra+rb, rounds_a=ra, rounds_b=rb,
                        response_rounds_a=p['responses'][a], response_rounds_b=p['responses'][b],
                        reply_links_a=p['replies'][a], reply_links_b=p['replies'][b],
                        mention_rounds_a=p['mentions'][a], mention_rounds_b=p['mentions'][b],
                        latency_kind='explicit_group_reply' if mode=='group' else 'private_sender_transition',
                        bidirectional_sessions=p['bidirectional_sessions'], sessions=p['sessions'],
                        messages_a=na, messages_b=nb, common_active_days=len(pair_days),
                        source_messages_a=na if mode=='c2c' else source_counts[a], source_messages_b=nb if mode=='c2c' else source_counts[b],
                        one_way_fraction=abs(ra-rb)/(ra+rb) if ra+rb else None,
                        median_response_seconds=(p['response_seconds'].median() if streamed else median(p['response_seconds'])) if p['response_seconds'] else None,
                        latency_samples=len(p['response_seconds']),
                        lexical_overlap_mean=(p['overlap'].total if streamed else sum(p['overlap']))/len(p['overlap']) if p['overlap'] else None,
                        lexical_samples=len(p['overlap']),
                        coactive_bins=len(bins[a] & bins[b]) if mode == 'group' else None,
                        common_neighbors=len(neighbors[a] & neighbors[b]) if mode == 'group' else None,
                        mention_rounds=p['mention_rounds'], reply_links=p['reply_links'],
                        repeated_messages=p['duplicates'], forwarded_messages=p['forwarded'],
                        structure_coverage=coverage, evidence=evidence))
    out.sort(key=lambda p: (-p['response_rounds'], -p['bidirectional_sessions'], p['a'], p['b']))
    bounds = observation or {}
    return dict(version=VERSION, pairs=out, rows=len(records), users=len(users),
                observation=dict(requested_start=bounds.get('start'), requested_end=bounds.get('end'),
                                 actual_start=records[0]['timestamp'].isoformat(' ') if records else None,
                                 actual_end=records[-1]['timestamp'].isoformat(' ') if records else None,
                                 timezone='Asia/Shanghai', session_gap_seconds=session_gap, coactive_bin_seconds=coactive_bin),
                coverage=dict(structured_rows=sum(known_counts.values()), total_rows=len(records), raw_rows=raw_rows,
                              duplicate_rows=duplicate_rows, conflicting_ids=len(conflicts),
                              unknown_rows=sum(r['structure']!='known' for r in records), unresolved_references=unresolved,
                              simultaneous_rows=records.simultaneous if streamed else sum(records[i]['timestamp']==records[i-1]['timestamp'] for i in range(1,len(records)))),
                validity='Real-data measurement validity has not been established; synthetic tests are structural checks.')
