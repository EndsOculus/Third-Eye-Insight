"""Read-only v2 adapter. Opaque NT payloads are NOT reverse-engineered as replies."""
import json
from datetime import datetime, timedelta, timezone
from database_access import open_database, DatabaseError

SHANGHAI = timezone(timedelta(hours=8))
MAX_PAYLOAD = 8192
MAX_TOTAL_PAYLOAD = 32 * 1024 * 1024


def date_bounds(start='', end=''):
    """Inclusive local dates -> Unix [start, end), independent of host timezone."""
    lo = int(datetime.fromisoformat(start).replace(tzinfo=SHANGHAI).timestamp()) if start else None
    hi = int((datetime.fromisoformat(end).replace(tzinfo=SHANGHAI) + timedelta(days=1)).timestamp()) if end else None
    if lo is not None and hi is not None and lo >= hi:
        raise ValueError('Invalid observation dates')
    return lo, hi


def decode_payload(raw, allow_structure=False):
    """Only the documented tei.message.v1 import envelope carries trusted structure.

    Ordinary NT text, arbitrary JSON/protobuf, textual @digits and opaque blobs
    cannot establish recipients. Unknown is distinct from a known empty target.
    No GBK conversion, longest-substring cleaning, or heuristic @ recognition.
    """
    result = dict(text='', structure='unknown', reply_to=None, mentions=[], forwarded=False)
    if not isinstance(raw, str):
        return result
    if len(raw.encode('utf-8')) > MAX_PAYLOAD:
        result['structure'] = 'oversize'
        return result
    result['text'] = raw
    if not allow_structure or not raw.lstrip().startswith('{'):
        return result
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict) or obj.get('schema') != 'tei.message.v1':
            return result
        return decode_metadata(obj, dict(result, text=obj.get('text', '')))
    except (ValueError, TypeError, RecursionError):
        result['structure'] = 'invalid'
        return result


def decode_metadata(obj, result):
    """Validate an already parsed envelope without reserializing separate fields."""
    if not isinstance(obj, dict) or obj.get('schema') != 'tei.message.v1':
        return result
    text = result['text']
    reply = obj.get('reply_to')
    mentions = obj.get('mentions', [])
    forwarded = obj.get('forwarded', False)
    if (not isinstance(text, str) or not isinstance(mentions, list) or len(mentions) > 300
            or any(not isinstance(v, str) or not v.isascii() or not v.isdigit() for v in mentions)
            or (reply is not None and (not isinstance(reply, str) or not reply or len(reply) > 128))
            or not isinstance(forwarded, bool)):
        result['structure'] = 'invalid'
        return result
    return dict(text=text, structure='known', reply_to=reply,
                mentions=sorted(set(mentions)), forwarded=forwarded)


def extract_interactions(db_path, identifier, mode='group', start='', end='', max_rows=None, cipher_config=None, source_format='nt', *, force_encrypted=False, scratch_dir=None, cancelled=lambda: False, progress=lambda n,t: None, sink=None):
    """Extract all message types in the selected period; no silent truncation.

    Numeric NT columns are read conservatively. 40001 is used as an optional
    opaque message key; a fixture/import envelope is not a claimed native decoder.
    """
    if mode not in ('group', 'c2c') or source_format not in ('nt', 'tei'):
        raise ValueError('Invalid mode')
    identifier = str(int(identifier))
    lo, hi = date_bounds(start, end)
    with open_database(db_path, force_encrypted=force_encrypted, config=cipher_config,
                       scratch_dir=scratch_dir, cancelled=cancelled, progress=progress) as (conn, encrypted, access):
        query = conn.query
        table = 'tei_messages_v1' if source_format == 'tei' else ('group_msg_table' if mode == 'group' else 'c2c_msg_table')
        columns = {str(r['name']) for r in query(f'PRAGMA table_info({table})')}
        required = ({'sender_id','peer_id','message_id','timestamp','conversation_id','mode','content','metadata'} if source_format == 'tei'
                    else {'40033', '40050', '40800', '40027' if mode == 'group' else '40030'})
        if not required <= columns:
            raise DatabaseError('schema')
        params = []
        def bind(value):
            if encrypted:
                return str(int(value))
            params.append(int(value))
            return '?'
        if source_format == 'tei':
            # Explicit user-selected import schema: metadata is a separate column,
            # never inferred from message content. conversation_id is the group or
            # owner ID; peer_id identifies the private thread.
            where = 'conversation_id = ' + bind(identifier) + " AND mode = '" + mode + "'"
        elif mode == 'group':
            where = '"40027" = ' + bind(identifier)
        else:
            # Standard local peer IDs identify the other participant. This also
            # includes receive-only threads; malformed sender/peer rows are flagged.
            where = '("40033" = ' + bind(identifier) + ' OR "40033" = "40030")'
        tscol = 'timestamp' if source_format == 'tei' else '"40050"'
        if lo is not None:
            where += ' AND ' + tscol + ' >= ' + bind(lo)
        if hi is not None:
            where += ' AND ' + tscol + ' < ' + bind(hi)
        msgid = 'CAST("40001" AS TEXT)' if '40001' in columns else 'NULL'
        peer = 'CAST("40030" AS TEXT)' if mode == 'c2c' else "''"
        sendercol, contentcol = '"40033"', '"40800"'
        metacol = 'NULL'
        if source_format == 'tei':
            msgid, peer, sendercol, contentcol, metacol = 'CAST(message_id AS TEXT)', 'CAST(peer_id AS TEXT)', 'sender_id', 'content', 'metadata'
        msgid = f'CASE WHEN length(CAST({msgid} AS BLOB)) <= 128 THEN {msgid} ELSE NULL END'
        peer = f'CASE WHEN length(CAST({peer} AS BLOB)) <= 32 THEN {peer} ELSE NULL END'
        sendercol = f'CASE WHEN length(CAST({sendercol} AS BLOB)) <= 32 THEN {sendercol} ELSE NULL END'
        size = f'length(CAST({contentcol} AS BLOB))'
        metasize = f'length(CAST({metacol} AS BLOB))'
        # Existing legacy adapter defines 40033 -> 40093 as sender -> QQ nickname.
        # Read only when the explicitly local GUI requests identities, never from message bodies.
        namecol = 'NULL'
        if sink is not None and getattr(sink, 'load_names', False) and source_format == 'nt' and '40093' in columns:
            namecol = 'CASE WHEN typeof("40093")=\'text\' AND length(CAST("40093" AS BLOB))<=320 THEN "40093" ELSE NULL END'
        # hex preserves arbitrary payload bytes through SQLite/SQLCipher adapters.
        sql = (f'SELECT CAST({sendercol} AS TEXT) AS sender_id, {peer} AS peer_id, '
               f'{msgid} AS message_id, CASE WHEN length(CAST({tscol} AS BLOB)) <= 32 THEN {tscol} ELSE NULL END AS timestamp, '
               f'{namecol} AS local_nickname, typeof({contentcol}) AS payload_type, {size} AS payload_length, '
               f'CASE WHEN {size} <= {MAX_PAYLOAD} THEN hex({contentcol}) ELSE NULL END AS payload_hex, '
               f'{metasize} AS metadata_length, CASE WHEN {metasize} <= {MAX_PAYLOAD} THEN {metacol} ELSE NULL END AS metadata '
               f'FROM {table} WHERE {where}')
        total = int(query(f'SELECT count(*) AS n FROM {table} WHERE {where}', tuple(params))[0]['n'])
        if max_rows is not None and total > max_rows:
            raise ValueError('Explicit caller record budget exceeded; no partial result was computed.')
        result = sink if sink is not None else []
        invalid_time = invalid_sender = native_rows = quote_rows = selected = payload_bytes = 0
        for index, row in enumerate(conn.iter_query(sql, tuple(params))):
            selected += 1
            if index % 1000 == 0:
                if cancelled(): raise DatabaseError('cancelled')
                progress(28 + int(16 * selected / max(total, 1)), f'Reading selected records: {selected:,} / {total:,}')
            if sink is None:
                payload_bytes += len(row['payload_hex'] or '') // 2 + len(row['metadata'] or '')
                if payload_bytes > MAX_TOTAL_PAYLOAD:
                    raise ValueError('In-memory API payload exceeds 32 MiB; use streaming analysis.')
            try:
                ts = datetime.fromtimestamp(float(row['timestamp']), SHANGHAI).replace(tzinfo=None)
            except (ValueError, TypeError, OverflowError, OSError):
                invalid_time += 1
                continue
            sender = str(row['sender_id'])
            peer_id = str(row['peer_id'])
            if not sender.isascii() or not sender.isdigit() or (mode == 'c2c' and (not peer_id.isascii() or not peer_id.isdigit() or peer_id == identifier or sender not in (identifier, peer_id))):
                invalid_sender += 1
                continue
            raw = None
            if row['payload_type'] == 'text' and row['payload_hex'] is not None:
                try:
                    raw = bytes.fromhex(row['payload_hex']).decode('utf-8')
                except (UnicodeError, ValueError):
                    pass
            payload = decode_payload(raw)
            if source_format == 'nt' and row['payload_type'] == 'blob' and row['payload_hex'] is not None:
                from native_payload import decode_native_text
                payload.update(decode_native_text(bytes.fromhex(row['payload_hex'])))
            if source_format == 'tei' and isinstance(row['metadata'], str) and raw is not None:
                try:
                    metadata = json.loads(row['metadata'])
                    payload = decode_metadata(metadata, payload)
                except (ValueError, RecursionError):
                    payload['structure'] = 'invalid'
            if source_format == 'tei' and (row['metadata_length'] or 0) > MAX_PAYLOAD:
                payload['structure'] = 'oversize'
            if row['payload_length'] is not None and int(row['payload_length']) > MAX_PAYLOAD:
                payload['structure'] = 'oversize'
            native_rows += payload.get('native_text_status') == 'parsed'
            quote_rows += bool(payload.get('native_quote_present', False))
            result.append(dict(sender=sender, peer=peer_id, timestamp=ts,
                               thread=('group:' + identifier) if mode == 'group' else ('c2c:' + peer_id),
                               message_id=str(row['message_id']) if row['message_id'] is not None else None,
                               order=index, local_nickname=row.get('local_nickname'), **payload))
        if sink is not None and getattr(sink, 'load_names', False):
            from contact_names import contact_names
            sink.local_names = contact_names(conn, encrypted, sink.users)
            for qq, (_, name, conflict) in sink.native_names.items():
                if name and not conflict: sink.local_names.setdefault(qq, name)
        return result, dict(selected_rows=selected, invalid_time=invalid_time, invalid_sender=invalid_sender,
                            start=start or None, end=end or None, timezone='Asia/Shanghai', truncated=False,
                            source=source_format, **access,
                            native_text_rows=native_rows, native_quote_rows=quote_rows)
